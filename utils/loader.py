#!/usr/bin/env python3
# =============================================================================
# utils/loader.py — Zarr → PyTorch Dataset (trainer-compatible)
#
# - Robust scaling for continuous (clip to [q01,q99] then z-score)
# - Robust min-max for NAIP per band using q01/q99 → [0,1]
# - Dense IDs for categorical: MISS=0, UNK=1, observed codes → 2...
# - Emits masks: cont_nan_mask, naip_nan_mask
# - Emits cat_target with IGNORE_INDEX for loss masking
#
# Trainer expectations satisfied:
#   - ctor: VQVAEDataset(zarr_path, schema_path:str|None, eager:bool, ignore_unk_in_loss:bool)
#   - attributes: cat_names, cont_names, schema_cat, naip
#   - method: class_weights_by_cat_name(name)->Tensor
#   - collate: default_collate_fn
#   - const: IGNORE_INDEX
# =============================================================================

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

from .loader_utils import (
    MISS_ID, UNK_ID,
    build_cat_maps,
    extract_cont_stats,
    encode_categorical_row,
    norm_continuous_row,
    scale_naip_patch,
)

IGNORE_INDEX = -100  # used by CE to ignore positions


class VQVAEDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        schema_path: Optional[str] = None,        # accepted for trainer compatibility
        *,
        eager: bool = False,
        ignore_unk_in_loss: bool = True,          # mark UNK as IGNORE_INDEX in cat_target
    ) -> None:
        # open lazily
        self.ds = xr.open_zarr(zarr_path, consolidated=True)

        # parse feature meta
        raw = self.ds.attrs.get("feature_meta", "{}")
        self.feature_meta = json.loads(raw) if isinstance(raw, str) else raw

        # feature axis descriptors
        self.feature_names: List[str] = [str(v) for v in self.ds["feature_names"].values.tolist()]
        self.feature_kinds: List[str] = [str(v) for v in self.ds["feature_kinds"].values.tolist()]

        self.cont_idx = [i for i, k in enumerate(self.feature_kinds) if k == "int"]
        self.cat_idx  = [i for i, k in enumerate(self.feature_kinds) if k == "cat"]

        self.cont_names = [self.feature_names[i] for i in self.cont_idx]
        self.cat_names  = [self.feature_names[i] for i in self.cat_idx]

        # stats/maps
        self.cont_stats = extract_cont_stats(self.feature_meta)      # {name: {min,max,mean,std,q01,q99}}
        self.cat_maps   = build_cat_maps(self.feature_meta)          # {name: {raw_code: dense_id}}

        # vocab sizes per categorical feature (include MISS/UNK)
        self.schema_cat: Dict[str, Dict[str, int]] = {}
        for name in self.cat_names:
            observed = self.cat_maps.get(name, {})
            num_ids = 2 + len(observed)  # MISS(0), UNK(1), then observed codes
            self.schema_cat[name] = {"num_ids": int(num_ids)}

        # NAIP attrs (q01/q99)
        self.has_naip = ("naip_patch" in self.ds.data_vars)
        if self.has_naip:
            na = self.ds["naip_patch"].attrs
            self.naip_q01 = na.get("q01", [])
            self.naip_q99 = na.get("q99", [])
            self.krow = int(self.ds.sizes["krow"])
            self.kcol = int(self.ds.sizes["kcol"])
            self.bands = int(self.ds.sizes["band"])

        # valid pixel index list (mask==1)
        mask = self.ds["mask"].values if eager else self.ds["mask"].data
        mask_np = np.asarray(mask) if isinstance(mask, np.ndarray) else mask.compute()
        ys, xs = np.where(mask_np == 1)
        self.xy: List[Tuple[int, int]] = list(zip(ys.tolist(), xs.tolist()))

        # sizes
        self.T = int(self.ds.sizes["time"])
        self.F = int(self.ds.sizes["feature"])

        # training behavior flags
        self.ignore_unk_in_loss = bool(ignore_unk_in_loss)

        # expose for trainer convenience
        if self.has_naip:
            self.naip = self.ds["naip_patch"]  # so code can do ds.naip.shape[-1] to get bands
        else:
            self.naip = None

        # optional schema ingest (not required for this loader to function)
        self.schema_path = schema_path
        
        # ---- chunk bucketing for locality-aware sampling ----
        # attrs_raw is [time, y, x, feature]; chunking is per-axis (ragged at edges)
        arr = self.ds["attrs_raw"].data  # dask array
        # dask chunks is a tuple per axis: (chunks_time, chunks_y, chunks_x, chunks_feature)
        chunks = getattr(arr, "chunks", None)
        if chunks is None:
          # fall back: assume single chunk covering full axis if not chunked
          y_chunk_sizes = (self.ds.sizes["y"],)
          x_chunk_sizes = (self.ds.sizes["x"],)
        else:
          _, y_chunk_sizes, x_chunk_sizes, _ = chunks

        from .chunking import compute_xy_chunks, pack_xy_by_chunk
        xy_np = np.array(self.xy, dtype=np.int64)  # shape [N,2] (y,x)
        y_bins, x_bins = compute_xy_chunks(xy_np, y_chunk_sizes, x_chunk_sizes)
        self.n_chunks_y = len(y_chunk_sizes)
        self.n_chunks_x = len(x_chunk_sizes)
        self.xy_by_chunk = pack_xy_by_chunk(
          xy_np, y_bins, x_bins, self.n_chunks_y, self.n_chunks_x
        )


    # -------------- class weights for CE (per-cat feature) ---------------
    def class_weights_by_cat_name(self, name: str) -> torch.Tensor:
        """
        Build simple inverse-frequency weights from feature_meta counts.
        ids: 0=MISS,1=UNK, then observed codes as per dense mapping.
        For CE with ignore_index set to IGNORE_INDEX, MISS/UNK weights are moot,
        but we still return a vector of correct length for convenience.
        """
        # default uniform
        num_ids = int(self.schema_cat[name]["num_ids"])
        w = np.ones((num_ids,), dtype=np.float32)

        # try to use counts if present
        feat = next((f for f in self.feature_meta.get("features", []) if f.get("name") == name and f.get("kind") == "cat"), None)
        if feat and "classes" in feat and feat["classes"]:
            # gather counts for observed codes
            code_counts = {int(c["code"]): int(c.get("count", 0)) for c in feat["classes"]}
            total = sum(max(1, c) for c in code_counts.values())
            # inverse frequency
            for raw_code, cnt in code_counts.items():
                dense_id = self.cat_maps.get(name, {}).get(raw_code, None)
                if dense_id is None or dense_id < 2 or dense_id >= num_ids:
                    continue
                freq = max(1.0, float(cnt))
                w[dense_id] = total / freq
            # normalize to mean=1 for stability
            w = w / max(1e-6, w.mean())

        return torch.tensor(w, dtype=torch.float32)

    # ------------------------------- dunder -------------------------------
    def __len__(self) -> int:
        return len(self.xy)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        y, x = self.xy[idx]

        # attrs_raw slice: [time, feature]
        arr = self.ds["attrs_raw"][:, y, x, :].values  # small vector, loads quickly

        # split arrays [T, C_*]
        if self.cont_idx:
            cont_raw = arr[:, self.cont_idx]
        else:
            cont_raw = np.empty((self.T, 0), np.float32)

        if self.cat_idx:
            cat_raw = arr[:, self.cat_idx]
        else:
            cat_raw = np.empty((self.T, 0), np.float32)

        # per-time processing
        cont_norm_list, cont_nan_mask_list, cat_ids_list = [], [], []
        for t in range(self.T):
            full_row = arr[t]
            # continuous robust z
            z_t, nanmask_t = norm_continuous_row(
                full_row,
                self.cont_idx,
                self.feature_names,
                self.cont_stats,
            )
            cont_norm_list.append(z_t)
            cont_nan_mask_list.append(nanmask_t)

            # categorical dense ids
            if self.cat_idx:
                ids_t = encode_categorical_row(
                    full_row,
                    self.cat_idx,
                    self.feature_names,
                    self.cat_maps,
                )
            else:
                ids_t = np.empty((0,), np.int64)
            cat_ids_list.append(ids_t)

        cont = np.stack(cont_norm_list, axis=0).astype(np.float32)          # [T, C_cont]
        cont_nan_mask = np.stack(cont_nan_mask_list, axis=0).astype(np.float32)
        cat_ids = np.stack(cat_ids_list, axis=0).astype(np.int64)           # [T, C_cat]

        # cat_target with IGNORE_INDEX where MISS/UNK
        cat_target = cat_ids.copy()
        miss_mask = (cat_target == MISS_ID)
        cat_target[miss_mask] = IGNORE_INDEX
        if self.ignore_unk_in_loss:
            unk_mask = (cat_target == UNK_ID)
            cat_target[unk_mask] = IGNORE_INDEX

        # NAIP patch → [bands,krow,kcol] float32 in [0,1] + mask
        if self.has_naip:
            patch = self.ds["naip_patch"][y, x].values  # [krow,kcol,band] or [band,krow,kcol]
            naip_scaled, naip_nan_mask = scale_naip_patch(
                patch, self.naip_q01, self.naip_q99
            )  # both [bands,krow,kcol]
        else:
            naip_scaled = np.zeros((0, 1, 1), dtype=np.float32)
            naip_nan_mask = np.zeros((0, 1, 1), dtype=np.float32)

        # years vector
        years = self.ds["years"].values.astype(np.int16)

        # pack tensors
        out: Dict[str, torch.Tensor] = {
            "cont": torch.from_numpy(cont),                              # [T,C_cont]
            "cont_nan_mask": torch.from_numpy(cont_nan_mask),            # [T,C_cont]
            "cat": torch.from_numpy(cat_ids),                            # [T,C_cat]
            "cat_target": torch.from_numpy(cat_target),                  # [T,C_cat]
            "naip": torch.from_numpy(naip_scaled.astype(np.float32)),    # [bands,krow,kcol]
            "naip_nan_mask": torch.from_numpy(naip_nan_mask.astype(np.float32)),
            "years": torch.from_numpy(years),                            # [T]
            "yx": torch.tensor([y, x], dtype=torch.int32),
        }

        return out


def default_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack dict-of-tensors along batch dimension. Keep 'years' from the first sample.
    """
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        if k == "years":
            out[k] = batch[0][k]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
