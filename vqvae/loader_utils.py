"""
utils/loader_utils.py
----------------------
Stateless feature encoding, normalization, and NAIP scaling helpers
used by `utils/loader.py`.

Purpose
    Provide pure, dependency-minimal utilities for converting raw feature
    vectors (from Zarr) into model-ready representations. These helpers are
    side-effect-free and deterministic, ensuring identical results for the
    same inputs across workers and epochs.

Used by
    - utils/loader.py (VQVAEDataset)
    - scripts/build_zarr.py for schema introspection or verification

Key functions
    build_cat_maps(feature_meta)
        → {feature_name: {raw_code: dense_id}}
        Dense IDs reserve 0=MISS, 1=UNK, observed codes begin at 2.

    encode_categorical_row(raw_vec, cat_indices, feature_names, cat_maps)
        → np.ndarray[int64], dense categorical vector per sample row.

    extract_cont_stats(feature_meta)
        → {feature_name: {min,max,mean,std,q01,q99}}
        Returns numeric summary stats used for normalization.

    norm_continuous_row(raw_vec, cont_indices, feature_names, cont_stats)
        → (z, nan_mask)
        Clips each value to [q01,q99], applies z-score (mean/std), replaces NaN with 0.

    denorm_continuous_row(z_vec, cont_indices, feature_names, cont_stats)
        → x_raw approximation from z-scores (inverse of z-score step; unclipped).

    scale_naip_patch(patch, q01, q99)
        → (scaled, nan_mask)
        Robust min–max scaling of NAIP imagery to [0,1] per band.

    build_id_maps_inverse(cat_maps)
        → {feature_name: {dense_id: raw_code}} for observed codes (ids ≥ 2).

    decode_categorical_row(dense_ids, cat_indices, feature_names, id_maps_inv)
        → float array of raw codes with NaN for MISS/UNK by default.

    decode_categorical_batch(dense_ids_2d, ...)
        → vectorized inverse for [T, C_cat].

Design notes
    - Stateless: no global I/O or Zarr deps.
    - Safe: guards against NaNs, degenerate stats, and missing quantiles.
    - Conventions: MISS_ID=0, UNK_ID=1; masks are float arrays of 1.0 where invalid.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

MISS_ID, UNK_ID = 0, 1  # reserved for categorical encodings

# ----------------------- CATEGORICAL ENCODING ---------------------------
def build_cat_maps(feature_meta: Dict) -> Dict[str, Dict[int, int]]:
    """
    Build code->dense-id maps per categorical feature from feature_meta.
    Dense IDs: MISS=0, UNK=1, observed codes → 2..(2+K-1).
    """
    maps: Dict[str, Dict[int, int]] = {}
    for f in feature_meta.get("features", []):
        if f.get("kind") != "cat":
            continue
        name = f["name"]
        classes = f.get("classes", [])
        code2id = {}
        next_id = 2
        for c in sorted(classes, key=lambda d: d["code"]):
            code2id[int(c["code"])] = next_id
            next_id += 1
        maps[name] = code2id
    return maps


def encode_categorical_row(
    raw_vec: np.ndarray,
    cat_indices: List[int],
    feature_names: List[str],
    cat_maps: Dict[str, Dict[int, int]],
) -> np.ndarray:
    """
    Convert raw integer codes (possibly NaN) into dense IDs with MISS/UNK.
    raw_vec: shape [F] (all features for a single (time,y,x) row)
    Returns array shape [C_cat] int64.
    """
    out = []
    for i in cat_indices:
        name = feature_names[i]
        v = raw_vec[i]
        if not np.isfinite(v):
            out.append(MISS_ID)
            continue
        code = int(v)
        m = cat_maps.get(name, {})
        out.append(m.get(code, UNK_ID))
    return np.asarray(out, dtype=np.int64)


# ----------------------- CATEGORICAL DECODING (inverse) ----------------
def build_id_maps_inverse(cat_maps: Dict[str, Dict[int, int]]) -> Dict[str, Dict[int, int]]:
    """
    Invert code->dense-id maps to dense-id->code for each categorical feature.
    Only observed codes (ids >= 2) are invertible. MISS/UNK handled by caller.

    Returns:
        { feature_name: {dense_id: raw_code, ...}, ... }
    """
    inv: Dict[str, Dict[int, int]] = {}
    for name, code2id in cat_maps.items():
        inv[name] = {dense_id: raw_code for raw_code, dense_id in code2id.items() if dense_id >= 2}
    return inv


def decode_categorical_row(
    dense_ids: np.ndarray,             # shape [C_cat], values in {0=MISS,1=UNK,>=2 observed}
    cat_indices: List[int],
    feature_names: List[str],
    id_maps_inv: Dict[str, Dict[int, int]],
    *,
    miss_value: float = np.nan,        # value to emit for MISS (0)
    unk_value: float = np.nan          # value to emit for UNK (1)
) -> np.ndarray:
    """
    Convert dense IDs back to raw categorical codes for a single row.
    Returns float array (so NaN is representable). Cast to int after handling NaNs if needed.
    """
    out = []
    for j, i in enumerate(cat_indices):
        name = feature_names[i]
        did = int(dense_ids[j])
        if did == MISS_ID:
            out.append(miss_value)
        elif did == UNK_ID:
            out.append(unk_value)
        else:
            code = id_maps_inv.get(name, {}).get(did, unk_value)
            out.append(float(code) if code is not np.nan else np.nan)
    return np.asarray(out, dtype=np.float32)


def decode_categorical_batch(
    dense_ids_2d: np.ndarray,          # shape [T, C_cat]
    cat_indices: List[int],
    feature_names: List[str],
    id_maps_inv: Dict[str, Dict[int, int]],
    *,
    miss_value: float = np.nan,
    unk_value: float = np.nan
) -> np.ndarray:
    """
    Vectorized inverse for a [T, C_cat] matrix of dense IDs.
    Returns float32 array shape [T, C_cat] with NaN for MISS/UNK by default.
    """
    T, C = dense_ids_2d.shape
    out = np.empty((T, C), dtype=np.float32)
    for t in range(T):
        out[t] = decode_categorical_row(
            dense_ids_2d[t], cat_indices, feature_names, id_maps_inv,
            miss_value=miss_value, unk_value=unk_value
        )
    return out


# ----------------------- CONTINUOUS NORMALIZATION ----------------------
def extract_cont_stats(feature_meta: Dict) -> Dict[str, Dict[str, float]]:
    """
    Pull per-feature stats for continuous vars from feature_meta.
    Returns { feature_name: {min,max,mean,std,q01,q99} }
    """
    out = {}
    for f in feature_meta.get("features", []):
        if f.get("kind") != "int":
            continue
        name = f["name"]
        out[name] = f.get("stats", {})
    return out


def norm_continuous_row(
    raw_vec: np.ndarray,
    cont_indices: List[int],
    feature_names: List[str],
    cont_stats: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust z-score using clip to [q01, q99], then (x - mean) / std.
    NaNs are replaced with 0 after normalization; a mask marks NaNs.
    Returns (z, nan_mask) where shapes are [C_cont].
    """
    z_list, nan_mask = [], []
    for i in cont_indices:
        name = feature_names[i]
        s = cont_stats.get(name, {})
        x = raw_vec[i]
        if not np.isfinite(x):
            z_list.append(0.0)
            nan_mask.append(1.0)
            continue
        lo = s.get("q01")
        hi = s.get("q99")
        mu = s.get("mean")
        sd = s.get("std")
        if lo is not None and hi is not None:
            x = min(max(float(x), float(lo)), float(hi))
        if mu is None or sd is None or sd == 0:
            z = 0.0
        else:
            z = (float(x) - float(mu)) / (float(sd) + 1e-6)
        z_list.append(z)
        nan_mask.append(0.0)
    return np.asarray(z_list, dtype=np.float32), np.asarray(nan_mask, dtype=np.float32)


def denorm_continuous_row(
    z_vec: np.ndarray,
    cont_indices: List[int],
    feature_names: List[str],
    cont_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    Inverse of z-score step: x ≈ z * std + mean, without re-applying clip.
    Useful for mapping normalized values back to original units for reporting.
    Returns array shape [C_cont].
    """
    x_list = []
    for i in cont_indices:
        name = feature_names[i]
        s = cont_stats.get(name, {})
        mu = float(s.get("mean", 0.0) or 0.0)
        sd = float(s.get("std", 1.0) or 1.0)
        x = float(z_vec[i]) * (sd if sd != 0 else 1.0) + mu
        x_list.append(x)
    return np.asarray(x_list, dtype=np.float32)


# ----------------------- NAIP SCALING ----------------------------------
def scale_naip_patch(
    patch: np.ndarray,  # shape [krow, kcol, band] or [band, krow, kcol]
    q01: List[float],
    q99: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust min-max scaling to [0,1] per band using q01/q99.
    Accepts patch in either [krow,kcol,band] or [band,krow,kcol].
    NaNs -> 0 after scaling; mask=1 where NaN.
    Returns (scaled, nan_mask) both with shape [band, krow, kcol].
    """
    arr = patch
    if arr.ndim != 3:
        raise ValueError(f"NAIP patch must be 3D, got {arr.shape}")
    # Reorder to [band,krow,kcol] if needed
    if arr.shape[0] != len(q01) and arr.shape[-1] == len(q01):
        arr = np.moveaxis(arr, -1, 0)  # [band,krow,kcol]
    bands, krow, kcol = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)
    nan_mask = np.zeros_like(arr, dtype=np.float32)
    for b in range(bands):
        x = arr[b]
        mask = ~np.isfinite(x)
        lo = float(q01[b]) if b < len(q01) and q01[b] is not None else np.nan
        hi = float(q99[b]) if b < len(q99) and q99[b] is not None else np.nan
        y = x.copy()
        if np.isfinite(lo):
            y = np.maximum(y, lo)
        if np.isfinite(hi):
            y = np.minimum(y, hi)
        denom = (hi - lo) if (np.isfinite(hi) and np.isfinite(lo) and hi > lo) else 1.0
        y = (y - (lo if np.isfinite(lo) else 0.0)) / (denom + 1e-6)
        y[mask] = 0.0
        out[b] = y.astype(np.float32, copy=False)
        nan_mask[b] = mask.astype(np.float32, copy=False)
    return out, nan_mask


# -----------------------------------------------------------------------------
# VQ-VAE postprocessing utilities
# -----------------------------------------------------------------------------

import torch
import numpy as np
from vqvae.loader_utils import denorm_continuous_row  # ← use the project helper

def decode_codebook_to_original_units(model, ds, device="cuda"):
    """
    Returns:
      cont_orig : [K, C_cont] float32 (denormalized to original units) or None if no continuous features
      cat_raw   : [K, C_cat] float32 (raw categorical codes; NaN for MISS/UNK if not present in map)
      canopy    : [K] float32 (unchanged scale)
    """
    model = model.to(device).eval()

    # 1) Extract codebook entries (works for EMA/ST quantizers since both expose .codebook)
    codes = model.quant.codebook.detach().to(device)             # [K, D]  
    zq = codes.unsqueeze(1)                                      # [K, 1, D]

    # 2) Decode each code as a 1-step sequence
    with torch.inference_mode():
        cont_pred, cat_logits, canopy = model.decoder(zq)        # cont:[K,1,C_cont], dict(name->[K,1,V]), canopy:[K]  

    # 3a) Continuous → original scale via denorm_continuous_row (row-by-row on NumPy)
    cont_orig = None
    if cont_pred is not None and cont_pred.shape[-1] > 0:
        cont_pred_np = cont_pred.squeeze(1).cpu().numpy()        # [K, C_cont]
        # denorm_continuous_row expects a single row; call it per code index   
        cont_indices   = list(range(len(ds.cont_names)))         # column order = ds.cont_names  
        feature_names  = ds.cont_names
        cont_stats     = ds.cont_stats                           # {name: {mean,std,q01,q99,...}}  
        cont_orig = np.stack([
            denorm_continuous_row(cont_pred_np[k], cont_indices, feature_names, cont_stats)
            for k in range(cont_pred_np.shape[0])
        ]).astype(np.float32)

    # 3b) Categorical logits → dense IDs → raw codes (invert ds.cat_maps)
    cat_raw = np.zeros((codes.shape[0], len(ds.cat_names)), dtype=np.float32) if ds.cat_names else np.zeros((codes.shape[0], 0), np.float32)
    if ds.cat_names:
        # dense IDs per categorical feature (one column per name, K rows)
        dense_cols = []
        for name in ds.cat_names:
            dense = cat_logits[name].argmax(dim=-1).squeeze(1).cpu().numpy().astype(np.int64)  # [K]
            dense_cols.append(dense)
        dense_ids_2d = np.stack(dense_cols, axis=1)  # [K, C_cat]

        # build inverse maps: dense_id -> raw_code (only observed codes; 0/1 = MISS/UNK)
        id_maps_inv = {name: {did: raw for raw, did in ds.cat_maps.get(name, {}).items()} for name in ds.cat_names}

        # vectorized decode over rows (K “time steps”); MISS/UNK become NaN by default
        # This mirrors the project’s batch decoder pattern. 
        T, C = dense_ids_2d.shape
        out = np.empty((T, C), dtype=np.float32)
        MISS_ID, UNK_ID = 0, 1
        for t in range(T):
            row = dense_ids_2d[t]
            decoded = []
            for j, name in enumerate(ds.cat_names):
                did = int(row[j])
                if did == MISS_ID or did == UNK_ID:
                    decoded.append(np.nan)
                else:
                    decoded.append(float(id_maps_inv.get(name, {}).get(did, np.nan)))
            out[t] = np.asarray(decoded, dtype=np.float32)
        cat_raw = out

    return cont_orig, cat_raw, canopy.squeeze(-1).cpu().numpy() if canopy.ndim > 1 else canopy.cpu().numpy()
