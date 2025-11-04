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

    scale_naip_patch(patch, q01, q99)
        → (scaled, nan_mask)
        Robust min–max scaling of NAIP imagery to [0,1] per band.

Design notes
    - Stateless: no global variables, no I/O, no Zarr dependencies.
    - Safe: guards against NaNs, degenerate stats, and missing quantiles.
    - Compatible: accepts both [krow,kcol,band] and [band,krow,kcol] NAIP layouts.

Conventions
    - MISS_ID = 0, UNK_ID = 1 are reserved and consistent across the repo.
    - Continuous normalization uses q01/q99 clipping, not min/max, to resist outliers.
    - Returned masks (e.g., nan_mask) are float arrays of 1.0 where invalid.

These routines define the mathematical contract for how features are transformed
between disk representation and model inputs.
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
