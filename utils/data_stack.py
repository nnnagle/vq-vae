"""
utils/data_stack.py
-------------
Builds feature stacks and metadata for the raster → Zarr training pipeline.

Purpose
    Provides the data-layer assembly functions that read per-feature rasters,
    validate cross-year consistency, and construct lazy Dask/xarray arrays
    suitable for downstream VQ-VAE training. Handles ragged chunks, mixed
    feature types (continuous and categorical), and metadata summarization.

Used by
    - build_zarr.py : orchestrates the cube creation using stack_attrs_raw_spatial()
    - train_vqvae.py : consumes the resulting Zarr archive

Design notes
    * Indexes input rasters via CSV descriptors; tolerant to header aliases.
    * Enforces consistent feature identity and order across all years.
    * Uses Dask-lazy operations for large rasters — no full in-memory load.
    * Computes per-feature statistics for schema and normalization.
    * Treats chunk boundaries explicitly to avoid misaligned samples.

Assistant guidance
    When extending:
        - Preserve idempotency of CSV parsing and array construction.
        - Do not eagerly compute Dask arrays; return lazily-evaluated graphs.
        - Keep feature kind semantics stable ("int" = continuous, "cat" = categorical).
"""

from typing import List, Dict, Tuple, Optional
import os
import csv
import re
import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed

from utils.log import log, warn, fail, ensure
from utils.raster_ops import read_into_mask_grid

# ----------------------------------------------------------------------
# Time selection: Select all years necessary for end_years and window_len
# ----------------------------------------------------------------------
def select_years(end_years: List[int], window_len: int) -> List[int]:
    years = set()
    for e in end_years:
        for y in range(e - window_len + 1, e + 1):
            years.add(int(y))
    out = sorted(years)
    ensure(len(out) > 0, "No years selected.")
    return out

# ----------------------------------------------------------------------
# Robust CSV reader for features list
# ----------------------------------------------------------------------
_HDR_ALIASES = {
    "year": {"year", "yr"},
    "kind": {"kind", "type"},
    "path": {"file_path", "filepath", "path"},
    "fid":  {"fid", "feature", "feature_id", "name"},
}

def _normalize_header(name: str) -> str:
    n = name.strip().lower()
    for key, alts in _HDR_ALIASES.items():
        if n in alts:
            return key
    return n

def _is_int(s: str) -> bool:
    try:
        int(s.strip())
        return True
    except Exception:
        return False

def _is_kind(s: str) -> bool:
    return s.strip().lower() in ("int", "cat")

def _is_path(s: str) -> bool:
    s = s.strip()
    return ("/" in s or "\\" in s or "." in os.path.basename(s))

def _stem_from_path(p: str) -> str:
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    return stem

def _sniff_dialect(f):
    head = f.read(4096)
    f.seek(0)
    try:
        return csv.Sniffer().sniff(head, delimiters=",\t; ")
    except Exception:
        # default to comma, but reader below tolerates tabs via split fallback
        class _D: delimiter = ","
        return _D()

def _iter_rows_loose(path: str):
    """
    Yield rows as lists of strings, handling comments and blank lines.
    Accepts comma, tab, or whitespace separated values.
    """
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # If comma or tab present, split on those; else split on whitespace
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            elif "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
            else:
                parts = re.split(r"\s+", line)
            yield parts

def index_inputs(features_csv: str, needed_years: List[int]) -> Dict[int, List[Dict]]:
    """
    Accepts both headered and headerless files.

    Headered:
      columns may be named with aliases:
        year|yr, kind|type, file_path|filepath|path, fid|feature|feature_id|name

    Headerless:
      3 or 4 columns in any order. We detect:
        - year: the value that parses as integer
        - kind: 'int' or 'cat' (case-insensitive)
        - path: looks like a file path (contains / or \ or has an extension)
        - fid: remaining column (optional). If missing, we derive from file stem.
    """
    # First, try headered parsing by peeking the first non-comment line
    # If recognizable headers exist, use DictReader. Otherwise, fall back to positional.
    # We implement our own tolerant header parsing instead of csv.DictReader directly.
    # Peek first non-comment line
    first_data_line = None
    with open(features_csv, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            first_data_line = s
            break

    ensure(first_data_line is not None, f"Empty features file: {features_csv}")

    def _parse_header_line(s: str) -> Optional[List[str]]:
        # Heuristic: if it contains any of our header aliases, treat as header
        candidates = re.split(r"[,\t\s]+", s.strip())
        norm = [_normalize_header(c) for c in candidates]
        has_yearish = any(c in _HDR_ALIASES["year"] or c == "year" for c in candidates)
        return candidates if has_yearish else None

    header_fields = _parse_header_line(first_data_line)

    y2f: Dict[int, List[Dict]] = {y: [] for y in needed_years}

    if header_fields is not None:
        # Headered route
        with open(features_csv, "r", encoding="utf-8") as f:
            dialect = _sniff_dialect(f)
            rdr = csv.reader(f, dialect)
            raw_header = next(rdr)
            header = [_normalize_header(h) for h in raw_header]

            # Build index map
            def _idx(name: str) -> Optional[int]:
                try:
                    return header.index(name)
                except ValueError:
                    return None

            yi = _idx("year"); ki = _idx("kind"); pi = _idx("path"); fi = _idx("fid")
            ensure(yi is not None and ki is not None and pi is not None,
                   f"Header must include year/kind/path (aliases ok). Got: {raw_header}")

            for row in rdr:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                # len(row) may be shorter than header if trailing blanks; pad
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                try:
                    y = int(row[yi])
                except Exception:
                    warn("Skipping row with non-integer year: %s", row)
                    continue
                kind = row[ki].strip().lower()
                ensure(kind in {"int", "cat"}, f"Unknown kind '{kind}' in row: {row}")
                path = row[pi].strip()
                ensure(path != "", f"Empty path in row: {row}")
                fid = row[fi].strip() if fi is not None and row[fi].strip() else _stem_from_path(path)
                if y in y2f:
                    y2f[y].append({"fid": fid, "kind": kind, "path": path})
    else:
        # Headerless positional route
        for parts in _iter_rows_loose(features_csv):
            # Ignore obvious header-like lines that slipped through
            if any(tok.lower() in ("year", "yr", "type", "kind") for tok in parts):
                continue

            # Identify columns
            year_idx = next((i for i, p in enumerate(parts) if _is_int(p)), None)
            kind_idx = next((i for i, p in enumerate(parts) if _is_kind(p)), None)
            path_idx = next((i for i, p in enumerate(parts) if _is_path(p)), None)

            ensure(year_idx is not None and kind_idx is not None and path_idx is not None,
                   f"Could not infer columns in row: {parts}")

            y = int(parts[year_idx])
            kind = parts[kind_idx].strip().lower()
            ensure(kind in {"int", "cat"}, f"Unknown kind '{kind}' in row: {parts}")
            path = parts[path_idx].strip()

            # fid is the remaining column if present; else derive
            remaining = [i for i in range(len(parts)) if i not in (year_idx, kind_idx, path_idx)]
            if len(remaining) >= 1:
                fid = parts[remaining[0]].strip()
                if fid == "" or _is_int(fid) or _is_kind(fid) or _is_path(fid):
                    fid = _stem_from_path(path)
            else:
                fid = _stem_from_path(path)

            if y in y2f:
                y2f[y].append({"fid": fid, "kind": kind, "path": path})

    # Stable per-year ordering by fid
    for y in y2f:
        y2f[y] = sorted(y2f[y], key=lambda r: r["fid"])
    # Sanity
    for y in needed_years:
        ensure(len(y2f[y]) > 0, f"No features found for year={y}. Check {features_csv}.")
    return y2f

def enforce_consistent_features(y2f: Dict[int, List[Dict]], needed_years: List[int]) -> Tuple[List[str], List[str]]:
    base = y2f[needed_years[0]]
    base_fids  = [r["fid"]  for r in base]
    base_kinds = [r["kind"] for r in base]
    for y in needed_years:
        fids  = [r["fid"]  for r in y2f[y]]
        kinds = [r["kind"] for r in y2f[y]]
        ensure(fids == base_fids and kinds == base_kinds,
               f"Inconsistent features in year={y}. "
               f"Expected fids={base_fids} kinds={base_kinds} but got fids={fids} kinds={kinds}")
    return base_fids, base_kinds

# ----------------------------------------------------------------------
# Lazy stacking (Dask) so we never materialize the full raster cube
# ----------------------------------------------------------------------
def _delayed_read_feature(path: str,
                          mask_da: xr.DataArray,
                          template: Dict,
                          out_dtype) -> delayed:
    H = int(mask_da.sizes["y"])
    W = int(mask_da.sizes["x"])

    def _reader():
        arr = read_into_mask_grid(
            ds_path=path,
            mask_shape=(H, W),
            mask_bounds=template["bounds"],
            mask_transform=template["transform"],
            dtype=out_dtype,
        )
        return arr  # shape (H, W), dtype=out_dtype

    return delayed(_reader)()

def stack_attrs_raw_spatial(
    y2f: Dict[int, List[Dict]],
    needed_years: List[int],
    mask_da: xr.DataArray,
    template: Dict,
    chunks: Dict[str, int],
    out_dtype=np.float32,
) -> xr.DataArray:
    """
    Build a Dask-backed array attrs_raw(time,y,x,feature). No full materialization.
    """
    H = int(mask_da.sizes["y"])
    W = int(mask_da.sizes["x"])

    tf_blocks: List[da.Array] = []
    for y in needed_years:
        feat_blocks: List[da.Array] = []
        for rec in y2f[y]:
            d = _delayed_read_feature(rec["path"], mask_da, template, out_dtype)
            a = da.from_delayed(d, shape=(H, W), dtype=out_dtype)
            feat_blocks.append(a)  # (H,W)
        fstack = da.stack(feat_blocks, axis=0)  # (F,H,W)
        tf_blocks.append(fstack)

    data = da.stack(tf_blocks, axis=0).transpose(0, 2, 3, 1)  # (T,H,W,F)

    data = data.rechunk({
        0: chunks.get("time", 1),
        1: chunks.get("y", 512),
        2: chunks.get("x", 512),
        3: chunks.get("feature", 64),
    })

    da_xr = xr.DataArray(
        data,
        dims=("time", "y", "x", "feature"),
        name="attrs_raw",
    )
    return da_xr

# ----------------------------------------------------------------------
# Feature metadata with streaming reductions
# ----------------------------------------------------------------------
def compute_feature_metadata(
    attrs_raw: xr.DataArray,
    feature_names: List[str],
    feature_kinds: List[str],
    mask_da: xr.DataArray,
) -> Dict:
    """
    Compute metadata for each feature, masked by mask_da==1:
      - For continuous ('int'): min,max,mean,std, q01,q25,250,q75,q99 (all Dask reductions)
      - For categorical ('cat'): histogram via Dask (fixed bins 0..254)
    Returns a JSON-serializable dict. No full array materialization.
    """
    ensure(attrs_raw.dims == ("time", "y", "x", "feature"),
           "attrs_raw dims must be (time,y,x,feature)")

    meta: Dict = {"features": []}
    m = mask_da.astype(bool)

    for i, (name, kind) in enumerate(zip(feature_names, feature_kinds)):
        v = attrs_raw.isel(feature=i).where(m)  # (time,y,x)
        if kind == "int":
            q = v.quantile([0.01, 0.25, 0.50, 0.75, 0.99], dim=("time", "y", "x"), skipna=True)
            stats = {
                "min": float(v.min(dim=("time", "y", "x"), skipna=True).compute().values),
                "max": float(v.max(dim=("time", "y", "x"), skipna=True).compute().values),
                "mean": float(v.mean(dim=("time", "y", "x"), skipna=True).compute().values),
                "std": float(v.std(dim=("time", "y", "x"), skipna=True).compute().values),
                "q01": float(q.sel(quantile=0.01).compute().values),
                "q25": float(q.sel(quantile=0.25).compute().values),
                "q50": float(q.sel(quantile=0.50).compute().values),
                "q75": float(q.sel(quantile=0.75).compute().values),
                "q99": float(q.sel(quantile=0.99).compute().values),
            }
            meta["features"].append({"name": name, "kind": "int", "stats": stats})
        else:  # 'cat' — codes are < 255
            arr = v.data
            valid = da.isfinite(arr)
            clean = da.where(valid, arr, -1)
            edges = np.arange(-0.5, 255.5, 1.0, dtype=np.float32)  # bins for 0..254
            hist, _ = da.histogram(clean, bins=edges, range=(-0.5, 254.5))
            counts = hist.compute()
            classes = [{"code": int(k), "count": int(n)} for k, n in enumerate(counts) if int(n) > 0]
            meta["features"].append({"name": name, "kind": "cat", "classes": classes})

    return meta

def compute_naip_metadata(
    naip_patch: xr.DataArray,
    mask_da: xr.DataArray,
    *,
    reduce_dims: Tuple[str, ...] = ("y", "x", "krow", "kcol"),
    include_source: Optional[str] = None,
) -> Dict:
    """
    Compute NAIP per-band robust quantiles (q01/q99) and basic structure, masked by mask_da==1.
    Streaming/Dask-safe: no full-array materialization.

    Parameters
      naip_patch : xr.DataArray with dims including ("y","x","krow","kcol","band")
      mask_da    : xr.DataArray (y,x) binary mask; broadcast-aligned automatically
      reduce_dims: dims to reduce over when computing quantiles (default: spatial + patch)
      include_source: optional absolute path to embed as metadata

    Returns
      {
        "q01": [float,...],     # per-band
        "q99": [float,...],     # per-band
        "bands": int,
        "kshape": (int,int),
        "dtype": str,
        "source": str (optional)
      }
    """
    # Basic structure
    bands = int(naip_patch.sizes.get("band", 0))
    krow  = int(naip_patch.sizes.get("krow", 0))
    kcol  = int(naip_patch.sizes.get("kcol", 0))
    out = {
        "bands": bands,
        "kshape": (krow, kcol),
        "dtype": str(naip_patch.dtype),
    }
    if include_source:
        out["source"] = include_source

    # Mask and quantiles (Dask reductions; no full materialization)
    m = mask_da.astype(bool)
    v = naip_patch.where(m)  # (y,x,krow,kcol,band) masked; xarray will align dims

    q = v.quantile([0.01, 0.25, 0.50, 0.75, 0.99], dim=reduce_dims, skipna=True)  # -> (quantile, band)
    # Compute to scalars/lists; still band-wise, not full array
    q01 = q.sel(quantile=0.01).compute().values.tolist()
    q25 = q.sel(quantile=0.25).compute().values.tolist()
    q50 = q.sel(quantile=0.50).compute().values.tolist()
    q75 = q.sel(quantile=0.75).compute().values.tolist()
    q99 = q.sel(quantile=0.99).compute().values.tolist()

    out["q01"] = [float(x) if x is not None else None for x in q01]
    out["q25"] = [float(x) if x is not None else None for x in q25]
    out["q50"] = [float(x) if x is not None else None for x in q50]
    out["q75"] = [float(x) if x is not None else None for x in q75]
    out["q99"] = [float(x) if x is not None else None for x in q99]
    return out
