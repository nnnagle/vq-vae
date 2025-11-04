#!/usr/bin/env python3
# =============================================================================
# utils/zarr_info.py — Inspect a VQ-VAE Zarr feature cube
#
# FUNCTIONS (quick index)
#   open_zarr(path)                  -> xr.Dataset
#   list_features(ds)                -> [(feature_name, kind), ...]
#   list_years(ds)                   -> [year, ...]
#   feature_meta(ds)                 -> dict (parsed from ds.attrs["feature_meta"])
#   print_dataset_attrs(ds)          -> prints CRS/transform/bounds/resolution/windowing
#   naip_info(ds)                    -> {"present":bool, "shape":(y,x,krow,kcol,band), "kshape":(3,3), "bands":int}
#   naip_band_stats(ds)              -> [{"band":i,"min":...,"max":...,"mean":...,"std":...}, ...]
#   print_continuous_stats(meta, top=None)
#   print_categorical_stats(meta, top_classes=10, top_feats=None)
#   dump_meta(ds, path)              -> write raw feature_meta JSON
#   export_counts(ds, path)          -> write CSV: feature,code,count,prop (per categorical feature)
#   export_manifest(ds, path)        -> write JSON summary of shapes/names/kinds/NAIP/attrs
#   summarize_zarr(path, ...)        -> main printer that calls the above pieces
#
# CLI USAGE
#   python -m utils.zarr_info /path/to/cube.zarr
#   python -m utils.zarr_info /path/to/cube.zarr --meta
#   python -m utils.zarr_info /path/to/cube.zarr --naip-stats
#   python -m utils.zarr_info /path/to/cube.zarr --cont-stats
#   python -m utils.zarr_info /path/to/cube.zarr --cat-stats --top 10 --top-feats 5
#   # optional exports for training pipeline (no policy decisions made here):
#   python -m utils.zarr_info /path/to/cube.zarr --dump-meta feature_meta.json
#   python -m utils.zarr_info /path/to/cube.zarr --export-counts counts.csv
#   python -m utils.zarr_info /path/to/cube.zarr --export-manifest manifest.json
#
# NOTES
#   • This module *does not* choose “kept codes”, collapse categories, or build vocabularies.
#     That belongs in the data loader / training code, where epoch/batch/coverage decisions exist.
# =============================================================================

import argparse
import csv
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


# ----------------------------- Core I/O ---------------------------------
def open_zarr(path: str) -> xr.Dataset:
    """
    Open a consolidated Zarr dataset with xarray. Raises if path is invalid.
    Workflow: hand path to xarray → return Dataset handle (lazily loads arrays).
    """
    return xr.open_zarr(path, consolidated=True)


def list_features(ds: xr.Dataset):
    """
    Return [(feature_name, feature_kind), ...].
    Workflow: read the two 1D coords the builder writes: feature_names, feature_kinds.
    """
    names = ds["feature_names"].values.tolist()
    kinds = ds["feature_kinds"].values.tolist()
    return list(zip(names, kinds))


def list_years(ds: xr.Dataset):
    """
    Return the list of years on the time axis.
    Workflow: read the 1D 'years' variable provided by the builder.
    """
    return ds["years"].values.tolist()


def feature_meta(ds: xr.Dataset) -> dict:
    """
    Parse JSON from ds.attrs['feature_meta'] into a Python dict.
    Workflow: pull the string attr → json.loads → return {} if missing/invalid.
    """
    raw = ds.attrs.get("feature_meta", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def print_dataset_attrs(ds: xr.Dataset):
    """
    Print dataset-level georeferencing and build parameters for a quick sanity check.
    Workflow: fetch common attrs and print tersely.
    """
    a = ds.attrs
    print("Dataset attrs:")
    print(f"  created     : {a.get('created')}")
    print(f"  crs         : {a.get('crs')}")
    print(f"  transform   : {a.get('transform')}")
    print(f"  bounds      : {a.get('bounds')}")
    print(f"  resolution  : {a.get('resolution')}")
    print(f"  window_len  : {a.get('window_len')}")
    print(f"  end_years   : {a.get('end_years')}")


# --------------------------- NAIP reporting -----------------------------
def naip_info(ds: xr.Dataset):
    """
    Describe the NAIP patch array if present.
    Returns:
      {"present": bool, "shape": (y,x,krow,kcol,band), "kshape": (krow,kcol), "bands": int}
    Workflow: detect 'naip_patch', extract dims/sizes; otherwise mark present=False.
    """
    if "naip_patch" not in ds:
        return {"present": False}
    arr = ds["naip_patch"]
    shp = tuple(arr.shape)             # (y, x, krow, kcol, band)
    krow = arr.sizes.get("krow", None)
    kcol = arr.sizes.get("kcol", None)
    bands = arr.sizes.get("band", None)
    return {"present": True, "shape": shp, "kshape": (krow, kcol), "bands": bands}


def naip_band_stats(ds: xr.Dataset):
    """
    Compute per-band summary stats over NAIP (ignoring NaNs).
    Workflow: transpose to put 'band' first → flatten other axes → stats per band.
    """
    if "naip_patch" not in ds:
        return []
    arr = ds["naip_patch"].transpose("band", "y", "x", "krow", "kcol")
    a = arr.to_numpy()  # (B, Y, X, 3, 3)
    B = a.shape[0]
    flat = a.reshape(B, -1)
    out = []
    for b in range(B):
        v = flat[b]
        v = v[~np.isnan(v)]
        if v.size == 0:
            out.append({"band": int(b), "min": None, "max": None, "mean": None, "std": None})
        else:
            out.append({
                "band": int(b),
                "min": float(np.nanmin(v)),
                "max": float(np.nanmax(v)),
                "mean": float(np.nanmean(v)),
                "std":  float(np.nanstd(v)),
            })
    return out


# --------------------- Continuous & categorical summaries ----------------
def print_continuous_stats(meta: dict, top: Optional[int] = None):
    """
    Print per-feature continuous statistics from feature_meta.
    Workflow: iterate 'features' where kind=='int' and display the stat dict.
    """
    feats = [f for f in meta.get("features", []) if f.get("kind") == "int"]
    print("Continuous feature stats (mask==1):")
    for f in feats[:top] if top else feats:
        name = f.get("name")
        s = f.get("stats") or {}
        print(f"  - {name}: min={s.get('min')}, max={s.get('max')}, mean={s.get('mean')}, "
              f"std={s.get('std')}, q01={s.get('q01')}, q99={s.get('q99')}")


def print_categorical_stats(meta: dict, top_classes: int = 10, top_feats: Optional[int] = None):
    """
    Print a compact summary of categorical class counts.
    Workflow: for each 'cat' feature, print #classes + top-N classes by count.
    """
    feats = [f for f in meta.get("features", []) if f.get("kind") == "cat"]
    print("Categorical feature class counts (mask==1):")
    for f in feats[:top_feats] if top_feats else feats:
        name = f.get("name")
        classes = f.get("classes", [])
        total = sum(int(c["count"]) for c in classes)
        print(f"  - {name}: {len(classes)} classes, total obs={total}")
        # sort by count desc
        top = sorted(classes, key=lambda c: int(c["count"]), reverse=True)[:top_classes]
        for c in top:
            print(f"      code={c['code']:<8} count={c['count']}")


# ---------------------------- Data exports ------------------------------
def dump_meta(ds: xr.Dataset, path: str):
    """
    Write the raw feature_meta JSON (verbatim) to a file.
    Workflow: parse ds.attrs['feature_meta'] → json.dump to path.
    """
    meta = feature_meta(ds)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def export_counts(ds: xr.Dataset, path: str):
    """
    Export categorical class counts to CSV with per-feature proportions.
    Columns: feature, code, count, prop
    Workflow: parse feature_meta → iterate 'cat' features → write tidy rows.
    """
    meta = feature_meta(ds)
    rows = []
    feats = meta.get("features", [])
    for f in feats:
        if f.get("kind") != "cat":
            continue
        classes = f.get("classes", [])
        total = sum(int(c["count"]) for c in classes) or 1
        for c in classes:
            rows.append([f["name"], int(c["code"]), int(c["count"]),
                         float(c["count"]) / float(total)])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "code", "count", "prop"])
        w.writerows(rows)


def export_manifest(ds: xr.Dataset, path: str):
    """
    Export a minimal manifest describing shapes, names/kinds, NAIP bands, and key attrs.
    Workflow: collect presence/shape/name metadata → write JSON for the trainer to consume.
    """
    manifest = {
        "attrs_raw_shape": tuple(ds["attrs_raw"].shape) if "attrs_raw" in ds else None,
        "years": ds["years"].values.tolist() if "years" in ds else None,
        "feature_names": ds["feature_names"].values.tolist() if "feature_names" in ds else None,
        "feature_kinds": ds["feature_kinds"].values.tolist() if "feature_kinds" in ds else None,
        "naip_shape": tuple(ds["naip_patch"].shape) if "naip_patch" in ds else None,
        "naip_bands": (
            int(ds["naip_patch"].sizes.get("band"))
            if "naip_patch" in ds else None
        ),
        "attrs": {
            "crs": ds.attrs.get("crs"),
            "transform": ds.attrs.get("transform"),
            "bounds": ds.attrs.get("bounds"),
            "resolution": ds.attrs.get("resolution"),
            "window_len": ds.attrs.get("window_len"),
            "end_years": ds.attrs.get("end_years"),
            "created": ds.attrs.get("created"),
        },
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ----------------------------- Summary ----------------------------------
def summarize_zarr(path: str,
                   show_meta: bool = False,
                   show_naip_stats: bool = False,
                   show_cont_stats: bool = False,
                   show_cat_stats: bool = False,
                   top_classes: int = 10,
                   top_feats: Optional[int] = None,
                   dump_meta_path: Optional[str] = None,
                   export_counts_path: Optional[str] = None,
                   export_manifest_path: Optional[str] = None):
    """
    Orchestrate the human-readable report and optional exports.
    Workflow:
      1) Open dataset; print core array shapes and feature roster.
      2) Print dataset attrs and NAIP layout (+ optional band stats).
      3) Print feature_meta-driven summaries (continuous/categorical).
      4) Optionally export raw meta, tidy counts CSV, or a compact manifest JSON.
    """
    ds = open_zarr(path)

    # Basic dataset structure
    print(f"\nZarr: {path}")
    if "attrs_raw" in ds:
        print(f"attrs_raw shape (time, y, x, feature): {tuple(ds['attrs_raw'].shape)}")
    else:
        print("attrs_raw: MISSING")

    # Years and features
    try:
        years = list_years(ds)
        print(f"Years: {years}")
    except Exception:
        print("Years: not found (expected variable 'years')")
    try:
        features = list_features(ds)
        print(f"Features ({len(features)}):")
        for name, kind in features:
            print(f"  - {name} ({kind})")
    except Exception:
        print("Features: not found (expected variables 'feature_names' and 'feature_kinds')")

    print_dataset_attrs(ds)

    # NAIP presence and layout
    info = naip_info(ds)
    if not info["present"]:
        print("NAIP: not present")
    else:
        y, x, krow, kcol, band = info["shape"]
        print(f"NAIP: present → shape (y, x, krow, kcol, band) = {info['shape']}")
        print(f"      patch = {info['kshape'][0]}×{info['kshape'][1]}  bands = {info['bands']}")
        if show_naip_stats:
            stats = naip_band_stats(ds)
            print("NAIP per-band stats (NaNs ignored):")
            for s in stats:
                print(f"  band {s['band']}: min={s['min']}, max={s['max']}, mean={s['mean']}, std={s['std']}")

    # Feature metadata-driven reports
    meta = feature_meta(ds)
    if show_meta:
        print("\nFeature metadata (JSON):")
        print(json.dumps(meta, indent=2))

    if show_cont_stats:
        print()
        print_continuous_stats(meta)

    if show_cat_stats:
        print()
        print_categorical_stats(meta, top_classes=top_classes, top_feats=top_feats)

    # Optional exports (no policy)
    if dump_meta_path:
        dump_meta(ds, dump_meta_path)
        print(f"\nWrote raw feature_meta JSON → {dump_meta_path}")
    if export_counts_path:
        export_counts(ds, export_counts_path)
        print(f"Wrote categorical counts CSV → {export_counts_path}")
    if export_manifest_path:
        export_manifest(ds, export_manifest_path)
        print(f"Wrote dataset manifest JSON → {export_manifest_path}")

    print()
    return ds


# ------------------------------- CLI ------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Inspect a VQ-VAE Zarr cube (features, years, NAIP, stats, exports).")
    p.add_argument("zarr_path", help="Path to the Zarr dataset directory")

    # Readouts
    p.add_argument("--meta", action="store_true", help="Print full feature_meta JSON")
    p.add_argument("--naip-stats", action="store_true", help="Print per-band NAIP stats")
    p.add_argument("--cont-stats", action="store_true", help="Print continuous stats from feature_meta")
    p.add_argument("--cat-stats", action="store_true", help="Print categorical class counts from feature_meta")
    p.add_argument("--top", type=int, default=10, help="Top-N classes per categorical to print with --cat-stats")
    p.add_argument("--top-feats", type=int, default=None, help="Limit number of categorical features printed")

    # Exports (policy-free)
    p.add_argument("--dump-meta", type=str, default=None, help="Write raw feature_meta JSON to this path")
    p.add_argument("--export-counts", type=str, default=None, help="Write categorical counts CSV (feature,code,count,prop)")
    p.add_argument("--export-manifest", type=str, default=None, help="Write dataset manifest JSON (shapes, names/kinds, NAIP, attrs)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        summarize_zarr(
            args.zarr_path,
            show_meta=args.meta,
            show_naip_stats=args.naip_stats,
            show_cont_stats=args.cont_stats,
            show_cat_stats=args.cat_stats,
            top_classes=args.top,
            top_feats=args.top_feats,
            dump_meta_path=args.dump_meta,
            export_counts_path=args.export_counts,
            export_manifest_path=args.export_manifest,
        )
    except Exception as e:
        sys.exit(f"Error: {e}")
