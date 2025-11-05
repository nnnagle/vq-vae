"""
scripts/build_zarr.py
---------------------
Construct a geospatial Zarr *feature cube* for VQ-VAE training by aligning all
inputs to a 30 m mask grid and streaming them into chunked, compressed arrays.

Purpose
    • Use a 30 m binary mask (1=keep) as the spatial template (CRS, transform, H×W).
    • For each end year and window length, stack yearly rasters (mixed int/cat)
      into a time axis to form: attrs_raw[time, y, x, feature].
    • Align a 10 m NAIP mosaic to the mask grid, and reshape into 3×3 patches
      per 30 m cell: naip_patch[y, x, krow=3, kcol=3, band].
    • Compute streaming feature metadata:
        – continuous: robust stats (q01, q99, mean, std)
        – categorical: per-code counts
    • Persist metadata into Zarr attrs (JSON) + sidecar JSON files.

Outputs (Zarr variables)
    • attrs_raw(time, y, x, feature)   – raw values (float, nodata→NaN), Dask-backed
    • mask(y, x)                        – uint8 (1=keep)
    • years(time)                       – int16
    • feature_names(feature)            – str
    • feature_kinds(feature)            – {"int"|"cat"}
    • naip_patch(y, x, krow, kcol, band)– float32/uint16 (3×3 per 30 m cell)

Performance & chunking
    • Fully streaming: reductions (stats/quantiles/histograms) are chunk-wise;
      no full-array materialization.
    • Chunk spec controls time/y/x/feature and NAIP krow/kcol partitioning.
    • Compression via Blosc (zstd/lz4/etc) with bit-shuffle.

CLI (examples)
    Minimal:
        pythion -m scripts.build_zarr \\
          --config scripts/config.yaml
        python -m scripts.build_zarr \\
          --mask data/mask_30m.tif \\
          --features_csv data/features.csv \\
          --naip_path /data/naip_2021_10m.tif \\
          --end_years 2020 2022 \\
          --window_len 4 \\
          --out_zarr out/cube.zarr \\
          --chunks time=1,y=1024,x=1024,feature=64,krow=3,kcol=3 \\
          --compress zstd:5 --naip_dtype float32

CSV schema (features_csv)
    Required: year, kind(int|cat), file_path
    Optional: fid (stable feature ID; defaults to file stem)
    Example:
        year,kind,fid,file_path
        2019,int,canopy_height,/data/30m/2019/canopy_height.tif
        2019,cat,landcover,/data/30m/2019/landcover_codes.tif
"""


import os
import json
import argparse
from typing import Dict, List

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

# --- local modules ------------------------------------------------------
from utils.data_stack import (
    select_years,
    index_inputs,
    enforce_consistent_features,
    stack_attrs_raw_spatial,     # Dask-lazy
    compute_feature_metadata,    # streaming reductions
    compute_naip_metadata,
)
from utils.raster_ops import (
    load_mask_template,
    align_naip_to_mask_as_patches,  # Dask-lazy per-chunk reads
)
from utils.log import log, ensure
from utils.argyaml import parse_args_with_yaml


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Build VQ-VAE Zarr feature cube")
    p.add_argument("--config", type=str, help="Path to YAML config file (with a 'build_zarr' section).")
    p.add_argument("--mask", required=True, help="30 m binary mask raster (1=keep)")
    p.add_argument("--features_csv", required=True,
                   help="CSV with columns: year,kind(int|cat),fid(optional),file_path")
    p.add_argument("--naip_path", required=True, help="Path to NAIP 10 m mosaic")
    p.add_argument("--end_years", required=True, nargs="+", type=int,
                   help="Window end years (integers)")
    p.add_argument("--window_len", required=True, type=int,
                   help="Length (in years) for each time window")
    p.add_argument("--out_zarr", required=True, help="Output Zarr store (directory)")
    p.add_argument("--chunks", default="time=1,y=512,x=512,feature=64,krow=3,kcol=3",
                   help="Chunk spec, e.g. time=1,y=512,x=512,feature=64; "
                        "krow/kcol ignored for attrs_raw but used for NAIP")
    p.add_argument("--naip_dtype", default="float32",
                   help="NAIP dtype to write (e.g., float32 or uint16)")
    p.add_argument("--compress", default="zstd:5",
                   help="Compressor spec 'zstd:5' or 'lz4:9'.")
    return parse_args_with_yaml(p, section="build_zarr")


def parse_chunk_spec(spec: str) -> Dict[str, int]:
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split("=")
        out[k.strip()] = int(v.strip())
    return out


def make_compressor(spec: str) -> Blosc:
    if ":" in spec:
        cname, clevel = spec.split(":")
        clevel = int(clevel)
    else:
        cname, clevel = spec, 5
    cname = cname.strip().lower()
    ensure(cname in {"zstd", "lz4", "zlib", "blosclz"}, f"Unsupported compressor: {cname}")
    return Blosc(cname=cname, clevel=clevel, shuffle=Blosc.BITSHUFFLE)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    args = parse_args()
    chunks = parse_chunk_spec(args.chunks)
    compressor = make_compressor(args.compress)

    # 1) Load mask template (H, W, transform, crs)
    log("Loading mask template from %s", args.mask)
    mask_da, template = load_mask_template(args.mask)  # DataArray dims=("y","x")
    H, W = int(mask_da.sizes["y"]), int(mask_da.sizes["x"])
    ensure(mask_da.dtype in (np.uint8, np.int8, np.bool_), "mask must be binary uint8/bool")
    log("Mask grid: H=%d, W=%d", H, W)

    # 2) Time axis from (end_years, window_len)
    needed_years = select_years(args.end_years, args.window_len)
    log("Needed years: %s", needed_years)

    # 3) Input index & consistency
    y2f = index_inputs(args.features_csv, needed_years)
    feature_ids, feature_kinds = enforce_consistent_features(y2f, needed_years)
    nfeat = len(feature_ids)
    log("Feature count: %d", nfeat)

    # 4) Dask-backed attrs_raw(time,y,x,feature)
    log("Building Dask-backed attrs_raw (lazy)...")
    attrs_raw = stack_attrs_raw_spatial(
        y2f=y2f,
        needed_years=needed_years,
        mask_da=mask_da,
        template=template,
        chunks={"time": chunks.get("time", 1),
                "y": chunks.get("y", 512),
                "x": chunks.get("x", 512),
                "feature": chunks.get("feature", 64)},
        out_dtype=np.float32,
    )

    # 5) Dask-backed NAIP patches aligned to mask grid
    log("Aligning NAIP to mask grid as 3×3 patches (lazy)...")
    naip_patch = align_naip_to_mask_as_patches(
        naip_path=args.naip_path,
        mask_da=mask_da,
        template=template,
        out_dtype=args.naip_dtype,
        kshape=(3, 3),
        chunks={"y": chunks.get("y", 512),
                "x": chunks.get("x", 512),
                "krow": chunks.get("krow", 3),
                "kcol": chunks.get("kcol", 3),
                "band": 1},
    )

    # 6) Persist NAIP metadata (source + structure + robust per-band quantiles)
    log("Attaching NAIP metadata and robust q01/q99 (streaming reductions)...")
    naip_meta = compute_naip_metadata(
      naip_patch=naip_patch,
      mask_da=mask_da,
      include_source=os.path.abspath(args.naip_path),
    )
    naip_patch.attrs.update(naip_meta)

    # 7) Feature metadata (all streaming)
    log("Computing feature metadata (continuous & categorical) with streaming reductions...")
    feature_meta = compute_feature_metadata(
        attrs_raw=attrs_raw,
        feature_names=feature_ids,
        feature_kinds=feature_kinds,
        mask_da=mask_da,
    )

    # 8) Assemble Dataset
    years_da = xr.DataArray(np.array(needed_years, dtype=np.int16), dims=("time",), name="years")
    feature_names_da = xr.DataArray(np.array(feature_ids, dtype="U"), dims=("feature",), name="feature_names")
    feature_kinds_da = xr.DataArray(np.array(feature_kinds, dtype="U"), dims=("feature",), name="feature_kinds")

    ds = xr.Dataset(
        data_vars=dict(
            attrs_raw=attrs_raw,
            naip_patch=naip_patch,
            mask=mask_da.astype(np.uint8),
            years=years_da,
            feature_names=feature_names_da,
            feature_kinds=feature_kinds_da,
        ),
        attrs={
            "feature_meta": json.dumps(feature_meta, separators=(",", ":"), ensure_ascii=False),
            "template_crs": str(template["crs"]),
            "template_transform": tuple(template["transform"][:6]),
        },
    )

    # 9) Write Zarr with explicit encoding (compressor preserved)
    log("Writing Zarr to: %s", args.out_zarr)
    enc = {
        "attrs_raw": {"compressor": compressor},
        "naip_patch": {"compressor": compressor},
        "mask": {"compressor": compressor},
        "years": {"compressor": compressor},
        "feature_names": {"compressor": compressor},
        "feature_kinds": {"compressor": compressor},
    }
    ds.to_zarr(args.out_zarr, mode="w", encoding=enc)
    zarr.convenience.consolidate_metadata(args.out_zarr)

    # 10) Sidecar JSON
    meta_path = os.path.join(args.out_zarr, "feature_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)
    log("Done. Zarr written and consolidated. Sidecar: %s", meta_path)

    naip_attrs = dict(ds["naip_patch"].attrs)
    naip_meta_path = os.path.join(args.out_zarr, "naip_meta.json")
    with open(naip_meta_path, "w") as f:
      json.dump(naip_attrs, f, indent=2)
    log(f"Wrote NAIP metadata to {naip_meta_path}")

if __name__ == "__main__":
    main()
