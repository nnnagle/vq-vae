#!/usr/bin/env python3
# ======================================================================
# File: build_zarr.py
# Purpose:
#   Orchestrate construction of a geospatial Zarr “feature cube” for
#   VQ-VAE training by **spatially cropping** inputs to a mask AOI.
#
# What it does:
#   • Uses a 30 m binary mask (1=keep, else drop) as the TEMPLATE
#     (CRS, transform, bounds, and (H,W) grid).
#   • Reads yearly 30 m single-band rasters (mixed continuous/categorical)
#     from a CSV (year,type(int|cat),file_path[,fid]) and crops them to the mask
#     via bounds intersection & raster windows (no reprojection).
#   • Reads a 10 m NAIP mosaic, aligns it to a 10 m template derived from
#     the mask (3× resolution), and reshapes into 3×3 patches per 30 m cell.
#   • Deduplicates years implied by (end_years, window_len) and writes:
#        attrs_raw(time,y,x,feature)     RAW values (float; nodata→NaN)
#        mask(y,x)                       uint8
#        years(time)                     int
#        feature_names(feature)          string
#        feature_kinds(feature)          {"int"|"cat"}
#        naip_patch(y,x,krow,kcol,band)  float32/uint16 (3×3 per 30 m cell)
#   • Stores skew-aware summaries for continuous vars and class counts for
#     categorical vars in dataset attrs as JSON, and sidecar JSON on disk.
#
# Streaming / memory behavior:
#   • attrs_raw and naip_patch are Dask-backed. All reductions (stats, histograms,
#     quantiles) are chunk-wise and streaming; the code never materializes the
#     full rasters in memory.
#
# Example usage
#   1) Minimal run with typical chunks and Zstd compression:
#        python build_zarr.py \
#          --mask data/mask_30m.tif \
#          --features_csv data/features.csv \
#          --naip_path /data/naip/naip_2021_10m.tif \
#          --end_years 2020 2022 \
#          --window_len 4 \
#          --out_zarr out/cube.zarr \
#          --chunks time=1,y=1024,x=1024,feature=64,krow=3,kcol=3 \
#          --compress zstd:5 \
#          --naip_dtype float32
#
#   2) Heavier spatial chunking for huge AOIs, faster band IO:
#        python build_zarr.py \
#          --mask data/mask_30m.tif \
#          --features_csv data/features.csv \
#          --naip_path /data/naip/naip_2021_10m.tif \
#          --end_years 2018 2019 2020 2021 2022 \
#          --window_len 5 \
#          --out_zarr out/cube_big.zarr \
#          --chunks time=1,y=2048,x=2048,feature=128,krow=3,kcol=3 \
#          --compress zstd:7
#
#   3) Integer NAIP storage (smaller on disk), still streaming:
#        python build_zarr.py \
#          --mask data/mask_30m.tif \
#          --features_csv data/features.csv \
#          --naip_path /data/naip/naip_2021_10m_uint16.tif \
#          --end_years 2022 \
#          --window_len 3 \
#          --out_zarr out/cube_uint16.zarr \
#          --naip_dtype uint16
#
# CSV format (features_csv):
#   Required columns: year, kind, file_path
#   Optional column:  fid  (stable feature ID/name; defaults to file stem)
#   Example:
#     year,kind,fid,file_path
#     2019,int,canopy_height,/data/30m/2019/canopy_height.tif
#     2019,cat,landcover,/data/30m/2019/landcover_codes.tif
#     2020,int,canopy_height,/data/30m/2020/canopy_height.tif
#     2020,cat,landcover,/data/30m/2020/landcover_codes.tif
# ======================================================================


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
    naip_attrs = {
        "source": os.path.abspath(args.naip_path),
        "kshape": (3, 3),
        "bands": int(naip_patch.sizes["band"]),
        "dtype": str(naip_patch.dtype),
    }
    try:
        m = mask_da.astype(bool)
        q = naip_patch.where(m).quantile([0.01, 0.99], dim=("y", "x", "krow", "kcol"), skipna=True)
        # Compute scalars; Dask reduces per-chunk, no full array materialization
        naip_attrs["q01"] = q.sel(quantile=0.01).compute().values.tolist()
        naip_attrs["q99"] = q.sel(quantile=0.99).compute().values.tolist()
    except Exception as e:
        log("NAIP quantiles skipped due to: %s", e)
    naip_patch.attrs.update(naip_attrs)

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
