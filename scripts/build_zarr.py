"""
build_zarr.py

Script to build a spatiotemporal feature cube in Zarr format from
large raster inputs, as specified by a YAML configuration file.

Design highlights
-----------------
- Uses xarray + dask + rasterio/rioxarray via utils.rasterio helpers.
- Never materializes full mosaics in memory.
- Handles different temporal layouts:
    * single-file multiband rasters spanning many years
    * per-year rasters with multiple bands
    * constructed temporal masks that encode data availability
- Separates features into continuous, categorical, and mask cubes.
- Encodes masking and normalization instructions in metadata only;
  it does NOT apply any normalization to stored values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import xarray as xr
import zarr
import dask
import yaml

# Local utilities
import utils.rasterio as urio

try:
    import log  # type: ignore
except Exception:  # pragma: no cover - fallback logger
    import logging as log  # type: ignore
from utils.log import log, warn

# -----------------------------------------------------------------------------
# Config / time helpers
# -----------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """Parse the YAML config at `path` and return it as a nested dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_full_time_axis(cfg: dict) -> np.ndarray:
    """
    Build the global dense time axis from cfg['out_zarr']['time']['window'].

    Expects structure like:
        dataset: 
          out_zarr:
            time:
              window:
                start: 2015
                end: 2024

    Returns a 1D np.ndarray of integer years [start, ..., end].
    """
    try:
        tw = cfg["dataset"]["out_zarr"]["time"]["window"]
        start = int(tw["start"])
        end = int(tw["end"])
    except Exception as exc:
        raise KeyError("Config missing out_zarr.time.window.start/end") from exc

    if end < start:
        raise ValueError(f"Invalid time window: start={start}, end={end}")
    years = np.arange(start, end + 1, dtype="int64")
    return years


# -----------------------------------------------------------------------------
# Template / static rasters
# -----------------------------------------------------------------------------


def open_spatial_template(
    template_path: str | Path,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Open the spatial template (AOI or mask) and return as 2D DataArray (y, x).

    The template defines CRS, transform, and spatial extent for all features.
    """
    da = urio.open_raster_as_dataarray(template_path, chunks=chunks)
    # Drop band dimension if present
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    return da


def open_static_raster_aligned(
    path: str | Path,
    template: xr.DataArray,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Open a static raster (aoi/strata/splits) and align it to the template grid.

    Returns a 2D DataArray (y, x) with CRS and transform matching template.
    """
    da = urio.open_raster_as_dataarray(path, chunks=chunks)
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    da = urio.align_to_template(da, template, resampling="nearest")
    return da


# -----------------------------------------------------------------------------
# File resolution helpers
# -----------------------------------------------------------------------------


def resolve_sharded_files(pattern: str | Path) -> Sequence[Path]:
    """
    Expand a glob pattern like '/path/lcms_lu*.tif' into existing files.

    Raises FileNotFoundError if no files match.
    """
    p = Path(pattern)
    files = sorted(p.parent.glob(p.name))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files


def resolve_multifile_year_patterns(
    file_pattern: str,
    years: Sequence[int],
) -> Dict[int, Sequence[Path]]:
    """
    Resolve a per-year glob pattern into {year: [files...]} mapping.

    `file_pattern` should contain '{year}' placeholder.
    Raises FileNotFoundError if a year in `years` produces no matches.
    """
    file_pattern = str(file_pattern)
    if "{year" not in file_pattern:
        raise ValueError(f"file_pattern must contain '{{year}}' placeholder: {file_pattern}")

    year_to_files: Dict[int, Sequence[Path]] = {}
    for year in years:
        pattern = file_pattern.format(year=year)
        p = Path(pattern)
        matches = sorted(p.parent.glob(p.name))
        if not matches:
            raise FileNotFoundError(f"No files found for pattern {pattern!r} (year={year})")
        year_to_files[year] = matches
    return year_to_files


# -----------------------------------------------------------------------------
# Time-variable constructors
# -----------------------------------------------------------------------------


def open_single_file_time_variable(
    files: Sequence[Path],
    bandname_template: str,
    years: Sequence[int],
    template: xr.DataArray,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Open a time-varying variable stored as one or more multiband rasters.

    Parameters
    ----------
    files : sequence of Path
        Sharded multiband rasters for the same logical layer.
    bandname_template : str
        Template for band names, e.g. 'lcms_lu_{year}'.
    years : sequence of int
        Global dense time axis (e.g. 2015..2024).
    template : DataArray
        Spatial template for alignment.
    chunks : dict, optional
        Chunking to pass through to lower-level IO.

    Returns
    -------
    xarray.DataArray
        Lazily-loaded DataArray(time, y, x), already aligned to template.
        Only years with matching bands will be present in the time axis.
    """
    da = urio.open_multiband_raster_for_years(
        files=files,
        bandname_template=bandname_template,
        years=years,
        chunks=chunks,
    )
    da = urio.align_to_template(da, template, resampling="nearest")
    return da


def open_multi_file_time_variable(
    year_to_files: Dict[int, Sequence[Path]],
    bandname: str,
    full_time: Sequence[int],
    template: xr.DataArray,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Open a time-varying variable stored as separate rasters per year.

    Returns
    -------
    xarray.DataArray(time, y, x) covering `full_time`, sparse positions = NaN.
    """
    # Load sparse-time data first
    da_sparse = urio.open_per_year_rasters(
        year_to_files=year_to_files,
        bandname=bandname,
        chunks=chunks,
    )
    da_sparse = urio.align_to_template(da_sparse, template, resampling="nearest")
    da = reindex_to_full_time(da_sparse, full_time)
    return da


# -----------------------------------------------------------------------------
# Temporal masks and reindexing
# -----------------------------------------------------------------------------


def build_temporal_availability_mask(
    feature_id: str,
    missing_years: Sequence[int],
    full_time: Sequence[int],
    template: xr.DataArray,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Build a boolean DataArray(time, y, x) describing data availability.

    True if year ∉ missing_years.
    False if year ∈ missing_years.
    Broadcast spatially to template grid.
    """
    missing_years = set(int(y) for y in missing_years)
    time_vals = np.array(full_time, dtype="int64")
    time_mask = np.array([int(y) not in missing_years for y in time_vals], dtype=bool)

    y = template.coords.get("y")
    x = template.coords.get("x")
    if y is None or x is None:
        y = np.arange(template.shape[-2])
        x = np.arange(template.shape[-1])

    data = xr.DataArray(
        np.broadcast_to(
            time_mask[:, None, None],
            (len(time_vals), template.shape[-2], template.shape[-1]),
        ),
        dims=("time", "y", "x"),
        coords={"time": time_vals, "y": y, "x": x},
        name=feature_id,
    )

    # Propagate non-dim coords from the template (e.g., 'spatial_ref')
    for cname, coord in template.coords.items():
        if cname not in data.coords:
            data = data.assign_coords({cname: coord})

    if chunks is not None:
        data = data.chunk(chunks)
    return data


def reindex_to_full_time(
    da: xr.DataArray,
    full_time: Sequence[int],
) -> xr.DataArray:
    """Reindex a time array to dense `full_time`, choosing a dtype-safe fill value.

    - Floats: fill with NaN (keeps float dtype)
    - Integers: fill with a sentinel (-1 for signed, 0 for unsigned)
    - Bools: fill with False
    """
    full_time = np.array(full_time, dtype="int64")
    if "time" not in da.coords:
        raise ValueError("DataArray has no 'time' coordinate for reindexing")

    # normalize existing time coord to ints
    da = da.assign_coords(time=[int(t) for t in da.coords["time"].values])

    dtype = da.dtype
    print(f"[REINDEX] name={da.name!r} BEFORE reindex dtype={dtype}")

    # choose fill_value based on dtype
    if np.issubdtype(dtype, np.bool_):
        fill_value = False
    elif np.issubdtype(dtype, np.integer):
        if np.issubdtype(dtype, np.signedinteger):
            # typical categorical nodata sentinel
            try:
                fill_value = dtype.type(-1)
            except Exception:
                fill_value = dtype.type(0)
        else:
            # unsigned ints can't hold -1, fall back to 0
            fill_value = dtype.type(0)
    else:
        # float or anything else: NaN is fine
        fill_value = np.nan
    print(f"[REINDEX] name={da.name!r} fill_value={fill_value!r}")
    da = da.reindex(time=full_time, fill_value=fill_value)
    print(f"[REINDEX] name={da.name!r} AFTER reindex dtype={da.dtype}")
    return da


def apply_missing_year_policy(
    da: xr.DataArray,
    kind: str,
    missing_fill: Dict[str, object],
) -> xr.DataArray:
    """
    Apply per-kind missing-year policy.

    continuous → fill=0.0 → float16
    categorical → fill=-1 → int16
    mask → False → bool
    """
    if kind not in missing_fill:
        raise KeyError(f"Missing fill value for kind={kind!r}")

    fill_value = missing_fill[kind]

    if kind == "continuous":
        da_out = da.fillna(fill_value).astype("float16")
    elif kind == "categorical":
        da_out = da.fillna(fill_value).astype("int16")
    elif kind == "mask":
        filled = da.copy()
        if np.issubdtype(filled.dtype, np.floating):
            filled = filled.fillna(float(bool(fill_value)))
        da_out = filled.astype("bool")
    else:
        raise ValueError(f"Unsupported kind: {kind!r}")
    return da_out


# -----------------------------------------------------------------------------
# Grouping & metadata helpers
# -----------------------------------------------------------------------------


def stack_features_by_kind(
    feature_arrays: Dict[str, xr.DataArray],
    kind: str,
    full_time: Sequence[int],
) -> xr.DataArray:
    """
    Stack individual feature rasters into DataArray(time, feature, y, x).
    """
    if not feature_arrays:
        raise ValueError(f"No feature arrays provided for kind={kind!r}")

    feature_ids = sorted(feature_arrays.keys())

    arrays = []
    for fid in feature_ids:
        da = feature_arrays[fid]
        print(f"[STACK] kind={kind!r} fid={fid!r} BEFORE reindex dtype={da.dtype}")
        da = reindex_to_full_time(da, full_time)
        print(f"[STACK] kind={kind!r} fid={fid!r} AFTER reindex dtype={da.dtype}")
        da = da.expand_dims(feature=[fid])
        arrays.append(da)

    result = xr.concat(arrays, dim="feature")
    print(
        f"[STACK] kind={kind!r} stacked result dtype={result.dtype} "
        f"feature_ids={feature_ids}"
    )
    return result


def attach_feature_attrs(
    da: xr.DataArray,
    feature_id: str,
    feature_cfg: dict,
) -> None:
    """Attach metadata from YAML config as attrs."""
    da.name = feature_id
    da.attrs.setdefault("id", feature_id)
    meta = feature_cfg.get("metadata", {}) or {}
    da.attrs.setdefault("kind", meta.get("kind"))

    if "normalization" in feature_cfg:
        da.attrs["normalization"] = feature_cfg["normalization"]

    mask_meta = meta.get("mask")
    if mask_meta is not None:
        da.attrs["masking"] = mask_meta

    if "NoData" in feature_cfg:
        da.attrs["nodata_value"] = feature_cfg["NoData"]


def collect_global_attrs(cfg: dict) -> dict:
    """Promote high-level config settings to dataset attrs."""
    out = {}
    out_zarr = cfg.get("out_zarr", {})
    time_cfg = out_zarr.get("time", {})
    window = time_cfg.get("window", {})
    out["time_window_start"] = window.get("start")
    out["time_window_end"] = window.get("end")
    out["missing_year_policy"] = out_zarr.get("missing_years", {})
    out["reproject_when"] = cfg.get("feature_layers", {}).get("reproject", {}).get("when", "if-necessary")
    out["reproject_target"] = cfg.get("feature_layers", {}).get("reproject", {}).get("target", "template")
    return out


# -----------------------------------------------------------------------------
# Zarr writing and stats
# -----------------------------------------------------------------------------


def write_feature_groups_to_zarr(
    zarr_path: str | Path,
    continuous: Optional[xr.DataArray],
    categorical: Optional[xr.DataArray],
    mask: Optional[xr.DataArray],
    static_rasters: Dict[str, xr.DataArray],
    global_attrs: dict,
    compressor_config: dict,
    chunk_config: Mapping[str, int],
) -> None:
    """Write continuous / categorical / mask groups + statics to Zarr."""
    p = Path(zarr_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ds_vars = {}
    if continuous is not None:
        print(f"[WRITE] input continuous dtype={continuous.dtype}")
        ds_vars["continuous"] = continuous
    if categorical is not None:
        # 1. Make sure there are truly no NaNs, use integer fill value
        categorical = categorical.fillna(np.int16(-1))

        # 2. Smash the dask graph into a clean int16-typed graph
        categorical = xr.apply_ufunc(
            lambda a: a.astype("int16"),
            categorical,
            dask="parallelized",
            output_dtypes=[np.int16],
        )

        # 3. Strip CF junk from attrs AND encoding
        for k in ("scale_factor", "add_offset", "missing_value", "_FillValue", "nodata_value"):
            if k in categorical.attrs:
                print(f"[CLEAN] dropping attr {k!r} from categorical")
                categorical.attrs.pop(k)
            if k in categorical.encoding:
                print(f"[CLEAN] dropping encoding {k!r} from categorical")
                categorical.encoding.pop(k)
        print("======= FULL CATEGORICAL METADATA DIGEST =======")
        print("categorical.dtype:", categorical.dtype)
        print("categorical.attrs:", categorical.attrs)
        print("categorical.encoding:", categorical.encoding)

        # Print the underlying Variable and its internal encoding
        v = categorical.variable
        print("variable.dtype:", v.dtype)
        print("variable.attrs:", v.attrs)           # rarely useful, but check
        print("variable.encoding:", v.encoding)     # THIS is where CF metadata hides
        print("variable._encoding:", getattr(v, "_encoding", None))  # some xarray versions use this
        print("================================================")
        print(f"[WRITE] input categorical dtype={categorical.dtype}")
        ds_vars["categorical"] = categorical
    if mask is not None:
        print(f"[WRITE] input mask dtype={mask.dtype}")
        # Drop CF scaling attrs; we want raw int codes, not decoded floats
        for k in ("scale_factor", "add_offset", "missing_value", "_FillValue", "nodata_value"):
            if k in mask.attrs:
                print(f"[CLEAN] dropping attr {k!r} from mask")
                mask.attrs.pop(k)
        ds_vars["mask"] = mask
    for name, da in static_rasters.items():
        print(f"[WRITE] input static[{name}] dtype={da.dtype}")
        ds_vars[name] = da

    if "categorical" in ds_vars:
        has_nan = bool(ds_vars["categorical"].astype("float32").isnull().any().compute())
        print("[DEBUG] categorical has any NaN:", has_nan)

    ds = xr.Dataset(ds_vars)
    ds.attrs.update(global_attrs)

    if "categorical" in ds:
        # Replace any NaNs produced by alignment with -1, then force int16
        ds["categorical"] = ds["categorical"].fillna(np.int16(-1)).astype("int16")
        print("[WRITE] ds['categorical'] dtype AFTER manual fix=", ds["categorical"].dtype)

    # log dtypes after constructing Dataset
    if "categorical" in ds:
        print(f"[WRITE] ds['categorical'] dtype AFTER Dataset={ds['categorical'].dtype}")
        print(f"[WRITE] ds['categorical'].encoding BEFORE to_zarr={ds['categorical'].encoding}")
        print(f"[WRITE] ds['categorical'].attrs BEFORE to_zarr={ds['categorical'].attrs}")

    # NEW: align dask chunks with desired Zarr chunks
    # only apply for dims that actually exist in this dataset
    chunk_for_ds = {d: chunk_config[d] for d in chunk_config if d in ds.dims}
    if chunk_for_ds:
        ds = ds.chunk(chunk_for_ds)

    if "categorical" in ds:
        print(f"[WRITE] ds['categorical'] dtype AFTER chunk={ds['categorical'].dtype}")

    encoding = {}
    for name, da in ds.data_vars.items():
        enc = {}
        dims = da.dims
        chunks = {d: chunk_config[d] for d in dims if d in chunk_config}
        if chunks:
            enc["chunks"] = tuple(chunks[d] for d in dims if d in chunks)
        if compressor_config:
            enc.update(compressor_config)
        encoding[name] = enc

    print(f"[WRITE] encoding['categorical']={encoding.get('categorical')}")
    ds.to_zarr(p, mode="w", encoding=encoding)

    # Immediately reopen and log what came back from disk
    verify = xr.open_zarr(p, consolidated=False)
    if "categorical" in verify:
        cat = verify["categorical"]
        print(f"[WRITE] verify['categorical'] dtype AFTER to_zarr/open_zarr={cat.dtype}")
        print(f"[WRITE] verify['categorical'].encoding AFTER open={cat.encoding}")
        print(f"[WRITE] verify['categorical'].attrs AFTER open={cat.attrs}")

def extract_feature_normalization_from_cfg(cfg: dict) -> dict[str, dict]:
    """
    Walk `variables` in the YAML and build a simple
    fid -> normalization_spec mapping.

    We normalize layout so that downstream code never has to care
    whether normalization lived under `metadata` or at top-level
    in the config.
    """
    out: dict[str, dict] = {}
    for feat in cfg.get("variables", []):
        fid = feat["id"]
        meta = feat.get("metadata", {}) or {}
        # allow both:
        #   metadata:
        #     normalization: { ... }
        # and
        #   normalization: { ... }
        norm_spec = meta.get("normalization") or feat.get("normalization")
        if norm_spec is None:
            continue

        # Optionally, normalize the schema a bit (e.g. ensure 'type' exists).
        ntype = norm_spec.get("type")
        if ntype is None:
            # e.g. categorical: normalization: embedding
            if isinstance(norm_spec, str):
                norm_spec = {"type": norm_spec}
            else:
                raise ValueError(f"Normalization spec for {fid!r} lacks 'type' field")

        out[fid] = norm_spec
    return out


def compute_and_save_feature_stats(
    zarr_path: str | Path,
    output_json_path: str | Path,
    kinds: Sequence[str] = ("continuous", "categorical", "mask"),
    feature_norm_cfg: dict[str, dict] | None = None,
) -> None:
    """Compute statistics per feature and write JSON sidecar."""
    p = Path(zarr_path)
    if not p.exists():
        raise FileNotFoundError(f"Zarr store not found: {p}")

    ds = xr.open_zarr(p, consolidated=False)

    stats: Dict[str, dict] = {}

    if "categorical" in ds:
        print(f"[STATS] ds['categorical'] dtype={ds['categorical'].dtype}")
    if "mask" in ds:
        print(f"[STATS] ds['mask'] dtype={ds['mask'].dtype}")

    # map dataset variable -> its feature dimension name
    kind_to_feature_dim = {
        "continuous":  "feature_continuous",
        "categorical": "feature_categorical",
        "mask":        "feature_mask",
    }

    # Optional debug: what features are present per group
    print("[DEBUG] group_features:")
    for grp, fdim in kind_to_feature_dim.items():
        if grp in ds and fdim in ds[grp].coords:
            fids = list(ds[grp].coords[fdim].values)
            print(f"  {grp}: {sorted(fids)}")
        else:
            print(f"  {grp}: (none)")

    # -----------------------------
    # 1. Continuous features
    # -----------------------------
    if "continuous" in ds and "continuous" in kinds:
        da = ds["continuous"].astype("float32")
        feature_dim = kind_to_feature_dim["continuous"]

        if feature_dim not in da.coords:
            raise KeyError(
                f"'continuous' variable is missing coord {feature_dim!r}; "
                "did you rename the dim before writing?"
            )

        # detect nodata (optional, same as before)
        nodata = (
            da.attrs.get("_FillValue", None)
            or da.attrs.get("missing_value", None)
            or da.encoding.get("nodatavals", [None])[0]
        )

        da_stats = da
        da_stats = da_stats.chunk({"time": 5, "y": 512, "x": 512})

        # if nodata is not None:
        #     da_stats = da_stats.where(da_stats != nodata)

        # reduce over everything except the feature dimension
        reduce_dims = [d for d in da.dims if d != feature_dim]
        quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]

        da_mean = da_stats.mean(dim=reduce_dims)
        da_std  = da_stats.std(dim=reduce_dims)
        da_min  = da_stats.min(dim=reduce_dims)
        da_max  = da_stats.max(dim=reduce_dims)
        da_q    = da_stats.quantile(quantiles, dim=reduce_dims)

        mean_v = da_mean.compute().values
        std_v  = da_std.compute().values
        min_v  = da_min.compute().values
        max_v  = da_max.compute().values
        q_v    = da_q.compute().values  # shape: (n_quantiles, n_features)

        feature_ids = list(da.coords[feature_dim].values)
        for i, fid in enumerate(feature_ids):
            feat_stats = stats.setdefault(fid, {})
            existing_kind = feat_stats.get("kind")
            if existing_kind is not None and existing_kind != "continuous":
                raise ValueError(
                    f"Feature {fid!r} appears as kind {existing_kind!r} and 'continuous'"
                )

            feat_stats["kind"] = "continuous"
            feat_stats["mean"] = float(mean_v[i])
            feat_stats["std"]  = float(std_v[i])
            feat_stats["min"]  = float(min_v[i])
            feat_stats["max"]  = float(max_v[i])
            feat_stats["quantiles"] = {
                str(q): float(q_v[j, i]) for j, q in enumerate(quantiles)
            }

    # -----------------------------
    # 2. Categorical & mask features
    # -----------------------------
    for group in ("categorical", "mask"):
        if group not in ds or group not in kinds:
            continue

        feature_dim = kind_to_feature_dim[group]
        da = ds[group]

        if feature_dim not in da.coords:
            raise KeyError(
                f"{group!r} variable is missing coord {feature_dim!r}; "
                "did you rename the dim before writing?"
            )

        feature_ids = list(da.coords[feature_dim].values)

        for i, fid in enumerate(feature_ids):
            flat = da.isel({feature_dim: i}).values.ravel()
            uniq, counts = np.unique(flat, return_counts=True)

            feat_stats = stats.setdefault(fid, {})

            existing_kind = feat_stats.get("kind")
            if existing_kind is not None and existing_kind != group:
                raise ValueError(
                    f"Feature {fid!r} appears as kind {existing_kind!r} and {group!r}"
                )

            feat_stats["kind"] = group
            feat_stats["value_counts"] = {
                str(v): int(c) for v, c in zip(uniq.tolist(), counts.tolist())
            }

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    # Also store in the Zarr itself
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["feature_stats"] = stats
    if feature_norm_cfg is not None:
        root.attrs["feature_normalization"] = feature_norm_cfg
    zarr.consolidate_metadata(str(zarr_path)) # keep consolidated metadata in sync


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Zarr feature cube from rasters.")
    parser.add_argument("config", help="Path to YAML config file describing inputs.")
    parser.add_argument("--out", dest="out_zarr", help="Override output Zarr path.")
    parser.add_argument("--no-stats", action="store_true", help="Skip computing stats JSON.")
    parser.add_argument("--stats-path", help="Explicit path for stats JSON.")
    parser.add_argument("--ncore", type=int, default=1, help="Number of cores to use.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    n = args.ncore
    if n > 1:
        from dask.distributed import Client, LocalCluster
        client = Client(LocalCluster(n_workers=n, threads_per_worker=1))
    else:
        import dask
        dask.config.set(scheduler="single-threaded")  # true single-core execution

    cfg = load_config(args.config)
    feature_norm_cfg = extract_feature_normalization_from_cfg(cfg)

    full_time = build_full_time_axis(cfg)

    out_cfg = cfg.get("dataset", {}).get("out_zarr", {})
    zarr_path = args.out_zarr or out_cfg.get("path")
    if zarr_path is None:
        raise KeyError("Output path missing: use --out or cfg['out_zarr']['path'].")

    chunk_cfg = out_cfg.get("chunks", {"time": 5, "feature": -1, "y": 256, "x": 256})
    compressor_cfg = out_cfg.get("compressor", {})

    tmpl_cfg = out_cfg.get("spatial_template", {})
    tmpl_path = tmpl_cfg.get("path")
    if tmpl_path is None:
        raise KeyError("Config missing spatial_template.path")
    template = open_spatial_template(
        tmpl_path,
        chunks={"y": chunk_cfg.get("y", 256), "x": chunk_cfg.get("x", 256)},
    )

    # Always look for aoi, strata, splits and create features if they exist.
    # We support two layouts:
    #   - nested under dataset.out_zarr.spatial_template.{aoi,strata,splits}
    #   - or dataset.{aoi,strata,splits}.path at top level.
    static_rasters: Dict[str, xr.DataArray] = {}
    ds_cfg = cfg.get("dataset", {})
    for key in ("aoi", "strata", "splits"):
        # Prefer nested under spatial_template, fall back to dataset-level path.
        path = tmpl_cfg.get(key) or ds_cfg.get(key, {}).get("path")
        if path:
            static_rasters[key] = open_static_raster_aligned(path, template)

    var_cfgs = cfg.get("variables", {})
    missing_year_policy = out_cfg.get(
        "missing_years", {"continuous": 0.0, "categorical": -1, "mask": False}
    )

    cont_vars: Dict[str, xr.DataArray] = {}
    cat_vars: Dict[str, xr.DataArray] = {}
    mask_vars: Dict[str, xr.DataArray] = {}

    for feat in var_cfgs:
        fid = feat["id"]
        meta = feat.get("metadata", {}) or {}
        kind = meta.get("kind")
        time_cfg = feat.get("time", {})
        t_type = time_cfg.get("type")

        try:
            log.info(f"Loading {fid} (kind={kind}, time.type={t_type})")
        except Exception:
            print(f"Loading {fid} (kind={kind}, time.type={t_type})")

        chunks_xy = {"y": chunk_cfg.get("y", 256), "x": chunk_cfg.get("x", 256)}

        if t_type == "single_file":
            files = resolve_sharded_files(time_cfg["files"])
            da = open_single_file_time_variable(
                files, time_cfg["bandname"], full_time, template, chunks_xy
            )

        elif t_type == "multi_file":
            if "years" in time_cfg:
                years = sorted(set(int(y) for y in time_cfg["years"]) & set(full_time))
            else:
                start = int(time_cfg.get("start", full_time[0]))
                end = int(time_cfg.get("end", full_time[-1]))
                years = [y for y in full_time if start <= y <= end]

            year_to_files = resolve_multifile_year_patterns(time_cfg["files"], years)
            da = open_multi_file_time_variable(
                year_to_files, time_cfg["bandname"], full_time, template, chunks_xy
            )

        elif t_type == "constructed":
            missing_years = feat.get("source", {}).get("missing-years", [])
            da = build_temporal_availability_mask(
                fid, missing_years, full_time, template, chunks_xy
            )
        else:
            raise ValueError(f"Unsupported time.type {t_type!r} for feature {fid!r}")

        attach_feature_attrs(da, fid, feat)

        # DEBUG: see how each feature is classified at build time
        try:
            log.info(
                "CLASSIFY feature=%s kind=%s time.type=%s dtype=%s",
                fid,
                kind,
                t_type,
                da.dtype,
            )
        except Exception:
            print(
                f"[CLASSIFY] feature={fid} kind={kind} time.type={t_type} dtype={da.dtype}"
            )

        if kind == "continuous":
            da = da.astype("float16")
            cont_vars[fid] = da
        elif kind == "categorical":
            if np.issubdtype(da.dtype, np.floating):
                warn(
                    "Feature %s (categorical) loaded as float dtype %s; casting to int16",
                    fid,
                    da.dtype,
                )
            da = da.fillna(np.int16(-1)).astype("int16")
            cat_vars[fid] = da
        elif kind == "mask":
            # Treat nodata (-32768 etc.) as False before casting
            nodata = (
                da.attrs.get("nodata_value", None)
                or da.attrs.get("_FillValue", None)
                or da.attrs.get("missing_value", None)
            )

            if nodata is not None:
                try:
                    print(f"[MASK] {fid}: mapping nodata={nodata!r} to 0 before bool cast")
                    # any pixel equal to nodata -> 0, everything else unchanged
                    da = da.where(da != nodata, other=0)
                except Exception as e:
                    print(f"[MASK] {fid}: failed to apply nodata mapping: {e!r}")

            if not np.issubdtype(da.dtype, np.bool_):
                warn(
                    "Feature %s (mask) loaded as non-bool dtype %s; casting to bool",
                    fid,
                    da.dtype,
                )
            da = da.astype("bool")
            mask_vars[fid] = da

        else:
            raise ValueError(f"Unknown kind={kind!r} for feature {fid!r}")

    print("CONT_VARS:", sorted(cont_vars.keys()))
    print("CAT_VARS:", sorted(cat_vars.keys()))
    print("MASK_VARS:", sorted(mask_vars.keys()))

    for fid, da in cat_vars.items():
        print(f"[MAIN] cat_vars[{fid!r}] dtype={da.dtype}")

    cont_da = stack_features_by_kind(cont_vars, "continuous", full_time) if cont_vars else None
    cat_da = stack_features_by_kind(cat_vars, "categorical", full_time) if cat_vars else None
    mask_da = stack_features_by_kind(mask_vars, "mask", full_time) if mask_vars else None

    # rename feature dim per kind so xarray does NOT try to align them
    if cont_da is not None:
        cont_da = cont_da.rename(feature="feature_continuous")
    if cat_da is not None:
        cat_da = cat_da.rename(feature="feature_categorical")
    if mask_da is not None:
        mask_da = mask_da.rename(feature="feature_mask")

    if cont_da is not None:
        cont_da = apply_missing_year_policy(cont_da, "continuous", missing_year_policy)
    if cat_da is not None:
        cat_da = apply_missing_year_policy(cat_da, "categorical", missing_year_policy)
    if mask_da is not None:
        mask_da = apply_missing_year_policy(mask_da, "mask", missing_year_policy)

    print("CONT_VARS:", sorted(cont_vars.keys()))
    print("CAT_VARS:", sorted(cat_vars.keys()))
    print("MASK_VARS:", sorted(mask_vars.keys()))

    if cat_da is not None:
        print(f"[MAIN] cat_da (stacked) dtype={cat_da.dtype}")

    global_attrs = collect_global_attrs(cfg)

    write_feature_groups_to_zarr(
        zarr_path,
        cont_da,
        cat_da,
        mask_da,
        static_rasters,
        global_attrs,
        compressor_cfg,
        chunk_cfg,
    )

    if not args.no_stats:
        print("stats:CONT_VARS:", sorted(cont_vars.keys()))
        print("stats:CAT_VARS:", sorted(cat_vars.keys()))
        print("stats:MASK_VARS:", sorted(mask_vars.keys()))
        print("\n======= VERIFY STORED ZARR FEATURES BEFORE STATS =======")
        verify = xr.open_zarr(zarr_path, consolidated=False)

        kind_to_feature_dim = {
            "continuous":  "feature_continuous",
            "categorical": "feature_categorical",
            "mask":        "feature_mask",
        }
        for group in ("continuous", "categorical", "mask"):
            if group in verify:
                fdim = kind_to_feature_dim[group]
                if fdim in verify[group].coords:
                    f = list(verify[group].coords[fdim].values)
                    print(f"{group.upper():<12} -> {f}")
                else:
                    print(f"{group.upper():<12} -> (no {fdim} coord)")
            else:
                print(f"{group.upper():<12} -> (not present)")
        print("=======================================================\n")
        stats_path = args.stats_path or Path(zarr_path).with_suffix(".stats.json")
        compute_and_save_feature_stats(zarr_path, stats_path,feature_norm_cfg=feature_norm_cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
