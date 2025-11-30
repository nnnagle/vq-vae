"""
Lightweight helpers around rasterio / rioxarray for lazy geospatial IO.

This module is intentionally narrow in scope:
- Never materialize full rasters in memory.
- Always return xarray.DataArray objects, chunked with dask when possible.
- Hide rasterio / rioxarray plumbing from higher-level code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import rasterio
from rasterio.io import DatasetReader
import xarray as xr


def _ensure_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster file not found: {p}")
    return p


def open_raster_as_dataarray(
    path: str | Path,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Open a raster file as an xarray.DataArray.

    For single-band rasters, returns (y, x).
    For multi-band rasters, returns (band, y, x).

    Parameters
    ----------
    path : str or Path
        Path to a raster readable by rasterio.
    chunks : dict, optional
        Dask chunking mapping (e.g. {"y": 256, "x": 256}). If provided,
        data will be loaded lazily with dask.

    Notes
    -----
    This uses rasterio + rioxarray. It does not compute() anything;
    callers remain responsible for triggering any actual reads.
    """
    p = _ensure_path(path)

    import rioxarray as rxr  # rioxarray required here

    da = rxr.open_rasterio(
        p,
        chunks=chunks,
    )
    return da


def get_bandname_map(path: str | Path) -> Dict[int, str]:
    """
    Return a mapping from band index (1-based) to band name.

    If a band has no explicit NAME tag, its value will be an empty string.
    """
    p = _ensure_path(path)
    band_map: Dict[int, str] = {}
    with rasterio.open(p) as ds:  # type: DatasetReader
        for bidx in range(1, ds.count + 1):
            tags = ds.tags(bidx) or {}
            band_map[bidx] = ds.descriptions[bidx - 1] or ""
    return band_map


def find_band_index_by_name(path: str | Path, bandname: str) -> int:
    """
    Find the 1-based band index in `path` whose name matches `bandname`.

    Matching is case-sensitive and exact. Raises ValueError if no band or
    more than one band matches.
    """
    band_map = get_bandname_map(path)
    matches = [b for b, name in band_map.items() if name == bandname]
    if not matches:
        raise ValueError(f"No band named {bandname!r} found in {path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple bands named {bandname!r} found in {path}: {matches}")
    return matches[0]


def open_sharded_rasters(
    files: Sequence[str | Path],
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Open multiple rasters that represent shards of one logical image.

    Returns
    -------
    xarray.DataArray
        A lazily stacked array with new dimension 'shard':
        (shard, y, x) or (shard, band, y, x) depending on inputs.

    Notes
    -----
    This does NOT mosaic shards into a single seamless raster. It simply
    stacks them along a new 'shard' dimension to avoid large in-memory
    mosaics. Higher-level code can decide how to interpret / combine them.
    """
    from dask import array as da

    paths = [_ensure_path(p) for p in files]
    if not paths:
        raise ValueError("open_sharded_rasters called with empty file list")

    arrays = []
    for p in paths:
        da_i = open_raster_as_dataarray(p, chunks=chunks)
        arrays.append(da_i)

    # xarray.concat will preserve band dimension if present.
    stacked = xr.concat(arrays, dim="shard")
    stacked = stacked.assign_coords(shard=range(len(arrays)))
    return stacked


def merge_sharded_bands_to_multiband(
    shards: xr.DataArray,
) -> xr.DataArray:
    """
    Convert (shard, band?, y, x) to a logical multiband (band, y, x).

    Because we do not want to materialize mosaics in memory, this function
    is conservative: if data appear to be spatial shards, it simply
    flattens shard+band into a single band dimension without resampling.

    In practice this is often enough if downstream operations are per-pixel
    and don't care which shard a pixel came from.

    Returns
    -------
    xarray.DataArray
        DataArray(band, y, x) where 'band' is a composite index.
    """
    if "band" in shards.dims:
        # Combine shard and band into a single composite band dimension.
        shards = shards.stack(composite_band=("shard", "band"))
        shards = shards.rename(composite_band="band")
    else:
        shards = shards.rename(shard="band")

    # Reassign a simple integer coordinate for bands
    shards = shards.assign_coords(band=range(1, shards.sizes["band"] + 1))
    return shards


def needs_reprojection(
    da: xr.DataArray,
    template: xr.DataArray,
) -> bool:
    """
    Return True if CRS or spatial transform differs between `da` and `template`.

    Assumes both DataArrays have rioxarray spatial metadata.
    """
    if not hasattr(da, "rio") or not hasattr(template, "rio"):
        raise AttributeError("Both arrays must be rioxarray-enabled (have .rio accessor).")

    da_crs = da.rio.crs
    tmpl_crs = template.rio.crs
    if da_crs != tmpl_crs:
        return True

    da_transform = da.rio.transform()
    tmpl_transform = template.rio.transform()
    return da_transform != tmpl_transform


def align_to_template(
    da: xr.DataArray,
    template: xr.DataArray,
    resampling: str = "nearest",
) -> xr.DataArray:
    """
    Reproject and resample `da` to match `template` grid if needed.

    If CRS and transform already match, returns `da` unchanged.
    """
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling

    if not needs_reprojection(da, template):
        return da

    try:
        resampling_enum = getattr(Resampling, resampling)
    except AttributeError as exc:
        raise ValueError(f"Unsupported resampling mode: {resampling!r}") from exc

    da = da.rio.reproject_match(template, resampling=resampling_enum)
    return da


def open_multiband_raster_for_years(
    files: Sequence[str | Path],
    bandname_template: str,
    years: Sequence[int],
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    From one or more multiband files, extract per-year bands based on
    bandname_template (e.g. "lcms_lu_{year}").

    Returns DataArray(time, y, x) for found years.
    """
    from collections import OrderedDict

    paths = [_ensure_path(p) for p in files]
    if not paths:
        raise ValueError("open_multiband_raster_for_years called with empty file list")

    year_to_da: "OrderedDict[int, xr.DataArray]" = OrderedDict()

    for year in years:
        bandname = bandname_template.format(year=year)
        found_da: Optional[xr.DataArray] = None
        for p in paths:
            try:
                bidx = find_band_index_by_name(p, bandname)
            except ValueError:
                continue
            da_full = open_raster_as_dataarray(p, chunks=chunks)
            
            if "band" in da_full.dims:
                # multiband: normal path
                da_band = da_full.sel(band=bidx, drop=True)
            else:
                # single-band raster: only makes sense if bidx == 1
                if bidx != 1:
                    raise ValueError(
                        f"Raster {p} has no 'band' dimension but requested band index {bidx}"
                    )
                    da_band = da_full  # already (y, x)

            found_da = da_band
            break

        if found_da is not None:
            year_to_da[year] = found_da

    if not year_to_da:
        raise ValueError(f"No matching bands for template {bandname_template!r}")

    stacked = xr.concat(list(year_to_da.values()), dim="time")
    stacked = stacked.assign_coords(time=list(year_to_da.keys()))
    return stacked


def open_per_year_rasters(
    year_to_files: Dict[int, Sequence[str | Path]],
    bandname: str,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Given mapping year -> files, open per-year rasters and extract one band.

    Returns DataArray(time, y, x) only for years with available data.
    """
    from collections import OrderedDict

    arrays: "OrderedDict[int, xr.DataArray]" = OrderedDict()

    for year, files in sorted(year_to_files.items()):
        if not files:
            continue
        paths = [_ensure_path(p) for p in files]
        found_da = None
        for p in paths:
            try:
                bidx = find_band_index_by_name(p, bandname)
            except ValueError:
                continue
            da_full = open_raster_as_dataarray(p, chunks=chunks)
            if "band" in da_full.dims:
                da_band = da_full.sel(band=bidx, drop=True)
            else:
                if bidx != 1:
                    raise ValueError(
                    f"Raster {p} has no 'band' dimension but requested band index {bidx}"
                    )
                da_band = da_full
            found_da = da_band
            break
        if found_da is not None:
            arrays[year] = found_da

    if not arrays:
        raise ValueError("open_per_year_rasters: no usable rasters found")

    stacked = xr.concat(list(arrays.values()), dim="time")
    stacked = stacked.assign_coords(time=list(arrays.keys()))
    return stacked
