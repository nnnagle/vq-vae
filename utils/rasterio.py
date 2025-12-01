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
    bandname_template: Optional[str],
    years: Sequence[int],
    chunks: Optional[Mapping[str, int]] = None,
    band_index_start: Optional[int] = None,
    band_index_year_start: Optional[int] = None,
) -> xr.DataArray:
    """
    From one or more multiband files, extract per-year bands.

    This supports two mutually exclusive selection modes:

    1. Name-based mode (existing behavior)
       -----------------------------------
       - `bandname_template` is a format string, e.g. "lcms_lu_{year}".
       - For each `year` in `years`, we:
           * compute bandname = bandname_template.format(year=year)
           * search each file in `files` for a band whose DESCRIPTION/NAME
             matches that string (via `find_band_index_by_name`).
           * Once found, we open that file lazily with `open_raster_as_dataarray`
             and select that band.

       This is what you use for something like LCMS multiband GeoTIFFs
       where each band has a meaningful name.

    2. Index-based mode (for datasets without band descriptions)
       ---------------------------------------------------------
       - `bandname_template` is None.
       - `band_index_start` gives the 1-based band index corresponding
         to `band_index_year_start`.
       - `band_index_year_start` is the year associated with that first band.
         For each year, we compute:
             band_index(year) = band_index_start + (year - band_index_year_start)
       - We assume *all* files in `files` share the same band layout, so we
         just use the first file as the template for band indices.

       This is exactly what you want for a VRT or multiband TIFF where bands
       are in strict year order (e.g. [1985, 1986, ..., 2024]) but have no
       DESCRIPTION tags.

    Parameters
    ----------
    files : sequence of str or Path
        Sharded multiband rasters for the same logical layer.
    bandname_template : str or None
        Template for band names, e.g. "lcms_lu_{year}", or None to use
        index-based selection.
    years : sequence of int
        Global dense time axis (e.g. 1985..2024). Only years for which we
        can find/select a band will be included in the output.
    chunks : dict, optional
        Dask chunking mapping (e.g. {"y": 256, "x": 256}) to pass to
        `open_raster_as_dataarray`.
    band_index_start : int, optional
        1-based band index for the year `band_index_year_start` when using
        index-based mode. Ignored if `bandname_template` is not None.
    band_index_year_start : int, optional
        Year corresponding to `band_index_start` in index-based mode.
        If omitted, defaults to the minimum of `years`.

    Returns
    -------
    xarray.DataArray
        Lazily-loaded DataArray(time, y, x) for the years where we found
        a usable band.

    Raises
    ------
    ValueError
        - If neither bandname_template nor band_index_start is provided.
        - If both are provided (ambiguous configuration).
        - If no matching bands can be found for any year.
        - If the requested band index is out of range.
    """
    from collections import OrderedDict

    paths = [_ensure_path(p) for p in files]
    if not paths:
        raise ValueError("open_multiband_raster_for_years called with empty file list")

    # Enforce exactly one selection mode.
    if bandname_template is None and band_index_start is None:
        raise ValueError(
            "open_multiband_raster_for_years requires either "
            "bandname_template (name-based mode) or band_index_start "
            "(index-based mode)."
        )
    if bandname_template is not None and band_index_start is not None:
        raise ValueError(
            "Provide either bandname_template OR band_index_start, not both "
            f"(got bandname_template={bandname_template!r}, "
            f"band_index_start={band_index_start!r})."
        )

    # Default anchor year for index-based mode if not explicitly provided.
    if bandname_template is None and band_index_start is not None:
        if band_index_year_start is None:
            band_index_year_start = int(min(years))

    year_to_da: "OrderedDict[int, xr.DataArray]" = OrderedDict()

    # Always process years in sorted order for reproducibility.
    for year in sorted(years):
        found_da: Optional[xr.DataArray] = None

        if bandname_template is not None:
            # -----------------------------
            # Name-based selection
            # -----------------------------
            bandname = bandname_template.format(year=year)
            for p in paths:
                try:
                    bidx = find_band_index_by_name(p, bandname)
                except ValueError:
                    # This file doesn't have that named band; try the next file.
                    continue

                da_full = open_raster_as_dataarray(p, chunks=chunks)

                if "band" in da_full.dims:
                    # multiband: normal path; rioxarray uses 1-based band coord
                    da_band = da_full.sel(band=bidx, drop=True)
                else:
                    # Single-band raster: only valid if we're effectively asking for band 1.
                    if bidx != 1:
                        raise ValueError(
                            f"Raster {p} has no 'band' dimension but requested band index {bidx}"
                        )
                    da_band = da_full  # already (y, x)

                found_da = da_band
                break  # stop after finding the first file with this band

        else:
            # -----------------------------
            # Index-based selection
            # -----------------------------
            assert band_index_start is not None
            assert band_index_year_start is not None

            # Convert year to a 1-based band index via a simple offset.
            bidx = int(band_index_start + (year - band_index_year_start))

            if bidx < 1:
                raise ValueError(
                    f"Computed band index {bidx} for year {year} is < 1; "
                    f"check band_index_start={band_index_start} and "
                    f"band_index_year_start={band_index_year_start}"
                )

            # In index-based mode we assume all shards share the same band layout,
            # so it's enough to inspect the first file.
            p0 = paths[0]
            da_full = open_raster_as_dataarray(p0, chunks=chunks)

            if "band" in da_full.dims:
                # Make sure the requested band index is in range.
                n_bands = int(da_full.sizes["band"])
                if bidx > n_bands:
                    raise ValueError(
                        f"Requested band index {bidx} for year {year} exceeds "
                        f"number of bands ({n_bands}) in {p0}"
                    )
                da_band = da_full.sel(band=bidx, drop=True)
            else:
                # Single-band raster; only band 1 is valid.
                if bidx != 1:
                    raise ValueError(
                        f"Raster {p0} has no 'band' dimension but requested band index {bidx}"
                    )
                da_band = da_full  # already (y, x)

            found_da = da_band

        if found_da is not None:
            year_to_da[year] = found_da

    if not year_to_da:
        # For name-based mode, this keeps the old error message. For index mode
        # this still gives a useful hint that nothing usable was found.
        raise ValueError(
            f"No matching bands found for template={bandname_template!r} "
            f"and/or band_index_start={band_index_start!r}"
        )

    # Stack along a new time dimension, preserving chronological order.
    sorted_years = sorted(year_to_da.keys())
    stacked = xr.concat([year_to_da[y] for y in sorted_years], dim="time")
    stacked = stacked.assign_coords(time=sorted_years)
    return stacked


def open_per_year_rasters(
    year_to_files: Dict[int, Sequence[str | Path]],
    bandname: Optional[str],
    band_index: Optional[int] = None,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.DataArray:
    """
    Given mapping year -> files, open per-year rasters and extract one band.

    Supports two mutually exclusive selection modes:

    1. Name-based mode (existing behavior)
       -----------------------------------
       - `bandname` is a string, e.g. "NIR_var_7m" or "lc_class".
       - For each file, we use `find_band_index_by_name(p, bandname)` to
         locate the desired band by DESCRIPTION/NAME metadata.

    2. Index-based mode (for rasters with no band descriptions)
       ---------------------------------------------------------
       - `bandname` is None.
       - `band_index` is an integer (1-based) indicating which band to use
         in every per-year file.
       - This is the natural choice for VRTs or TIFFs where all tiles share
         the same band layout but have no useful band names.

    For each year:
      - We iterate over that year's tiles, extract the desired band
        lazily from each, and stack them along a 'shard' dimension.
      - We then lazily "mosaic" across tiles using .max(dim="shard"),
        which works well when tiles are non-overlapping and nodata is NaN
        (rioxarray default with mask_and_scale=True).

    Returns
    -------
    xarray.DataArray
        DataArray(time, y, x) only for years with available data.

    Notes
    -----
    - All IO is lazy via dask: we open rasters with chunks and never
      call .compute() here.
    - Years with no usable tiles are skipped entirely.
    """
    from collections import OrderedDict

    # Enforce a single selection mode.
    if bandname is None and band_index is None:
        raise ValueError(
            "open_per_year_rasters requires either a bandname "
            "(name-based mode) or a band_index (index-based mode)."
        )
    if bandname is not None and band_index is not None:
        raise ValueError(
            "Provide either bandname OR band_index, not both "
            f"(got bandname={bandname!r}, band_index={band_index!r})."
        )

    arrays: "OrderedDict[int, xr.DataArray]" = OrderedDict()

    for year, files in sorted(year_to_files.items()):
        if not files:
            continue

        paths = [_ensure_path(p) for p in files]
        shard_arrays = []

        for p in paths:
            # Decide how to get the band index for this file.
            if bandname is not None:
                # Name-based: try to find a band with the given DESCRIPTION.
                try:
                    bidx = find_band_index_by_name(p, bandname)
                except ValueError:
                    # this file doesn't have the band we want; skip it
                    continue
            else:
                # Index-based: same band index for every file.
                assert band_index is not None
                bidx = int(band_index)
                if bidx < 1:
                    raise ValueError(
                        f"Invalid band_index={bidx} for file {p}; "
                        "band indices are 1-based."
                    )

            da_full = open_raster_as_dataarray(p, chunks=chunks)

            if "band" in da_full.dims:
                # multiband: normal path; rioxarray uses 1-based band coords
                n_bands = int(da_full.sizes["band"])
                if bidx > n_bands:
                    # This tile doesn't have that many bands; skip it.
                    continue
                da_band = da_full.sel(band=bidx, drop=True)
            else:
                # Single-band raster: only valid if we're effectively asking for band 1
                if bidx != 1:
                    raise ValueError(
                        f"Raster {p} has no 'band' dimension but requested band index {bidx}"
                    )
                da_band = da_full  # already (y, x)

            shard_arrays.append(da_band)

        if not shard_arrays:
            # no usable tiles for this year; move on
            continue

        # Stack all tiles for this year along a 'shard' dimension.
        # This is fully lazy: each shard is dask-backed, and concat
        # just builds a combined graph.
        shard_stack = xr.concat(shard_arrays, dim="shard")
        shard_stack = shard_stack.assign_coords(shard=range(len(shard_arrays)))

        # rioxarray.open_rasterio usually decodes nodata to NaN;
        # reductions like .max() ignore NaN by default.
        # So we can lazily "mosaic" tiles by taking max across shards.
        # For non-overlapping tiles, each pixel comes from exactly one shard.
        mosaic = shard_stack.max(dim="shard")

        arrays[year] = mosaic

    if not arrays:
        raise ValueError("open_per_year_rasters: no usable rasters found")

    # Stack per-year mosaics into a time axis. Still lazy.
    stacked = xr.concat(list(arrays.values()), dim="time")
    stacked = stacked.assign_coords(time=list(arrays.keys()))
    return stacked
