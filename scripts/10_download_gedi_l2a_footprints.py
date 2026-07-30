#!/usr/bin/env python3
# ================================================================
# Script: scripts/10_download_gedi_l2a_footprints.py
#
# Purpose:
#   Download GEDI L2A (Elevation and Height Metrics) footprints for the
#   state of Virginia from NASA Earthdata via the `earthaccess` library,
#   subset the individual shots to the Virginia boundary, snap each shot
#   onto the project's 30 m Albers (AEA_WGS84) raster grid, keep a curated
#   set of relative-height (RH) metrics plus the quality/geolocation
#   fields needed to actually use the data, and write the result to a
#   grid-aligned GeoParquet "data store".
#
#   GEDI L2A is distributed as HDF5 granules. Each granule holds up to
#   8 BEAM groups; every beam contains one record per laser shot. The
#   `rh` dataset is a [n_shots x 101] array giving relative height at
#   percentiles 0..100 (metres, relative to elev_lowestmode / ground).
#
# Why grid-snapping (the "ingest" step):
#   The training data cube is a single, heavily chunked Zarr raster on a
#   fixed 30 m Albers grid (see scripts/09_extract_topo_to_gcs.py for the
#   canonical CRS + transform). ForestDatasetV2 reads dense patch windows
#   and every loss samples integer (row, col) pixel indices on that grid.
#   GEDI footprints are sparse points, so we DO NOT rasterize them into
#   the cube. Instead each shot is reprojected onto that exact grid and
#   given integer (row, col) pixel indices (plus sub-pixel offsets, since
#   GEDI's ~10 m geolocation error is a third of a pixel). The output is a
#   sparse, join-ready table: downstream code joins footprints to embeddings
#   / cube pixels on (row, col[, year]) without ever building a dense array.
#
# What "footprint" means here:
#   One row == one GEDI shot (~25 m footprint on the ground), located at
#   (lon_lowestmode, lat_lowestmode). This is the true point geometry,
#   not a rasterized/gridded product.
#
# Pipeline:
#   1. Authenticate to NASA Earthdata (earthaccess.login()).
#   2. Search the GEDI L2A collection over the Virginia bounding box and
#      an optional date range.
#   3. Process in batches (--batch-size): download a batch of granules
#      concurrently (--threads), then for each granule + beam read the
#      requested RH percentiles and curated metadata, quality-filter, clip
#      to the Virginia boundary (bbox, then precise polygon if provided),
#      reproject to the AEA_WGS84 grid, snap to (row, col, x_off, y_off),
#      write the batch's GeoParquet, and delete the batch's .h5 before the
#      next batch. Peak disk stays ~batch_size GB (not the ~1.8 TB archive)
#      and the run is resumable — completed part files survive a crash.
#   4. Output is partitioned by year (default; year=YYYY/part-*.parquet),
#      per-granule, or a single combined file.
#
# Why batches: the job is NETWORK-bound, not CPU-bound. The parallelism that
# helps is concurrent downloads (--threads), which earthaccess does within a
# batch. Adding CPU cores or running many processes in parallel does not help
# (one shared network link) and risks tripping NASA rate limits.
#
# Grid source (how the target grid is defined):
#   The sidecar table is meaningless without the cube it indexes into, so by
#   default the grid is read straight from the training Zarr — its x/y cell
#   coordinates and spatial_ref CRS are the single source of truth. If the
#   grid ever changes, the sidecar re-syncs automatically.
#     - default: --zarr PATH (or $ZARR_ROOT/va_vae_dataset.zarr). Reads
#       CRS + transform + shape from the cube's coordinates.
#     - --grid-template PATH: lift the grid from a template raster (rasterio).
#     - --grid-source constants: hardcoded AEA_WGS84 CRS + TARGET_TRANSFORM
#       below (offline fallback; kept in sync with scripts/09).
#
# Dependencies (not part of the base env — install as needed):
#   pip install earthaccess h5py numpy pandas geopandas shapely pyproj pyarrow
#   # --grid-template also needs: rasterio
#
# Earthdata credentials (any one of):
#   - interactive: earthaccess.login()  will prompt
#   - ~/.netrc entry for urs.earthdata.nasa.gov
#   - env vars EARTHDATA_USERNAME / EARTHDATA_PASSWORD
#
# ----------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------
#   # 1. PRODUCTION: full GEDI-era Virginia archive, quality shots only,
#   #    batched streaming (bounded disk, resumable), year-partitioned.
#   #    Point --cache-dir at scratch, NOT a synced folder (Dropbox etc.).
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a \
#       --cache-dir /scratch/$USER/gedi_cache \
#       --start 2019-04-01 --end 2025-01-01 \
#       --quality-only \
#       --grid-source constants \
#       --batch-size 32 --threads 16
#
#   #    Re-running the same command resumes: cached granules are skipped and
#   #    existing part-*.parquet files are left in place (new granules add new
#   #    part files). Delete the out-dir to start clean.
#
#   # 2. Restrict to a date range and a precise state-boundary clip:
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a \
#       --start 2019-04-01 --end 2023-12-31 \
#       --boundary data/aoi/virginia.geojson \
#       --quality-only
#
#   # 3. Read the snap grid straight from the training Zarr cube
#   #    (recommended — keeps the sidecar aligned to the actual data):
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a \
#       --zarr $ZARR_ROOT/va_vae_dataset.zarr \
#       --keep-h5
#
#   # 3b. Or define the grid from a template raster / offline constants:
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a --grid-template data/mask/mask.tif
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a --grid-source constants
#
#   # 4. Custom RH percentiles, single combined output file:
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a \
#       --rh 25 50 75 90 95 98 100 \
#       --partition single
#
#   # 5. Debug: just the first 3 granules, keep everything (no clip/quality):
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir /tmp/gedi_test --limit 3 --keep-off-grid
# ================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# utils.log is the house logging style used across scripts/
try:
    from utils.log import log, warn, fail
except Exception:  # allow running the file from outside the repo root
    def log(msg, *a):  print("[INFO]", msg % a if a else msg)
    def warn(msg, *a): print("[WARN]", msg % a if a else msg)
    def fail(msg, code=1): print("[ERROR]", msg); sys.exit(code)


# ---------------------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------------------

# GEDI L2A short name + version on NASA Earthdata (CMR).
GEDI_L2A_SHORT_NAME = "GEDI02_A"
GEDI_L2A_VERSION = "002"

# Virginia bounding box (lon_min, lat_min, lon_max, lat_max), WGS84.
# Generous padding so no boundary shots are missed before the precise clip.
VIRGINIA_BBOX = (-83.70, 36.53, -75.23, 39.47)

# ---- Target grid (must match scripts/09_extract_topo_to_gcs.py) ----
# Project's custom Albers Equal Area on WGS84. Snapping uses this so
# footprints land on the same pixel grid as the training data cube.
TARGET_CRS_WKT = (
    'PROJCS["AEA_WGS84",'
    '  GEOGCS["GCS_WGS_1984",'
    '    DATUM["WGS_1984",'
    '      SPHEROID["WGS_84",6378137,298.257223563]],'
    '    PRIMEM["Greenwich",0],'
    '    UNIT["Degree",0.0174532925199433]],'
    '  PROJECTION["Albers_Conic_Equal_Area"],'
    '  PARAMETER["False_Easting",0],'
    '  PARAMETER["False_Northing",0],'
    '  PARAMETER["Central_Meridian",-96],'
    '  PARAMETER["Standard_Parallel_1",29.5],'
    '  PARAMETER["Standard_Parallel_2",45.5],'
    '  PARAMETER["Latitude_Of_Origin",23],'
    '  UNIT["Meter",1]]'
)
# GDAL-style affine: [pixel_w, row_rot, x_origin, col_rot, pixel_h, y_origin].
# x_origin/y_origin are the UPPER-LEFT corner of pixel (0, 0); pixel_h < 0.
TARGET_TRANSFORM = [30.0, 0.0, 1089315.0, 0.0, -30.0, 1966485.0]
# Grid extent in projected coords: (x_min, y_min, x_max, y_max).
# Matches PADDED_REGION in scripts/09; used to derive n_rows / n_cols.
TARGET_EXTENT = (1089315.0, 1574805.0, 1795875.0, 1966485.0)

# The eight GEDI beams. 0000/0001/0010/0011 are "coverage" beams (lower
# power); 0101/0110/1000/1011 are "full-power" beams. Full-power beams
# penetrate dense canopy better — see `is_full_power` column below.
GEDI_BEAMS = [
    "BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011",
    "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011",
]
FULL_POWER_BEAMS = {"BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"}

# RH percentiles the user asked for, plus a few extras that are standard
# for canopy work. rh[p] is column index p in the [n_shots x 101] array.
DEFAULT_RH_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 98, 100]

# Per-shot scalar datasets to carry through. These are the fields that
# make GEDI footprints actually usable: geolocation, timing, quality,
# beam sensitivity, and the ground/canopy-top elevations.
#   dataset name in HDF5   ->  output column name
SCALAR_FIELDS = {
    "shot_number":        "shot_number",
    "lat_lowestmode":     "latitude",       # footprint ground location
    "lon_lowestmode":     "longitude",
    "lat_highestreturn":  "lat_highestreturn",
    "lon_highestreturn":  "lon_highestreturn",
    "elev_lowestmode":    "elev_ground",    # ground elevation (m, ref ellipsoid)
    "elev_highestreturn": "elev_canopy_top",
    "quality_flag":       "quality_flag",   # 1 = good, 0 = poor
    "degrade_flag":       "degrade_flag",   # >0 = degraded pointing/geoloc
    "sensitivity":        "sensitivity",    # beam sensitivity (0-1)
    "solar_elevation":    "solar_elevation",# <0 => night acquisition
    "num_detectedmodes":  "num_detectedmodes",
    "selected_algorithm": "selected_algorithm",
    "delta_time":         "delta_time",     # seconds since 2018-01-01
}

# GEDI epoch for delta_time -> UTC timestamp.
GEDI_EPOCH = np.datetime64("2018-01-01T00:00:00")


# ---------------------------------------------------------------------
# GRID DEFINITION
# ---------------------------------------------------------------------
# Subgroups the cube is likely to store x/y/spatial_ref under. build_zarr.py
# writes each dataset group (static, annual, ...) as its own xarray Dataset via
# rioxarray, so the coordinates live inside the groups, not at the root.
_ZARR_GRID_GROUPS = ["static", "annual", "irregular", "strata", "aoi"]


def _zarr_group_names(path: str) -> List[str]:
    """Best-effort list of subgroup names in a zarr store (version-agnostic)."""
    names: List[str] = []
    try:
        import zarr
        root = zarr.open_group(str(path), mode="r")
        for attr in ("group_keys", "groups"):
            fn = getattr(root, attr, None)
            if callable(fn):
                try:
                    for item in fn():
                        names.append(item[0] if isinstance(item, tuple) else item)
                    if names:
                        break
                except Exception:
                    continue
    except Exception:
        pass
    return names


def _crs_wkt_from_dataset(ds) -> Optional[str]:
    """Extract a CRS WKT from an xarray Dataset (rio accessor or spatial_ref)."""
    # 1) rioxarray, if installed and the CRS decoded.
    try:
        import rioxarray  # noqa: F401
        crs = ds.rio.crs
        if crs is not None:
            return crs.to_wkt()
    except Exception:
        pass
    # 2) The spatial_ref coordinate's attrs (rioxarray/CF convention).
    for cname in ("spatial_ref", "crs"):
        if cname in ds.coords or cname in ds.variables:
            attrs = ds[cname].attrs
            for key in ("crs_wkt", "spatial_ref", "esri_pe_string"):
                val = attrs.get(key)
                if isinstance(val, str) and val:
                    return val
    return None


def _read_grid_from_zarr(path: str):
    """Open the cube with xarray and return (x_vals, y_vals, crs_wkt).

    Tries the root and each known/discovered subgroup until one exposes 1-D
    `x` and `y` coordinates. Uses xarray so it is agnostic to the installed
    zarr-python version's Group API. Any element may be None if not found.
    """
    import xarray as xr

    candidates = [None] + _ZARR_GRID_GROUPS
    for name in _zarr_group_names(path):
        if name not in candidates:
            candidates.append(name)

    for grp in candidates:
        try:
            ds = xr.open_zarr(path, group=grp, consolidated=False,
                              mask_and_scale=False, decode_times=False)
        except Exception:
            continue
        if "x" in ds.coords and "y" in ds.coords:
            x = np.asarray(ds["x"].values, dtype="float64")
            y = np.asarray(ds["y"].values, dtype="float64")
            crs_wkt = _crs_wkt_from_dataset(ds)
            ds.close()
            log("Read grid from zarr group %r (%d x %d).",
                grp if grp is not None else "<root>", y.size, x.size)
            return x, y, crs_wkt
        ds.close()
    return None, None, None


class TargetGrid:
    """Target raster grid used to snap footprints to (row, col) pixel indices.

    Holds a CRS (as WKT/authority string), a GDAL-style affine transform, and
    the grid size (n_rows, n_cols). Provides world -> pixel snapping.
    """

    def __init__(self, crs, transform: List[float], n_rows: int, n_cols: int):
        self.crs = crs
        self.transform = list(transform)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)

    @classmethod
    def from_constants(cls) -> "TargetGrid":
        px_w, _, x0, _, px_h, y0 = TARGET_TRANSFORM
        x_min, y_min, x_max, y_max = TARGET_EXTENT
        n_cols = int(round((x_max - x_min) / abs(px_w)))
        n_rows = int(round((y_max - y_min) / abs(px_h)))
        return cls(TARGET_CRS_WKT, TARGET_TRANSFORM, n_rows, n_cols)

    @classmethod
    def from_zarr(cls, path: str) -> "TargetGrid":
        """Read the grid from the training Zarr's x/y coords + spatial_ref CRS.

        The cube is written by build_zarr.py via rioxarray, so it follows the
        CF/rioxarray convention: 1-D `x` and `y` cell-CENTRE coordinate arrays
        and a `spatial_ref` variable carrying the CRS WKT — stored inside each
        dataset subgroup (static, annual, ...), not at the root. The affine
        transform is reconstructed from the cell spacing and the UL corner.
        """
        x, y, crs_wkt = _read_grid_from_zarr(str(path))
        if x is None or y is None:
            fail(f"Could not find 'x'/'y' coordinates in zarr: {path}. "
                 "Pass --grid-template or --grid-source constants instead.")

        if x.size < 2 or y.size < 2:
            fail("Zarr x/y coordinates too small to infer pixel size.")

        px_w = float(x[1] - x[0])          # +ve, west->east
        px_h = float(y[1] - y[0])          # -ve when north->south (typical)
        # x/y are cell centres; shift to the upper-left corner of pixel (0,0).
        x0 = float(x[0]) - px_w / 2.0
        y0 = float(y[0]) - px_h / 2.0
        transform = [px_w, 0.0, x0, 0.0, px_h, y0]

        if crs_wkt is None:
            warn("No spatial_ref/CRS found in zarr; falling back to AEA_WGS84 WKT.")
            crs_wkt = TARGET_CRS_WKT
        return cls(crs_wkt, transform, y.size, x.size)

    @classmethod
    def from_template(cls, path: str) -> "TargetGrid":
        try:
            import rasterio
        except ImportError:
            fail("--grid-template requires rasterio. pip install rasterio")
        with rasterio.open(path) as ds:
            t = ds.transform  # affine.Affine(a, b, c, d, e, f)
            transform = [t.a, t.b, t.c, t.d, t.e, t.f]
            crs = ds.crs.to_wkt()
            return cls(crs, transform, ds.height, ds.width)

    def world_to_pixel(self, x: np.ndarray, y: np.ndarray):
        """Map projected (x, y) to fractional pixel coords.

        Returns (col_f, row_f) as floats. Assumes an axis-aligned transform
        (no rotation), which holds for this project's grid.
        """
        px_w, _, x0, _, px_h, y0 = self.transform
        col_f = (x - x0) / px_w
        row_f = (y - y0) / px_h  # px_h is negative
        return col_f, row_f


def snap_to_grid(gdf, grid: "TargetGrid", keep_off_grid: bool):
    """Reproject footprints to the grid CRS and add pixel-index columns.

    Adds:
      row, col            integer pixel indices (int32)
      x_off, y_off        sub-pixel offset within the pixel, in metres,
                          measured from the pixel's upper-left corner (0..30)
      x_aea, y_aea        projected coordinates (metres)
      in_grid             bool: pixel index falls inside the cube extent

    With keep_off_grid=False, rows where in_grid is False are dropped.
    """
    import geopandas as gpd  # noqa: F401  (gdf is already a GeoDataFrame)

    proj = gdf.to_crs(grid.crs)
    x = proj.geometry.x.to_numpy()
    y = proj.geometry.y.to_numpy()

    col_f, row_f = grid.world_to_pixel(x, y)
    col = np.floor(col_f).astype("int64")
    row = np.floor(row_f).astype("int64")

    px_w = grid.transform[0]
    px_h = grid.transform[4]
    x_off = (col_f - col) * px_w
    y_off = (row_f - row) * abs(px_h)

    in_grid = (row >= 0) & (row < grid.n_rows) & (col >= 0) & (col < grid.n_cols)

    out = gdf.copy()
    out["x_aea"] = x.astype("float64")
    out["y_aea"] = y.astype("float64")
    out["row"] = row.astype("int32")
    out["col"] = col.astype("int32")
    out["x_off"] = x_off.astype("float32")
    out["y_off"] = y_off.astype("float32")
    out["in_grid"] = in_grid

    if not keep_off_grid:
        out = out.loc[in_grid].reset_index(drop=True)
    return out if len(out) else None


# ---------------------------------------------------------------------
# EARTHDATA SEARCH / DOWNLOAD
# ---------------------------------------------------------------------
def earthdata_login():
    """Authenticate to NASA Earthdata. Tries netrc/env, then interactive."""
    try:
        import earthaccess
    except ImportError:
        fail("earthaccess not installed. Run: pip install earthaccess h5py "
             "geopandas shapely pyproj pyarrow")
    try:
        auth = earthaccess.login(strategy="netrc")
    except Exception:
        auth = None
    if not auth or not getattr(auth, "authenticated", False):
        auth = earthaccess.login(strategy="interactive", persist=True)
    if not getattr(auth, "authenticated", False):
        fail("Earthdata authentication failed. Check ~/.netrc or "
             "EARTHDATA_USERNAME / EARTHDATA_PASSWORD.")
    log("Authenticated to NASA Earthdata.")
    return earthaccess


def search_granules(earthaccess, bbox, start: Optional[str], end: Optional[str]):
    """Search the GEDI L2A collection over the Virginia bbox + date range."""
    kwargs = dict(
        short_name=GEDI_L2A_SHORT_NAME,
        version=GEDI_L2A_VERSION,
        bounding_box=bbox,
    )
    if start and end:
        kwargs["temporal"] = (start, end)
    results = earthaccess.search_data(**kwargs)
    log("Found %d GEDI L2A granules intersecting Virginia.", len(results))
    return results


def granule_stem(result) -> Optional[str]:
    """Best-effort .h5 filename stem for a search result (for resume skipping).

    Returns None if it can't be determined, in which case the granule is not
    skipped (safe: it just gets re-processed, which is idempotent).
    """
    from urllib.parse import urlparse
    try:
        for url in result.data_links():
            name = Path(urlparse(url).path).name
            if name.endswith(".h5"):
                return Path(name).stem
    except Exception:
        pass
    return None


def download_granules(earthaccess, results, cache_dir: Path,
                      threads: int = 8) -> List[Path]:
    """Download a set of granules to a local cache; returns local .h5 paths.

    `threads` controls how many granules download concurrently (earthaccess
    uses a thread pool). This is the parallelism that matters for this
    workload — it is network-bound, not CPU-bound. Already-cached files are
    skipped, so this is safe to re-run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(results, local_path=str(cache_dir),
                                 threads=threads)
    h5s = [Path(p) for p in paths if str(p).endswith(".h5")]
    return h5s


# ---------------------------------------------------------------------
# HDF5 -> DataFrame
# ---------------------------------------------------------------------
def read_beam(h5, beam: str, rh_percentiles: List[int]) -> Optional[pd.DataFrame]:
    """Read one BEAM group into a DataFrame of shots. None if empty/missing."""
    if beam not in h5:
        return None
    g = h5[beam]
    if "rh" not in g:
        return None

    n = g["rh"].shape[0]
    if n == 0:
        return None

    cols = {}
    # RH metrics (index p == pth percentile).
    rh = g["rh"][:]  # [n, 101]
    for p in rh_percentiles:
        cols[f"rh{p}"] = rh[:, p].astype("float32")

    # Scalar fields.
    for src, dst in SCALAR_FIELDS.items():
        if src in g:
            cols[dst] = g[src][:]
        else:
            cols[dst] = np.full(n, np.nan)

    df = pd.DataFrame(cols)
    df["beam"] = beam
    df["is_full_power"] = beam in FULL_POWER_BEAMS
    return df


def granule_to_df(path: Path, rh_percentiles: List[int]) -> Optional[pd.DataFrame]:
    """Read every beam in a granule into a single DataFrame."""
    import h5py
    frames = []
    try:
        with h5py.File(path, "r") as h5:
            for beam in GEDI_BEAMS:
                df = read_beam(h5, beam, rh_percentiles)
                if df is not None and len(df):
                    frames.append(df)
    except Exception as e:
        warn("Failed to read %s: %s", path.name, e)
        return None
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["source_granule"] = path.name
    # Absolute UTC time from delta_time (seconds since 2018-01-01).
    if "delta_time" in out:
        out["time_utc"] = GEDI_EPOCH + (out["delta_time"].to_numpy()
                                        * 1e9).astype("timedelta64[ns]")
        out["year"] = pd.DatetimeIndex(out["time_utc"]).year.astype("int16")
    return out


# ---------------------------------------------------------------------
# FILTERING / SUBSETTING
# ---------------------------------------------------------------------
def apply_quality(df: pd.DataFrame, quality_only: bool,
                  min_sensitivity: float) -> pd.DataFrame:
    """Standard GEDI quality gate: quality_flag==1, degrade_flag==0, sens>=thr."""
    if not quality_only:
        return df
    m = (df["quality_flag"] == 1) & (df["degrade_flag"] == 0)
    if min_sensitivity is not None:
        m &= df["sensitivity"] >= min_sensitivity
    return df.loc[m].reset_index(drop=True)


def clip_to_virginia(df: pd.DataFrame, bbox, boundary_path: Optional[str]):
    """Bbox filter, then precise point-in-polygon clip if a boundary is given.

    Returns a GeoDataFrame (EPSG:4326) with point geometry, or None if empty.
    """
    import geopandas as gpd

    lon_min, lat_min, lon_max, lat_max = bbox
    m = (df["longitude"].between(lon_min, lon_max) &
         df["latitude"].between(lat_min, lat_max))
    df = df.loc[m].reset_index(drop=True)
    if df.empty:
        return None

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    if boundary_path:
        boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        poly = boundary.union_all() if hasattr(boundary, "union_all") \
            else boundary.unary_union
        gdf = gdf[gdf.within(poly)].reset_index(drop=True)
    return gdf if len(gdf) else None


# ---------------------------------------------------------------------
# WRITING
#
# Streaming-friendly: 'per-granule' and 'year' write immediately per batch so
# a crash never loses more than the current batch and the run is resumable.
# 'single' must accumulate (one file can't be appended to), so it holds
# footprints in memory and writes once at the end.
# ---------------------------------------------------------------------
def write_batch(gdf, out_dir: Path, partition: str, tag: str,
                accumulator: list):
    """Write a batch's footprints immediately, or accumulate for 'single'.

    `tag` is a unique, stable label for this batch (used in the output file
    name) so concurrent/streamed batches never clobber each other.
    """
    if partition == "single":
        accumulator.append(gdf)
        return

    if partition == "per-granule":
        out_path = out_dir / f"{tag}.parquet"
        gdf.to_parquet(out_path, index=False)
        log("Wrote %d footprints -> %s", len(gdf), out_path.name)
        return

    # partition == "year": one part file per year present in this batch.
    # Hive convention: the partition column lives in the path, NOT in the
    # file (an in-file `year` collides with the path-derived one on read).
    for yr, sub in gdf.groupby("year"):
        ydir = out_dir / f"year={int(yr)}"
        ydir.mkdir(parents=True, exist_ok=True)
        out_path = ydir / f"part-{tag}.parquet"
        sub.drop(columns=["year"]).to_parquet(out_path, index=False)
        log("Wrote %d footprints -> %s", len(sub), out_path)


def finalize(accumulator: list, out_dir: Path, partition: str):
    """Flush accumulated footprints for 'single' mode (no-op otherwise)."""
    if partition != "single" or not accumulator:
        return
    import geopandas as gpd
    allgdf = gpd.GeoDataFrame(pd.concat(accumulator, ignore_index=True),
                              crs=accumulator[0].crs)
    out_path = out_dir / "gedi_l2a_virginia.parquet"
    allgdf.to_parquet(out_path, index=False)
    log("Wrote %d total footprints -> %s", len(allgdf), out_path)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download GEDI L2A footprints (RH metrics) for Virginia "
                    "from NASA Earthdata, snap them to the project's 30 m "
                    "Albers grid, and write a join-ready GeoParquet store.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for GeoParquet footprint files.")
    p.add_argument("--cache-dir", default=None,
                   help="Where to cache downloaded .h5 granules "
                        "(default: <out-dir>/_granules).")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD.")
    p.add_argument("--bbox", nargs=4, type=float, default=list(VIRGINIA_BBOX),
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="Search/clip bounding box (WGS84).")
    p.add_argument("--boundary", default=None,
                   help="Optional polygon (GeoJSON/SHP/GPKG) for a precise "
                        "point-in-polygon clip to Virginia.")
    p.add_argument("--zarr", default=None,
                   help="Training Zarr cube to read the snap grid from "
                        "(default: $ZARR_ROOT/va_vae_dataset.zarr if it exists).")
    p.add_argument("--grid-template", default=None,
                   help="Raster whose CRS+transform+extent define the snap grid "
                        "(overrides --zarr).")
    p.add_argument("--grid-source", choices=["zarr", "constants"], default=None,
                   help="Force the grid source. 'constants' uses the hardcoded "
                        "AEA_WGS84 grid (offline). Default: auto (zarr if found, "
                        "else constants).")
    p.add_argument("--rh", type=int, nargs="+", default=DEFAULT_RH_PERCENTILES,
                   help="RH percentiles to keep (0-100).")
    p.add_argument("--quality-only", action="store_true",
                   help="Keep only quality_flag==1 & degrade_flag==0 shots.")
    p.add_argument("--min-sensitivity", type=float, default=0.9,
                   help="Min beam sensitivity when --quality-only (default 0.9).")
    p.add_argument("--keep-off-grid", action="store_true",
                   help="Keep shots outside the grid extent (flagged in_grid=False) "
                        "instead of dropping them.")
    p.add_argument("--partition", choices=["year", "per-granule", "single"],
                   default="year",
                   help="Output layout (default: year=YYYY/ partitions).")
    p.add_argument("--keep-h5", action="store_true",
                   help="Keep raw .h5 granules after extraction (default: delete). "
                        "Iteration/debug only — a full run keeps ~1.8 TB of HDF5.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Granules downloaded+processed+deleted per batch. Bounds "
                        "peak disk to ~batch_size GB and makes the run resumable "
                        "(default: 32).")
    p.add_argument("--threads", type=int, default=8,
                   help="Concurrent downloads within a batch (default: 8). This is "
                        "the only parallelism that helps — the job is network-bound, "
                        "not CPU-bound. Raise cautiously; too many trips NASA limits.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N granules (debugging).")
    return p.parse_args(argv)


def resolve_grid(args) -> "TargetGrid":
    """Pick the snap grid: explicit template/constants > zarr > auto-default."""
    import os

    if args.grid_source == "constants":
        log("Grid source: hardcoded AEA_WGS84 constants.")
        return TargetGrid.from_constants()
    if args.grid_template:
        log("Grid source: template raster %s", args.grid_template)
        return TargetGrid.from_template(args.grid_template)

    # Resolve a zarr path: explicit flag, then $ZARR_ROOT default.
    zarr_path = args.zarr
    if zarr_path is None:
        root = os.environ.get("ZARR_ROOT")
        if root:
            candidate = Path(root) / "va_vae_dataset.zarr"
            if candidate.exists():
                zarr_path = str(candidate)

    if args.grid_source == "zarr" and not zarr_path:
        fail("--grid-source zarr requires --zarr PATH (or $ZARR_ROOT set).")

    if zarr_path:
        if not Path(zarr_path).exists():
            fail(f"Zarr path not found: {zarr_path}")
        log("Grid source: training zarr %s", zarr_path)
        return TargetGrid.from_zarr(zarr_path)

    warn("No zarr found; falling back to hardcoded AEA_WGS84 constants. "
         "Pass --zarr to bind the sidecar grid to the actual cube.")
    return TargetGrid.from_constants()


def main(argv=None):
    args = parse_args(argv)

    for p in args.rh:
        if not 0 <= p <= 100:
            fail(f"RH percentile out of range 0-100: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "_granules"

    grid = resolve_grid(args)
    log("Target grid: %d rows x %d cols, transform=%s",
        grid.n_rows, grid.n_cols, grid.transform)

    bbox = tuple(args.bbox)
    ea = earthdata_login()
    results = search_granules(ea, bbox, args.start, args.end)
    if not results:
        fail("No granules found for the given bbox/date range.")
    if args.limit:
        results = results[: args.limit]

    # Resume: skip granules already recorded as processed in a prior run.
    # 'single' mode can't resume (it writes one file at the very end), so the
    # manifest is only consulted for the streaming (per-granule / year) modes.
    manifest_path = out_dir / "_processed_granules.txt"
    done: set = set()
    if args.partition != "single" and manifest_path.exists():
        done = set(manifest_path.read_text().split())
    if done:
        before = len(results)
        results = [r for r in results if granule_stem(r) not in done]
        log("Resume: skipping %d already-processed granules; %d remain.",
            before - len(results), len(results))
    if not results:
        log("Nothing to do — all granules already processed. Output under %s", out_dir)
        return

    # Batched streaming: download a batch of granules concurrently, extract +
    # snap + write, delete the batch, then move on. Peak disk stays ~batch_size
    # granules (~GB each) instead of the full ~1.8 TB archive, and a crash only
    # costs the current batch — completed year=YYYY/part-*.parquet files remain.
    n_total = len(results)
    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    log("Processing %d granules in %d batch(es) of up to %d (threads=%d).",
        n_total, n_batches, args.batch_size, args.threads)

    manifest = (None if args.partition == "single"
                else open(manifest_path, "a"))
    accumulator: list = []
    n_shots = 0
    for bi in range(n_batches):
        batch = results[bi * args.batch_size:(bi + 1) * args.batch_size]
        log("Batch %d/%d: downloading %d granules...", bi + 1, n_batches, len(batch))
        h5_paths = download_granules(ea, batch, cache_dir, threads=args.threads)

        for path in h5_paths:
            df = granule_to_df(path, args.rh)
            if df is not None:
                df = apply_quality(df, args.quality_only, args.min_sensitivity)
                if not df.empty:
                    gdf = clip_to_virginia(df, bbox, args.boundary)
                    if gdf is not None and not gdf.empty:
                        gdf = snap_to_grid(gdf, grid, args.keep_off_grid)
                        if gdf is not None and not gdf.empty:
                            n_shots += len(gdf)
                            # Tag part files by granule stem so batches, and
                            # re-runs, never collide within a partition dir.
                            write_batch(gdf, out_dir, args.partition,
                                        path.stem, accumulator)

            # Mark processed (even if it yielded no footprints) so a resume
            # doesn't re-download it, then drop the .h5.
            if manifest is not None:
                manifest.write(path.stem + "\n")
                manifest.flush()
            if not args.keep_h5:
                try:
                    path.unlink()
                except OSError:
                    pass

    if manifest is not None:
        manifest.close()
    finalize(accumulator, out_dir, args.partition)

    if n_shots == 0:
        fail("No footprints survived filtering / clipping / grid snapping.")
    log("Done. %d Virginia GEDI L2A footprints written under %s",
        n_shots, out_dir)


if __name__ == "__main__":
    main()
