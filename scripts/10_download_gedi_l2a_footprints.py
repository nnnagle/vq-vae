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
#   3. Stream/download the matching .h5 granules to a local cache.
#   4. For each granule + beam, read the requested RH percentiles and the
#      curated metadata columns, apply quality filtering, and clip to the
#      Virginia boundary (bbox, then precise polygon if provided).
#   5. Reproject to the AEA_WGS84 grid and snap to (row, col, x_off, y_off);
#      clip to the grid extent (or flag off-grid shots with --keep-off-grid).
#   6. Write GeoParquet, partitioned by year (default), per-granule, or as a
#      single combined file.
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
#   # 1. Full Virginia archive, quality shots only, partitioned by year:
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a \
#       --quality-only
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
def _find_grid_in_zarr(root):
    """Locate 1-D 'x'/'y' coordinate arrays and a CRS WKT in a zarr tree.

    Walks the group hierarchy (build_zarr writes nested groups) and returns
    (x_array, y_array, crs_wkt). Any element may be None if not found.
    """
    x = y = crs_wkt = None

    def crs_from_attrs(attrs):
        for key in ("crs_wkt", "spatial_ref", "crs", "esri_pe_string"):
            val = attrs.get(key)
            if isinstance(val, str) and val:
                return val
        return None

    def visit(node):
        nonlocal x, y, crs_wkt
        # Arrays directly under this group.
        arrays = getattr(node, "arrays", None)
        items = arrays() if callable(arrays) else []
        for name, arr in items:
            base = name.split("/")[-1]
            if base == "x" and x is None and arr.ndim == 1:
                x = arr
            elif base == "y" and y is None and arr.ndim == 1:
                y = arr
            if crs_wkt is None:
                found = crs_from_attrs(dict(arr.attrs))
                if found:
                    crs_wkt = found
        # Group-level attrs may also carry the CRS.
        if crs_wkt is None:
            found = crs_from_attrs(dict(getattr(node, "attrs", {})))
            if found:
                crs_wkt = found
        # Recurse into subgroups.
        groups = getattr(node, "groups", None)
        subs = groups() if callable(groups) else []
        for _, g in subs:
            if x is not None and y is not None and crs_wkt is not None:
                break
            visit(g)

    try:
        visit(root)
    except Exception as e:  # pragma: no cover - defensive
        warn("Error while scanning zarr for grid metadata: %s", e)
    return x, y, crs_wkt


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
        and a `spatial_ref` variable carrying the CRS WKT. The affine transform
        is reconstructed from the cell spacing and the upper-left corner.
        """
        import zarr

        root = zarr.open(str(path), mode="r")
        x, y, crs_wkt = _find_grid_in_zarr(root)
        if x is None or y is None:
            fail(f"Could not find 'x'/'y' coordinate arrays in zarr: {path}. "
                 "Pass --grid-template or --grid-source constants instead.")

        x = np.asarray(x[:], dtype="float64")
        y = np.asarray(y[:], dtype="float64")
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


def download_granules(earthaccess, results, cache_dir: Path) -> List[Path]:
    """Download granules to a local cache; returns local .h5 paths."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(results, local_path=str(cache_dir))
    h5s = [Path(p) for p in paths if str(p).endswith(".h5")]
    log("Downloaded / cached %d granule files in %s", len(h5s), cache_dir)
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
# ---------------------------------------------------------------------
def write_output(gdf, out_dir: Path, partition: str, granule_stem: str,
                 accumulator: list):
    """Write (or accumulate) a granule's footprints per the partition mode."""
    if partition == "per-granule":
        out_path = out_dir / f"{granule_stem}.parquet"
        gdf.to_parquet(out_path, index=False)
        log("Wrote %d footprints -> %s", len(gdf), out_path.name)
    else:
        # 'year' and 'single' are both finalized after all granules are read.
        accumulator.append(gdf)


def finalize(accumulator: list, out_dir: Path, partition: str):
    """Flush accumulated footprints for 'year' / 'single' partition modes."""
    if partition == "per-granule" or not accumulator:
        return
    import geopandas as gpd
    allgdf = gpd.GeoDataFrame(pd.concat(accumulator, ignore_index=True),
                              crs=accumulator[0].crs)
    if partition == "single":
        out_path = out_dir / "gedi_l2a_virginia.parquet"
        allgdf.to_parquet(out_path, index=False)
        log("Wrote %d total footprints -> %s", len(allgdf), out_path)
    elif partition == "year":
        for yr, sub in allgdf.groupby("year"):
            ydir = out_dir / f"year={int(yr)}"
            ydir.mkdir(parents=True, exist_ok=True)
            out_path = ydir / "gedi_l2a_virginia.parquet"
            # Hive convention: the partition column lives in the path, NOT in
            # the file. Keeping an in-file `year` too makes the dataset reader
            # see two incompatible `year` columns (path dictionary vs int16).
            sub.drop(columns=["year"]).to_parquet(out_path, index=False)
            log("Wrote %d footprints -> %s", len(sub), out_path)


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
                   help="Keep raw .h5 granules after extraction (default: delete).")
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

    h5_paths = download_granules(ea, results, cache_dir)

    accumulator: list = []
    n_shots = 0
    for path in h5_paths:
        df = granule_to_df(path, args.rh)
        if df is None:
            continue
        df = apply_quality(df, args.quality_only, args.min_sensitivity)
        if df.empty:
            continue
        gdf = clip_to_virginia(df, bbox, args.boundary)
        if gdf is None or gdf.empty:
            continue
        gdf = snap_to_grid(gdf, grid, args.keep_off_grid)
        if gdf is None or gdf.empty:
            continue

        n_shots += len(gdf)
        write_output(gdf, out_dir, args.partition, path.stem, accumulator)

        if not args.keep_h5:
            try:
                path.unlink()
            except OSError:
                pass

    finalize(accumulator, out_dir, args.partition)

    if n_shots == 0:
        fail("No footprints survived filtering / clipping / grid snapping.")
    log("Done. %d Virginia GEDI L2A footprints written under %s",
        n_shots, out_dir)


if __name__ == "__main__":
    main()
