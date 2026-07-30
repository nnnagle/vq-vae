#!/usr/bin/env python3
# ================================================================
# Script: scripts/10_download_gedi_l2a_footprints.py
#
# Purpose:
#   Download GEDI L2A (Elevation and Height Metrics) footprints for the
#   state of Virginia from NASA Earthdata via the `earthaccess` library,
#   subset the individual shots to the Virginia boundary, keep a curated
#   set of relative-height (RH) metrics plus the quality/geolocation
#   fields needed to actually use the data, and write the result to a
#   GeoParquet "data store".
#
#   GEDI L2A is distributed as HDF5 granules. Each granule holds up to
#   8 BEAM groups; every beam contains one record per laser shot. The
#   `rh` dataset is a [n_shots x 101] array giving relative height at
#   percentiles 0..100 (metres, relative to elev_lowestmode / ground).
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
#   4. For each granule + beam, read the requested RH percentiles and
#      the curated metadata columns, apply quality filtering, and clip
#      to the Virginia boundary (bbox, then precise polygon if provided).
#   5. Concatenate and write GeoParquet (one file per granule by default,
#      or a single combined file with --single-file).
#
# Dependencies (not part of the base env — install as needed):
#   pip install earthaccess h5py numpy pandas geopandas shapely pyarrow
#
# Earthdata credentials (any one of):
#   - interactive: earthaccess.login()  will prompt
#   - ~/.netrc entry for urs.earthdata.nasa.gov
#   - env vars EARTHDATA_USERNAME / EARTHDATA_PASSWORD
#
# Example:
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a \
#       --start 2019-04-01 --end 2023-12-31 \
#       --quality-only
#
#   # Precise clip to a Virginia boundary polygon instead of just bbox:
#   python scripts/10_download_gedi_l2a_footprints.py \
#       --out-dir data/gedi/va_l2a --boundary data/aoi/virginia.geojson
# ================================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

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
# EARTHDATA SEARCH / DOWNLOAD
# ---------------------------------------------------------------------
def earthdata_login():
    """Authenticate to NASA Earthdata. Tries netrc/env, then interactive."""
    try:
        import earthaccess
    except ImportError:
        fail("earthaccess not installed. Run: pip install earthaccess h5py "
             "geopandas shapely pyarrow")
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
            arr = g[src][:]
            cols[dst] = arr
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

    Returns a GeoDataFrame (EPSG:4326) with point geometry.
    """
    import geopandas as gpd
    from shapely.geometry import box as shp_box

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
    else:
        # Keep a tidy bbox rectangle intersection (already bbox-filtered).
        _ = shp_box(*bbox)
    return gdf if len(gdf) else None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download GEDI L2A footprints (RH metrics) for Virginia "
                    "from NASA Earthdata into a GeoParquet data store.")
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
    p.add_argument("--rh", type=int, nargs="+", default=DEFAULT_RH_PERCENTILES,
                   help="RH percentiles to keep (0-100).")
    p.add_argument("--quality-only", action="store_true",
                   help="Keep only quality_flag==1 & degrade_flag==0 shots.")
    p.add_argument("--min-sensitivity", type=float, default=0.9,
                   help="Min beam sensitivity when --quality-only (default 0.9).")
    p.add_argument("--single-file", action="store_true",
                   help="Write one combined GeoParquet instead of per-granule.")
    p.add_argument("--keep-h5", action="store_true",
                   help="Keep raw .h5 granules after extraction (default: keep).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N granules (debugging).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    for p in args.rh:
        if not 0 <= p <= 100:
            fail(f"RH percentile out of range 0-100: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "_granules"

    bbox = tuple(args.bbox)
    ea = earthdata_login()
    results = search_granules(ea, bbox, args.start, args.end)
    if not results:
        fail("No granules found for the given bbox/date range.")
    if args.limit:
        results = results[: args.limit]

    h5_paths = download_granules(ea, results, cache_dir)

    combined = []
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

        n_shots += len(gdf)
        if args.single_file:
            combined.append(gdf)
        else:
            out_path = out_dir / (path.stem + ".parquet")
            gdf.to_parquet(out_path, index=False)
            log("Wrote %d footprints -> %s", len(gdf), out_path.name)

        if not args.keep_h5:
            try:
                path.unlink()
            except OSError:
                pass

    if args.single_file:
        if not combined:
            fail("No footprints survived filtering/clipping.")
        import geopandas as gpd
        allgdf = gpd.GeoDataFrame(pd.concat(combined, ignore_index=True),
                                  crs="EPSG:4326")
        out_path = out_dir / "gedi_l2a_virginia.parquet"
        allgdf.to_parquet(out_path, index=False)
        log("Wrote %d total footprints -> %s", len(allgdf), out_path)

    log("Done. %d Virginia GEDI L2A footprints written under %s",
        n_shots, out_dir)


if __name__ == "__main__":
    main()
