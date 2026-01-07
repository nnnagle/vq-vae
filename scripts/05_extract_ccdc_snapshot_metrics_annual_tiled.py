#!/usr/bin/env python3
"""
05_extract_ccdc_snapshot_metrics_annual_tiled.py

Export annual snapshot metrics (Sections 1 + 2) for tiled CCDC assets.

For each year in 2010-2024, this script:
  - Extracts snapshot state metrics (reflectance, indices, tasseled cap)
  - Extracts snapshot harmonic amplitude (if available)
  - Extracts snapshot trajectory metrics (velocity, duration, RMSE, derivatives)
  - Exports an annual GeoTIFF to GCS for every tile

Snapshot date: September 31 (falls back to the last valid day of September).
"""

from __future__ import annotations

import calendar
import datetime as _dt
import math
from dataclasses import dataclass

import ee

from utils.gee import ee_init, export_img_to_gcs

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ee_init()

# CCDC asset configuration (matches fitting script)
ASSET_ROOT = "projects/ee-nnnagle/assets/ccdc/ccdc_1986_2024"
TILE_PX = 4096
OVERLAP_PX = 0
SCALE_M = 30.0

# GCS export configuration
GCS_BUCKET = "va_rasters"
GCS_DIR = "ccdc_snapshots_annual_tiled"
BASE_NAME_PREFIX = "ccdc_snapshot_metrics"

# Target projection (matches fitting script)
TARGET_CRS = (
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

# Full padded region (matches fitting script)
PADDED_REGION = ee.Geometry.Rectangle(
    [1089315, 1574805, 1795875, 1966485],
    proj=TARGET_CRS,
    geodesic=False,
)
TARGET_TRANSFORM = [30, 0, 1089315, 0, -30, 1966485]

# Spectral bands for trajectory calculation
SPECTRAL_BANDS = ["GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
HARMONIC_BANDS = ["RED", "NIR", "SWIR1", "SWIR2"]

# Snapshot date configuration
EXTRACT_MONTH = 9
EXTRACT_DAY = 31
SNAPSHOT_YEARS = list(range(2010, 2025))

MAX_PIXELS = 1e13
EPS = 1e-6
MISSING = -9999


# -----------------------------------------------------------------------------
# SNAPSHOT DATES (Sep 31 targets -> last valid day of September)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SnapshotDate:
    key: str
    t_snap: float
    actual_date: _dt.date


def safe_date(year: int, month: int, day: int) -> _dt.date:
    """Return a valid date; fall back to the last day of the month if needed."""
    try:
        return _dt.date(year, month, day)
    except ValueError:
        last_day = calendar.monthrange(year, month)[1]
        return _dt.date(year, month, last_day)


def frac_year(date_obj: _dt.date) -> float:
    doy = date_obj.timetuple().tm_yday
    days = 366 if _dt.date(date_obj.year, 12, 31).timetuple().tm_yday == 366 else 365
    return date_obj.year + (doy - 1) / days


def build_snapshots(years: list[int]) -> dict[int, SnapshotDate]:
    snapshots: dict[int, SnapshotDate] = {}
    for year in years:
        actual_date = safe_date(year, EXTRACT_MONTH, EXTRACT_DAY)
        key = f"{year}_{EXTRACT_MONTH:02d}{EXTRACT_DAY:02d}"
        snapshots[year] = SnapshotDate(
            key=key,
            t_snap=frac_year(actual_date),
            actual_date=actual_date,
        )
    return snapshots


SNAPSHOTS = build_snapshots(SNAPSHOT_YEARS)


# -----------------------------------------------------------------------------
# TILING LOGIC (copied from fitting script)
# -----------------------------------------------------------------------------
def make_aligned_tiles(
    region: ee.Geometry,
    crs: str,
    crs_transform: list[float],
    scale_m: float,
    tile_px: int,
    overlap_px: int,
):
    """
    Create a client-side list of (row, col, geom, x_ul, y_ul) tiles aligned to crs_transform.

    Assumes north-up grid:
      crs_transform = [xScale, 0, xOrigin, 0, -yScale, yOrigin]
    """
    x_scale, x_shear, x0, y_shear, y_scale, y0 = crs_transform
    if x_shear != 0 or y_shear != 0:
        raise ValueError("This tiler assumes no rotation/shear.")
    if abs(x_scale) != scale_m or abs(y_scale) != scale_m:
        raise ValueError("crs_transform scale does not match scale_m.")

    tile_m = tile_px * scale_m
    overlap_m = overlap_px * scale_m

    # Region bounds in target CRS (client-side)
    ring = region.bounds(proj=crs, maxError=1).coordinates().get(0).getInfo()
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def snap_down(val, origin, step):
        return origin + math.floor((val - origin) / step) * step

    def snap_up(val, origin, step):
        return origin + math.ceil((val - origin) / step) * step

    # X: increasing east
    x_start = snap_down(xmin, x0, tile_m)
    x_end = snap_up(xmax, x0, tile_m)

    # Y: y0 is origin at top; y decreases south because y_scale is negative.
    y_start = snap_up(ymax, y0, tile_m)
    y_end = snap_down(ymin, y0, tile_m)

    tiles = []
    row = 0
    y_ul = y_start
    while y_ul > y_end:
        col = 0
        x_ul = x_start
        while x_ul < x_end:
            x1 = x_ul - overlap_m
            x2 = x_ul + tile_m + overlap_m
            y1 = y_ul - tile_m - overlap_m
            y2 = y_ul + overlap_m

            geom = ee.Geometry.Rectangle([x1, y1, x2, y2], proj=crs, geodesic=False)

            tiles.append((row, col, geom, x_ul, y_ul))
            col += 1
            x_ul += tile_m
        row += 1
        y_ul -= tile_m

    return tiles


def tile_transform(x_ul: float, y_ul: float, scale_m: float) -> list[float]:
    """Return a north-up crsTransform for a tile with given upper-left corner."""
    return [scale_m, 0, x_ul, 0, -scale_m, y_ul]


# -----------------------------------------------------------------------------
# CORE ARRAY UTILITIES - All array operations in one place
# -----------------------------------------------------------------------------
class ArrayOps:
    """Centralized array operations for CCDC data."""

    @staticmethod
    def segment_mask_at_t(t_start: ee.Image, t_end: ee.Image, t: ee.Image) -> ee.Image:
        """Boolean [segment] mask where tStart <= t <= tEnd."""
        return t_start.lte(t).And(t_end.gte(t))

    @staticmethod
    def has_any_segment(mask_1d: ee.Image) -> ee.Image:
        """Check if any segment matches mask -> scalar boolean."""
        return mask_1d.arrayReduce(ee.Reducer.anyNonZero(), [0]).arrayGet([0])

    @staticmethod
    def select_coefs_at_t(coefs_2d: ee.Image, mask_1d: ee.Image) -> ee.Image:
        """
        Select coefficients from segment matching mask.
        Input: coefs_2d [segment, coef], mask_1d [segment] boolean
        Output: [coef] 1D array
        """
        mask_float = mask_1d.toFloat()

        def get_coef_value(coef_idx):
            coef_idx = ee.Number(coef_idx)
            coef_col = coefs_2d.arraySlice(1, coef_idx, coef_idx.add(1))
            coef_1d = coef_col.arrayProject([0])
            masked = coef_1d.multiply(mask_float)
            return masked.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])

        coef_indices = ee.List([0, 1, 2, 3, 4, 5, 6, 7])
        coef_images = coef_indices.map(get_coef_value)
        result = ee.ImageCollection.fromImages(coef_images).toBands().toArray()

        has_match = ArrayOps.has_any_segment(mask_1d.gt(0.5))
        missing_array = ee.Image(
            ee.Array([MISSING, MISSING, MISSING, MISSING, MISSING, MISSING, MISSING, MISSING])
        ).toArray(0)

        return result.where(has_match.Not(), missing_array)

    @staticmethod
    def select_value_at_t(arr_1d: ee.Image, mask_1d: ee.Image) -> ee.Image:
        """
        Select value from segment matching mask.
        Input: arr_1d [segment], mask_1d [segment] boolean
        Output: scalar
        """
        masked = arr_1d.arrayMask(mask_1d)
        return masked.arrayReduce(ee.Reducer.firstNonNull(), [0]).arrayGet([0])


# -----------------------------------------------------------------------------
# SPECTRAL INDICES AND DERIVATIVES
# -----------------------------------------------------------------------------
def calc_ndvi(nir: ee.Image, red: ee.Image) -> ee.Image:
    """Calculate NDVI: (NIR - RED) / (NIR + RED)."""
    return nir.subtract(red).divide(nir.add(red).add(EPS))


def calc_nbr(nir: ee.Image, swir2: ee.Image) -> ee.Image:
    """Calculate NBR: (NIR - SWIR2) / (NIR + SWIR2)."""
    return nir.subtract(swir2).divide(nir.add(swir2).add(EPS))


def calc_ndmi(nir: ee.Image, swir1: ee.Image) -> ee.Image:
    """Calculate NDMI: (NIR - SWIR1) / (NIR + SWIR1)."""
    return nir.subtract(swir1).divide(nir.add(swir1).add(EPS))


def calc_index_derivative(A: ee.Image, B: ee.Image, dA: ee.Image, dB: ee.Image) -> ee.Image:
    """
    Calculate d/dt [(A-B)/(A+B)] = 2*(B*dA - A*dB)/(A+B)^2
    Used for exact derivatives of NDVI, NBR, NDMI.
    """
    denom = A.add(B).add(EPS)
    return B.multiply(dA).subtract(A.multiply(dB)).multiply(2.0).divide(denom.pow(2).add(EPS))


def tasseled_cap_5band(g: ee.Image, r: ee.Image, n: ee.Image, s1: ee.Image, s2: ee.Image, prefix: str) -> ee.Image:
    """
    5-band Tasseled Cap (no blue): Brightness/Greenness/Wetness.
    Coefficients: Zhai et al. (2022), RSE 274:112992, doi:10.1016/j.rse.2022.112992
    """
    tcb = (
        g.multiply(0.4596)
        .add(r.multiply(0.5046))
        .add(n.multiply(0.5458))
        .add(s1.multiply(0.4114))
        .add(s2.multiply(0.2589))
    ).rename(f"{prefix}_tcb")

    tcg = (
        g.multiply(-0.3374)
        .add(r.multiply(-0.4901))
        .add(n.multiply(0.7909))
        .add(s1.multiply(0.0177))
        .add(s2.multiply(-0.1416))
    ).rename(f"{prefix}_tcg")

    tcw = (
        g.multiply(0.2254)
        .add(r.multiply(0.3681))
        .add(n.multiply(0.2250))
        .add(s1.multiply(-0.6053))
        .add(s2.multiply(-0.6298))
    ).rename(f"{prefix}_tcw")

    return ee.Image.cat([tcb, tcg, tcw])


# -----------------------------------------------------------------------------
# SNAPSHOT METRICS - Prediction at specific dates
# -----------------------------------------------------------------------------
class SnapshotMetrics:
    """Generate snapshot metrics for a specific date."""

    def __init__(self, ccdc: ee.Image, t_snap: float, key: str):
        self.ccdc = ccdc
        self.t_snap = t_snap
        self.t = ee.Image.constant(t_snap)
        self.key = key
        self.prefix = f"snap_{key}"

        # Common arrays
        self.t_start = ccdc.select("tStart")
        self.t_end = ccdc.select("tEnd")
        self.dur = self.t_end.subtract(self.t_start)

        # Segment mask for this date
        self.seg_mask = ArrayOps.segment_mask_at_t(self.t_start, self.t_end, self.t)
        self.has_seg = ArrayOps.has_any_segment(self.seg_mask)

    def predict_band(self, band: str) -> ee.Image:
        """Predict band value at snapshot date (trend-only)."""
        coefs = self.ccdc.select(f"{band}_coefs")
        coefs_sel = ArrayOps.select_coefs_at_t(coefs, self.seg_mask)

        c0 = coefs_sel.arrayGet([0])
        c1 = coefs_sel.arrayGet([1])
        pred = c0.add(c1.multiply(self.t))

        return ee.Image(pred).where(self.has_seg.Not(), MISSING)

    def get_state_metrics(self) -> ee.Image:
        """Get snapshot state: reflectance, indices, tasseled cap."""
        g = self.predict_band("GREEN").rename(f"{self.prefix}_green")
        r = self.predict_band("RED").rename(f"{self.prefix}_red")
        n = self.predict_band("NIR").rename(f"{self.prefix}_nir")
        s1 = self.predict_band("SWIR1").rename(f"{self.prefix}_swir1")
        s2 = self.predict_band("SWIR2").rename(f"{self.prefix}_swir2")

        ndvi = calc_ndvi(n, r).rename(f"{self.prefix}_ndvi")
        nbr = calc_nbr(n, s2).rename(f"{self.prefix}_nbr")
        ndmi = calc_ndmi(n, s1).rename(f"{self.prefix}_ndmi")

        tc = tasseled_cap_5band(g, r, n, s1, s2, self.prefix)

        return ee.Image.cat([g, r, n, s1, s2, ndvi, nbr, ndmi, tc])

    def get_harmonic_metrics(self) -> ee.Image:
        """Get snapshot harmonics: amplitude for each band."""
        bands_list = []

        for band in HARMONIC_BANDS:
            coefs = self.ccdc.select(f"{band}_coefs")
            coefs_sel = ArrayOps.select_coefs_at_t(coefs, self.seg_mask)

            cos1 = coefs_sel.arrayGet([2])
            sin1 = coefs_sel.arrayGet([3])
            amp = cos1.pow(2).add(sin1.pow(2)).sqrt()

            amp = ee.Image(amp).where(self.has_seg.Not(), MISSING)
            bands_list.append(amp.rename(f"{self.prefix}_seasonal_amp_{band.lower()}"))

        return ee.Image.cat(bands_list)

    def get_trajectory_metrics(self) -> ee.Image:
        """Get snapshot trajectory: velocity, duration, RMSE, index derivatives."""
        slopes = {}
        slope_sq_list = []

        for band in SPECTRAL_BANDS:
            coefs = self.ccdc.select(f"{band}_coefs")
            coefs_sel = ArrayOps.select_coefs_at_t(coefs, self.seg_mask)
            slope = coefs_sel.arrayGet([1])
            slopes[band] = ee.Image(slope).where(self.has_seg.Not(), MISSING)
            slope_sq_list.append(slopes[band].pow(2))

        vel_sq = slope_sq_list[0]
        for sq in slope_sq_list[1:]:
            vel_sq = vel_sq.add(sq)
        vel = vel_sq.sqrt().rename(f"{self.prefix}_spectral_velocity")

        seg_dur = ArrayOps.select_value_at_t(self.dur, self.seg_mask)
        seg_dur = ee.Image(seg_dur).where(self.has_seg.Not(), MISSING).rename(f"{self.prefix}_segment_duration")

        rmse_list = []
        for band in SPECTRAL_BANDS:
            rmse = self.ccdc.select(f"{band}_rmse")
            rmse_val = ArrayOps.select_value_at_t(rmse, self.seg_mask)
            rmse_list.append(rmse_val)

        rmse_mean = rmse_list[0]
        for rmse_val in rmse_list[1:]:
            rmse_mean = rmse_mean.add(rmse_val)
        rmse_mean = (
            ee.Image(rmse_mean.divide(len(SPECTRAL_BANDS)))
            .where(self.has_seg.Not(), MISSING)
            .rename(f"{self.prefix}_rmse_mean")
        )

        nir_t = self.predict_band("NIR")
        red_t = self.predict_band("RED")
        swir1_t = self.predict_band("SWIR1")
        swir2_t = self.predict_band("SWIR2")

        dndvi = calc_index_derivative(nir_t, red_t, slopes["NIR"], slopes["RED"]).rename(f"{self.prefix}_dndvi_dt")
        dndmi = calc_index_derivative(nir_t, swir1_t, slopes["NIR"], slopes["SWIR1"]).rename(f"{self.prefix}_dndmi_dt")
        dnbr = calc_index_derivative(nir_t, swir2_t, slopes["NIR"], slopes["SWIR2"]).rename(f"{self.prefix}_dnbr_dt")

        return ee.Image.cat([vel, seg_dur, rmse_mean, dndvi, dndmi, dnbr])


# -----------------------------------------------------------------------------
# Main metrics builder
# -----------------------------------------------------------------------------
def build_snapshot_metrics(ccdc: ee.Image, t_snap: float, key: str, has_harmonics: bool) -> ee.Image:
    """Extract snapshot metrics for a single date (Sections 1 + 2)."""
    snap = SnapshotMetrics(ccdc, t_snap, key)
    parts = [snap.get_state_metrics()]
    if has_harmonics:
        parts.append(snap.get_harmonic_metrics())
    parts.append(snap.get_trajectory_metrics())
    return ee.Image.cat(parts).toFloat()


# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def process_tile(row: int, col: int, x_ul: float, y_ul: float, geom: ee.Geometry, dry_run: bool = False) -> None:
    """Process a single tile: load CCDC asset, extract metrics, export to GCS."""
    asset_id = f"{ASSET_ROOT}_r{row:03d}_c{col:03d}"

    print(f"\n{'='*80}")
    print(f"Processing tile r{row:03d}_c{col:03d}")
    print(f"  Asset: {asset_id}")
    print(f"  Upper-left: ({x_ul}, {y_ul})")
    print(f"{'='*80}")

    try:
        ccdc_img = ee.Image(asset_id)
        band_list = ccdc_img.bandNames().getInfo()
        bands = set(band_list)
        print(f"  Found {len(band_list)} bands")
    except Exception as e:
        print(f"  ERROR: Could not load asset: {e}")
        print(f"  Skipping tile r{row:03d}_c{col:03d}")
        return

    required_coef_bands = [f"{band}_coefs" for band in SPECTRAL_BANDS]
    missing_bands = [b for b in required_coef_bands if b not in bands]

    if missing_bands:
        print(f"  ERROR: Missing required bands: {missing_bands}")
        print(f"  Skipping tile r{row:03d}_c{col:03d}")
        return

    has_harmonics = all(f"{band}_coefs" in bands for band in HARMONIC_BANDS)
    tile_tr = tile_transform(x_ul, y_ul, SCALE_M)

    for year in SNAPSHOT_YEARS:
        snapshot = SNAPSHOTS[year]
        base_name = f"{BASE_NAME_PREFIX}_{snapshot.key}_r{row:03d}_c{col:03d}"
        print(f"  Building snapshot metrics for {snapshot.actual_date.isoformat()} ({snapshot.key})...")

        try:
            summary_img = build_snapshot_metrics(ccdc_img, snapshot.t_snap, snapshot.key, has_harmonics)
            output_bands = summary_img.bandNames().getInfo()
            print(f"    Generated {len(output_bands)} metric bands")
        except Exception as e:
            print(f"    ERROR: Failed to build metrics for {snapshot.key}: {e}")
            continue

        if dry_run:
            print("    DRY RUN: Would export snapshot metrics to GCS")
            print(f"      Bucket: {GCS_BUCKET}")
            print(f"      Path: {GCS_DIR}/{base_name}")
            continue

        print("    Exporting to GCS...")
        try:
            export_img_to_gcs(
                img=ee.Image(summary_img),
                aoi=geom,
                bucket=GCS_BUCKET,
                base_name=base_name,
                gcs_dir=GCS_DIR,
                crs=TARGET_CRS,
                crsTransform=tile_tr,
                maxPixels=MAX_PIXELS,
            )
            print(f"    ✓ Export task started for r{row:03d}_c{col:03d} {snapshot.key}")
        except Exception as e:
            print(f"    ERROR: Export failed for {snapshot.key}: {e}")


def main(dry_run: bool = False, max_tiles: int | None = None) -> None:
    """
    Process all CCDC tiles and export annual snapshot metrics to GCS.

    Args:
        dry_run: If True, don't actually start export tasks
        max_tiles: If set, only process first N tiles (for testing)
    """
    print("Generating tile grid...")
    tiles = make_aligned_tiles(
        region=PADDED_REGION,
        crs=TARGET_CRS,
        crs_transform=TARGET_TRANSFORM,
        scale_m=SCALE_M,
        tile_px=TILE_PX,
        overlap_px=OVERLAP_PX,
    )

    total_tiles = len(tiles)
    print(f"Found {total_tiles} tiles to process")

    if max_tiles:
        tiles = tiles[:max_tiles]
        print(f"Limiting to first {max_tiles} tiles for testing")

    if dry_run:
        print("\n*** DRY RUN MODE - No exports will be started ***\n")

    success_count = 0
    error_count = 0

    for i, (row, col, geom, x_ul, y_ul) in enumerate(tiles):
        print(f"\nProgress: {i + 1}/{len(tiles)}")

        try:
            process_tile(row, col, x_ul, y_ul, geom, dry_run=dry_run)
            success_count += 1
        except Exception as e:
            print(f"  FATAL ERROR processing tile r{row:03d}_c{col:03d}: {e}")
            error_count += 1
            continue

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total tiles: {len(tiles)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")

    if not dry_run and success_count > 0:
        print("\nExport tasks started successfully!")
        print("Check Earth Engine Tasks tab for progress")
        print(f"Outputs will be in: gs://{GCS_BUCKET}/{GCS_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract annual CCDC snapshot metrics from tiled assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Preview what would be exported without starting tasks",
    )

    parser.add_argument(
        "--max-tiles",
        "--max_tiles",
        type=int,
        default=None,
        metavar="N",
        help="Process only first N tiles (for testing)",
    )

    args = parser.parse_args()
    main(dry_run=args.dry_run, max_tiles=args.max_tiles)
