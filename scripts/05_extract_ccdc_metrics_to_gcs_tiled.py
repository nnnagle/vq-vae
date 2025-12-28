#!/usr/bin/env python3
"""
05_extract_ccdc_metrics_tiled.py

Process ALL tiled CCDC assets and export comprehensive forest metrics to GCS.
Uses the same tiling logic as the fitting script to discover and process all tiles.

CCDC INPUT:
- Spectral bands: GREEN, RED, NIR, SWIR1, SWIR2
- Time period: 1986-2024
- Date format: Fractional years (e.g., 2015.5 = mid-2015)
- Coefficients convention assumed:
    coef[0] = intercept (at year 0)
    coef[1] = slope (per year)
    coef[2..] = harmonics (cos/sin pairs), if present

IMPORTANT CHANGE (this version):
- Snapshot metrics are no longer evaluated at tEnd of the last segment.
- Snapshot metrics are evaluated at *explicit target dates* (end-of-summer):
    Aug 31, 2024; Aug 31, 2022; Aug 31, 2020.
- Snapshot reflectance predictions are TREND-ONLY:
    y_hat(t) = c0 + c1 * t
  (No harmonic terms are added to reflectance for the snapshot predictions.)

TASSELED CAP (no blue band):
- We compute 5-band Tasseled Cap Brightness/Greenness/Wetness using:
  GREEN, RED, NIR, SWIR1, SWIR2 (blue omitted).
- Coefficients from:
  Zhai, Y., Roy, D.P., Martins, V.S., Zhang, H.K., Yan, L., Li, Z. (2022).
  "Conterminous United States Landsat-8 top of atmosphere and surface reflectance
   tasseled cap transformation coefficients." Remote Sensing of Environment 274:112992.
  DOI: 10.1016/j.rse.2022.112992

OUTPUT METRICS (band count depends on availability of magnitudes/harmonics):
--------------------------------------------------------------------------

SECTION 1: SNAPSHOT STATE (3 dates; 11 bands per date = 33 bands)
  For each date D in {2024-08-31, 2022-08-31, 2020-08-31}:
    Predicted reflectance (trend-only) at date D:
      snap_<YYYY_0831>_green
      snap_<YYYY_0831>_red
      snap_<YYYY_0831>_nir
      snap_<YYYY_0831>_swir1
      snap_<YYYY_0831>_swir2

    Derived indices (from predicted bands):
      snap_<YYYY_0831>_ndvi
      snap_<YYYY_0831>_nbr
      snap_<YYYY_0831>_ndmi

    Tasseled Cap (5-band, no blue; from predicted bands):
      snap_<YYYY_0831>_tcb
      snap_<YYYY_0831>_tcg
      snap_<YYYY_0831>_tcw

SECTION 1b: SNAPSHOT HARMONICS (3 dates; 4 bands per date = 12 bands)
  For each date D:
    Seasonal amplitude (first harmonic pair) from the segment that CONTAINS D:
      snap_<YYYY_0831>_seasonal_amp_red
      snap_<YYYY_0831>_seasonal_amp_nir
      snap_<YYYY_0831>_seasonal_amp_swir1
      snap_<YYYY_0831>_seasonal_amp_swir2

SECTION 2: SNAPSHOT TRAJECTORY (3 dates; 6 bands per date = 18 bands)
  For each date D (segment containing D):
      snap_<YYYY_0831>_spectral_velocity     # ||slope vector|| across GREEN,RED,NIR,SWIR1,SWIR2
      snap_<YYYY_0831>_segment_duration      # (tEnd - tStart) of the containing segment
      snap_<YYYY_0831>_rmse_mean             # mean RMSE across spectral bands for containing segment

      snap_<YYYY_0831>_dndvi_dt              # exact trend-only d/dt NDVI at D
      snap_<YYYY_0831>_dndmi_dt              # exact trend-only d/dt NDMI at D
      snap_<YYYY_0831>_dnbr_dt               # exact trend-only d/dt NBR  at D

SECTION 3: LONG-TERM STATE (unchanged from prior script)
  Duration-weighted mean reflectance (evaluated at segment midpoints):
    mean_green, mean_red, mean_nir, mean_swir1, mean_swir2
  Duration-weighted mean indices:
    mean_ndvi, mean_nbr, mean_ndmi
  Duration-weighted mean phenology (if harmonic coefs present):
    mean_seasonal_amp_red/nir/swir1/swir2

SECTION 3b: HARMONIC QUALITY & CONSISTENCY (unchanged from prior script; if present)
  SNR, variance, and summary metrics derived across all segments.

SECTION 4..8: LONG-TERM TRAJECTORY, DISTURBANCE HISTORY, VARIABILITY,
              RAPID LOSS EVENTS, LOSS RECOVERY METRICS (unchanged semantics)

NOTES:
- Segment selection for snapshot date D is per-pixel:
    choose the segment where tStart <= D <= tEnd
  If no segment contains D (rare but possible), snapshot outputs are set to -9999.
- Magnitudes use Euclidean distance: sqrt(sum(band_mag²)) where used.
- Missing values: -9999 for snapshot/state metrics when insufficient data.
"""

from __future__ import annotations

import math
import datetime as _dt
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
GCS_DIR = "ccdc_summaries_tiled"
BASE_NAME_PREFIX = "ccdc_metrics_1986_2024"

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

MAX_PIXELS = 1e13
EPS = 1e-6
MISSING = -9999


# -----------------------------------------------------------------------------
# SNAPSHOT DATES (Aug 31 targets) -> fractional years
# -----------------------------------------------------------------------------
def frac_year(year: int, month: int, day: int) -> float:
    d = _dt.date(year, month, day)
    doy = d.timetuple().tm_yday
    days = 366 if _dt.date(year, 12, 31).timetuple().tm_yday == 366 else 365
    return year + (doy - 1) / days


SNAPSHOTS = {
    "2024_0831": frac_year(2024, 8, 31),
    "2022_0831": frac_year(2022, 8, 31),
    "2020_0831": frac_year(2020, 8, 31),
}


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
    xScale, xShear, x0, yShear, yScale, y0 = crs_transform
    if xShear != 0 or yShear != 0:
        raise ValueError("This tiler assumes no rotation/shear.")
    if abs(xScale) != scale_m or abs(yScale) != scale_m:
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

    # Y: y0 is origin at top; y decreases south because yScale is negative.
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
    def get_coef(coefs_2d: ee.Image, coef_idx: int) -> ee.Image:
        """Extract coefficient at index from [segment, coef] -> [segment]."""
        return coefs_2d.arraySlice(1, coef_idx, coef_idx + 1).arrayProject([0])
    
    @staticmethod
    def get_intercept(coefs_2d: ee.Image) -> ee.Image:
        """Extract intercept (coef[0]) from [segment, coef] -> [segment]."""
        return ArrayOps.get_coef(coefs_2d, 0)
    
    @staticmethod
    def get_slope(coefs_2d: ee.Image) -> ee.Image:
        """Extract slope (coef[1]) from [segment, coef] -> [segment]."""
        return ArrayOps.get_coef(coefs_2d, 1)
    
    @staticmethod
    def get_harmonic_amp(coefs_2d: ee.Image) -> ee.Image:
        """Calculate first harmonic amplitude from [segment, coef] -> [segment]."""
        cos1 = ArrayOps.get_coef(coefs_2d, 2)
        sin1 = ArrayOps.get_coef(coefs_2d, 3)
        return cos1.pow(2).add(sin1.pow(2)).sqrt()
    
    @staticmethod
    def first_element(arr_1d: ee.Image) -> ee.Image:
        """Get first element from [segment] array -> scalar."""
        return arr_1d.arrayGet([0])
    
    @staticmethod
    def last_element(arr_1d: ee.Image) -> ee.Image:
        """Get last element from [segment] array -> scalar."""
        n = arr_1d.arrayLength(0)
        return arr_1d.arraySlice(0, n.subtract(1), n).arrayGet([0])
    
    @staticmethod
    def reduce_mean(arr_1d: ee.Image) -> ee.Image:
        """Mean of [segment] array -> scalar."""
        return arr_1d.arrayReduce(ee.Reducer.mean(), [0]).arrayGet([0])
    
    @staticmethod
    def reduce_sum(arr_1d: ee.Image) -> ee.Image:
        """Sum of [segment] array -> scalar."""
        return arr_1d.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
    
    @staticmethod
    def reduce_max(arr_1d: ee.Image) -> ee.Image:
        """Max of [segment] array -> scalar."""
        return arr_1d.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0])
    
    @staticmethod
    def reduce_min(arr_1d: ee.Image) -> ee.Image:
        """Min of [segment] array -> scalar."""
        return arr_1d.arrayReduce(ee.Reducer.min(), [0]).arrayGet([0])
    
    @staticmethod
    def reduce_variance(arr_1d: ee.Image) -> ee.Image:
        """Variance of [segment] array -> scalar."""
        return arr_1d.arrayReduce(ee.Reducer.variance(), [0]).arrayGet([0])
    
    @staticmethod
    def reduce_stddev(arr_1d: ee.Image) -> ee.Image:
        """Standard deviation of [segment] array -> scalar."""
        return arr_1d.arrayReduce(ee.Reducer.stdDev(), [0]).arrayGet([0])
    
    @staticmethod
    def segment_mask_at_t(tStart: ee.Image, tEnd: ee.Image, t: ee.Image) -> ee.Image:
        """Boolean [segment] mask where tStart <= t <= tEnd."""
        return tStart.lte(t).And(tEnd.gte(t))
    
    @staticmethod
    def has_any_segment(mask_1d: ee.Image) -> ee.Image:
        """Check if any segment matches mask -> scalar boolean."""
        return mask_1d.arrayReduce(ee.Reducer.anyNonZero(), [0]).arrayGet([0])
    
    @staticmethod
    def select_coefs_at_t(coefs_2d: ee.Image, mask_1d: ee.Image) -> ee.Image:
        """
        Select coefficients from segment matching mask.
        Input: coefs_2d [segment, coef], mask_1d [segment] boolean
        Output: [coef] 1D array (or array of MISSING if no match)
        """
        # Convert boolean mask to 0/1
        mask_numeric = mask_1d.toFloat()
    
        # For each coefficient, multiply by mask and sum
        # This gives us the matching segment's value (all others are 0)
        coef_list = []
        for i in range(8):  # Standard CCDC has 8 coefficients
            # Get coefficient i for all segments: [segment]
            coef_i = coefs_2d.arraySlice(1, i, i+1).arrayProject([0])
            
            # Multiply by mask (zeros out non-matching segments)
            masked_i = coef_i.multiply(mask_numeric)
            
            # Sum - gives us the value from the matching segment (or 0 if no match)
            value_i = masked_i.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
        
            coef_list.append(value_i)
    
        # Combine into 1D array [coef]
        result = ee.Image(coef_list).toArray()
    
        # Check if we had a match - if not, replace with MISSING
        has_match = ArrayOps.has_any_segment(mask_1d)
        
        # Use where() instead of If() - it's safer for server-side operations
        missing_array = ee.Image([MISSING] * 8).toArray()
        
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
        self.tStart = ccdc.select("tStart")
        self.tEnd = ccdc.select("tEnd")
        self.dur = self.tEnd.subtract(self.tStart)
        
        # Segment mask for this date
        self.seg_mask = ArrayOps.segment_mask_at_t(self.tStart, self.tEnd, self.t)
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
        # Predict all bands
        g = self.predict_band("GREEN").rename(f"{self.prefix}_green")
        r = self.predict_band("RED").rename(f"{self.prefix}_red")
        n = self.predict_band("NIR").rename(f"{self.prefix}_nir")
        s1 = self.predict_band("SWIR1").rename(f"{self.prefix}_swir1")
        s2 = self.predict_band("SWIR2").rename(f"{self.prefix}_swir2")
        
        # Calculate indices
        ndvi = calc_ndvi(n, r).rename(f"{self.prefix}_ndvi")
        nbr = calc_nbr(n, s2).rename(f"{self.prefix}_nbr")
        ndmi = calc_ndmi(n, s1).rename(f"{self.prefix}_ndmi")
        
        # Tasseled cap
        tc = tasseled_cap_5band(g, r, n, s1, s2, self.prefix)
        
        return ee.Image.cat([g, r, n, s1, s2, ndvi, nbr, ndmi, tc])
    
    def get_harmonic_metrics(self) -> ee.Image | None:
        """Get snapshot harmonics: amplitude for each band."""
        bands_list = []
        
        for band in HARMONIC_BANDS:
            coef_name = f"{band}_coefs"
            if coef_name not in self.ccdc.bandNames().getInfo():
                return None
            
            coefs = self.ccdc.select(coef_name)
            coefs_sel = ArrayOps.select_coefs_at_t(coefs, self.seg_mask)
            
            cos1 = coefs_sel.arrayGet([2])
            sin1 = coefs_sel.arrayGet([3])
            amp = cos1.pow(2).add(sin1.pow(2)).sqrt()
            
            amp = ee.Image(amp).where(self.has_seg.Not(), MISSING)
            bands_list.append(amp.rename(f"{self.prefix}_seasonal_amp_{band.lower()}"))
        
        return ee.Image.cat(bands_list) if bands_list else None
    
    def get_trajectory_metrics(self) -> ee.Image:
        """Get snapshot trajectory: velocity, duration, RMSE, index derivatives."""
        # Get slopes for all spectral bands
        slopes = {}
        slope_sq_list = []
        
        for band in SPECTRAL_BANDS:
            coefs = self.ccdc.select(f"{band}_coefs")
            coefs_sel = ArrayOps.select_coefs_at_t(coefs, self.seg_mask)
            slope = coefs_sel.arrayGet([1])
            slopes[band] = ee.Image(slope).where(self.has_seg.Not(), MISSING)
            slope_sq_list.append(slopes[band].pow(2))
        
        # Spectral velocity (magnitude of slope vector)
        vel_sq = slope_sq_list[0]
        for sq in slope_sq_list[1:]:
            vel_sq = vel_sq.add(sq)
        vel = vel_sq.sqrt().rename(f"{self.prefix}_spectral_velocity")
        
        # Segment duration
        seg_dur = ArrayOps.select_value_at_t(self.dur, self.seg_mask)
        seg_dur = ee.Image(seg_dur).where(self.has_seg.Not(), MISSING).rename(f"{self.prefix}_segment_duration")
        
        # Mean RMSE across bands
        rmse_list = []
        for band in SPECTRAL_BANDS:
            rmse = self.ccdc.select(f"{band}_rmse")
            rmse_val = ArrayOps.select_value_at_t(rmse, self.seg_mask)
            rmse_list.append(rmse_val)
        
        rmse_mean = rmse_list[0]
        for r in rmse_list[1:]:
            rmse_mean = rmse_mean.add(r)
        rmse_mean = (
            ee.Image(rmse_mean.divide(len(SPECTRAL_BANDS)))
            .where(self.has_seg.Not(), MISSING)
            .rename(f"{self.prefix}_rmse_mean")
        )
        
        # Index derivatives (exact, trend-only)
        nir_t = self.predict_band("NIR")
        red_t = self.predict_band("RED")
        swir1_t = self.predict_band("SWIR1")
        swir2_t = self.predict_band("SWIR2")
        
        dndvi = calc_index_derivative(nir_t, red_t, slopes["NIR"], slopes["RED"]).rename(f"{self.prefix}_dndvi_dt")
        dndmi = calc_index_derivative(nir_t, swir1_t, slopes["NIR"], slopes["SWIR1"]).rename(f"{self.prefix}_dndmi_dt")
        dnbr = calc_index_derivative(nir_t, swir2_t, slopes["NIR"], slopes["SWIR2"]).rename(f"{self.prefix}_dnbr_dt")
        
        return ee.Image.cat([vel, seg_dur, rmse_mean, dndvi, dndmi, dnbr])


# -----------------------------------------------------------------------------
# LONG-TERM METRICS - Statistics across all segments
# -----------------------------------------------------------------------------
class LongTermMetrics:
    """Generate long-term metrics across all CCDC segments."""
    
    def __init__(self, ccdc: ee.Image):
        self.ccdc = ccdc
        self.tStart = ccdc.select("tStart")
        self.tEnd = ccdc.select("tEnd")
        self.tBreak = ccdc.select("tBreak")
        self.dur = self.tEnd.subtract(self.tStart)
        self.total_dur = ArrayOps.reduce_sum(self.dur)
        self.span = ArrayOps.reduce_max(self.tEnd).subtract(ArrayOps.reduce_min(self.tStart))
        self.t_mid = self.tStart.add(self.tEnd).divide(2)
    
    def get_state_metrics(self) -> ee.Image:
        """Duration-weighted mean reflectance and indices."""
        bands_list = []
        
        # Mean reflectance at segment midpoints
        for band in SPECTRAL_BANDS:
            coefs = self.ccdc.select(f"{band}_coefs")
            c0 = ArrayOps.get_intercept(coefs)
            c1 = ArrayOps.get_slope(coefs)
            refl_mid = c0.add(c1.multiply(self.t_mid))
            
            mean_refl = (
                ArrayOps.reduce_sum(refl_mid.multiply(self.dur))
                .divide(self.total_dur.add(EPS))
                .rename(f"mean_{band.lower()}")
            )
            bands_list.append(mean_refl)
        
        # Mean indices at segment midpoints
        nir_coefs = self.ccdc.select("NIR_coefs")
        red_coefs = self.ccdc.select("RED_coefs")
        swir1_coefs = self.ccdc.select("SWIR1_coefs")
        swir2_coefs = self.ccdc.select("SWIR2_coefs")
        
        nir_mid = ArrayOps.get_intercept(nir_coefs).add(ArrayOps.get_slope(nir_coefs).multiply(self.t_mid))
        red_mid = ArrayOps.get_intercept(red_coefs).add(ArrayOps.get_slope(red_coefs).multiply(self.t_mid))
        swir1_mid = ArrayOps.get_intercept(swir1_coefs).add(ArrayOps.get_slope(swir1_coefs).multiply(self.t_mid))
        swir2_mid = ArrayOps.get_intercept(swir2_coefs).add(ArrayOps.get_slope(swir2_coefs).multiply(self.t_mid))
        
        ndvi_seg = calc_ndvi(nir_mid, red_mid)
        nbr_seg = calc_nbr(nir_mid, swir2_mid)
        ndmi_seg = calc_ndmi(nir_mid, swir1_mid)
        
        mean_ndvi = ArrayOps.reduce_sum(ndvi_seg.multiply(self.dur)).divide(self.total_dur.add(EPS)).rename("mean_ndvi")
        mean_nbr = ArrayOps.reduce_sum(nbr_seg.multiply(self.dur)).divide(self.total_dur.add(EPS)).rename("mean_nbr")
        mean_ndmi = ArrayOps.reduce_sum(ndmi_seg.multiply(self.dur)).divide(self.total_dur.add(EPS)).rename("mean_ndmi")
        
        bands_list.extend([mean_ndvi, mean_nbr, mean_ndmi])
        
        # Store for use in variability metrics
        self.ndvi_seg = ndvi_seg
        self.nir_mid = nir_mid
        
        return ee.Image.cat(bands_list)
    
    def get_harmonic_metrics(self) -> dict | None:
        """Harmonic/phenology metrics: mean amplitudes, SNR, variance, summary."""
        # Check if all harmonic bands exist
        if not all(f"{b}_coefs" in self.ccdc.bandNames().getInfo() for b in HARMONIC_BANDS):
            return None
        
        results = {"mean": [], "snr": [], "variance": [], "summary": []}
        
        for band in HARMONIC_BANDS:
            coefs = self.ccdc.select(f"{band}_coefs")
            rmse = self.ccdc.select(f"{band}_rmse")
            amp_seg = ArrayOps.get_harmonic_amp(coefs)
            
            band_lower = band.lower()
            
            # Duration-weighted mean amplitude
            mean_amp = (
                ArrayOps.reduce_sum(amp_seg.multiply(self.dur))
                .divide(self.total_dur.add(EPS))
                .rename(f"mean_seasonal_amp_{band_lower}")
            )
            results["mean"].append(mean_amp)
            
            # SNR
            mean_rmse = ArrayOps.reduce_mean(rmse)
            snr = mean_amp.divide(mean_rmse.add(EPS)).rename(f"mean_seasonal_snr_{band_lower}")
            results["snr"].append(snr)
            
            # Variance
            var_amp = ArrayOps.reduce_variance(amp_seg).rename(f"variance_seasonal_amp_{band_lower}")
            results["variance"].append(var_amp)
        
        # Summary metrics for NIR and SWIR1
        for band in ["NIR", "SWIR1"]:
            coefs = self.ccdc.select(f"{band}_coefs")
            amp_seg = ArrayOps.get_harmonic_amp(coefs)
            band_lower = band.lower()
            
            max_amp = ArrayOps.reduce_max(amp_seg).rename(f"max_seasonal_amp_{band_lower}")
            results["summary"].append(max_amp)
            
            amp_first = ArrayOps.first_element(amp_seg)
            amp_last = ArrayOps.last_element(amp_seg)
            net_change = amp_last.subtract(amp_first).rename(f"net_change_seasonal_amp_{band_lower}")
            results["summary"].append(net_change)
        
        return results
    
    def get_trajectory_metrics(self) -> ee.Image:
        """Spectral trajectory: distance, velocity, net change, recovery/decline rates."""
        bands_list = []
        
        # Spectral velocity magnitude across segments
        slope_sq_list = []
        for band in SPECTRAL_BANDS:
            coefs = self.ccdc.select(f"{band}_coefs")
            slope = ArrayOps.get_slope(coefs)
            slope_sq_list.append(slope.pow(2))
        
        vel_sq = slope_sq_list[0]
        for sq in slope_sq_list[1:]:
            vel_sq = vel_sq.add(sq)
        slope_mag = vel_sq.sqrt()
        
        seg_traj_len = slope_mag.multiply(self.dur)
        total_traj_len = ArrayOps.reduce_sum(seg_traj_len).rename("total_spectral_distance")
        traj_rate = total_traj_len.divide(self.span.divide(10).max(EPS)).rename("spectral_distance_per_decade")
        mean_velocity = ArrayOps.reduce_mean(slope_mag).rename("mean_spectral_velocity")
        
        bands_list.extend([total_traj_len, traj_rate, mean_velocity])
        
        # Net change (first segment start to last segment end)
        nir_coefs = self.ccdc.select("NIR_coefs")
        red_coefs = self.ccdc.select("RED_coefs")
        
        nir_c0 = ArrayOps.get_intercept(nir_coefs)
        nir_c1 = ArrayOps.get_slope(nir_coefs)
        red_c0 = ArrayOps.get_intercept(red_coefs)
        red_c1 = ArrayOps.get_slope(red_coefs)
        
        t_first = ArrayOps.first_element(self.tStart)
        t_last = ArrayOps.last_element(self.tEnd)
        
        nir_first = ArrayOps.first_element(nir_c0).add(ArrayOps.first_element(nir_c1).multiply(t_first))
        red_first = ArrayOps.first_element(red_c0).add(ArrayOps.first_element(red_c1).multiply(t_first))
        ndvi_first = calc_ndvi(nir_first, red_first)
        
        nir_last = ArrayOps.last_element(nir_c0).add(ArrayOps.last_element(nir_c1).multiply(t_last))
        red_last = ArrayOps.last_element(red_c0).add(ArrayOps.last_element(red_c1).multiply(t_last))
        ndvi_last = calc_ndvi(nir_last, red_last)
        
        net_change_ndvi = ndvi_last.subtract(ndvi_first).rename("net_change_ndvi")
        net_change_nir = nir_last.subtract(nir_first).rename("net_change_nir")
        
        bands_list.extend([net_change_ndvi, net_change_nir])
        
        # Recovery/decline rates (from NDVI slope proxy)
        ndvi_slope_approx = ArrayOps.get_slope(nir_coefs).subtract(ArrayOps.get_slope(red_coefs))
        mean_recovery = ArrayOps.reduce_mean(ndvi_slope_approx.max(0)).rename("mean_recovery_rate")
        mean_decline = ArrayOps.reduce_mean(ndvi_slope_approx.min(0).abs()).rename("mean_decline_rate")
        
        bands_list.extend([mean_recovery, mean_decline])
        
        return ee.Image.cat(bands_list)
    
    def get_disturbance_metrics(self) -> ee.Image:
        """Disturbance history: segment counts, durations, break magnitudes."""
        bands_list = []
        
        # Segment statistics
        num_segments = self.tStart.arrayLength(0).toFloat().rename("num_segments")
        
        break_mask = self.tBreak.gt(0)
        break_count = ArrayOps.reduce_sum(break_mask).rename("break_count")
        break_rate = break_count.divide(self.span.divide(10).max(EPS)).rename("break_rate_per_decade")
        
        bands_list.extend([num_segments, break_count, break_rate])
        
        # Duration statistics
        years_since_last = ArrayOps.last_element(self.dur).rename("years_since_last_break")
        mean_dur = ArrayOps.reduce_mean(self.dur).rename("mean_segment_duration")
        max_dur = ArrayOps.reduce_max(self.dur).rename("max_segment_duration")
        min_dur = ArrayOps.reduce_min(self.dur).rename("min_segment_duration")
        
        bands_list.extend([years_since_last, mean_dur, max_dur, min_dur])
        
        # Break magnitudes (if available)
        if all(f"{b}_magnitude" in self.ccdc.bandNames().getInfo() for b in SPECTRAL_BANDS):
            mag_list = []
            for band in SPECTRAL_BANDS:
                mag = self.ccdc.select(f"{band}_magnitude").abs()
                mag_list.append(mag)
            
            mag_sum = mag_list[0]
            for m in mag_list[1:]:
                mag_sum = mag_sum.add(m)
            
            mean_mag = mag_sum.divide(len(SPECTRAL_BANDS))
            mean_break_mag = ArrayOps.reduce_mean(mean_mag).rename("mean_break_magnitude")
            max_break_mag = ArrayOps.reduce_max(mean_mag).rename("max_break_magnitude")
            
            bands_list.extend([mean_break_mag, max_break_mag])
        
        return ee.Image.cat(bands_list)
    
    def get_variability_metrics(self) -> ee.Image:
        """Variability: variance in indices, RMSE, duration."""
        bands_list = []
        
        # Index variance (requires state metrics to be computed first)
        var_ndvi = ArrayOps.reduce_variance(self.ndvi_seg).rename("variance_ndvi")
        var_nir = ArrayOps.reduce_variance(self.nir_mid).rename("variance_nir")
        bands_list.extend([var_ndvi, var_nir])
        
        # Duration variability
        dur_mean = ArrayOps.reduce_mean(self.dur)
        dur_std = ArrayOps.reduce_stddev(self.dur)
        cv_duration = dur_std.divide(dur_mean.add(EPS)).rename("cv_segment_duration")
        bands_list.append(cv_duration)
        
        # RMSE statistics
        rmse_list = []
        for band in SPECTRAL_BANDS:
            rmse = self.ccdc.select(f"{band}_rmse")
            rmse_list.append(rmse)
        
        rmse_sum = rmse_list[0]
        for r in rmse_list[1:]:
            rmse_sum = rmse_sum.add(r)
        
        rmse_mean_bands = rmse_sum.divide(len(SPECTRAL_BANDS))
        mean_rmse = ArrayOps.reduce_mean(rmse_mean_bands).rename("mean_rmse_joint")
        max_rmse = ArrayOps.reduce_max(rmse_mean_bands).rename("max_rmse_joint")
        
        bands_list.extend([mean_rmse, max_rmse])
        
        return ee.Image.cat(bands_list)
    
    def get_rapid_loss_metrics(self) -> ee.Image | None:
        """Detect rapid substantial forest loss events."""
        # Check for magnitude bands
        if not all(f"{b}_magnitude" in self.ccdc.bandNames().getInfo() for b in SPECTRAL_BANDS):
            return None
        
        # Calculate spectral magnitude
        mag_sq_list = []
        for band in SPECTRAL_BANDS:
            mag = self.ccdc.select(f"{band}_magnitude")
            mag_sq_list.append(mag.pow(2))
        
        mag_sq_sum = mag_sq_list[0]
        for m in mag_sq_list[1:]:
            mag_sq_sum = mag_sq_sum.add(m)
        
        spectral_mag = mag_sq_sum.sqrt()
        
        # Define loss events: rapid (<1.5 yr), substantial (>0.1), negative NIR
        nir_mag = self.ccdc.select("NIR_magnitude")
        is_loss = self.dur.lt(1.5).And(spectral_mag.gt(0.10)).And(nir_mag.lt(0))
        
        # Count and rate
        loss_count = ArrayOps.reduce_sum(is_loss)
        loss_rate = loss_count.divide(self.span.divide(10).max(EPS)).rename("rapid_loss_rate_per_decade")
        
        # Mean magnitude of loss events
        has_events = loss_count.gt(0)
        loss_mags = spectral_mag.arrayMask(is_loss)
        mean_mag = (
            ArrayOps.reduce_mean(loss_mags)
            .where(has_events.Not(), 0)
            .rename("rapid_loss_mean_magnitude")
        )
        
        # Years of 4 most recent events
        loss_years = self.tBreak.arrayMask(is_loss)
        loss_years_sorted = loss_years.arraySort()
        n_events = loss_years_sorted.arrayLength(0)
        
        def get_nth_recent(n):
            idx = n_events.subtract(n)
            year = loss_years_sorted.arraySlice(0, idx, idx.add(1)).arrayGet([0])
            return ee.Image(year).where(n_events.lt(n), MISSING).rename(f"rapid_loss_year_{n}")
        
        year_1 = get_nth_recent(1)
        year_2 = get_nth_recent(2)
        year_3 = get_nth_recent(3)
        year_4 = get_nth_recent(4)
        
        return ee.Image.cat([loss_rate, mean_mag, year_1, year_2, year_3, year_4])
    
    def get_loss_recovery_metrics(self) -> ee.Image | None:
        """Calculate intervals between loss events."""
        # Check for magnitude bands
        if not all(f"{b}_magnitude" in self.ccdc.bandNames().getInfo() for b in SPECTRAL_BANDS):
            return None
        
        # Calculate spectral magnitude (same as rapid loss)
        mag_sq_list = []
        for band in SPECTRAL_BANDS:
            mag = self.ccdc.select(f"{band}_magnitude")
            mag_sq_list.append(mag.pow(2))
        
        mag_sq_sum = mag_sq_list[0]
        for m in mag_sq_list[1:]:
            mag_sq_sum = mag_sq_sum.add(m)
        
        spectral_mag = mag_sq_sum.sqrt()
        nir_mag = self.ccdc.select("NIR_magnitude")
        
        is_loss = self.dur.lt(1.5).And(spectral_mag.gt(0.10)).And(nir_mag.lt(0))
        
        # Get loss break times and calculate intervals
        loss_tBreak = self.tBreak.arrayMask(is_loss)
        n_losses = loss_tBreak.arrayLength(0)
        n_intervals = n_losses.subtract(1)
        
        next_loss = loss_tBreak.arraySlice(0, 1, None)
        curr_loss = loss_tBreak.arraySlice(0, 0, -1)
        intervals = next_loss.subtract(curr_loss)
        
        # Mean interval
        has_data = n_intervals.gte(1)
        mean_recovery = (
            ArrayOps.reduce_mean(intervals)
            .where(has_data.Not(), MISSING)
            .rename("mean_loss_recovery_duration")
        )
        
        # Std dev (requires at least 3 intervals for reliability)
        has_reliable = n_intervals.gte(3)
        std_recovery = (
            ArrayOps.reduce_stddev(intervals)
            .where(has_reliable.Not(), MISSING)
            .rename("std_loss_recovery_duration")
        )
        
        return ee.Image.cat([mean_recovery, std_recovery])


# -----------------------------------------------------------------------------
# Main metrics builder
# -----------------------------------------------------------------------------
def build_comprehensive_forest_metrics(ccdc: ee.Image, bands: set[str]) -> ee.Image:
    """
    Extract comprehensive forest characterization metrics.
    Captures both snapshot state/trajectory and long-term patterns.
    """
    parts: list[ee.Image] = []
    
    # ================================================================
    # SECTION 1 & 1b & 2: SNAPSHOT METRICS (state, harmonics, trajectory)
    # ================================================================
    for key, t_snap in SNAPSHOTS.items():
        snap = SnapshotMetrics(ccdc, t_snap, key)
        
        # State (reflectance, indices, TC)
        parts.append(snap.get_state_metrics())
        
        # Harmonics (if available)
        harm = snap.get_harmonic_metrics()
        if harm is not None:
            parts.append(harm)
        
        # Trajectory (velocity, duration, derivatives)
        parts.append(snap.get_trajectory_metrics())
    
    # ================================================================
    # SECTION 3-8: LONG-TERM METRICS
    # ================================================================
    lt = LongTermMetrics(ccdc)
    
    # Section 3: State (mean reflectance, indices)
    parts.append(lt.get_state_metrics())
    
    # Section 3b: Harmonics (mean, SNR, variance, summary)
    harm_metrics = lt.get_harmonic_metrics()
    if harm_metrics is not None:
        for category in ["mean", "snr", "variance", "summary"]:
            if harm_metrics[category]:
                parts.append(ee.Image.cat(harm_metrics[category]))
    
    # Section 4: Trajectory (distance, velocity, net change, recovery/decline)
    parts.append(lt.get_trajectory_metrics())
    
    # Section 5: Disturbance history (segments, breaks, durations)
    parts.append(lt.get_disturbance_metrics())
    
    # Section 6: Variability (variance in indices, RMSE, duration)
    parts.append(lt.get_variability_metrics())
    
    # Section 7: Rapid loss events
    rapid_loss = lt.get_rapid_loss_metrics()
    if rapid_loss is not None:
        parts.append(rapid_loss)
    
    # Section 8: Loss recovery intervals
    loss_recovery = lt.get_loss_recovery_metrics()
    if loss_recovery is not None:
        parts.append(loss_recovery)
    
    # ================================================================
    # Combine all parts
    # ================================================================
    return ee.Image.cat(parts).toFloat()


# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def process_tile(row: int, col: int, x_ul: float, y_ul: float, geom: ee.Geometry, dry_run: bool = False) -> None:
    """Process a single tile: load CCDC asset, extract metrics, export to GCS."""
    asset_id = f"{ASSET_ROOT}_r{row:03d}_c{col:03d}"
    base_name = f"{BASE_NAME_PREFIX}_r{row:03d}_c{col:03d}"

    print(f"\n{'='*80}")
    print(f"Processing tile r{row:03d}_c{col:03d}")
    print(f"  Asset: {asset_id}")
    print(f"  Upper-left: ({x_ul}, {y_ul})")
    print(f"{'='*80}")

    # Check if asset exists
    try:
        ccdc_img = ee.Image(asset_id)
        band_list = ccdc_img.bandNames().getInfo()
        bands = set(band_list)
        print(f"  Found {len(band_list)} bands")
    except Exception as e:
        print(f"  ERROR: Could not load asset: {e}")
        print(f"  Skipping tile r{row:03d}_c{col:03d}")
        return

    # Verify required bands
    required_coef_bands = [f"{band}_coefs" for band in SPECTRAL_BANDS]
    missing_bands = [b for b in required_coef_bands if b not in bands]

    if missing_bands:
        print(f"  ERROR: Missing required bands: {missing_bands}")
        print(f"  Skipping tile r{row:03d}_c{col:03d}")
        return

    # Build metrics
    print("  Building comprehensive forest metrics...")
    try:
        summary_img = build_comprehensive_forest_metrics(ccdc_img, bands)
        output_bands = summary_img.bandNames().getInfo()
        print(f"  Generated {len(output_bands)} metric bands")
    except Exception as e:
        print(f"  ERROR: Failed to build metrics: {e}")
        print(f"  Skipping tile r{row:03d}_c{col:03d}")
        return

    if dry_run:
        print(f"  DRY RUN: Would export {len(output_bands)} bands to GCS")
        print(f"    Bucket: {GCS_BUCKET}")
        print(f"    Path: {GCS_DIR}/{base_name}")
        return

    # Export to GCS
    print("  Exporting to GCS...")
    tile_tr = tile_transform(x_ul, y_ul, SCALE_M)

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
        print(f"  ✓ Export task started for r{row:03d}_c{col:03d}")
    except Exception as e:
        print(f"  ERROR: Export failed: {e}")


def main(dry_run: bool = False, max_tiles: int | None = None) -> None:
    """
    Process all CCDC tiles and export metrics to GCS.

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
    skip_count = 0
    error_count = 0

    for i, (row, col, geom, x_ul, y_ul) in enumerate(tiles):
        print(f"\nProgress: {i+1}/{len(tiles)}")

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
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")

    if not dry_run and success_count > 0:
        print("\nExport tasks started successfully!")
        print("Check Earth Engine Tasks tab for progress")
        print(f"Outputs will be in: gs://{GCS_BUCKET}/{GCS_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract comprehensive CCDC metrics from tiled assets",
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
