#!/usr/bin/env python3
"""
Export LS8DAY seasonal / hydro-year structure metrics as one multi-band COG per hydro year.

For each hydro_year and AOI, we produce a single image with bands:

  NDVI_summer_p95            # 95th percentile NDVI during summer (Jun–Aug)
  NDVI_winter_max            # max NDVI during previous winter (Dec–Feb), clear pixels only
  NDVI_amplitude             # NDVI_summer_p95 - NDVI_winter_max

  NBR_annual_min             # minimum NBR over the full hydro-year (Oct–Sep window)
  NBR_summer_p95             # 95th percentile NBR during summer (Jun–Aug)

  NDMI_summer_mean           # mean NDMI during summer (Jun–Aug)
  EVI2_summer_p95            # 95th percentile EVI2 during summer (Jun–Aug)

  winter_obs_count         # count of valid observations in winter window
  summer_obs_count         # count of valid observations in summer window

Hydro-year definition:
  hydro_year Y covers [(Y-1)-10-01, Y-10-01). Seasonal subsets are:
    - previous winter: Dec(Y-1)–Mar(Y)
    - current summer: Jun(Y)–Sep(Y)

Each hydro-year image is exported as a COG GeoTIFF to GCS (one file per hydro year).
"""


from __future__ import annotations

import ee
from utils.gee import ee_init, export_img_to_gcs, get_state_geometry

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ee_init()
# Years to process (inclusive)
YEARS = list(range(2010, 2025))  # 2010–2024

# AOI: either a state geometry or a specific rectangle
STATE_NAME = "Virginia"
USE_STATE_GEOM = False  # Set True for full state; False keeps small debug AOI

TARGET_CRS = (
    "PROJ4:+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

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

TARGET_TRANSFORM = [30, 0, 1089315, 0, -30, 1966485]

# Region is calculated to give grid size as multiple of 256
PADDED_REGION = ee.Geometry.Rectangle(
    [1089315, 1574805, 1795875, 1966485],
    proj=TARGET_CRS,
    geodesic=False,
)

# Small debug AOI (subset over Virginia, safe for development / testing)
#ee_init()
#DEBUG_AOI = ee.Geometry.Rectangle(
#    [-79.0, 37.9, -78.9, 38.0]  # lon_min, lat_min, lon_max, lat_max
#)

# GCS export configuration
GCS_BUCKET = "va_rasters"
GCS_DIR = "ls8day"

LS8DAY = "LANDSAT/COMPOSITES/C02/T1_L2_8DAY"

# Seasonal definitions as month-day strings, used to build YYYY-MM-DD ranges
YEAR_PREV_START_MD = "10-01"
YEAR_END_MD = "10-01" # Oct 1 (exclusive)

WINTER_PREV_START_MD = "12-01"  # Dec 1 of previous year
WINTER_PREV_END_MD   = "03-01"  # Mar 1 of hydro_year (exclusive)

SUMMER_START_MD      = "06-01"  # Jun 1
SUMMER_END_MD        = "09-01"  # Sep 1 (exclusive)

# # Nodata values for metrics
# # Index nodata: NDVI/EVI/FI/NBR metrics and their percentiles
# NODATA_INDEX = -9999.0
# # DOY nodata: integer-safe sentinel, but we store as float in this script
# NODATA_DOY = -32768.0

# ---------------------------------------------------------------------
# PER-IMAGE PROCESSING: SCALE, MASK, INDICES
# ---------------------------------------------------------------------

def safe_ndvi(nir: ee.Image, red: ee.Image) -> ee.Image:
    """
    Compute NDVI with basic sanity checks:

      NDVI = (NIR - RED) / (NIR + RED)

    Mask out:
      - very low brightness (NIR + RED <= 0.02)
      - non-positive NIR
      - out-of-range NDVI (|NDVI| > 1).
    """
    sum_ = nir.add(red)
    raw = nir.subtract(red).divide(sum_)
    mask = (
        nir.gt(0)
        .And(sum_.gt(0.02))
        .And(raw.abs().lte(1))
    )
    return raw.updateMask(mask)

def safe_evi(nir: ee.Image, red: ee.Image, blue: ee.Image) -> ee.Image:
    """
    Compute EVI with basic sanity checks:

      EVI = 2.5 * (NIR - RED) /
            (NIR + 6*RED - 7.5*BLUE + 1)

    Mask out:
      - denominator near zero
      - non-positive NIR
      - absurd values (|EVI| > 2.5).
    """
    denom = nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    raw = nir.subtract(red).multiply(2.5).divide(denom)
    mask = (
        denom.gt(0.02)
        .And(nir.gt(0))
        .And(raw.abs().lte(2.5))
    )
    return raw.updateMask(mask)

def safe_evi2(nir: ee.Image, red: ee.Image) -> ee.Image:
    """
    EVI2:

      EVI2 = 2.5 * (NIR - RED) / (NIR + 2.4*RED + 1)

    Mask out:
      - denominator near zero
      - non-positive NIR
      - absurd values (|EVI2| > 2.5).
    """
    denom = nir.add(red.multiply(2.4)).add(1)
    raw = nir.subtract(red).multiply(2.5).divide(denom)
    mask = (
        denom.gt(0.02)
        .And(nir.gt(0))
        .And(raw.abs().lte(2.5))
    )
    return raw.updateMask(mask)

def safe_nbr(nir: ee.Image, swir2: ee.Image) -> ee.Image:
    """
    Compute NBR with basic sanity checks:

      NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Mask out:
      - non-positive NIR or SWIR2
      - very low brightness
      - out-of-range values (|NBR| > 1).
    """
    sum_ = nir.add(swir2)
    raw = nir.subtract(swir2).divide(sum_)
    mask = (
        nir.gt(0)
        .And(swir2.gt(0))
        .And(sum_.gt(0.02))
        .And(raw.abs().lte(1))
    )
    return raw.updateMask(mask)

def safe_ndmi(nir: ee.Image, swir1: ee.Image) -> ee.Image:
    """
    Normalized Difference Moisture Index:

      NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Mask out:
      - non-positive NIR or SWIR1
      - very low brightness
      - out-of-range values (|NDMI| > 1).
    """
    sum_ = nir.add(swir1)
    raw = nir.subtract(swir1).divide(sum_)
    mask = (
        nir.gt(0)
        .And(swir1.gt(0))
        .And(sum_.gt(0.02))
        .And(raw.abs().lte(1))
    )
    return raw.updateMask(mask)


def add_indices_ls8day(img: ee.Image) -> ee.Image:
    """
    Add NDVI, NBR, NDMI, EVI2, and DOY to a Landsat 8-day composite image.
    Input bands: blue, green, red, nir, swir1, swir2, thermal.
    """
    blue  = img.select("blue").toFloat().rename("BLUE")
    red   = img.select("red").toFloat().rename("RED")
    nir   = img.select("nir").toFloat().rename("NIR")
    swir1 = img.select("swir1").toFloat().rename("SWIR1")
    swir2 = img.select("swir2").toFloat().rename("SWIR2")

    ndvi = safe_ndvi(nir, red).rename("NDVI")
    nbr  = safe_nbr(nir, swir2).rename("NBR")
    ndmi = safe_ndmi(nir, swir1).rename("NDMI")
    evi2 = safe_evi2(nir, red).rename("EVI2")

    ts = img.get("system:time_start")
    doy_num = ee.Number(
        ee.Algorithms.If(ts, ee.Date(ts).getRelative("day", "year"), 0)
    )
    doy = ee.Image.constant(doy_num).rename("DOY").toInt16()

    out = img.addBands([ndvi, nbr, ndmi, evi2, doy], overwrite=True)

    # keep original properties
    return out.copyProperties(img, img.propertyNames())


# ---------------------------------------------------------------------
# HLS COLLECTION + YEARLY SERIES
# ---------------------------------------------------------------------
def get_ls8day_collection_range(start: str, end: str, aoi: ee.Geometry) -> ee.ImageCollection:
    return (
        ee.ImageCollection(LS8DAY)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.notNull(["system:time_start"]))
    )

def build_seasonal_series(season_year: int, aoi: ee.Geometry) -> ee.ImageCollection:
    full_start = f"{season_year-1}-{YEAR_PREV_START_MD}"
    full_end   = f"{season_year}-{YEAR_END_MD}"

    coll = get_ls8day_collection_range(full_start, full_end, aoi)
    return coll.map(add_indices_ls8day).select(["NDVI", "NBR", "NDMI", "EVI2", "DOY"])


def compute_annual_struct_metrics(hydro_year: int, aoi: ee.Geometry) -> ee.Image:
    """
    For a given hydro_year and AOI, compute:

      NDVI_summer_p95
      NDVI_winter_max
      NDVI_amplitude

      NBR_annual_min
      NBR_summer_p95

      NDMI_summer_mean

      EVI2_summer_p95

    Where:
      - "winter" = previous winter: Dec(hydro_year-1) to early hydro_year
      - "summer" = current summer: mid hydro_year
    """
    series = build_seasonal_series(hydro_year, aoi)

    # Full hydro-year series (prev winter + current summer)
    ic_full = series

    # Explicit date ranges for subsets
    winter_prev_start = f"{hydro_year - 1}-{WINTER_PREV_START_MD}"
    winter_prev_end   = f"{hydro_year}-{WINTER_PREV_END_MD}"

    summer_start      = f"{hydro_year}-{SUMMER_START_MD}"
    summer_end        = f"{hydro_year}-{SUMMER_END_MD}"

    ic_winter = series.filterDate(winter_prev_start, winter_prev_end)
    ic_summer = series.filterDate(summer_start, summer_end)

    def make_all_masked() -> ee.Image:
        placeholder = ee.Image.constant([0] * 9).rename([
            "NDVI_summer_p95",
            "NDVI_winter_max",
            "NDVI_amplitude",
            "NBR_annual_min",
            "NBR_summer_p95",
            "NDMI_summer_mean",
            "EVI2_summer_p95",
            "winter_obs_count",
            "summer_obs_count",
        ])
        mask = ee.Image.constant(0)
        return placeholder.updateMask(mask).toFloat()

    def compute_metrics() -> ee.Image:
        first = ee.Image(ic_full.first())

        empty_like_ndvi = (
            first.select("NDVI")
            .multiply(0)
            .updateMask(ee.Image.constant(0))
        )
        empty_like_nbr = (
            first.select("NBR")
            .multiply(0)
            .updateMask(ee.Image.constant(0))
        )
        empty_like_ndmi = (
            first.select("NDMI")
            .multiply(0)
            .updateMask(ee.Image.constant(0))
        )
        empty_like_evi2 = (
            first.select("EVI2")
            .multiply(0)
            .updateMask(ee.Image.constant(0))
        )
        
        # ---- Count valid pixels in each season ----
        winter_obs_count = ee.Image(
            ee.Algorithms.If(
                ic_winter.size().gt(0),
                ic_winter.select("NDVI").count(),  # any band works
                empty_like_ndvi.rename("winter_obs_count"),
            )
        ).rename("winter_obs_count").toFloat()

        summer_obs_count = ee.Image(
            ee.Algorithms.If(
                ic_summer.size().gt(0),
                ic_summer.select("NDVI").count(),
                empty_like_ndvi.rename("summer_obs_count"),
            )
        ).rename("summer_obs_count").toFloat()


        # NDVI_summer_p95
        ndvi_summer_p95_img = ee.Image(
            ee.Algorithms.If(
                ic_summer.size().gt(0),
                ic_summer
                .select("NDVI")
                .reduce(ee.Reducer.percentile([95])),
                empty_like_ndvi.rename("NDVI_p95"),
            )
        )
        ndvi_summer_p95 = (
            ndvi_summer_p95_img
            .select("NDVI_p95")
            .rename("NDVI_summer_p95")
        )

        # NDVI_winter_max_clean (prev winter only)
        ndvi_winter_max_img = ee.Image(
            ee.Algorithms.If(
                ic_winter.size().gt(0),
                ic_winter.qualityMosaic("NDVI"),
                empty_like_ndvi,
            )
        )
        ndvi_winter_max = (
            ndvi_winter_max_img
            .select("NDVI")
            .rename("NDVI_winter_max")
        )

        # NDVI_amplitude
        ndvi_amplitude = ndvi_summer_p95.subtract(ndvi_winter_max)
        ndvi_amplitude = ndvi_amplitude.rename("NDVI_amplitude")

        # NBR_annual_min over full hydro-year
        def add_neg_nbr(im: ee.Image) -> ee.Image:
            return im.addBands(
                im.select("NBR").multiply(-1).rename("negNBR")
            )

        ic_nbr_neg = ic_full.map(add_neg_nbr)
        nbr_min_img = ee.Image(
            ee.Algorithms.If(
                ic_nbr_neg.size().gt(0),
                ic_nbr_neg.qualityMosaic("negNBR"),
                empty_like_nbr,
            )
        )
        nbr_annual_min = (
            nbr_min_img
            .select("NBR")
            .rename("NBR_annual_min")
        )

        # NBR_summer_p95
        nbr_summer_p95_img = ee.Image(
            ee.Algorithms.If(
                ic_summer.size().gt(0),
                ic_summer
                .select("NBR")
                .reduce(ee.Reducer.percentile([95])),
                empty_like_nbr.rename("NBR_p95"),
            )
        )
        nbr_summer_p95 = (
            nbr_summer_p95_img
            .select("NBR_p95")
            .rename("NBR_summer_p95")
        )

        # NDMI_summer_mean
        ndmi_summer_mean_img = ee.Image(
            ee.Algorithms.If(
                ic_summer.size().gt(0),
                ic_summer.select("NDMI").mean(),
                empty_like_ndmi,
            )
        )
        ndmi_summer_mean = (
            ndmi_summer_mean_img
            .rename("NDMI_summer_mean")
        )

        # EVI2_summer_p95
        evi2_summer_p95_img = ee.Image(
            ee.Algorithms.If(
                ic_summer.size().gt(0),
                ic_summer
                .select("EVI2")
                .reduce(ee.Reducer.percentile([95])),
                empty_like_evi2.rename("EVI2_p95"),
            )
        )
        evi2_summer_p95 = (
            evi2_summer_p95_img
            .select("EVI2_p95")
            .rename("EVI2_summer_p95")
        )

        out = ee.Image.cat([
            ndvi_summer_p95,
            ndvi_winter_max,
            ndvi_amplitude,
            nbr_annual_min,
            nbr_summer_p95,
            ndmi_summer_mean,
            evi2_summer_p95,
            winter_obs_count,
            summer_obs_count,
        ])

        return out.toFloat()

    return ee.Image(
        ee.Algorithms.If(
            ic_full.size().gt(0),
            compute_metrics(),
            make_all_masked(),
        )
    ).set("hydro_year", hydro_year)

# ---------------------------------------------------------------------
# MAIN: EXPORT ONE IMAGE PER YEAR
# ---------------------------------------------------------------------

def main():
    """
    Entry point:

      1. Initialize Earth Engine.
      2. Choose AOI (state geometry or debug rectangle).
      3. For each year:
           - build annual metrics image
           - export as COG GeoTIFF to GCS via export_img_to_gcs().
    """
    ee_init()

#    if USE_STATE_GEOM:
    aoi = get_state_geometry(STATE_NAME)
#    else:
#        aoi = DEBUG_AOI

    for nominal_year in YEARS:
        print(f"Building hydro year metrics for hydro year={nominal_year}")
        annual_img = compute_annual_struct_metrics(nominal_year, aoi)

        base_name = f"ls8day_metrics_{nominal_year}"
        print(f"Exporting {base_name} to gs://{GCS_BUCKET}/{GCS_DIR}/")

        export_img_to_gcs(
            img=annual_img,
            aoi=PADDED_REGION,
            bucket=GCS_BUCKET,
            base_name=base_name,
            gcs_dir=GCS_DIR,
            crs=TARGET_CRS,
            crsTransform=TARGET_TRANSFORM,
            # Do not pass crs/crsTransform here if export_img_to_gcs
            # already sets them for you.
        )


if __name__ == "__main__":
    main()
