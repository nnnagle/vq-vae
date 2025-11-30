#!/usr/bin/env python3
"""
Export HLS index metrics (2015–2024) as one multi-band COG *per year*.

For each year and AOI, we produce a single image with bands:

  NDVI_max, NDVI_max_DOY, NDVI_min, NDVI_min_DOY, NDVI_p05, NDVI_p95
  EVI_max,  EVI_max_DOY,  EVI_min,  EVI_min_DOY,  EVI_p05,  EVI_p95
  FI_max,   FI_max_DOY,   FI_min,   FI_min_DOY,   FI_p05,   FI_p95
  NBR_max,  NBR_max_DOY,  NBR_min,  NBR_min_DOY,  NBR_p05,  NBR_p95

Each annual image is exported as a COG GeoTIFF to GCS (one file per year).
You can then stack these in Python/xarray into a space–time Zarr (e.g., for a VAE).
"""

from __future__ import annotations

import ee
from utils.gee import ee_init, export_img_to_gcs, get_state_geometry

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ee_init()
# Years to process (inclusive)
YEARS = list(range(2015, 2025))  # 2015–2024

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
GCS_DIR = "hls"

# HLS v002 collections (Landsat and Sentinel-2)
HLS_L30 = "NASA/HLS/HLSL30/v002"
HLS_S30 = "NASA/HLS/HLSS30/v002"

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


def make_clear_land_mask(fmask: ee.Image) -> ee.Image:
    """
    Return a mask image where 1 = clear land, 0 = anything bad.

    HLS Fmask bit layout (LSDS v2 style):

      bit 0: cirrus
      bit 1: cloud
      bit 2: cloud adjacency
      bit 3: cloud shadow
      bit 4: snow/ice
      bit 5: water

    We mark pixels as "bad" if ANY of these bits is 1.
    Everything else is considered clear land.
    """

    def bit(img: ee.Image, b: int) -> ee.Image:
        # Extract bit b: (value >> b) & 1
        return img.rightShift(b).bitwiseAnd(1)

    cirrus   = bit(fmask, 0)
    cloud    = bit(fmask, 1)
    adjacent = bit(fmask, 2)
    shadow   = bit(fmask, 3)
    snow     = bit(fmask, 4)
    water    = bit(fmask, 5)

    # Pixels to reject if ANY of these is 1
    bad = (
        cirrus
        .Or(cloud)
        .Or(adjacent)
        .Or(shadow)
        .Or(snow)
        .Or(water)
    )

    clear = bad.Not()  # 1 where all bits 0–5 are 0
    return clear


def add_indices(img: ee.Image) -> ee.Image:
    """
    Per-image HLS processing for v002:

      - Mask to clear land pixels (Fmask == 0 across bits 0–5)
      - Use HLS surface reflectance bands:
          * Blue:  B2  (both L30 and S30)
          * Red:   B4  (both L30 and S30)
          * NIR:   B5 (L30) / B8A (S30) / B8 (S30 fallback)
          * SWIR1: B6 (L30) / B11 (S30)
          * SWIR2: B7 (L30) / B12 (S30)
      - Compute NDVI, EVI, FI, NBR
      - Add DOY (day-of-year) band.

    Returns the original image with additional bands, masked to clear land.
    """
    fmask = img.select("Fmask")
    clear = make_clear_land_mask(fmask)

    bands = img.bandNames()

    # Cast raw DN to float so we don't get integer division anywhere.
    blue_raw = img.select("B2").toFloat()
    red_raw  = img.select("B4").toFloat()

    # NIR depends on platform / sensor; handle L30 vs S30
    nir_raw = ee.Image(
        ee.Algorithms.If(
            bands.contains("B5"),
            img.select("B5").toFloat(),   # L30
            ee.Algorithms.If(
                bands.contains("B8A"),
                img.select("B8A").toFloat(),  # S30
                img.select("B8").toFloat(),   # S30 fallback
            ),
        )
    )

    swir1_raw = ee.Image(
        ee.Algorithms.If(
            bands.contains("B6"),
            img.select("B6").toFloat(),   # L30
            img.select("B11").toFloat(),  # S30
        )
    )

    swir2_raw = ee.Image(
        ee.Algorithms.If(
            bands.contains("B7"),
            img.select("B7").toFloat(),   # L30
            img.select("B12").toFloat(),  # S30
        )
    )

    # Rename to semantic band names, for clarity
    nir   = nir_raw.rename("NIR")
    red   = red_raw.rename("RED")
    blue  = blue_raw.rename("BLUE")
    swir1 = swir1_raw.rename("SWIR1")
    swir2 = swir2_raw.rename("SWIR2")

    # Vegetation / burn indices
    ndvi = safe_ndvi(nir, red).rename("NDVI")
    evi  = safe_evi(nir, red, blue).rename("EVI")
    # Simple "Foliage Index": mean of NIR + SWIR1 + SWIR2
    fi   = nir.add(swir1).add(swir2).divide(3.0).rename("FI")
    nbr  = safe_nbr(nir, swir2).rename("NBR")

    # DOY from system time; default to 0 if missing (rare / pathological)
    ts = img.get("system:time_start")
    doy_num = ee.Number(
        ee.Algorithms.If(
            ts,
            ee.Date(ts).getRelative("day", "year"),
            0,
        )
    )
    doy = ee.Image.constant(doy_num).rename("DOY").toInt16()

    # Add derived bands to original image, overwriting any conflicting names
    out = img.addBands(
        [blue, red, nir, swir1, swir2, ndvi, evi, fi, nbr, doy],
        overwrite=True,
    )

    # Apply clear-land mask
    out = out.updateMask(clear)

    # Preserve original properties (including time, tile ID, etc.)
    return out.copyProperties(img, img.propertyNames())


# ---------------------------------------------------------------------
# HLS COLLECTION + YEARLY SERIES
# ---------------------------------------------------------------------

def get_hls_collection(year: int, aoi: ee.Geometry) -> ee.ImageCollection:
    """
    Retrieve merged HLS L30 + S30 ImageCollection for a given year and AOI.

    Filters:
      - spatially by AOI
      - temporally [year-01-01, (year+1)-01-01)
      - removes scenes with null system:time_start
    """
    start = f"{year}-01-01"
    end   = f"{year + 1}-01-01"

    l30 = (
        ee.ImageCollection(HLS_L30)
        .filterBounds(aoi)
        .filterDate(start, end)
    )
    s30 = (
        ee.ImageCollection(HLS_S30)
        .filterBounds(aoi)
        .filterDate(start, end)
    )

    # Defensive filter: ensure time_start is present
    time_filter = ee.Filter.notNull(["system:time_start"])
    l30 = l30.filter(time_filter)
    s30 = s30.filter(time_filter)

    merged = l30.merge(s30)
    return merged


def build_yearly_series(year: int, aoi: ee.Geometry) -> ee.ImageCollection:
    """
    For a given year and AOI, return a per-image HLS series with:

      - NDVI, EVI, FI, NBR, DOY

    This series is the input to the annual metrics computation.
    """
    coll = get_hls_collection(year, aoi)
    return coll.map(add_indices).select(["NDVI", "EVI", "FI", "NBR", "DOY"])


# ---------------------------------------------------------------------
# METRICS PER INDEX (ROBUST TO EMPTY YEARS)
# ---------------------------------------------------------------------
def _metrics_for_index(series: ee.ImageCollection, idx_name: str) -> ee.Image:
    """
    Given an ImageCollection with bands [idx_name, DOY],
    compute per-year metrics:

      idx_max, idx_max_DOY,
      idx_min, idx_min_DOY,
      idx_p05, idx_p95

    If the collection is empty (no scenes / no clear pixels), returns an
    image where all bands are present but fully masked. When exported,
    those pixels become nodata in the GeoTIFF, which you can treat as
    NaN downstream (e.g., in xarray).
    """
    ic = series.select([idx_name, "DOY"])

    out_names = [
        f"{idx_name}_max",
        f"{idx_name}_max_DOY",
        f"{idx_name}_min",
        f"{idx_name}_min_DOY",
        f"{idx_name}_p05",
        f"{idx_name}_p95",
    ]

    def make_all_masked() -> ee.Image:
        """
        Build an image with the correct band names but fully masked
        everywhere. Values are irrelevant because the mask is 0
        for all pixels.
        """
        placeholder = ee.Image.constant([0] * 6).rename(out_names)
        mask = ee.Image.constant(0)  # mask == 0 -> everything masked
        return placeholder.updateMask(mask).toFloat()

    def compute_metrics() -> ee.Image:
        """
        Compute max/min/percentiles and associated DOY when the collection
        is non-empty.
        """
        # Max index + DOY of that max (qualityMosaic chooses highest idx_name)
        max_img  = ic.qualityMosaic(idx_name)
        idx_max  = max_img.select(idx_name).rename(f"{idx_name}_max")
        doy_max  = max_img.select("DOY").rename(f"{idx_name}_max_DOY")

        # Min index + DOY of that min via negative qualityMosaic
        def add_neg(im: ee.Image) -> ee.Image:
            return im.addBands(
                im.select(idx_name).multiply(-1).rename("negIdx")
            )

        ic_neg  = ic.map(add_neg)
        min_img = ic_neg.qualityMosaic("negIdx")
        idx_min = min_img.select(idx_name).rename(f"{idx_name}_min")
        doy_min = min_img.select("DOY").rename(f"{idx_name}_min_DOY")

        # Percentiles across the year (5th and 95th)
        pct = ic.select(idx_name).reduce(ee.Reducer.percentile([5, 95]))
        idx_p05 = pct.select(f"{idx_name}_p5").rename(f"{idx_name}_p05")
        idx_p95 = pct.select(f"{idx_name}_p95").rename(f"{idx_name}_p95")

        metrics = ee.Image.cat(
            [idx_max, doy_max, idx_min, doy_min, idx_p05, idx_p95]
        )

        return metrics.toFloat()

    # If there are no images in the collection for this index/year,
    # return a fully masked image with the right schema.
    return ee.Image(
        ee.Algorithms.If(
            ic.size().gt(0),
            compute_metrics(),
            make_all_masked(),
        )
    )


def compute_annual_metrics(year: int, aoi: ee.Geometry) -> ee.Image:
    """
    For a given year and AOI, compute all metrics for NDVI, EVI, FI, and NBR.

    Returns a single image with 24 bands:
      NDVI_* (6), EVI_* (6), FI_* (6), NBR_* (6)

    Band naming convention matches the header docstring.
    """
    series = build_yearly_series(year, aoi)

    ndvi_metrics = _metrics_for_index(series, "NDVI")
    evi_metrics  = _metrics_for_index(series, "EVI")
    fi_metrics   = _metrics_for_index(series, "FI")
    nbr_metrics  = _metrics_for_index(series, "NBR")

    out = ee.Image.cat([ndvi_metrics, evi_metrics, fi_metrics, nbr_metrics])
    return out.toFloat().set("year", year)


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

    for year in YEARS:
        print(f"Building annual metrics for year={year}")
        annual_img = compute_annual_metrics(year, aoi)

        base_name = f"hls_metrics_{year}"
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
