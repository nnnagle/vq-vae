#!/usr/bin/env python3
"""
Export NAIP-derived texture/NDVI features for Virginia, snapped to a 30 m target grid.

For each year in YEARS:

  1. Load NAIP imagery over the AOI.
  2. Add DOY (day-of-year) and YEAR bands to each image.
  3. Filter NAIP for that year and SORT by acquisition date (system:time_start).
  4. Mosaic NAIP for that year.
  5. Compute NDVI and texture features on NIR and NDVI:
       - NDVI
       - NIR_var_7m
       - NIR_var_15m
       - NIR_ent_21m
       - NIR_lac_21m
       - NDVI_var_15m
       - DOY
  6. Aggregate from NAIP resolution → 5 m → 30 m and snap to the target grid.
  7. Export a multi-band GeoTIFF to GCS.

Resulting rasters are NAIP-based feature stacks aligned with 30 m pixels,
ready for downstream modeling.
"""

from __future__ import annotations

import ee
from utils.gee import get_state_geometry, ee_init, export_img_to_gcs

ee_init()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STATE_NAME = "Virginia"

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

# Your GCS bucket + output subdirectory
GCS_BUCKET = "va_rasters"
GCS_DIR = "naip"

# NAIP years to process
YEARS = [2011, 2012, 2014, 2016, 2018, 2021, 2023]
# YEARS = [2011]   # debug: single year

# ---------------------------------------------------------------------------
# HELPERS: TEMPORAL METADATA (DOY / YEAR)
# ---------------------------------------------------------------------------

def add_doy(img: ee.Image) -> ee.Image:
    """
    Add DOY (day-of-year) and YEAR bands to a NAIP image.

    Notes:
      - DOY is 1-based (Jan 1 = 1), derived from system:time_start.
      - YEAR is the calendar year of the observation.
    """
    img = ee.Image(img)
    date = img.date()

    # DOY: getRelative('day', 'year') is 0-based → +1
    doy = (
        ee.Image.constant(date.getRelative("day", "year").add(1))
        .rename("DOY")
        .toInt16()
    )

    year = (
        ee.Image.constant(date.get("year"))
        .rename("YEAR")
        .toInt16()
    )

    return (
        img.addBands([doy, year])
           .set("DOY", date.getRelative("day", "year").add(1))
           .set("YEAR", date.get("year"))
    )

# ---------------------------------------------------------------------------
# HELPERS: TEXTURE / LACUNARITY
# ---------------------------------------------------------------------------

def focal_variance(band_img: ee.Image, radius_m: float, out_name: str) -> ee.Image:
    """Compute local variance within a circular neighborhood."""
    return band_img.reduceNeighborhood(
        reducer=ee.Reducer.variance(),
        kernel=ee.Kernel.circle(radius_m, "meters"),
        skipMasked=True,
    ).rename(out_name)

def focal_entropy(band_img: ee.Image, radius_m: float, out_name: str) -> ee.Image:
    """Compute local entropy using a circular kernel."""
    kernel = ee.Kernel.circle(radius_m, "meters")
    return band_img.entropy(kernel).rename(out_name)

def focal_lacunarity(img: ee.Image, radius_m: float, out_name: str) -> ee.Image:
    """
    Lacunarity = var(sum) / mean^2 + 1 over a moving circular window.
    """
    kernel = ee.Kernel.circle(radius_m, "meters")

    sum_img = img.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=kernel,
        skipMasked=True,
    ).rename("sum")

    mean_img = img.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=kernel,
        skipMasked=True,
    ).rename("mean")

    sum_var = sum_img.reduceNeighborhood(
        reducer=ee.Reducer.variance(),
        kernel=kernel,
        skipMasked=True,
    ).rename("sum_var")

    return sum_var.divide(mean_img.multiply(mean_img)).add(1).rename(out_name)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Entry point:

      1. Initialize Earth Engine.
      2. Prepare NAIP collection and reference projection.
      3. Loop over YEARS:
           - filter & sort NAIP by year/time
           - mosaic
           - compute NDVI + texture features
           - aggregate to 5 m, then snap to 30 m target grid
           - export as GeoTIFF to GCS
    """
    ee_init()

    # --------------------------
    # NAIP ImageCollection
    # --------------------------
    naip = ee.ImageCollection("USDA/NAIP/DOQQ")

    # Grab NAIP projection for setting a default before reduceResolution.
    # This ensures our intermediate 5 m grid is consistent.
    naip_proj = naip.first().projection()

    # Add DOY/YEAR bands ONCE (per-image, based on each image's own timestamp)
    naip_with_doy = naip.map(add_doy)

    # ----------------
    # Loop over years
    # ----------------
    for yr in YEARS:
        print(f"Processing NAIP year {yr}")

        # Filter to the target year and SORT by acquisition time.
        # NOTE: mosaic() uses the collection order to decide which pixels "win"
        # where images overlap. Sorting makes that choice deterministic.
        # - sort ascending: later images tend to win (because later in the stack)
        # If you want earliest images to win instead, sort descending.
        naip_y = (
            naip_with_doy
            .filter(ee.Filter.eq("YEAR", yr))
            .sort("system:time_start")   # ascending
        )

        # Mosaic all NAIP tiles for that year into a single image, then clip
        mosaic = naip_y.mosaic().clip(PADDED_REGION)

        # ------------------
        # Compute NDVI
        # ------------------
        nir = mosaic.select("N")
        red = mosaic.select("R")

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        ndvi = ndvi.updateMask(nir.add(red).gt(0))  # mask out bad/zero denominators

        mosaic_ndvi = mosaic.addBands(ndvi)

        # ------------------
        # Texture features
        # ------------------
        nir_b  = mosaic_ndvi.select("N")
        ndvi_b = mosaic_ndvi.select("NDVI")

        NIR_var_7m   = focal_variance(nir_b, 7,  "NIR_var_7m")
        NIR_var_15m  = focal_variance(nir_b, 15, "NIR_var_15m")
        NIR_ent_21m  = focal_entropy(nir_b, 21,  "NIR_ent_21m")
        NIR_lac_21m  = focal_lacunarity(nir_b, 21, "NIR_lac_21m")
        NDVI_var_15m = focal_variance(ndvi_b, 15, "NDVI_var_15m")

        mosaic_tex = mosaic_ndvi.addBands([
            NIR_var_7m,
            NIR_var_15m,
            NIR_ent_21m,
            NIR_lac_21m,
            NDVI_var_15m,
        ])

        # ------------------
        # Build feature image (still at NAIP resolution)
        # ------------------
        feature_img = mosaic_tex.select([
            "NDVI",
            "NIR_var_7m",
            "NIR_var_15m",
            "NIR_ent_21m",
            "NIR_lac_21m",
            "NDVI_var_15m",
            "DOY",
        ])

        # ---------------------------------------------------------
        # Set working projection (5 m) BEFORE reduceResolution
        # ---------------------------------------------------------
        coarse_proj = naip_proj.atScale(5)
        feature_img = feature_img.setDefaultProjection(coarse_proj)

        # ---------------------------------------------------------
        # Aggregate NAIP→5 m, then snap 5 m→30 m target grid
        # ---------------------------------------------------------
        feature_30m = (
            feature_img
            .reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=16384,
            )
            .reproject(crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM)
            .toFloat()
        )

        # ------------------
        # Export per-year GeoTIFF to GCS
        # ------------------
        base_name = f"naip_{yr}"

        export_img_to_gcs(
            img=feature_30m,
            aoi=PADDED_REGION,
            bucket=GCS_BUCKET,
            base_name=base_name,
            gcs_dir=GCS_DIR,
            crs=TARGET_CRS,
            crsTransform=TARGET_TRANSFORM,
        )

        print(f"Started export for {yr}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
