#!/usr/bin/env python3
"""
04_extract_lcms_to_gcs.py
Export LCMS bands (1985–2024) for a chosen AOI as multi-band COG GeoTIFFs.

For each LCMS band in BANDS:

    - Extract one annual image for each year in YEARS
    - Stack these into a single multi-band raster (one band per year)
    - Export ONE Cloud-Optimized GeoTIFF to GCS

Example outputs:

    lcms_chg_1985_2024.tif
    lcms_lc_1985_2024.tif
    lcms_lu_1985_2024.tif
    lcms_lc_p_trees_1985_2024.tif
    lcms_qa_interp_1985_2024.tif

Useful for downstream ML, temporal modeling, or Python/xarray/Zarr workflows.
"""

from __future__ import annotations

import ee
from utils.gee import get_state_geometry, ee_init, export_img_to_gcs

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ee_init()
ee.Initialize()
# Full LCMS temporal range used by the v2024-10 release
YEARS = list(range(1985, 2025))

LCMS_COLLECTION_ID = "USFS/GTAC/LCMS/v2024-10"

# Destination in Google Cloud Storage
GCS_BUCKET = "va_rasters"
GCS_DIR = "lcms"

# AOI default; you can swap to full state geometry in main().
#STATE_NAME = "Virginia"



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

# Long LCMS band names → short stems for filenames.
# The stems define the output band prefix in the final multiband GeoTIFF.
BAND_STEMS = {
    "Change": "chg",
    "Land_Cover": "lc",
    "Land_Use": "lu",
    "Change_Raw_Probability_Slow_Loss": "chg_p_slowloss",
    "Change_Raw_Probability_Fast_Loss": "chg_p_fastloss",
    "Change_Raw_Probability_Gain": "chg_p_gain",
    "Land_Cover_Raw_Probability_Trees": "lc_p_trees",
    "Land_Cover_Raw_Probability_Tall-Shrubs-and-Trees-Mix": "lc_p_tallshrubs_treesmix",
    "Land_Cover_Raw_Probability_Shrubs-and-Trees-Mix": "lc_p_shrubs_treesmix",
    "Land_Cover_Raw_Probability_Grass-Forb-Herb-and-Trees-Mix": "lc_p_grassforb_treesmix",
    "Land_Cover_Raw_Probability_Barren-and-Trees-Mix": "lc_p_barren_treesmix",
    "Land_Cover_Raw_Probability_Tall-Shrubs": "lc_p_tallshrubs",
    "Land_Cover_Raw_Probability_Shrubs": "lc_p_shrubs",
    # "Land_Cover_Raw_Probability_Grass-Forb-Herb-and-Shrubs-Mix": "lc_p_grassforb_shrubsmix",
    # "Land_Cover_Raw_Probability_Barren-and-Shrubs-Mix": "lc_p_barren_shrubsmix",
    # "Land_Cover_Raw_Probability_Grass-Forb-Herb": "lc_p_grassforb",
    # "Land_Cover_Raw_Probability_Barren-and-Grass-Forb-Herb-Mix": "lc_p_barren_grassforbmix",
    "Land_Cover_Raw_Probability_Barren-or-Impervious": "lc_p_barren_impervious",
    # "Land_Cover_Raw_Probability_Snow-or-Ice": "lc_p_snowice",
    "Land_Cover_Raw_Probability_Water": "lc_p_water",
    "Land_Use_Raw_Probability_Agriculture": "lu_p_ag",
    "Land_Use_Raw_Probability_Developed": "lu_p_dev",
    "Land_Use_Raw_Probability_Forest": "lu_p_forest",
    # "Land_Use_Raw_Probability_Other": "lu_p_other",
    "Land_Use_Raw_Probability_Rangeland-or-Pasture": "lu_p_rangeland_pasture",
}

# All bands we will export; comment some out here if you want to limit output.
BANDS = list(BAND_STEMS.keys())

# Short stem for the per-year QA mask time series
QA_STEM = "qa_interp"    # bit 0 == 1 → valid, 0 → invalid/interpolated


# ---------------------------------------------------------------------------
# CORE HELPERS
# ---------------------------------------------------------------------------

def get_lcms_collection() -> ee.ImageCollection:
    """
    Return LCMS ImageCollection filtered to the YEAR range and the CONUS study area.

    LCMS metadata:
      - Each image contains many thematic bands.
      - Each image has a 'year' property indicating the observation year.
    """
    start_year = min(YEARS)
    end_year = max(YEARS)

    coll = (
        ee.ImageCollection(LCMS_COLLECTION_ID)
        .filter(ee.Filter.eq("study_area", "CONUS"))
        .filter(ee.Filter.calendarRange(start_year, end_year, "year"))
    )
    return coll


def build_qa_mask(img: ee.Image) -> ee.Image:
    """
    Build a simple QA mask using LCMS QA_Bits band.

    LCMS QA_Bits interpretation (simplified):
      - bit0 == 1 → “good” / non-interpolated pixel
      - bit0 == 0 → interpolated / lower-confidence pixel

    We produce a uint8 mask where:
      1 = good
      0 = bad

    This mask is stacked per-year into a QA time series.
    """
    qa = img.select("QA_Bits")
    bit0 = qa.bitwiseAnd(1)  # extract LSB
    return bit0.toUint8()


def build_multiyear_stack_for_band(
    base_coll: ee.ImageCollection,
    band_name: str,
    stem: str,
) -> ee.Image:
    """
    Build a multi-band LCMS time series for a *single* LCMS band.

    Output band names: <stem>_<year>, e.g. chg_1992, lc_p_trees_2010.
    """
    years_list = ee.List(YEARS)
    coll = base_coll.select(band_name)

    def year_to_img(y):
        y = ee.Number(y)
        img = ee.Image(
            coll.filter(ee.Filter.eq("year", y)).first()
        )
        # Just select the band; no rename using y inside the mapped function
        return img.select(band_name)

    imgs = years_list.map(year_to_img)
    ic = ee.ImageCollection(imgs)
    stacked = ic.toBands()

    # Now rename on the client side using the plain Python YEARS list
    band_names = [f"{stem}_{year}" for year in YEARS]
    stacked = stacked.rename(band_names)

    return stacked


def build_multiyear_stack_for_qa(
    base_coll: ee.ImageCollection,
    stem: str,
) -> ee.Image:
    """
    Build a multi-band QA mask time series using build_qa_mask().

    For each year:
      - Extract LCMS image for that year.
      - Compute QA mask (0/1).
      - Stack into one multi-band image with bands:
            <stem>_<year>

    Returns
    -------
    ee.Image
        Multi-band QA image with per-year quality flags.
    """
    years_list = ee.List(YEARS)

    def year_to_qa(y):
        y = ee.Number(y)
        img = ee.Image(
            base_coll.filter(ee.Filter.eq("year", y)).first()
        )
        # build_qa_mask returns an ee.Image (uint8); no rename here either
        return build_qa_mask(img)

    imgs = years_list.map(year_to_qa)
    ic = ee.ImageCollection(imgs)
    stacked = ic.toBands()

    band_names = [f"{stem}_{year}" for year in YEARS]
    stacked = stacked.rename(band_names)

    return stacked
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Workflow:

      1. Initialize Earth Engine.
      2. Load LCMS multi-year ImageCollection.
      3. Choose AOI (state geometry or debug rectangle).
      4. For each LCMS band in BANDS:
            - Build multi-year stacked image.
            - Export as Cloud-Optimized GeoTIFF.
      5. Export the QA mask time series.

    This produces one COG per band plus one COG for QA.
    """
    ee_init()

    # Debug AOI: tiny rectangle in western/central Virginia.
    # Swap with the full state for production export:
    #
    # aoi = get_state_geometry(STATE_NAME)
    #
    #aoi = ee.Geometry.Rectangle([-79.0, 37.9, -78.9, 38.0])

    base_coll = get_lcms_collection()

    # 1) Export all configured LCMS bands
    for band_name in BANDS:
        stem = BAND_STEMS.get(band_name)
        if stem is None:
            # Fallback sanitization if BAND_STEMS is incomplete
            stem = band_name.replace(" ", "_").replace("/", "_")

        print(f"Building multi-year stack for LCMS band='{band_name}', stem='{stem}'")
        stacked_img = build_multiyear_stack_for_band(
            base_coll=base_coll,
            band_name=band_name,
            stem=stem,
        )

        start_year = min(YEARS)
        end_year = max(YEARS)
        base_name = f"lcms_{stem}_{start_year}_{end_year}"

        export_img_to_gcs(
            img=stacked_img,
            aoi=PADDED_REGION,
            bucket=GCS_BUCKET,
            base_name=base_name,
            gcs_dir=GCS_DIR,
            crs=TARGET_CRS,
            crsTransform=TARGET_TRANSFORM,
        )

    # 2) Export QA stack
    print("Building multi-year QA mask stack…")
    qa_stacked = build_multiyear_stack_for_qa(
        base_coll=base_coll,
        stem=QA_STEM,
    )

    start_year = min(YEARS)
    end_year = max(YEARS)
    base_name = f"lcms_{QA_STEM}_{start_year}_{end_year}"

    export_img_to_gcs(
        img=qa_stacked,
        aoi=PADDED_REGION,
        bucket=GCS_BUCKET,
        base_name=base_name,
        gcs_dir=GCS_DIR,
        crs=TARGET_CRS,
        crsTransform=TARGET_TRANSFORM,
    )


if __name__ == "__main__":
    main()
