#!/usr/bin/env python3
# ================================================================
# File: 06_download_nlcd_lc_from_gee.py
# Purpose: Download NLCD land cover and forest cover (canopy) layers
#          from Google Earth Engine for all available years (>=2000)
#          using the most recent NLCD collection.
#
# Dataset:
#   - USGS National Land Cover Database (NLCD)
#     https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2021_REL_NLCD
#
# Notes:
#   - Land cover products are available for 2001, 2004, 2006, 2008,
#     2011, 2013, 2016, 2019, 2021 
#   - This script uses the Earth Engine Python API to export both
#     layers to Google Drive or Cloud Storage for a user-defined AOI.
#
# Inputs:
#   - State name or custom GeoJSON defining the area of interest (AOI)
#
# Outputs:
#   - GeoTIFF rasters for each NLCD year:
#       * NLCD_LandCover_<YEAR>.tif
#       * NLCD_TreeCanopy_<YEAR>.tif
#
# Requirements:
#   - Google Earth Engine Python API authenticated (`earthengine authenticate`)
#   - Environment variable: GOOGLE_APPLICATION_CREDENTIALS (if using service account)
#   - Python modules: ee, utils.log, utils.io, utils.gee
#
# ================================================================

import ee
from utils.log import log, fail, ensure
from utils.io import ensure_dir
from utils.gee import get_state_geometry

# ------------------------------------------------
# Initialize GEE
# ------------------------------------------------
try:
    ee.Initialize(project="ee-nnnagle")
    log("Initialized Earth Engine API successfully.")
except Exception as e:
    fail(f"Failed to initialize Earth Engine: {e}")

# ------------------------------------------------
# Configurable parameters
# ------------------------------------------------
STATE_NAME = "Virginia"         # change this as needed
OUT_DIR = "data/nlcd"
ensure_dir(OUT_DIR)

region = get_state_geometry(STATE_NAME)

# ------------------------------------------------
# NLCD Collections
# ------------------------------------------------
nlcd_landcover = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")


#nlcd_year_range = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
nlcd_year_range = [2011, 2013, 2016, 2019, 2021]

# ------------------------------------------------
# Export function
# ------------------------------------------------
def export_nlcd(image, year, layer_type):
    desc = f"NLCD_{layer_type}_{year}_{STATE_NAME.replace(' ', '_')}"
    out_path = f"{OUT_DIR}/{desc}.tif"
    log("Preparing export: %s", desc)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=desc,
        folder="nlcd_exports",
        fileNamePrefix=desc,
        region=region,
        scale=30,
        crs="EPSG:5070",
        maxPixels=1e13
    )
    task.start()
    log("Export started for %s", desc)

# ------------------------------------------------
# Loop over years and export both layers
# ------------------------------------------------

img = nlcd_landcover.first()
props = img.propertyNames().getInfo()
print("NLCD LC Metadata keys:", props)


for key in props:
    print(key, ":", img.get(key).getInfo())

for yr in nlcd_year_range:
    img = nlcd_landcover.filter(ee.Filter.eq("system:index",str(yr)))
    if img.size().getInfo() == 0:
        log("WARNING: No NLCD image for year %d", yr)
        continue
    img = img.first().clip(region)
    export_nlcd(img.select("landcover"), yr, "LandCover")


log("All export tasks have been submitted to Earth Engine.")

