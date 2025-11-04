#!/usr/bin/env python3
# ================================================================
# File: 05_download_lcms_from_gee.py
# Purpose: Download LCMS land use, cover, and change layers
#          from Google Earth Engine for all available years (>=2010)
#          using the most recent LCMS collection.
# NOTE: THIS DOESN'T DOWNLOAD OFF OF GOOGLE DRIVE, YOU NEED TO DO THAT
#
# Dataset:
#   - USFS Landscape Change Monitoring System (LCMS)
#     https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_LCMS_v2024-10
#
# Notes:
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
# SERVICE_ACCOUNT = "gee-vqvae@ee-nnnagle.iam.gserviceaccount.com"
# KEY_FILE = "/home/nnagle/.config/earthengine/service-account.json"

# credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
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
lcms = ee.ImageCollection("USFS/GTAC/LCMS/v2024-10")

lcms_year_range = list(range(2011, 2012)) 

# ------------------------------------------------
# Export function
# ------------------------------------------------
def export_lcms(image, year, layer_type):
  desc = f"LCMS_{layer_type}_{year}_{STATE_NAME.replace(' ', '_')}"
  out_path = f"{OUT_DIR}/{desc}.tif"
  log("Preparing export: %s", desc)
  task = ee.batch.Export.image.toDrive(
    image=image,
    description=desc,
    folder="lcms_exports",
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
img = lcms.first()
props = img.propertyNames().getInfo()
print("LCMS Metadata keys:", props)

for key in props:
    print(key, ":", img.get(key).getInfo())

for yr in lcms_year_range:
  img = lcms.filter(ee.Filter.eq("year",int(yr))).filter(ee.Filter.eq("study_area", "CONUS"))
  if img.size().getInfo() == 0:
    log("WARNING: No LCMS image for year %d over study_area=CONUS", yr)
    continue
  img = img.first().clip(region)
  export_lcms(img.select("Land_Cover"), yr, "LandCover")
  export_lcms(img.select("Land_Use"), yr, "LandUse")
  export_lcms(img.select("Change"), yr, "Change")


log("All export tasks have been submitted to Earth Engine.")

