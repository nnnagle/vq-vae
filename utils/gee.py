#!/usr/bin/env python3
# ================================================================
# Module: utils/gee.py
# Purpose: Convenience wrappers for working with Google Earth Engine (EE)
#          from Python. Provides initialization, geometry loading,
#          year discovery, and export helpers.
#
# Description:
#   This module encapsulates common GEE operations used across
#   all data-fetch scripts (NLCD, LCMS, NAIP, etc.).  It provides
#   a consistent interface for:
#     - Initializing Earth Engine with a project or default credentials
#     - Loading state or custom AOI geometries
#     - Discovering available years in ImageCollections
#     - Exporting EE images to Google Drive or Cloud Storage
#
# Usage:
#   from utils.gee import ee_init, get_state_geometry, load_geojson_geometry
#   from utils.gee import export_image_drive, export_image_gcs, discover_years
#
# Dependencies:
#   - earthengine-api
#   - utils.log (for logging and error handling)
#
# Table of Contents:
#   1. Initialization
#   2. AOI helpers
#   3. Collection / metadata helpers
#   4. Export helpers (Drive / GCS)
#
# Function Manifest:
#   ee_init(project=None)
#       Initialize the Earth Engine API for the current user or project.
#
#   get_state_geometry(state_name)
#       Retrieve a U.S. state boundary polygon from the TIGER dataset.
#
#   load_geojson_geometry(path)
#       Load a GeoJSON file (Geometry, Feature, or FeatureCollection) and
#       return a unified ee.Geometry.
#
#   clip_to_region(image, region)
#       Clip an ee.Image to a given ee.Geometry.
#
#   discover_years(coll, prop="year")
#       Extract unique year values from an ee.ImageCollection.
#
#   export_image_drive(image, desc, region, ...)
#       Export an ee.Image to Google Drive (returns ee.batch.Task).
#
#   export_image_gcs(image, desc, region, bucket, ...)
#       Export an ee.Image to Google Cloud Storage (returns ee.batch.Task).
#
# Notes:
#   - All functions are idempotent: safe to call multiple times.
#   - Errors are fatal via utils.log.fail(); no unhandled exceptions escape.
#   - Coordinate Reference System (CRS) defaults to EPSG:5070 (NLCD/CONUS standard).
#
# ================================================================

# utils/gee.py
import json
from typing import Optional, List

import ee

from utils.log import log, fail


# ---------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------
def ee_init(project: Optional[str] = None) -> None:
    """
    Initialize the Earth Engine API.
    Usage:
        ee_init()                  # default credentials
        ee_init(project="ee-xyz")  # with explicit project
    """
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        log("Initialized Earth Engine API%s.",
            f" (project={project})" if project else "")
    except Exception as e:
        fail(f"Failed to initialize Earth Engine: {e}")


# ---------------------------------------------------------------------
# 2. AOI helpers
# ---------------------------------------------------------------------
def get_state_geometry(state_name: str) -> ee.Geometry:
    """
    Fetch a US state polygon from TIGER dataset (by NAME).
    Returns ee.Geometry (usually a MultiPolygon).
    """
    try:
        states = ee.FeatureCollection("TIGER/2018/States")
        fc = states.filter(ee.Filter.eq("NAME", state_name))
        feature = ee.Feature(fc.first())
        geom = ee.Geometry(feature.geometry())
    except Exception as e:
        fail(f"Unable to load TIGER geometry for state '{state_name}': {e}")
    log("Loaded TIGER geometry for state: %s", state_name)
    return geom


def load_geojson_geometry(path: str) -> ee.Geometry:
    """
    Load an AOI from a GeoJSON file that may contain:
      - a bare Geometry object,
      - a Feature with 'geometry',
      - a FeatureCollection (uses the union of all geometries).
    Returns ee.Geometry.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        fail(f"Failed to read GeoJSON '{path}': {e}")

    # Geometry
    if "type" in data and data["type"] in {"Polygon", "MultiPolygon", "Point",
                                           "MultiPoint", "LineString", "MultiLineString",
                                           "GeometryCollection"}:
        return ee.Geometry(data)

    # Feature
    if data.get("type") == "Feature" and "geometry" in data:
        return ee.Geometry(data["geometry"])

    # FeatureCollection
    if data.get("type") == "FeatureCollection" and "features" in data:
        geoms = [ee.Feature(feat).geometry() for feat in data["features"] if "geometry" in feat]
        if not geoms:
            fail(f"No geometries found in FeatureCollection '{path}'.")
        union = ee.Geometry(geoms[0])
        for g in geoms[1:]:
            union = union.union(ee.Geometry(g), maxError=1)
        return union

    fail(f"Unsupported GeoJSON structure in '{path}'.")
    # (unreachable, but keeps type checkers happy)
    return ee.Geometry.Point([0, 0])


def clip_to_region(image: ee.Image, region: ee.Geometry) -> ee.Image:
    """Clip a GEE image to a region (simple wrapper)."""
    return image.clip(region)


# ---------------------------------------------------------------------
# 3. Collection/metadata helpers
# ---------------------------------------------------------------------
def discover_years(coll: ee.ImageCollection, prop: str = "year") -> List[int]:
    """
    Discover unique 'year' values (or any numeric property) in a collection.
    Returns sorted list of ints.
    """
    try:
        vals = coll.aggregate_array(prop).getInfo() or []
        years = sorted(set(int(v) for v in vals))
        log("Discovered %d distinct %s values: %s", len(years), prop, years)
        return years
    except Exception as e:
        fail(f"Failed to discover '{prop}' values from collection: {e}")
        return []


# ---------------------------------------------------------------------
# Export helpers (Drive / GCS)
# ---------------------------------------------------------------------
def export_image_drive(
    image: ee.Image,
    desc: str,
    region: ee.Geometry,
    *,
    folder: str = "exports",
    scale: int = 30,
    crs: str = "EPSG:5070",
    max_pixels: float = 1e13
) -> ee.batch.Task:
    """
    Start a Drive export; returns the created ee.batch.Task.
    """
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=desc,
            folder=folder,
            fileNamePrefix=desc,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=max_pixels,
        )
        task.start()
        log("Drive export started: desc=%s, folder=%s, scale=%s, crs=%s, task_id=%s",
            desc, folder, scale, crs, task.id)
        return task
    except Exception as e:
        fail(f"Drive export failed for '{desc}': {e}")

import subprocess
import ee


def gcs_file_exists(bucket: str, prefix: str) -> bool:
    """
    Return True if gs://bucket/prefix.tif already exists.

    `prefix` can include "directories", e.g. "va/lcms_lc_2016_2024".
    """
    uri = f"gs://{bucket}/{prefix}.tif"
    try:
        result = subprocess.run(
            ["gsutil", "ls", uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        # gsutil not available; fall back to "assume does not exist"
        print("WARNING: gsutil not found; skipping GCS existence check.")
        return False


def export_img_to_gcs(
    img: ee.Image,
    aoi: ee.Geometry,
    bucket: str,
    base_name: str,
    scale: int = 30,
    maxPixels: int=1e9,
    gcs_dir: str | None = None,
    crs: str | None = None,
    crsTransform: list[float] | None = None,
):
    """
    Export an image as a single COG GeoTIFF to GCS.

    - Does *not* overwrite: if the target object exists, it skips the export.
      Because GCS doesn't overwrite anyway. 
    - If `gcs_dir` is provided, the file is written under that prefix.

    Final object path (no trailing slash on gcs_dir):
      gs://<bucket>/<gcs_dir>/<base_name>.tif
    or, if gcs_dir is None:
      gs://<bucket>/<base_name>.tif
    """
    img = ee.Image(img)
    aoi = ee.Geometry(aoi)
    # Build the GCS object prefix (without extension)
    if gcs_dir:
        gcs_dir = gcs_dir.rstrip("/")  # avoid double slashes
        prefix = f"{gcs_dir}/{base_name}"
    else:
        prefix = base_name

    # Clip to AOI once
    patch = img.clip(aoi)

    # Check for existing file
    if gcs_file_exists(bucket, prefix):
        print(f"WARNING: GCS file already exists: gs://{bucket}/{prefix}.tif")
        print("Skipping export.")
        return

    # Common export args
    export_kwargs = dict(
        image=patch,
        description=base_name,
        bucket=bucket,
        fileNamePrefix=prefix,
        region=aoi,
        maxPixels=maxPixels,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )

    #Handle crs / crsTransform vs scale:
    #- If crs_transform is provided: use crs/crsTransform, do NOT set scale.
    #- Else: use scale (and optional crs).
    if crsTransform is not None:
        export_kwargs["crsTransform"] = crsTransform  # EE expects camelCase
        if crs is not None:
            export_kwargs["crs"] = crs
    else:
        export_kwargs["scale"] = scale
        if crs is not None:
            export_kwargs["crs"] = crs

    # Kick off export
    task = ee.batch.Export.image.toCloudStorage(**export_kwargs)
    task.start()
    print(f"Started Tif export task: gs://{bucket}/{prefix}.tif, task_id={task.id}")


