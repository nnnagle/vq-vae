#!/usr/bin/env python3
"""
Export an ML-ready static stack (topography + soils) to GCS as two GeoTIFFs:

1) FEATURES (float32): 8-band stack, each band masked by its corresponding QC mask.
2) MASKS (uint8): 5-band stack of explicit 0/1 QC masks, unmasked everywhere (no nodata holes).

Why two exports?
- Earth Engine exports GeoTIFFs with a single pixel type per image. If you cat() float + uint8,
  masks usually get promoted to float. Splitting keeps masks truly uint8 on disk.

Outputs aggregated to your 30m AEA grid via median (features) and max (masks).

FEATURE BANDS:
  - elevation          (m)              [masked by dem_mask]
  - slope_deg          (deg)            [masked by dem_mask]
  - northness          (cos(aspect))    [masked by dem_mask]
  - eastness           (sin(aspect))    [masked by dem_mask]
  - HAND               (m)              [masked by hand_mask]
  - TWI                (log(a/tan(β)))  [masked by twi_mask]
  - TPI_500m           (m)              [masked by tpi_mask]
  - ksat_log10_0_5     (log10(cm/hr))   [masked by ksat_mask]

MASK BANDS (uint8 0/1, unmasked everywhere):
  - dem_mask
  - tpi_mask
  - hand_mask
  - twi_mask
  - ksat_mask

Rule:
- Compute derivatives on each asset's native grid.
- Only at the end aggregate to your 30m AEA export grid.
"""

from __future__ import annotations

import math
import ee

from utils.gee import ee_init, export_img_to_gcs, get_state_geometry

# ---------------------------------------------------------------------
# CONFIG: AOI + OUTPUT GRID (your exact settings)
# ---------------------------------------------------------------------
ee_init()

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

PADDED_REGION = ee.Geometry.Rectangle(
    [1089315, 1574805, 1795875, 1966485],
    proj=TARGET_CRS,
    geodesic=False,
)

GCS_BUCKET = "va_rasters"
GCS_DIR = "topo"

# ---------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------
DEM_ASSET     = "USGS/3DEP/10m_collection"   # elevation band: "elevation" (m) - ImageCollection
HAND_ASSET    = "MERIT/Hydro/v1_0_1"         # band: "hnd" (m)
FLOWACC_ASSET = "WWF/HydroSHEDS/15ACC"       # band: "b1" upstream cells

# POLARIS Ksat (log10(cm/hr)) – depth slices are images in a collection, by system:index
# NOTE: Only _mean collections exist, not _p5 or _p95
POLARIS_KSAT_MEAN = "projects/sat-io/open-datasets/polaris/ksat_mean"
POLARIS_DEPTH_ID  = "ksat_0_5"

# TWI numerical stability
EPS_SLOPE_RAD = 1e-4
MIN_TAN = 1e-6

# TPI window
TPI_RADIUS_M = 500  # radius (meters)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def median_to_output_grid(native_img: ee.Image) -> ee.Image:
    """Aggregate from native projection -> your 30m AEA grid using median."""
    return (
        native_img
        .reduceResolution(reducer=ee.Reducer.median(), maxPixels=4096)
        .reproject(crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM)
    )

def max_to_output_grid(native_mask: ee.Image, native_proj: ee.Projection) -> ee.Image:
    """
    Aggregate a 0/1 mask to output grid using max (logical OR).
    Ensures masks remain explicit 0/1 everywhere (unmasked).
    """
    return (
        native_mask
        .setDefaultProjection(native_proj)
        .reduceResolution(reducer=ee.Reducer.max(), maxPixels=4096)
        .reproject(crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM)
        .unmask(0)
        .toUint8()
    )

def pick_depth(ic: ee.ImageCollection, depth_id: str) -> ee.Image:
    """Select the image in the collection with system:index == depth_id."""
    return ee.Image(ic.filter(ee.Filter.eq("system:index", depth_id)).first())

# ---------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------
def compute_stack_ml_ready(aoi: ee.Geometry) -> tuple[ee.Image, ee.Image]:
    # ----------------------------
    # DEM-native: elevation, slope/aspect, TPI
    # ----------------------------
    dem_ic = ee.ImageCollection(DEM_ASSET).select("elevation")
    dem_ref = ee.Image(dem_ic.first()).select("elevation")
    dem_proj = dem_ref.projection()

    # Mosaic can lose default projection; force a valid default projection before any ops
    dem = (
        dem_ic.mosaic()
        .setDefaultProjection(dem_proj)
        .clip(aoi)
        .toFloat()
    )

    slope_deg  = ee.Terrain.slope(dem).rename("slope_deg").toFloat()
    aspect_deg = ee.Terrain.aspect(dem).toFloat()
    aspect_rad = aspect_deg.multiply(math.pi / 180.0)

    northness = aspect_rad.cos().rename("northness").toFloat()
    eastness  = aspect_rad.sin().rename("eastness").toFloat()

    kernel = ee.Kernel.circle(radius=TPI_RADIUS_M, units="meters", normalize=False)
    dem_mean_500 = dem.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=kernel,
        skipMasked=True
    )
    tpi_500 = dem.subtract(dem_mean_500).rename("TPI_500m").toFloat()

    # ----------------------------
    # HAND-native
    # ----------------------------
    hand = (
        ee.Image(HAND_ASSET)
        .select("hnd")
        .clip(aoi)
        .rename("HAND")
        .toFloat()
    )
    hand_proj = hand.projection()

    # ----------------------------
    # Flowacc-native: upstream area (m^2) on its native grid
    # ----------------------------
    flowacc = ee.Image(FLOWACC_ASSET).select("b1").clip(aoi).toFloat()
    flow_proj = flowacc.projection()

    upstream_area_m2 = (
        flowacc.add(1)
        .multiply(ee.Image.pixelArea().reproject(flow_proj))
        .rename("upstream_area_m2")
        .setDefaultProjection(flow_proj)
    )

    # ----------------------------
    # TWI computed on DEM grid (combine slope from DEM + upstream area from flowacc)
    # ----------------------------
    slope_rad = slope_deg.multiply(math.pi / 180.0)
    tan_slope = slope_rad.max(EPS_SLOPE_RAD).tan().max(MIN_TAN)

    upstream_area_on_dem = upstream_area_m2.reproject(dem_proj)
    cellsize_m = ee.Number(dem_proj.nominalScale())
    a_specific = upstream_area_on_dem.divide(cellsize_m)

    twi = a_specific.divide(tan_slope).log().rename("TWI").toFloat()

    # ----------------------------
    # POLARIS Ksat (log10 space) - mean only
    # ----------------------------
    ksat_mean = (
        pick_depth(ee.ImageCollection(POLARIS_KSAT_MEAN), POLARIS_DEPTH_ID)
        .rename("ksat_log10_0_5")
        .toFloat()
        .clip(aoi)
    )
    ksat_proj = ksat_mean.projection()

    # ----------------------------
    # Per-layer sanity masks (make explicit 0/1 and unmasked)
    # ----------------------------
    dem_mask = (
        dem.mask()
        .And(slope_deg.mask())
        .And(northness.mask())
        .And(eastness.mask())
        .And(dem.gt(-200).And(dem.lt(6000)))
        .And(slope_deg.gte(0).And(slope_deg.lte(80)))
    ).rename("dem_mask")
    dem_mask_u = dem_mask.unmask(0).toUint8()

    tpi_mask = (
        tpi_500.mask()
        .And(tpi_500.gt(-500).And(tpi_500.lt(500)))
    ).rename("tpi_mask")
    tpi_mask_u = tpi_mask.unmask(0).toUint8()

    hand_mask = (
        hand.mask()
        .And(hand.gte(-5).And(hand.lte(200)))
    ).rename("hand_mask")
    hand_mask_u = hand_mask.unmask(0).toUint8()

    twi_mask = (
        twi.mask()
        .And(twi.gt(-50).And(twi.lt(50)))
    ).rename("twi_mask")
    twi_mask_u = twi_mask.unmask(0).toUint8()

    ksat_mask = (
        ksat_mean.mask()
        .And(ksat_mean.gt(-6).And(ksat_mean.lt(3)))
    ).rename("ksat_mask")
    ksat_mask_u = ksat_mask.unmask(0).toUint8()

    # ----------------------------
    # Apply masks to features (updateMask expects a mask image; nonzero => valid)
    # ----------------------------
    dem_features = (
        ee.Image.cat([
            dem.rename("elevation"),
            slope_deg,
            northness,
            eastness,
        ])
        .updateMask(dem_mask_u)
        .setDefaultProjection(dem_proj)
    )

    tpi_masked  = tpi_500.updateMask(tpi_mask_u).setDefaultProjection(dem_proj)
    hand_masked = hand.updateMask(hand_mask_u).setDefaultProjection(hand_proj)
    twi_masked  = twi.updateMask(twi_mask_u).setDefaultProjection(dem_proj)
    ksat_masked = ksat_mean.updateMask(ksat_mask_u).setDefaultProjection(ksat_proj)

    # ----------------------------
    # Aggregate FEATURES to your 30m output grid (median)
    # ----------------------------
    elev_30  = median_to_output_grid(dem_features.select("elevation")).toFloat()
    slope_30 = median_to_output_grid(dem_features.select("slope_deg")).toFloat()
    north_30 = median_to_output_grid(dem_features.select("northness")).toFloat()
    east_30  = median_to_output_grid(dem_features.select("eastness")).toFloat()
    hand_30  = median_to_output_grid(hand_masked).toFloat()
    twi_30   = median_to_output_grid(twi_masked).toFloat()
    tpi_30   = median_to_output_grid(tpi_masked).toFloat()
    ksat_30  = median_to_output_grid(ksat_masked).toFloat()

    features_out = ee.Image.cat([
        elev_30.rename("elevation"),
        slope_30.rename("slope_deg"),
        north_30.rename("northness"),
        east_30.rename("eastness"),
        hand_30.rename("HAND"),
        twi_30.rename("TWI"),
        tpi_30.rename("TPI_500m"),
        ksat_30.rename("ksat_log10_0_5"),
    ]).set({
        "rule": "native_derivatives_then_median_to_output_grid",
        "tpi_radius_m": TPI_RADIUS_M,
        "ksat_space": "log10(cm/hr)",
        "dem_asset": DEM_ASSET,
        "hand_asset": HAND_ASSET,
        "flowacc_asset": FLOWACC_ASSET,
        "polaris_ksat_mean": POLARIS_KSAT_MEAN,
        "polaris_depth": POLARIS_DEPTH_ID,
        "stack_type": "features_float",
    })

    # ----------------------------
    # Aggregate MASKS to output grid (max) and ensure explicit 0/1 everywhere
    # ----------------------------
    dem_mask_30  = max_to_output_grid(dem_mask_u, dem_proj).rename("dem_mask")
    tpi_mask_30  = max_to_output_grid(tpi_mask_u, dem_proj).rename("tpi_mask")
    hand_mask_30 = max_to_output_grid(hand_mask_u, hand_proj).rename("hand_mask")
    twi_mask_30  = max_to_output_grid(twi_mask_u, dem_proj).rename("twi_mask")
    ksat_mask_30 = max_to_output_grid(ksat_mask_u, ksat_proj).rename("ksat_mask")

    masks_out = ee.Image.cat([
        dem_mask_30,
        tpi_mask_30,
        hand_mask_30,
        twi_mask_30,
        ksat_mask_30,
    ]).toUint8().set({
        "stack_type": "masks_uint8",
        "mask_values": "0=invalid_or_missing,1=valid",
    })

    return features_out, masks_out

# ---------------------------------------------------------------------
# MAIN: EXPORT
# ---------------------------------------------------------------------
def main():
    ee_init()
    aoi = get_state_geometry(STATE_NAME)

    features_img, masks_img = compute_stack_ml_ready(aoi)

    base_features = "topo_soils_features_30m_aea"
    base_masks    = "topo_soils_masks_30m_aea"

    print(f"Exporting {base_features} to gs://{GCS_BUCKET}/{GCS_DIR}/")
    export_img_to_gcs(
        img=features_img,
        aoi=PADDED_REGION,
        bucket=GCS_BUCKET,
        base_name=base_features,
        gcs_dir=GCS_DIR,
        crs=TARGET_CRS,
        crsTransform=TARGET_TRANSFORM,
    )

    print(f"Exporting {base_masks} to gs://{GCS_BUCKET}/{GCS_DIR}/")
    export_img_to_gcs(
        img=masks_img,
        aoi=PADDED_REGION,
        bucket=GCS_BUCKET,
        base_name=base_masks,
        gcs_dir=GCS_DIR,
        crs=TARGET_CRS,
        crsTransform=TARGET_TRANSFORM,
    )

if __name__ == "__main__":
    main()
