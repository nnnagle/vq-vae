#!/usr/bin/env python3
# 04_fit_ccdc_to_ee.py

from __future__ import annotations

import math
import ee

ee.Initialize()

# ----------------------------
# CONFIG
# ----------------------------
ASSET_ROOT = "projects/ee-nnnagle/assets/ccdc/ccdc_1986_2024"
TILE_PX = 4096
OVERLAP_PX = 0  # use 16/32 if you plan edge-safe postprocessing later

START_DATE = "1986-01-01"
END_DATE   = "2025-01-01"

SCALE_M = 30.0

# Safe ceiling; EE hard-fails if you exceed maxPixels
MAX_PIXELS_GLOBAL = 1e13

# ----------------------------
# AOI / PROJECTION
# ----------------------------
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

# ----------------------------
# Full padded region (eventual VA / CONUS)
# ----------------------------
PADDED_REGION = ee.Geometry.Rectangle(
    [1089315, 1574805, 1795875, 1966485],
    proj=TARGET_CRS,
    geodesic=False,
)
TARGET_TRANSFORM = [30, 0, 1089315, 0, -30, 1966485]

# ----------------------------
# Test tile (1024 × 1024) — override for now
# ----------------------------
PADDED_REGION_1024 = ee.Geometry.Rectangle(
    [1457955, 1724565, 1488675, 1755285],
    proj=TARGET_CRS,
    geodesic=False,
)
TARGET_TRANSFORM_1024 = [30, 0, 1457955, 0, -30, 1755285]

#PADDED_REGION = PADDED_REGION_1024
#TARGET_TRANSFORM = TARGET_TRANSFORM_1024

# ----------------------------
# Landsat C2 L2 collections (SR)
# ----------------------------
COLL_L5 = "LANDSAT/LT05/C02/T1_L2"
COLL_L7 = "LANDSAT/LE07/C02/T1_L2"
COLL_L8 = "LANDSAT/LC08/C02/T1_L2"
COLL_L9 = "LANDSAT/LC09/C02/T1_L2"

SR_SCALE = 0.0000275
SR_OFFSET = -0.2


def mask_landsat_c2_l2(img: ee.Image) -> ee.Image:
    """Mask clouds/shadows/snow/water/fill + saturation using QA_PIXEL and QA_RADSAT."""
    qa = img.select("QA_PIXEL")
    radsat = img.select("QA_RADSAT")

    def bit(b: int) -> ee.Image:
        return qa.bitwiseAnd(1 << b).neq(0)

    mask = (
        bit(0).Not()         # fill
        .And(bit(1).Not())   # dilated cloud
        .And(bit(2).Not())   # cirrus
        .And(bit(3).Not())   # cloud
        .And(bit(4).Not())   # cloud shadow
        .And(bit(5).Not())   # snow
        .And(bit(7).Not())   # water
        .And(radsat.eq(0))   # no saturation
    )
    return img.updateMask(mask)


def scale_sr(img: ee.Image) -> ee.Image:
    """Apply C2 SR scaling: refl = DN*0.0000275 - 0.2."""
    sr = img.select("SR_B.*").multiply(SR_SCALE).add(SR_OFFSET)
    return img.addBands(sr, overwrite=True)


def rename_bands_l57(img: ee.Image) -> ee.Image:
    """Rename L5/L7 surface reflectance bands to common names."""
    return img.select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],  
        ["GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
    )

def rename_bands_l89(img: ee.Image) -> ee.Image:
    """Rename L8/L9 surface reflectance bands to common names."""
    return img.select(
        ["SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        ["GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
    )


def add_ndvi_nbr(img: ee.Image) -> ee.Image:
    """Add NDVI and NBR bands."""
    ndvi = img.normalizedDifference(["NIR", "RED"]).rename("NDVI")
    nbr  = img.normalizedDifference(["NIR", "SWIR2"]).rename("NBR")
    return img.addBands([ndvi, nbr])


def prep_collection(
    coll_id: str,
    aoi: ee.Geometry,
    start: str,
    end: str,
    is_l89: bool
) -> ee.ImageCollection:
    """Prepare a Landsat C2 L2 SR collection: mask, scale, rename, add indices."""
    return (
        ee.ImageCollection(coll_id)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.notNull(["system:time_start"]))
        .map(mask_landsat_c2_l2)
        .map(scale_sr)
        .map(rename_bands_l89 if is_l89 else rename_bands_l57)
        #.map(add_ndvi_nbr)
        .select(["GREEN","RED","NIR","SWIR1", "SWIR2"])
    )


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
    x_end   = snap_up(xmax, x0, tile_m)

    # Y: y0 is origin at top; y decreases south because yScale is negative.
    # We’ll iterate from north (ymax) to south (ymin) in steps of tile_m.
    y_start = snap_up(ymax, y0, tile_m)      # northward snap
    y_end   = snap_down(ymin, y0, tile_m)    # southward snap target

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


def run_ccdc(tile_geom: ee.Geometry) -> ee.Image:
    """Run CCDC on NDVI+NBR for a single tile geometry."""
    ts = (
        prep_collection(COLL_L5, tile_geom, START_DATE, END_DATE, is_l89=False)
        .merge(prep_collection(COLL_L7, tile_geom, START_DATE, END_DATE, is_l89=False))
        .merge(prep_collection(COLL_L8, tile_geom, START_DATE, END_DATE, is_l89=True))
        .merge(prep_collection(COLL_L9, tile_geom, START_DATE, END_DATE, is_l89=True))
        .sort("system:time_start")
    )

    ccdc = ee.Algorithms.TemporalSegmentation.Ccdc(
        collection=ts,
        breakpointBands=['GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'],
        tmaskBands=["GREEN","SWIR2"],
        minObservations=6,
        chiSquareProbability=0.99,
        minNumOfYearsScaler=1.33,
        dateFormat=1,            # 0: days since 1/1/70, 1: fractional years
        **{"lambda": .002},
        maxIterations=25000,
    )
    return ccdc


def main():
    tiles = make_aligned_tiles(
        region=PADDED_REGION,
        crs=TARGET_CRS,
        crs_transform=TARGET_TRANSFORM,
        scale_m=SCALE_M,
        tile_px=TILE_PX,
        overlap_px=OVERLAP_PX,
    )

    print(f"Tiles: {len(tiles)}")

    # conservative per-tile maxPixels (tile + overlap)
    tile_side_px = TILE_PX + 2 * OVERLAP_PX
    max_pixels_tile = float(tile_side_px * tile_side_px) * 4.0  # ×4 safety (masking/pyramids etc.)
    max_pixels_tile = min(max_pixels_tile, MAX_PIXELS_GLOBAL)

    for row, col, geom, x_ul, y_ul in tiles:
        asset_id = f"{ASSET_ROOT}_r{row:03d}_c{col:03d}"
        desc = f"ccdc_r{row:03d}_c{col:03d}"

        ccdc = run_ccdc(geom)

        # Force strict grid alignment for this tile export
        tr = tile_transform(x_ul, y_ul, SCALE_M)

        task = ee.batch.Export.image.toAsset(
            image=ccdc,
            description=desc,
            assetId=asset_id,
            region=geom,
            crs=TARGET_CRS,
            crsTransform=tr,
            maxPixels=max_pixels_tile,
            pyramidingPolicy={".default": "sample"},  # CRITICAL for array images
        )
        task.start()
        print("Started:", desc, "->", asset_id, "task:", task.id)


if __name__ == "__main__":
    main()
