#!/usr/bin/env python3
"""
02_extract_years_since_change_to_gcs.py
Compute LCMS disturbance history for a given AOI and export as COGs.

Outputs (1985–2024 by default):

- ysfc_value_<YEAR>:
    Years since the most recent fast change, as a LOWER BOUND.
    For pixels with no fast change yet observed by YEAR, this is:
        YEAR - BASE_NO_CHANGE_YEAR
    which should be interpreted as "at least this many years since last event".

- ysfc_censored_<YEAR>:
    1 where ysfc_value is a censored lower bound
      (no fast change yet observed in [START_YEAR, YEAR]).
    0 where ysfc_value is an exact time since the last observed fast change.

- cum_event_count:
    Static (single-band) image: total number of fast changes in
    [START_YEAR, END_YEAR] at each pixel.

All exports are Cloud Optimized GeoTIFFs to GCS.
"""

from __future__ import annotations

import ee
from utils.gee import ee_init, export_img_to_gcs, get_state_geometry

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ee_init() 

# Turn on to print per-year diagnostics; useful when debugging logic
DEBUG = False  # set True to print yearly histograms / counts

# LCMS temporal range to process (inclusive)
START_YEAR = 1985
END_YEAR = 2024

# LCMS change product (v2024-10) in Earth Engine
LCMS_COLLECTION_ID = "USFS/GTAC/LCMS/v2024-10"

# Fast-change LCMS "Change" codes (1–16) for LCMS v2024-10:
#  1 Wind
#  2 Hurricane
#  3 Snow or Ice Transition
#  4 Desiccation
#  5 Inundation
#  6 Prescribed Fire
#  7 Wildfire
#  8 Mechanical Land Transformation
#  9 Tree Removal
# (Codes not listed are either slow change or other categories we ignore here.)
FAST_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Base value for pixels that have not yet experienced any fast change.
# Conceptually: "no fast change yet" up to the current year.
BASE_NO_CHANGE_YEAR = 1980

# Export / AOI config
GCS_BUCKET = "va_rasters"
GCS_DIR = "lcms_ysfc"

# Base names of exported rasters (no file extension)
OUTPUT_PREFIX_YSFC_VALUE = "lcms_ysfc_value_1985_2024"
OUTPUT_PREFIX_YSFC_CENSORED = "lcms_ysfc_censored_1985_2024"
OUTPUT_PREFIX_CUM_EVENTS = "lcms_cum_event_count_1985_2024"

# AOI: name used by get_state_geometry() in utils.gee
STATE_NAME = "Virginia"

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

# Alternate debugging AOI (commented out)
# USE_RECT_AOI = True
# RECT_AOI = [-79.0, 37.5, -78.5, 38.0]  # lon_min, lat_min, lon_max, lat_max

# EE reduceRegion limits
MAX_PIXELS = 1e8          # for histogram summaries
EXPORT_MAX_PIXELS = 1e13  # for exports; should comfortably exceed AOI * years
EXPORT_SCALE = 30         # LCMS nominal resolution (m)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_lcms_change_collection(start_year: int, end_year: int) -> ee.ImageCollection:
    """
    Return LCMS Change-band ImageCollection for CONUS and given year range.

    The collection is filtered to:
      - study_area == "CONUS"
      - calendar year in [start_year, end_year]
    and then reduced to the 'Change' band only.
    """
    if start_year > end_year:
        raise ValueError("START_YEAR must be <= END_YEAR")

    coll = (
        ee.ImageCollection(LCMS_COLLECTION_ID)
        .filter(ee.Filter.eq("study_area", "CONUS"))
        .filter(ee.Filter.calendarRange(start_year, end_year, "year"))
        .select("Change")
    )
    return coll


def get_year_image(lcms_change: ee.ImageCollection, year: int) -> ee.Image:
    """
    Return LCMS Change image for a given year, with 'year' and
    'system:time_start' properties set.
    """
    y = ee.Number(year)
    img = ee.Image(
        lcms_change.filter(ee.Filter.eq("year", y)).first()
    )

    img = img.set("year", y)
    # Use July 1 of the given year as the nominal time_start
    img = img.set("system:time_start", ee.Date.fromYMD(year, 7, 1).millis())
    return img


def build_last_fast_change_year_collection(
    start_year: int,
    end_year: int,
    fast_codes: list[int],
) -> tuple[ee.ImageCollection, ee.Image, ee.Image]:
    """
    Build:
      - an ImageCollection where each image has one band: 'LastFastChangeYear'
      - a union NPA mask across all years
      - a static cum_event_count image

    Concept:
      We iterate over years Y in [start_year, end_year], maintaining:
        A(Y) = most recent FAST change year observed up to and including Y.
        E(Y) = cumulative number of fast-change events up to and including Y.

      If no fast change has ever occurred at a pixel up to year Y,
      that pixel keeps the BASE_NO_CHANGE_YEAR value in A(Y).

    Returns:
        lfcy_collection: ImageCollection with one image per year
                         (1985–2024 by default) with band 'LastFastChangeYear'
        npa_union:       ee.Image mask where Change == 16 in ANY year
                         (Non-Processing Area union across years)
        cum_events:      ee.Image with band 'CumEventCount' (static)
    """

    # LCMS Change images for requested period
    lcms_change = get_lcms_change_collection(start_year, end_year)
    years = ee.List.sequence(start_year, end_year)

    # Use the first LCMS Change image as a projection/footprint template
    template = ee.Image(lcms_change.first()).select("Change")
    # ------------------------------------------------------------------
    # Union NPA mask:
    # Code 16 = Non-Processing Area (NPA). This mask is 1 anywhere that
    # any year has Change == 16. We later invert this to mask out NPAs.
    # ------------------------------------------------------------------
    npa_union = (
        lcms_change
        .map(lambda img: ee.Image(img).eq(16))  # 1 where Change == 16, else 0
        .sum()                                 # sum across years
        .gt(0)                                 # > 0 => NPA at least one year
    )

    # ------------------------------------------------------------------
    # Initialize iterative state for fold:
    # A   = cumulative LastFastChangeYear (starts at BASE_NO_CHANGE_YEAR)
    # E   = cumulative event count (starts at 0)
    # col = growing ImageCollection of per-year outputs
    # ------------------------------------------------------------------
    init_state = ee.Dictionary({
      # A: BASE_NO_CHANGE_YEAR everywhere, but in LCMS projection/footprint
      "A": template.multiply(0).add(BASE_NO_CHANGE_YEAR).rename("LastFastChangeYear"),
      # E: 0 everywhere, same projection
      "E": template.multiply(0).rename("CumEventCount"),
      "col": ee.ImageCollection([])
    })

    def yearly_step(year, state):
        """
        Fold step for each year:
          - Take previous A (last fast-change year) and E (cum events)
          - Identify pixels with a fast change in current year
          - Update A where fast change occurs
          - Update E by adding fast_mask
          - Append current A to output collection
        """
        state = ee.Dictionary(state)
        A = ee.Image(state.get("A"))
        E = ee.Image(state.get("E"))
        col = ee.ImageCollection(state.get("col"))

        year = ee.Number(year)
        img = get_year_image(lcms_change, year)
        change = img.select("Change")

        # Fast-change mask: 1 where Change is in fast_codes, else 0
        fast_mask = change.remap(
            fast_codes,           # input codes we care about
            [1] * len(fast_codes),# all mapped to 1
            0                     # default: 0
        )

        # Update last-fast-change year where a fast change occurs this year
        updated_A = A.where(fast_mask.eq(1), ee.Image.constant(year))

        # Update cumulative event count
        updated_E = E.add(fast_mask).rename("CumEventCount")

        # The output for this year is the *current* value of A
        out = updated_A.rename("LastFastChangeYear").copyProperties(
            img,
            ["year", "system:time_start"]
        )

        new_col = col.merge(ee.ImageCollection([out]))
        return ee.Dictionary({"A": updated_A, "E": updated_E, "col": new_col})

    # Run the fold across years, accumulating A, E, and the collection
    result = ee.Dictionary(years.iterate(yearly_step, init_state))
    lfcy_collection = ee.ImageCollection(result.get("col"))
    cum_events = ee.Image(result.get("E")).rename("CumEventCount")

    return lfcy_collection, npa_union, cum_events


def stack_years_to_multiband(
    collection: ee.ImageCollection,
    band_name: str,
    prefix: str,
) -> ee.Image:
    """
    Stack an annual ImageCollection into a single multi-band Image,
    with band names like '<prefix>1985', '<prefix>1986', etc., using toBands().
    """

    def rename_for_stack(img):
        """Rename the target band to prefix+year for clean stacking."""
        img = ee.Image(img)
        year_int = ee.Number(img.get("year")).int()
        year_str = year_int.format()  # e.g. "1985"
        new_band_name = ee.String(prefix).cat(year_str)
        return img.select(band_name).rename(new_band_name)

    # Sort by year for predictable band ordering, then map renaming
    renamed = collection.sort("year").map(rename_for_stack)
    stacked = ee.ImageCollection(renamed).toBands()

    # Normalize band names to just prefix+year via regex.
    band_names = stacked.bandNames()

    def strip_prefix(bn):
        bn = ee.String(bn)
        return bn.replace('.*' + prefix, prefix)

    cleaned_names = band_names.map(strip_prefix)
    stacked = stacked.rename(cleaned_names)

    return stacked

def stack_years_to_multiband_with_prefix(
    collection: ee.ImageCollection,
    prefix: str,
) -> ee.Image:
    """
    Stack a collection of single-band images whose band names all contain
    a known prefix (e.g., 'ysfc_value_'), and strip any toBands() cruft.
    """
    stacked = ee.ImageCollection(collection).toBands()
    band_names = stacked.bandNames()

    def strip_prefix(bn):
        bn = ee.String(bn)
        # e.g. '2_0_ysfc_censored_2024' -> 'ysfc_censored_2024'
        return bn.replace('.*' + prefix, prefix)

    cleaned = band_names.map(strip_prefix)
    stacked = stacked.rename(cleaned)
    return stacked


def build_ysfc_value_and_censored(
    lfcy_collection: ee.ImageCollection,
    base_no_change_year: int,
) -> tuple[ee.ImageCollection, ee.ImageCollection]:
    """
    From a LastFastChangeYear collection, build:

      - ysfc_value collection:
          Y - LastFastChangeYear
          For pixels where LastFastChangeYear == base_no_change_year,
          this is a LOWER BOUND (no fast change observed yet).

      - ysfc_censored collection:
          1 where LastFastChangeYear == base_no_change_year
          0 otherwise.

    Both collections carry 'year' and 'system:time_start' properties.
    """

    def build_value(img):
        img = ee.Image(img)
        year = ee.Number(img.get("year"))
        last_year = img.select("LastFastChangeYear")
        new_band = ee.String("ysfc_value_").cat(year.int().format('%d'))
        ys_value = last_year.expression(
            'y - last',
            {'y': year, 'last': last_year}
        ).rename(new_band)
        return ys_value.copyProperties(
            img,
            ["year", "system:time_start"]
        )

    def build_censored(img):
        img = ee.Image(img)
        year = ee.Number(img.get("year"))
        last_year = img.select("LastFastChangeYear")
        new_band = ee.String("ysfc_censored_").cat(year.int().format('%d'))
        censored = last_year.eq(base_no_change_year).rename(new_band)
        return censored.copyProperties(
            img,
            ["year", "system:time_start"]
        )

    ys_value_coll = lfcy_collection.map(build_value)
    ys_censored_coll = lfcy_collection.map(build_censored)

    return ys_value_coll, ys_censored_coll


def debug_yearly_counts(
    aoi: ee.Geometry,
    lfcy_collection: ee.ImageCollection,
    fast_codes: list[int],
):
    """
    Print per-year diagnostics for LCMS Change and the cumulative state.
    """
    print("=== DEBUG: yearly LCMS Change / fast-change / state counts ===")

    lcms_change = get_lcms_change_collection(START_YEAR, END_YEAR)

    for year in range(START_YEAR, END_YEAR + 1):
        # LCMS Change image for the year, clipped to AOI
        img = (
            lcms_change
            .filter(ee.Filter.eq("year", year))
            .first()
            .select("Change")
            .clip(aoi)
        )

        # Histogram of all Change codes within AOI
        change_hist = img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=EXPORT_SCALE,
            maxPixels=MAX_PIXELS,
        ).get("Change")

        # Fast-change mask and total count of fast-change pixels that year
        fast_mask = img.remap(
            fast_codes,
            [1] * len(fast_codes),
            0
        )
        fast_count = fast_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=EXPORT_SCALE,
            maxPixels=MAX_PIXELS,
        ).get("remapped")

        # Current LastFastChangeYear image for this year
        lf_img = (
            lfcy_collection
            .filter(ee.Filter.eq("year", year))
            .first()
            .select("LastFastChangeYear")
            .clip(aoi)
        )

        # Pixels where the state has been updated (i.e., had a fast change
        # in some year <= current year)
        state_updated_mask = lf_img.neq(BASE_NO_CHANGE_YEAR)
        state_count = state_updated_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=EXPORT_SCALE,
            maxPixels=MAX_PIXELS,
        ).get("LastFastChangeYear")

        # Materialize client-side for logging
        print(f"Year {year}:")
        print("  Change histogram:", change_hist.getInfo())
        print("  Fast-change pixel count:", fast_count.getInfo())
        print("  Pixels with updated state (A != BASE_NO_CHANGE_YEAR):", state_count.getInfo())


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Entry point:
      1. Initialize EE and AOI
      2. Build LastFastChangeYear ImageCollection + NPA mask + cum_event_count
      3. (Optional) print debug stats
      4. Build ysfc_value and ysfc_censored collections
      5. Stack ysfc_value and ysfc_censored into multiband images
      6. Mask NPAs, cast to int16
      7. Export ysfc_value, ysfc_censored, and cum_event_count to GCS
    """
    # Initialize Earth Engine authentication/session
    ee_init()

    # AOI geometry (state boundary by name). For custom AOI, swap this out.
    # aoi = get_state_geometry(STATE_NAME)
    # if USE_RECT_AOI:
    #     aoi = ee.Geometry.Rectangle(RECT_AOI)
    aoi = PADDED_REGION

    # 1) Build cumulative LastFastChangeYear collection + NPA mask + cum events
    lfcy_collection, npa_union, cum_events = build_last_fast_change_year_collection(
        start_year=START_YEAR,
        end_year=END_YEAR,
        fast_codes=FAST_CODES,
    )

    # Optional: debug yearly stats (for sanity checks on small AOIs)
    if DEBUG:
        debug_yearly_counts(aoi, lfcy_collection, FAST_CODES)

    # 2) Build ysfc_value and ysfc_censored collections from lfcy_collection
    ys_value_coll, ys_censored_coll = build_ysfc_value_and_censored(
        lfcy_collection=lfcy_collection,
        base_no_change_year=BASE_NO_CHANGE_YEAR,
    )

    ys_value_stacked = stack_years_to_multiband_with_prefix(
        ys_value_coll,
        prefix="ysfc_value_",
    )

    ys_censored_stacked = stack_years_to_multiband_with_prefix(
        ys_censored_coll,
        prefix="ysfc_censored_",
    )

    # 4) Apply union NPA mask and cast to int16
    # npa_union == 1 where pixel is NPA in any year; we invert to mask them out.
    mask = npa_union.Not()

    ys_value_stacked = ys_value_stacked.updateMask(mask).toInt16()
    ys_censored_stacked = ys_censored_stacked.updateMask(mask).toInt16()
    cum_events_masked = cum_events.updateMask(mask).toInt16()

    # 5) Export ysfc_value stack
    export_img_to_gcs(
        img=ys_value_stacked,
        aoi=aoi,
        bucket=GCS_BUCKET,
        base_name=OUTPUT_PREFIX_YSFC_VALUE,
        gcs_dir=GCS_DIR,
        crs=TARGET_CRS,
        crsTransform=TARGET_TRANSFORM,
    )

    # 6) Export ysfc_censored stack
    export_img_to_gcs(
        img=ys_censored_stacked,
        aoi=aoi,
        bucket=GCS_BUCKET,
        base_name=OUTPUT_PREFIX_YSFC_CENSORED,
        gcs_dir=GCS_DIR,
        crs=TARGET_CRS,
        crsTransform=TARGET_TRANSFORM,
    )

    # 7) Export cum_event_count (static)
    export_img_to_gcs(
        img=cum_events_masked,
        aoi=aoi,
        bucket=GCS_BUCKET,
        base_name=OUTPUT_PREFIX_CUM_EVENTS,
        gcs_dir=GCS_DIR,
        crs=TARGET_CRS,
        crsTransform=TARGET_TRANSFORM,
    )


if __name__ == "__main__":
    main()
