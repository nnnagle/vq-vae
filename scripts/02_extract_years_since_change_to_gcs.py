#!/usr/bin/env python3
"""
02_extract_years_since_change_to_gcs.py
Compute LCMS disturbance history for a given AOI and export as COGs.

Outputs (1985–2024 by default):

- lfcy_<YEAR>: LastFastChangeYear
    Cumulative "year of most recent fast change" up to and including YEAR.
    Pixels that have never experienced a fast change retain BASE_NO_CHANGE_YEAR.

- ysfc_<YEAR>: YearsSinceFastChange
    YEAR - LastFastChangeYear, with SENTINEL_NEVER_CHANGED for pixels that
    have never experienced a fast change in the record.

Both stacks are exported as Cloud Optimized GeoTIFFs to GCS.
"""

from __future__ import annotations

import ee
from utils.gee import ee_init, export_img_to_gcs, get_state_geometry

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

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

# Sentinel for pixels that NEVER experience a fast change in [START_YEAR, END_YEAR]
# This is only applied in the YearsSinceFastChange product.
SENTINEL_NEVER_CHANGED = -1

# Export / AOI config
GCS_BUCKET = "va_rasters"
GCS_DIR = "lcms_ysfc"

# Base names of exported rasters (no file extension)
OUTPUT_PREFIX_LFCY = "lcms_last_fast_change_year_1985_2024"
OUTPUT_PREFIX_YSFC = "lcms_years_since_fast_change_1985_2024"

# AOI: name used by get_state_geometry() in utils.gee
STATE_NAME = "Virginia"

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

    This is the core LCMS disturbance classification we build everything from.
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

    The underlying LCMS collection already has a 'year' property; we just:
      - filter for the requested year
      - take .first() (there should be exactly one)
      - explicitly add 'year' and a synthetic mid-year timestamp for
        consistent temporal metadata.
    """
    y = ee.Number(year)
    img = ee.Image(
        lcms_change.filter(ee.Filter.eq("year", y)).first()
    )

    # Note: .first() always returns an ee.Image; if there is truly nothing,
    # downstream operations will just be fully masked. We keep a sanity check
    # but don't rely on it for EE logic.
    img = img.set("year", y)
    # Use July 1 of the given year as the nominal time_start
    img = img.set("system:time_start", ee.Date.fromYMD(year, 7, 1).millis())
    return img


def build_last_fast_change_year_collection(
    start_year: int,
    end_year: int,
    fast_codes: list[int],
) -> tuple[ee.ImageCollection, ee.Image]:
    """
    Build an ImageCollection where each image has one band: 'LastFastChangeYear'.

    Concept:
      We iterate over years Y in [start_year, end_year], maintaining a
      cumulative state A per pixel:

        A(Y) = most recent FAST change year observed up to and including Y.

      If no fast change has ever occurred at a pixel up to year Y,
      that pixel keeps the BASE_NO_CHANGE_YEAR value.

    Returns:
        lfcy_collection: ImageCollection with one image per year
                         (1985–2024 by default) with band 'LastFastChangeYear'
        npa_union:       ee.Image mask where Change == 16 in ANY year
                         (Non-Processing Area union across years)
    """

    # LCMS Change images for requested period
    lcms_change = get_lcms_change_collection(start_year, end_year)
    years = ee.List.sequence(start_year, end_year)

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
    # col = growing ImageCollection of per-year outputs
    # ------------------------------------------------------------------
    init_state = ee.Dictionary({
        "A": ee.Image.constant(BASE_NO_CHANGE_YEAR).rename("LastFastChangeYear"),
        "col": ee.ImageCollection([])
    })

    def yearly_step(year, state):
        """
        Fold step for each year:
          - Take previous state A (cumulative last-change year)
          - Identify pixels with a fast change in current year
          - Update A where fast change occurs
          - Append current A to output collection
        """
        state = ee.Dictionary(state)
        A = ee.Image(state.get("A"))      # cumulative last-fast-change year
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

        # If this year is a fast change at a pixel, update A := current year there.
        updated_A = A.where(fast_mask.eq(1), ee.Image.constant(year))

        # The output for this year is the *current* value of A
        out = updated_A.rename("LastFastChangeYear").copyProperties(
            img,
            ["year", "system:time_start"]
        )

        new_col = col.merge(ee.ImageCollection([out]))
        return ee.Dictionary({"A": updated_A, "col": new_col})

    # Run the fold across years, accumulating both A and the collection
    result = ee.Dictionary(years.iterate(yearly_step, init_state))
    lfcy_collection = ee.ImageCollection(result.get("col"))

    return lfcy_collection, npa_union


def stack_years_to_multiband(
    collection: ee.ImageCollection,
    band_name: str,
    prefix: str,
) -> ee.Image:
    """
    Stack an annual ImageCollection into a single multi-band Image,
    with band names like '<prefix>1985', '<prefix>1986', etc., using toBands().

    Parameters
    ----------
    collection : ee.ImageCollection
        Per-year images, each containing `band_name` and a 'year' property.
    band_name : str
        Name of the band to stack from each annual image.
    prefix : str
        Prefix for output band names, e.g. "lfcy_" or "ysfc_".

    Returns
    -------
    ee.Image
        Multi-band image with one band per year, named prefix+YYYY.
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

    # After toBands(), EE tends to prepend image IDs etc. to band names.
    # Band names will look like '..._lfcy_1985', etc.
    # Normalize to just prefix+year via regex.
    band_names = stacked.bandNames()

    def strip_prefix(bn):
        bn = ee.String(bn)
        # Replace everything up to the final occurrence of prefix with prefix itself.
        # The pattern '.*' + prefix eats all leading junk.
        return bn.replace('.*' + prefix, prefix)

    cleaned_names = band_names.map(strip_prefix)
    stacked = stacked.rename(cleaned_names)

    return stacked


def build_years_since_from_lfcy(
    lfcy_collection: ee.ImageCollection,
    base_no_change_year: int,
    sentinel_never_changed: int,
) -> ee.ImageCollection:
    """
    From a LastFastChangeYear collection, build a YearsSinceFastChange collection.

    For each year Y:

        YearsSinceFastChange = Y - LastFastChangeYear

    but where LastFastChangeYear == base_no_change_year (i.e., no fast change
    has ever happened up to that year), we use sentinel_never_changed instead.

    This preserves information about "never changed in entire record".
    """

    def per_year(img):
        """Compute YearsSinceFastChange for one year image."""
        img = ee.Image(img)
        year = ee.Number(img.get("year"))
        last_year = img.select("LastFastChangeYear")

        # Raw years since last change: Y - LastFastChangeYear
        ys_raw = ee.Image.constant(year).subtract(last_year)

        # Replace values that are still at base_no_change_year with sentinel
        ys = ys_raw.where(
            last_year.eq(base_no_change_year),
            sentinel_never_changed
        ).rename("YearsSinceFastChange")

        return ys.copyProperties(
            img,
            ["year", "system:time_start"]
        )

    return lfcy_collection.map(per_year)


# ---------------------------------------------------------------------------
# Optional: manual export helper for a COG
# (commented out because utils.gee.export_img_to_gcs is used instead)
# ---------------------------------------------------------------------------

# def export_multiband_cog_to_gcs(
#     img: ee.Image,
#     aoi: ee.Geometry,
#     bucket: str,
#     prefix: str,
#     scale: int = EXPORT_SCALE,
# ):
#     """
#     Export a multi-band image as a single COG GeoTIFF to GCS.
#
#     This is a lower-level version using ee.batch.Export.image.toCloudStorage.
#     In practice, we rely on export_img_to_gcs() from utils.gee instead.
#     """
#     patch = img.clip(aoi)
#
#     task = ee.batch.Export.image.toCloudStorage(
#         image=patch,
#         description=prefix,
#         bucket=bucket,
#         fileNamePrefix=prefix,
#         region=aoi,
#         scale=scale,
#         maxPixels=EXPORT_MAX_PIXELS,
#         fileFormat="GeoTIFF",
#         formatOptions={
#             "cloudOptimized": True
#         }
#     )
#     task.start()
#     print(f"Started COG export task: {prefix}, task_id={task.id}")


def debug_yearly_counts(
    aoi: ee.Geometry,
    lfcy_collection: ee.ImageCollection,
    fast_codes: list[int],
):
    """
    Print per-year diagnostics for LCMS Change and the cumulative state.

    For each year, prints:
      - histogram of Change codes
      - number of pixels with fast change that year
      - number of pixels whose LastFastChangeYear has been updated from
        BASE_NO_CHANGE_YEAR (i.e., have *ever* had a fast change up to that year)

    This is all client-side .getInfo() stuff, so use only on small AOIs
    or when you enjoy waiting.
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
      2. Build LastFastChangeYear ImageCollection + NPA mask
      3. (Optional) print debug stats
      4. Stack LastFastChangeYear into multiband image
      5. Derive YearsSinceFastChange per year
      6. Stack YearsSinceFastChange into multiband image
      7. Mask NPAs, cast to int16
      8. Export both stacks to GCS as COG-ready GeoTIFFs (via export_img_to_gcs)
    """
    # Initialize Earth Engine authentication/session
    ee_init()

    # AOI geometry (state boundary by name). For custom AOI, swap this out.
    aoi = get_state_geometry(STATE_NAME)
    # if USE_RECT_AOI:
    #     aoi = ee.Geometry.Rectangle(RECT_AOI)

    # 1) Build cumulative LastFastChangeYear collection + NPA mask
    lfcy_collection, npa_union = build_last_fast_change_year_collection(
        start_year=START_YEAR,
        end_year=END_YEAR,
        fast_codes=FAST_CODES,
    )

    # Optional: debug yearly stats (for sanity checks on small AOIs)
    if DEBUG:
        debug_yearly_counts(aoi, lfcy_collection, FAST_CODES)

    # 2) Stack LastFastChangeYear into a multiband image
    lfcy_stacked = stack_years_to_multiband(
        lfcy_collection,
        band_name="LastFastChangeYear",
        prefix="lfcy_",
    )

    # 3) Build YearsSinceFastChange collection from lfcy_collection
    ys_collection = build_years_since_from_lfcy(
        lfcy_collection=lfcy_collection,
        base_no_change_year=BASE_NO_CHANGE_YEAR,
        sentinel_never_changed=SENTINEL_NEVER_CHANGED,
    )

    # 4) Stack YearsSinceFastChange into a multiband image
    ysfc_stacked = stack_years_to_multiband(
        ys_collection,
        band_name="YearsSinceFastChange",
        prefix="ysfc_",
    )

    # 5) Apply union NPA mask and cast to int16 (same mask for both stacks)
    # npa_union == 1 where pixel is NPA in any year; we invert to mask them out.
    lfcy_stacked = lfcy_stacked.updateMask(npa_union.Not()).toInt16()
    ysfc_stacked = ysfc_stacked.updateMask(npa_union.Not()).toInt16()

    # 6) Export YearsSinceFastChange stack (primary for downstream modeling)
    export_img_to_gcs(
        img=ysfc_stacked,
        aoi=aoi,
        bucket=GCS_BUCKET,
        base_name=OUTPUT_PREFIX_YSFC,
        gcs_dir=GCS_DIR,
        # Do not pass crs/crsTransform here if export_img_to_gcs
        # already sets them for you.
    )

    # Optional: also export LastFastChangeYear stack
    export_img_to_gcs(
        img=lfcy_stacked,
        aoi=aoi,
        bucket=GCS_BUCKET,
        base_name=OUTPUT_PREFIX_LFCY,
        gcs_dir=GCS_DIR,
        # Do not pass crs/crsTransform here if export_img_to_gcs
        # already sets them for you.
    )


if __name__ == "__main__":
    main()
