import os
import json
import uuid
import pytest
import ee

from utils.gee import (
    ee_init,
    get_state_geometry,
    load_geojson_geometry,
    discover_years,
    export_image_drive,
    export_image_gcs,
)

# ---------- Constants ----------
# NLCD TCC 2023 release (stable and small footprint per image)
NLCD_TCC_COLL = "USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5"

# ---------- Pytest markers ----------
# We’ll use "integration" for optional export tests
pytestmark = pytest.mark.usefixtures("ee_ready")


# ---------- Fixtures ----------
@pytest.fixture(scope="session")
def ee_ready():
    """
    Ensure EE is initialized and credentials are available.
    If not authenticated, skip the entire test session cleanly.
    """
    try:
        # Try to initialize with default project/context
        ee_init()
        # A lightweight call to confirm the token works
        _ = ee.ImageCollection(NLCD_TCC_COLL).size().getInfo()
    except Exception as e:
        pytest.skip(f"Earth Engine not authenticated or unreachable: {e}")


@pytest.fixture(scope="session")
def small_region():
    """
    A tiny AOI inside Virginia to keep exports and queries minimal.
    """
    va = get_state_geometry("Virginia")
    # 5 km buffer around centroid (units are meters in default spherical metric)
    # Using a small buffer keeps export thumbnails tiny.
    return va.centroid().buffer(5000)


@pytest.fixture
def tmp_geojson(tmp_path):
    """
    Create a minimal GeoJSON polygon file to test load_geojson_geometry.
    """
    gj = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-79.5, 37.5],
                [-79.5, 37.6],
                [-79.4, 37.6],
                [-79.4, 37.5],
                [-79.5, 37.5],
            ]]
        }
    }
    p = tmp_path / "aoi.geojson"
    p.write_text(json.dumps(gj))
    return str(p)


# ---------- Unit-like tests (fast) ----------
def test_ee_init_no_raise():
    # Should not raise
    ee_init()


def test_get_state_geometry_virginia():
    geom = get_state_geometry("Virginia")
    assert isinstance(geom, ee.geometry.Geometry), "Expected an ee.Geometry"
    # Sanity: Virginia bbox should be wide enough
    bounds = ee.Feature(geom.bounds()).geometry().coordinates().getInfo()
    assert bounds, "Bounds should be available"


def test_load_geojson_geometry(tmp_geojson):
    geom = load_geojson_geometry(tmp_geojson)
    assert isinstance(geom, ee.geometry.Geometry), "Expected an ee.Geometry"


def test_discover_years_on_nlcd_tcc():
    coll = ee.ImageCollection(NLCD_TCC_COLL)
    years = discover_years(coll, prop="year")
    assert isinstance(years, list) and all(isinstance(y, int) for y in years)
    # Expect at least one of these common years to exist
    assert any(y in years for y in (2019, 2021, 2023)), f"Unexpected year set: {years}"


# ---------- Optional integration tests (disabled by default) ----------
@pytest.mark.integration
def test_export_image_gcs_integration(small_region):
    """
    Runs only if EE_TEST_GCS=1 and EE_GCS_BUCKET are set.
    Starts a small export and cancels it immediately to avoid cost.
    """
    if os.getenv("EE_TEST_GCS") != "1":
        pytest.skip("Set EE_TEST_GCS=1 and EE_GCS_BUCKET to enable this test.")

    bucket = os.getenv("EE_GCS_BUCKET")
    if not bucket:
        pytest.skip("EE_GCS_BUCKET not set.")

    # Pick a tiny NLCD TCC image (first available year) and clip to small region
    coll = ee.ImageCollection(NLCD_TCC_COLL).filter(ee.Filter.eq("study_area", "CONUS"))
    year_list = discover_years(coll, "year")
    assert year_list, "No years discovered for NLCD TCC."
    y = year_list[-1]
    img = coll.filter(ee.Filter.eq("year", int(y))).first().select("Science_Percent_Tree_Canopy_Cover").clip(small_region)

    desc = f"unittest_nlcd_tcc_{y}_{uuid.uuid4().hex[:8]}"
    task = export_image_gcs(
        image=img,
        desc=desc,
        region=small_region,
        bucket=bucket,
        prefix="unittest",
        scale=30,
        crs="EPSG:5070",
    )
    # Ensure a task id was produced, then cancel to avoid running the full export
    assert task and task.id, "Expected a valid EE task"
    try:
        task_status = task.status()
        assert task_status, "Task status should be retrievable"
    finally:
        # Cancel to avoid quota/time usage
        task.cancel()


@pytest.mark.integration
def test_export_image_drive_integration(small_region):
    """
    Runs only if EE_TEST_DRIVE=1 is set.
    Starts a small export and cancels it immediately to avoid cost.
    """
    if os.getenv("EE_TEST_DRIVE") != "1":
        pytest.skip("Set EE_TEST_DRIVE=1 to enable this test.")

    coll = ee.ImageCollection(NLCD_TCC_COLL).filter(ee.Filter.eq("study_area", "CONUS"))
    years = discover_years(coll, "year")
    assert years, "No years discovered for NLCD TCC."
    y = years[0]
    img = coll.filter(ee.Filter.eq("year", int(y))).first().select("Science_Percent_Tree_Canopy_Cover").clip(small_region)

    desc = f"unittest_drive_nlcd_tcc_{y}_{uuid.uuid4().hex[:8]}"
    task = export_image_drive(
        image=img,
        desc=desc,
        region=small_region,
        folder="nlcd_unittest",
        scale=30,
        crs="EPSG:5070",
    )
    assert task and task.id
    try:
        _ = task.status()
    finally:
        task.cancel()
