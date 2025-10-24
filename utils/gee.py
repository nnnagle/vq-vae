import json
import ee
from utils.log import log, fail

def get_state_geometry(state_name: str) -> ee.Geometry:
    """Fetch US state polygon from TIGER dataset."""
    states = ee.FeatureCollection("TIGER/2018/States")
    fc = states.filter(ee.Filter.eq("NAME", state_name))
    feature = ee.Feature(fc.first())
    geom = ee.Geometry(feature.geometry())
    log("Loaded TIGER geometry for state: %s", state_name)
    return geom

def load_geojson_geometry(path: str) -> ee.Geometry:
    """Load AOI geometry from GeoJSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return ee.Geometry(data)

def clip_to_region(image: ee.Image, region: ee.Geometry) -> ee.Image:
    """Clip a GEE image to a region."""
    return image.clip(region)

