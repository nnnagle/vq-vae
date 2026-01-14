"""
build_zarr.py - v2.0

Build a hierarchical spatiotemporal feature cube in Zarr format from
large raster inputs, as specified by a YAML configuration file.

Design principles
-----------------
- Hierarchical structure: aoi, strata, annual/, irregular/, static/
- Within each temporal category, groups contain data/quality/mask subsections
- Each band stored as individual named variable for flexible access
- Semantic types (continuous, categorical, mask) control dtype and statistics
- Annual data: dense time axis, clipped/padded to match global time window
- Irregular data: sparse snapshots with year coordinate mapping
- Static data: no time dimension
- Statistics computed per-variable and embedded in zarr attrs + JSON sidecar

Usage:
  python build_zarr.py config.yaml [--out path/to/output.zarr] [--ncore 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

import numpy as np
import xarray as xr
import zarr
import yaml
from dataclasses import dataclass

# Import rioxarray to register .rio accessor on xarray objects
import rioxarray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    log.warning("tqdm not available, progress bars disabled")

# Suppress common warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SpatialConfig:
    """Spatial reference configuration from YAML."""
    crs_wkt: str
    resolution: float
    transform: List[float]
    bounds: Dict[str, float]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Calculate (height, width) from bounds and resolution."""
        width = int((self.bounds['xmax'] - self.bounds['xmin']) / self.resolution)
        height = int((self.bounds['ymax'] - self.bounds['ymin']) / abs(self.transform[4]))
        return (height, width)


@dataclass 
class BandSpec:
    """Specification for a single band/variable."""
    id: str
    source_band: int
    fill_value: Optional[Dict[str, Any]] = None
    path: Optional[str] = None  # For per-band file paths


@dataclass
class GroupSpec:
    """Specification for a data group (e.g., ls8day, naip)."""
    name: str
    category: str  # 'annual', 'irregular', 'static'
    subsection: str  # 'data', 'quality', 'mask'
    semantic_type: str  # 'continuous', 'categorical', 'mask'
    path: Optional[str] = None  # Group-level path pattern
    bands: List[BandSpec] = None
    years: Optional[List[int]] = None  # For annual/irregular
    year_range: Optional[Tuple[int, int]] = None  # Alternative to explicit years
    fill_value: Optional[Dict[str, Any]] = None  # Group-level default fill value


# =============================================================================
# Configuration Parsing
# =============================================================================

def load_config(path: str | Path) -> dict:
    """Load and parse YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    
    log.info(f"Loaded configuration from {p}")
    return cfg


def parse_spatial_config(cfg: dict) -> SpatialConfig:
    """Extract spatial configuration from YAML."""
    spatial = cfg['dataset']['spatial']
    
    return SpatialConfig(
        crs_wkt=spatial['crs']['wkt'],
        resolution=spatial['resolution'],
        transform=spatial['transform'],
        bounds=spatial['bounds']
    )


def get_global_time_window(cfg: dict) -> Tuple[int, int]:
    """Get the global time window for annual data."""
    time_cfg = cfg['dataset']['time']['continuous']
    return (time_cfg['start'], time_cfg['end'])


def get_dtype_for_semantic_type(semantic_type: str, dtype_cfg: dict) -> np.dtype:
    """
    Map semantic type to numpy dtype based on config.
    
    Args:
        semantic_type: 'continuous', 'categorical', or 'mask'
        dtype_cfg: The dtype configuration dict (cfg['dataset']['dtype'])
    
    Returns:
        numpy dtype object
    """
    # Handle both full config and dtype-only config
    if 'dataset' in dtype_cfg:
        # Full config was passed
        dtype_map = {
            'continuous': dtype_cfg['dataset']['dtype']['continuous'],
            'categorical': dtype_cfg['dataset']['dtype']['categorical'],
            'mask': dtype_cfg['dataset']['dtype']['mask'],
        }
    else:
        # Just dtype config was passed
        dtype_map = {
            'continuous': dtype_cfg.get('continuous', 'float16'),
            'categorical': dtype_cfg.get('categorical', 'int16'),
            'mask': dtype_cfg.get('mask', 'uint8'),
        }
    
    dtype_str = dtype_map.get(semantic_type, 'float16')
    return np.dtype(dtype_str)


def parse_band_list(band_cfg: Any) -> List[BandSpec]:
    """
    Parse band configuration.
    
    Expected format:
    {
        'semantic_type': 'continuous',  # optional, used elsewhere
        'bands': [
            {'id': 'var1', 'source_band': 1, ...},
            {'id': 'var2', 'source_band': 2, ...}
        ]
    }
    
    Or legacy format (list directly):
    [
        {'id': 'var1', 'source_band': 1, ...},
        ...
    ]
    """
    if isinstance(band_cfg, list):
        # Legacy: direct list of bands
        return [BandSpec(id=b['id'], 
                        source_band=b.get('source_band'),
                        fill_value=b.get('fill_value'),
                        path=b.get('path')) 
                for b in band_cfg]
    elif isinstance(band_cfg, dict) and 'bands' in band_cfg:
        # New explicit format: dict with 'bands' key
        return [BandSpec(id=b['id'], 
                        source_band=b.get('source_band'),
                        fill_value=b.get('fill_value'),
                        path=b.get('path')) 
                for b in band_cfg['bands']]
    else:
        raise ValueError(
            f"Band configuration must be a list or dict with 'bands' key. "
            f"Got: {type(band_cfg).__name__}"
        )


def get_semantic_type(section_cfg: Any, subsection: str) -> str:
    """Determine semantic type from configuration."""
    # Default semantic types by subsection
    defaults = {
        'data': 'continuous',
        'quality': 'continuous', 
        'mask': 'mask'
    }
    
    if isinstance(section_cfg, dict) and 'semantic_type' in section_cfg:
        return section_cfg['semantic_type']
    
    return defaults.get(subsection, 'continuous')


def parse_group_specs(cfg: dict) -> List[GroupSpec]:
    """Parse all group specifications from config."""
    specs = []
    
    # Parse annual groups
    for group in cfg.get('annual', []):
        group_name = group['group']
        group_path = group.get('path')
        group_fill_value = group.get('fill_value')  # Get group-level fill_value
        
        # Parse years
        years = None
        year_range = None
        if 'years' in group:
            year_cfg = group['years']
            if isinstance(year_cfg, dict):
                year_range = (year_cfg['start'], year_cfg['end'])
            else:
                years = year_cfg
        
        # Parse each subsection (data, quality, mask)
        for subsection in ['data', 'quality', 'mask']:
            if subsection not in group:
                continue
            
            section_cfg = group[subsection]
            semantic_type = get_semantic_type(section_cfg, subsection)
            bands = parse_band_list(section_cfg)
            
            specs.append(GroupSpec(
                name=group_name,
                category='annual',
                subsection=subsection,
                semantic_type=semantic_type,
                path=group_path,
                bands=bands,
                years=years,
                year_range=year_range,
                fill_value=group_fill_value  # Pass group-level fill_value
            ))
    
    # Parse irregular groups
    for group in cfg.get('irregular', []):
        group_name = group['group']
        group_path = group.get('path')
        years = group.get('years', [])
        group_fill_value = group.get('fill_value')  # Get group-level fill_value
        
        for subsection in ['data', 'quality', 'mask']:
            if subsection not in group:
                continue
            
            section_cfg = group[subsection]
            semantic_type = get_semantic_type(section_cfg, subsection)
            bands = parse_band_list(section_cfg)
            
            specs.append(GroupSpec(
                name=group_name,
                category='irregular',
                subsection=subsection,
                semantic_type=semantic_type,
                path=group_path,
                bands=bands,
                years=years,
                fill_value=group_fill_value  # Pass group-level fill_value
            ))
    
    # Parse static groups
    for group in cfg.get('static', []):
        group_name = group['group']
        group_path = group.get('path')
        group_fill_value = group.get('fill_value')  # Get group-level fill_value
        
        for subsection in ['data', 'quality', 'mask']:
            if subsection not in group:
                continue
            
            section_cfg = group[subsection]
            semantic_type = get_semantic_type(section_cfg, subsection)
            bands = parse_band_list(section_cfg)
            
            specs.append(GroupSpec(
                name=group_name,
                category='static',
                subsection=subsection,
                semantic_type=semantic_type,
                path=group_path,
                bands=bands,
                fill_value=group_fill_value  # Pass group-level fill_value
            ))
    
    log.info(f"Parsed {len(specs)} group specifications")
    return specs


# =============================================================================
# File Resolution
# =============================================================================

def detect_file_pattern(group_spec: GroupSpec) -> str:
    """
    Detect file pattern type:
    - 'per_year': path contains {yyyy} or {year}
    - 'multiband': single file with multiple bands
    - 'per_band': separate file per band
    """
    if group_spec.path:
        path = group_spec.path
        if '{yyyy}' in path or '{year}' in path:
            return 'per_year'
        else:
            return 'multiband'
    elif all(b.path for b in group_spec.bands):
        return 'per_band'
    else:
        raise ValueError(f"Cannot determine file pattern for group {group_spec.name}")


def resolve_file_path(pattern: str, year: Optional[int] = None) -> Path:
    """Resolve file path pattern, optionally substituting year."""
    if year is not None:
        pattern = pattern.replace('{yyyy}', str(year))
        pattern = pattern.replace('{year}', str(year))
    
    path = Path(pattern)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return path


def get_year_list(group_spec: GroupSpec, global_window: Tuple[int, int]) -> List[int]:
    """
    Get list of years for a group.
    
    For annual: intersection of group years and global window
    For irregular: explicit year list
    For static: empty list
    """
    if group_spec.category == 'static':
        return []
    
    if group_spec.years:
        # Explicit year list (irregular)
        return sorted(group_spec.years)
    
    elif group_spec.year_range:
        # Year range (annual)
        start, end = group_spec.year_range
        group_years = list(range(start, end + 1))
        
        if group_spec.category == 'annual':
            # Clip to global window
            global_start, global_end = global_window
            return [y for y in group_years if global_start <= y <= global_end]
        else:
            return group_years
    
    else:
        # Use global window for annual
        if group_spec.category == 'annual':
            global_start, global_end = global_window
            return list(range(global_start, global_end + 1))
        else:
            raise ValueError(f"No year information for group {group_spec.name}")


# =============================================================================
# Raster I/O with rioxarray
# =============================================================================

def open_raster_band(
    file_path: Path,
    band_index: Optional[int] = None,
    chunks: Optional[Dict[str, int]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> xr.DataArray:
    """
    Open a single band from a raster file using rioxarray.
    
    Args:
        file_path: Path to raster file
        band_index: 1-based band index (None for single-band files)
        chunks: Chunk specification for dask (only spatial dims used)
        bounds: Optional (minx, miny, maxx, maxy) to read only a spatial window
    
    Returns:
        DataArray with spatial dimensions (y, x)
    """
    # Filter chunks to only spatial dimensions that exist in rasters
    spatial_chunks = None
    if chunks:
        spatial_chunks = {k: v for k, v in chunks.items() if k in ('x', 'y', 'band')}
    
    # Open with rioxarray
    # If bounds provided, clip during read (much faster than loading then clipping)
    if bounds:
        minx, miny, maxx, maxy = bounds
        da = rioxarray.open_rasterio(
            file_path, 
            chunks=spatial_chunks,
            masked=True
        )
        # Clip immediately while reading
        da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    else:
        da = rioxarray.open_rasterio(file_path, chunks=spatial_chunks)
    
    # Select band if multi-band
    if band_index is not None:
        if 'band' in da.dims:
            da = da.sel(band=band_index)
    else:
        # Single band - drop band dimension
        if 'band' in da.dims and len(da.band) == 1:
            da = da.isel(band=0, drop=True)
    
    return da


def align_to_template(
    da: xr.DataArray,
    template: xr.DataArray,
    method: str = 'nearest'
) -> xr.DataArray:
    """
    Align a DataArray to match the template's spatial grid.
    
    Uses spatial windowing for same-CRS clipping, or reprojection if CRS differs.
    Preserves dask arrays (doesn't force computation).
    """
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    
    # Map string to Resampling enum
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
        'mode': Resampling.mode,
    }
    
    resampling = resampling_map.get(method, Resampling.nearest)
    
    # Check if alignment needed
    crs_match = da.rio.crs == template.rio.crs
    transform_match = da.rio.transform() == template.rio.transform()
    shape_match = da.rio.shape == template.rio.shape
    
    log.debug(f"        Alignment check: CRS={crs_match}, Transform={transform_match}, Shape={shape_match}")
    
    if crs_match and transform_match and shape_match:
        log.debug(f"        ✓ Already aligned, skipping")
        return da
    
    # If same CRS but different bounds/shape, use clip instead of reproject
    if crs_match:
        log.debug(f"        Same CRS, using clip instead of reproject")
        try:
            # Get template bounds
            minx, miny, maxx, maxy = template.rio.bounds()
            
            # Clip to bounds (lazy operation)
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            
            log.debug(f"        ✓ Clipped from {da.rio.shape} to {clipped.rio.shape}")
            return clipped
            
        except Exception as e:
            log.warning(f"        Clip failed ({e}), falling back to reproject")
            # Fall through to reproject
    
    # Different CRS or clip failed - use reproject
    log.debug(f"Reprojecting (method={method})")
    log.debug(f"  Source CRS: {da.rio.crs}")
    log.debug(f"  Target CRS: {template.rio.crs}")
    log.debug(f"  Source shape: {da.rio.shape}")
    log.debug(f"  Target shape: {template.rio.shape}")
    
    aligned = da.rio.reproject_match(template, resampling=resampling)
    
    return aligned


def create_spatial_template(spatial_cfg: SpatialConfig, chunks: Dict[str, int]) -> xr.DataArray:
    """Create a spatial template DataArray from config."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    
    height, width = spatial_cfg.shape
    
    # Create coordinate arrays
    transform = Affine(*spatial_cfg.transform)
    x_coords = np.arange(width) * transform.a + transform.c + transform.a / 2
    y_coords = np.arange(height) * transform.e + transform.f + transform.e / 2
    
    # Create empty template
    data = np.zeros((height, width), dtype=np.float32)
    
    template = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={
            'y': y_coords,
            'x': x_coords
        }
    )
    
    # Add spatial reference info
    template.rio.write_crs(spatial_cfg.crs_wkt, inplace=True)
    template.rio.write_transform(transform, inplace=True)
    
    if chunks:
        template = template.chunk(chunks)
    
    return template


# =============================================================================
# Data Loading
# =============================================================================

def load_static_band(
    band_spec: BandSpec,
    group_spec: GroupSpec,
    template: xr.DataArray,
    target_dtype: np.dtype,
    chunks: Dict[str, int]
) -> xr.DataArray:
    """Load a single static (non-temporal) band."""
    
    # Determine file path
    if band_spec.path:
        file_path = resolve_file_path(band_spec.path)
    elif group_spec.path:
        file_path = resolve_file_path(group_spec.path)
    else:
        raise ValueError(f"No path specified for band {band_spec.id}")
    
    # Open raster
    da = open_raster_band(file_path, band_spec.source_band, chunks)
    
    # Align to template
    da = align_to_template(da, template)
    
    # Handle nodata/fill values
    da = handle_fill_values(da, band_spec, group_spec)
    
    # Convert dtype
    da = da.astype(target_dtype)
    
    # Set name
    da.name = band_spec.id
    
    return da


def load_annual_band(
    band_spec: BandSpec,
    group_spec: GroupSpec,
    template: xr.DataArray,
    target_dtype: np.dtype,
    global_window: Tuple[int, int],
    chunks: Dict[str, int]
) -> xr.DataArray:
    """Load a single annual (time-varying) band."""
    
    # Get years for this group
    group_years = get_year_list(group_spec, global_window)
    
    # Determine file pattern
    pattern_type = detect_file_pattern(group_spec)
    
    if pattern_type == 'per_year':
        # Load from per-year files
        da = load_per_year_band(band_spec, group_spec, group_years, template, chunks)
    
    elif pattern_type == 'multiband':
        # Load from single multiband file
        da = load_multiband_annual(band_spec, group_spec, group_years, template, chunks)
    
    elif pattern_type == 'per_band':
        # Load from per-band VRT/file
        log.debug(f"      Loading per-band annual for {band_spec.id}")
        log.debug(f"      Group year_range: {group_spec.year_range}")
        log.debug(f"      Requested years: {group_years}")
        da = load_per_band_annual(band_spec, group_spec, group_years, template, chunks)
        log.debug(f"      Loaded shape: {da.shape}, time coords: {da.time.values[:5]}...{da.time.values[-3:]}")
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Ensure time dimension
    if 'time' not in da.dims:
        raise ValueError(f"Annual band {band_spec.id} missing time dimension")
    
    # Clip/pad to global window
    global_start, global_end = global_window
    global_years = list(range(global_start, global_end + 1))
    
    log.debug(f"      Before temporal align: min={da.min().values:.3f}, max={da.max().values:.3f}")
    da = align_temporal_to_window(da, global_years, group_spec.semantic_type)
    log.debug(f"      After temporal align: min={da.min().values:.3f}, max={da.max().values:.3f}")
    
    # Handle nodata/fill values
    da = handle_fill_values(da, band_spec, group_spec)
    log.debug(f"      After fill handling: min={da.min().values:.3f}, max={da.max().values:.3f}")
    
    # Convert dtype
    da = da.astype(target_dtype)
    log.debug(f"      After dtype convert: min={da.min().values:.3f}, max={da.max().values:.3f}")
    
    # Set name
    da.name = band_spec.id
    
    return da


def load_per_year_band(
    band_spec: BandSpec,
    group_spec: GroupSpec,
    years: List[int],
    template: xr.DataArray,
    chunks: Dict[str, int]
) -> xr.DataArray:
    """Load band from per-year files (e.g., naip_{yyyy}.vrt)."""
    
    log.debug(f"      Loading {len(years)} years for {band_spec.id}")
    arrays = []
    
    for year in years:
        log.debug(f"        Opening year {year}...")
        file_path = resolve_file_path(group_spec.path, year)
        da = open_raster_band(file_path, band_spec.source_band, chunks)
        log.debug(f"        Aligning year {year}...")
        da = align_to_template(da, template)
        arrays.append(da)
        log.debug(f"        ✓ Year {year} ready")
    
    # Stack along time
    log.debug(f"      Concatenating {len(arrays)} years...")
    combined = xr.concat(arrays, dim='time')
    log.debug(f"      Assigning coordinates...")
    combined = combined.assign_coords(time=years)
    log.debug(f"      ✓ Concatenation complete")
    
    return combined


def load_multiband_annual(
    band_spec: BandSpec,
    group_spec: GroupSpec,
    years: List[int],
    template: xr.DataArray,
    chunks: Dict[str, int]
) -> xr.DataArray:
    """
    Load from single multiband file where bands represent years.
    
    Assumes band ordering matches year ordering.
    """
    file_path = resolve_file_path(group_spec.path)
    
    # Filter chunks to only spatial dimensions
    spatial_chunks = None
    if chunks:
        spatial_chunks = {k: v for k, v in chunks.items() if k in ('x', 'y', 'band')}
    
    # Open all bands
    da_full = rioxarray.open_rasterio(file_path, chunks=spatial_chunks)
    
    # Map years to band indices
    # This assumes bands are ordered by year
    if len(da_full.band) < len(years):
        raise ValueError(
            f"File {file_path} has {len(da_full.band)} bands but "
            f"config specifies {len(years)} years"
        )
    
    # Select bands corresponding to years
    # Assuming 1-indexed bands and sequential year mapping
    year_start = years[0]
    band_start = band_spec.source_band if band_spec.source_band else 1
    
    band_indices = [band_start + (year - year_start) for year in years]
    da = da_full.sel(band=band_indices)
    
    # Rename band dimension to time
    da = da.rename({'band': 'time'})
    da = da.assign_coords(time=years)
    
    # Align to template
    da = align_to_template(da, template)
    
    return da


def load_per_band_annual(
    band_spec: BandSpec,
    group_spec: GroupSpec,
    years: List[int],
    template: xr.DataArray,
    chunks: Dict[str, int]
) -> xr.DataArray:
    """Load from per-band file (VRT spanning years)."""
    
    file_path = resolve_file_path(band_spec.path)
    log.debug(f"      Opening file: {file_path}")
    
    # Filter chunks to only spatial dimensions
    spatial_chunks = None
    if chunks:
        spatial_chunks = {k: v for k, v in chunks.items() if k in ('x', 'y', 'band')}
    
    # Open multiband VRT
    da = rioxarray.open_rasterio(file_path, chunks=spatial_chunks)
    log.debug(f"      After open: shape={da.shape}, min={da.min().values:.3f}, max={da.max().values:.3f}")
    
    # Rename band to time
    if 'band' in da.dims:
        da = da.rename({'band': 'time'})
        
        # Assign year coordinates
        # If band count doesn't match, infer actual years from group spec
        if len(da.time) != len(years):
            log.warning(
                f"Band count ({len(da.time)}) doesn't match requested years ({len(years)}) "
                f"for {band_spec.id} - using group's full year range"
            )
            # Try to get actual year range from group spec
            if group_spec.year_range:
                actual_start, actual_end = group_spec.year_range
                actual_years = list(range(actual_start, actual_end + 1))
                log.debug(f"      year_range: {actual_start}-{actual_end} = {len(actual_years)} years")
                log.debug(f"      band count: {len(da.time)}")
                if len(actual_years) == len(da.time):
                    log.debug(f"      ✓ Assigning actual years: {actual_start}-{actual_end}")
                    da = da.assign_coords(time=actual_years)
                else:
                    log.error(
                        f"      ✗ Group year_range {actual_start}-{actual_end} ({len(actual_years)}) "
                        f"still doesn't match band count ({len(da.time)})"
                    )
                    log.error(f"      Falling back to band indices [0..{len(da.time)-1}]")
                    da = da.assign_coords(time=list(range(len(da.time))))
            else:
                log.warning(f"      No year_range in group spec, using band indices")
                da = da.assign_coords(time=list(range(len(da.time))))
        else:
            # Counts match - use requested years
            log.debug(f"      Counts match, assigning requested years")
            da = da.assign_coords(time=years)
    
    log.debug(f"      Time coords assigned: {da.time.values[:3]}...{da.time.values[-3:]}")
    log.debug(f"      After time assign: min={da.min().values:.3f}, max={da.max().values:.3f}")
    
    # Align to template
    da = align_to_template(da, template)
    log.debug(f"      After align: min={da.min().values:.3f}, max={da.max().values:.3f}")
    
    return da


def load_irregular_band(
    band_spec: BandSpec,
    group_spec: GroupSpec,
    template: xr.DataArray,
    target_dtype: np.dtype,
    chunks: Dict[str, int]
) -> xr.DataArray:
    """Load a single irregular (snapshot) band."""
    
    years = group_spec.years
    if not years:
        raise ValueError(f"Irregular group {group_spec.name} missing years list")
    
    # Determine file pattern
    pattern_type = detect_file_pattern(group_spec)
    
    if pattern_type == 'per_year':
        da = load_per_year_band(band_spec, group_spec, years, template, chunks)
    else:
        raise ValueError(f"Irregular data only supports per_year pattern, got {pattern_type}")
    
    # Rename time to snapshot (integer index)
    snapshot_indices = list(range(len(years)))
    da = da.rename({'time': 'snapshot'})
    da = da.assign_coords(snapshot=snapshot_indices)
    
    # Add year coordinate
    da = da.assign_coords(snapshot_year=('snapshot', years))
    
    # Handle nodata/fill values  
    da = handle_fill_values(da, band_spec, group_spec)
    
    # Convert dtype
    da = da.astype(target_dtype)
    
    # Set name
    da.name = band_spec.id
    
    return da


def handle_fill_values(
    da: xr.DataArray,
    band_spec: BandSpec,
    group_spec: GroupSpec
) -> xr.DataArray:
    """
    Handle fill/nodata values according to band spec and semantic type.
    
    Priority order:
    1. Band-specific fill_value (if specified)
    2. Group-level fill_value (if specified)
    3. Raster metadata nodata (if present)
    4. No replacement
    
    Stores the target fill value in da.attrs['_FillValue'] for use when
    writing to Zarr.
    """
    
    # Check for explicit fill_value in band spec (highest priority)
    if band_spec.fill_value:
        source_val = band_spec.fill_value.get('source')
        target_val = band_spec.fill_value.get('target')
        
        if target_val == 'na' or target_val is None:
            # Replace with NaN
            da = da.where(da != source_val, np.nan)
            # Store NaN as the fill value for Zarr
            da.attrs['_FillValue'] = np.nan
        else:
            da = da.where(da != source_val, target_val)
            # Store the target value as fill value for Zarr
            da.attrs['_FillValue'] = target_val
        
        return da
    
    # Check for group-level fill_value (second priority)
    if group_spec.fill_value:
        source_val = group_spec.fill_value.get('source')
        target_val = group_spec.fill_value.get('target')
        
        if target_val == 'na' or target_val is None:
            # Replace with NaN
            da = da.where(da != source_val, np.nan)
            # Store NaN as the fill value for Zarr
            da.attrs['_FillValue'] = np.nan
        else:
            da = da.where(da != source_val, target_val)
            # Store the target value as fill value for Zarr
            da.attrs['_FillValue'] = target_val
        
        return da
    
    # Check for nodata in raster metadata (third priority)
    nodata = da.attrs.get('_FillValue') or da.attrs.get('nodata')
    
    if nodata is not None:
        if group_spec.semantic_type == 'continuous':
            # Replace with NaN
            da = da.where(da != nodata, np.nan)
            # Store NaN as the fill value
            da.attrs['_FillValue'] = np.nan
        elif group_spec.semantic_type in ('categorical', 'mask'):
            # Replace with 0 or False
            fill = 0 if group_spec.semantic_type == 'categorical' else False
            da = da.where(da != nodata, fill)
            # Store the fill value
            da.attrs['_FillValue'] = fill
    
    return da


def align_temporal_to_window(
    da: xr.DataArray,
    target_years: List[int],
    semantic_type: str
) -> xr.DataArray:
    """
    Align temporal data to target year window.
    
    - Clips if da extends beyond target
    - Pads if da doesn't cover full target
    """
    current_years = da.time.values.tolist()
    
    # Determine fill value for padding
    if semantic_type == 'continuous':
        fill_value = 0.0
    elif semantic_type == 'categorical':
        fill_value = -1
    elif semantic_type == 'mask':
        fill_value = False
    else:
        fill_value = 0
    
    # Reindex to target years
    da = da.reindex(time=target_years, fill_value=fill_value)
    
    return da


# =============================================================================
# Variable Construction
# =============================================================================

def build_variable_hierarchy(
    group_specs: List[GroupSpec],
    template: xr.DataArray,
    global_window: Tuple[int, int],
    dataset_cfg: dict,
    chunks: Dict[str, int]
) -> Dict[str, xr.DataArray]:
    """
    Build all variables according to hierarchical structure.
    
    Args:
        group_specs: List of group specifications
        template: Spatial template DataArray
        global_window: (start_year, end_year) tuple
        dataset_cfg: cfg['dataset'] dictionary
        chunks: Chunk specification
    
    Returns dict mapping full paths to DataArrays:
    {
        'annual/ls8day/data/NDVI_summer_p95': DataArray(...),
        'irregular/naip/data/NDVI': DataArray(...),
        'static/topo/data/elevation': DataArray(...),
    }
    """
    variables = {}
    
    # Extract dtype config
    dtype_cfg = dataset_cfg.get('dtype', {})
    
    # Group specs by category/group/subsection for progress reporting
    total_bands = sum(len(spec.bands) for spec in group_specs)
    
    log.info(f"Building {total_bands} variables from {len(group_specs)} group specs")
    log.info("")
    
    iterator = tqdm(group_specs, desc="Loading groups") if HAS_TQDM else group_specs
    
    band_counter = 0
    for spec in iterator:
        # Log group header
        log.info(f"{'─' * 80}")
        log.info(f"Loading: {spec.category}/{spec.name}/{spec.subsection}")
        log.info(f"  Type: {spec.semantic_type}")
        log.info(f"  Bands: {len(spec.bands)}")
        
        # Determine target dtype
        target_dtype = get_dtype_for_semantic_type(spec.semantic_type, dtype_cfg)
        log.info(f"  Target dtype: {target_dtype}")
        
        # Build path prefix
        path_prefix = f"{spec.category}/{spec.name}/{spec.subsection}"
        
        # Load each band
        for i, band in enumerate(spec.bands, 1):
            band_counter += 1
            var_path = f"{path_prefix}/{band.id}"
            
            import time
            start_time = time.time()
            log.info(f"  [{band_counter}/{total_bands}] {band.id}...")
            
            try:
                if spec.category == 'static':
                    da = load_static_band(band, spec, template, target_dtype, chunks)
                
                elif spec.category == 'annual':
                    da = load_annual_band(
                        band, spec, template, target_dtype, global_window, chunks
                    )
                
                elif spec.category == 'irregular':
                    da = load_irregular_band(band, spec, template, target_dtype, chunks)
                
                else:
                    raise ValueError(f"Unknown category: {spec.category}")
                
                # Attach metadata
                da.attrs['semantic_type'] = spec.semantic_type
                da.attrs['category'] = spec.category
                da.attrs['group'] = spec.name
                da.attrs['subsection'] = spec.subsection
                
                variables[var_path] = da
                
                # Log shape info
                shape_str = ' × '.join(str(s) for s in da.shape)
                elapsed = time.time() - start_time
                log.info(f"      ✓ Loaded: {shape_str} {da.dtype} ({elapsed:.1f}s)")
                
            except Exception as e:
                elapsed = time.time() - start_time
                log.error(f"      ✗ Failed to load {var_path} after {elapsed:.1f}s: {e}")
                raise
    
    log.info(f"{'─' * 80}")
    log.info(f"✓ Successfully loaded {len(variables)} variables")
    log.info("")
    return variables


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_variable_statistics(
    da: xr.DataArray,
    semantic_type: str,
    aoi_mask: Optional[xr.DataArray] = None
) -> Dict[str, Any]:
    """
    Compute statistics for a single variable using dask for memory efficiency.
    
    Args:
        da: Variable data (can be dask-backed)
        semantic_type: continuous, categorical, or mask
        aoi_mask: Optional boolean mask to exclude pixels
    
    Returns:
        Dictionary of statistics appropriate for semantic type
    """
    import dask.array as dask_array
    
    stats = {
        'semantic_type': semantic_type,
        'shape': list(da.shape),
        'dtype': str(da.dtype)
    }
    
    # Apply AOI mask if provided
    if aoi_mask is not None:
        # Broadcast mask to match da dimensions
        if 'time' in da.dims:
            mask_broadcast = aoi_mask.broadcast_like(da)
        elif 'snapshot' in da.dims:
            mask_broadcast = aoi_mask.broadcast_like(da)
        else:
            mask_broadcast = aoi_mask
        
        da_masked = da.where(mask_broadcast)
        
        # Check if any valid pixels remain
        valid_count = mask_broadcast.sum().compute().item()
        total_count = mask_broadcast.size
        log.debug(f"  Mask: {valid_count}/{total_count} pixels valid ({100*valid_count/total_count:.1f}%)")
        
        if valid_count == 0:
            log.warning(f"  ⚠️  All pixels masked out by AOI - statistics will be invalid")
    else:
        da_masked = da
    
    # Rechunk for efficient computation (avoid tiny chunks)
    if hasattr(da_masked.data, 'chunks'):
        # Rechunk to reasonable size for statistics
        if 'time' in da_masked.dims:
            da_masked = da_masked.chunk({'time': -1, 'y': 512, 'x': 512})
        elif 'snapshot' in da_masked.dims:
            da_masked = da_masked.chunk({'snapshot': -1, 'y': 512, 'x': 512})
        else:
            da_masked = da_masked.chunk({'y': 512, 'x': 512})
    
    if semantic_type == 'continuous':
        # Use dask/xarray methods that don't load into memory
        log.debug(f"Computing continuous stats for {da.name}")
        
        # Cast to float32 to avoid overflow in reductions (float16 overflows easily)
        da_stats = da_masked.astype('float32')
        
        # Check if data has any non-NaN values
        valid_data_count = (~np.isnan(da_stats)).sum().compute().item()
        if valid_data_count == 0:
            log.warning(f"  ⚠️  No valid data for {da.name} - returning zero stats")
            stats.update({
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'q02': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 0.0, 'q98': 0.0, 'sd': 0.0,
                'warning': 'no_valid_data'
            })
            return stats
        
        # Basic statistics using xarray (dask-aware)
        stats['mean'] = float(da_stats.mean().compute().values)
        stats['std'] = float(da_stats.std().compute().values)
        stats['min'] = float(da_stats.min().compute().values)
        stats['max'] = float(da_stats.max().compute().values)
        
        # Quantiles - need to be more careful
        # Use dask.array.percentile if available
        if hasattr(da_stats.data, 'compute'):
            # It's a dask array
            data_flat = da_stats.data.ravel()
            data_flat = data_flat[~dask_array.isnan(data_flat)]
            
            # Compute quantiles in one go
            quantiles_vals = dask_array.percentile(
                data_flat, 
                [2, 25, 50, 75, 98]
            ).compute()
            
            stats['q02'] = float(quantiles_vals[0])
            stats['q25'] = float(quantiles_vals[1])
            stats['q50'] = float(quantiles_vals[2])
            stats['q75'] = float(quantiles_vals[3])
            stats['q98'] = float(quantiles_vals[4])
        else:
            # Not dask, can use numpy
            data_flat = da_stats.values.ravel()
            data_flat = data_flat[~np.isnan(data_flat)]
            
            if len(data_flat) > 0:
                stats['q02'] = float(np.percentile(data_flat, 2))
                stats['q25'] = float(np.percentile(data_flat, 25))
                stats['q50'] = float(np.percentile(data_flat, 50))
                stats['q75'] = float(np.percentile(data_flat, 75))
                stats['q98'] = float(np.percentile(data_flat, 98))
        
        stats['sd'] = stats['std']  # Alias
    
    elif semantic_type == 'categorical':
        # Compute histogram - need to be careful with memory
        log.debug(f"Computing categorical stats for {da.name}")
        
        if hasattr(da_masked.data, 'compute'):
            # Dask array - compute in chunks and aggregate
            # For categorical data, we need the full histogram
            # Use a streaming approach
            from collections import Counter
            
            histogram = Counter()
            
            # Process chunk by chunk
            for chunk in da_masked.data.to_delayed().ravel():
                chunk_data = chunk.compute()
                chunk_data = chunk_data[~np.isnan(chunk_data)]
                chunk_data = chunk_data[chunk_data != -1]  # Filter fill value
                
                unique, counts = np.unique(chunk_data, return_counts=True)
                for val, count in zip(unique, counts):
                    histogram[int(val)] += int(count)
            
            stats['histogram'] = dict(histogram)
            stats['num_classes'] = len(histogram)
        else:
            # Not dask
            data_flat = da_masked.values.ravel()
            data_flat = data_flat[~np.isnan(data_flat)]
            
            unique, counts = np.unique(data_flat, return_counts=True)
            
            # Filter out fill value (-1)
            valid_mask = unique != -1
            unique = unique[valid_mask]
            counts = counts[valid_mask]
            
            stats['histogram'] = {
                int(val): int(count) for val, count in zip(unique, counts)
            }
            stats['num_classes'] = len(unique)
    
    elif semantic_type == 'mask':
        # Compute boolean statistics
        log.debug(f"Computing mask stats for {da.name}")
        
        # Debug: check data type and sample values
        log.debug(f"  Mask dtype: {da_masked.dtype}")
        sample_vals = da_masked.values.flat[:100] if hasattr(da_masked, 'values') else da_masked.data.compute().flat[:100]
        unique_sample = np.unique(sample_vals[~np.isnan(sample_vals)])
        log.debug(f"  Unique values in sample: {unique_sample}")
        
        # Use xarray methods (dask-aware)
        # For mask data stored as uint8 (0/1), check for non-zero
        if da_masked.dtype in (np.uint8, np.int8, np.int16, np.float16, np.float32):
            true_count = int((da_masked > 0).sum().compute().values)
            false_count = int((da_masked == 0).sum().compute().values)
        else:
            # For actual boolean dtype
            true_count = int((da_masked == True).sum().compute().values)
            false_count = int((da_masked == False).sum().compute().values)
        
        total = true_count + false_count
        
        stats['true_count'] = true_count
        stats['false_count'] = false_count
        
        if total > 0:
            stats['true_fraction'] = true_count / total
        else:
            stats['true_fraction'] = 0.0
    
    return stats


def compute_all_statistics(
    variables: Dict[str, xr.DataArray],
    aoi: Optional[xr.DataArray] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics for all variables.
    
    Note: For very large datasets, consider computing statistics from the
    written zarr store instead of from in-memory dask arrays.
    """
    
    stats_dict = {}
    
    log.info(f"Computing statistics for {len(variables)} variables")
    iterator = tqdm(variables.items(), desc="Computing stats") if HAS_TQDM else variables.items()
    
    for var_path, da in iterator:
        semantic_type = da.attrs.get('semantic_type', 'continuous')
        
        try:
            stats = compute_variable_statistics(da, semantic_type, aoi)
            stats_dict[var_path] = stats
        except Exception as e:
            log.warning(f"Failed to compute stats for {var_path}: {e}")
            stats_dict[var_path] = {
                'semantic_type': semantic_type,
                'error': str(e)
            }
    
    return stats_dict


def compute_statistics_from_zarr(
    zarr_path: Path,
    aoi: Optional[xr.DataArray] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics from written zarr store.
    
    This is safer for very large datasets as it reads data in chunks
    directly from disk rather than holding dask arrays in memory.
    
    Args:
        zarr_path: Path to zarr store
        aoi: Optional AOI mask for spatial filtering
    
    Returns:
        Dictionary mapping variable paths to statistics
    """
    import zarr
    
    stats_dict = {}
    root = zarr.open_group(str(zarr_path), mode='r')
    
    # Walk the zarr hierarchy
    variable_paths = []
    
    def walk_group(group, path=""):
        for key in group.keys():
            item = group[key]
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(item, zarr.Array):
                variable_paths.append(current_path)
            elif isinstance(item, zarr.Group):
                walk_group(item, current_path)
    
    walk_group(root)
    
    # Filter out aoi and strata (root level)
    variable_paths = [p for p in variable_paths if '/' in p]
    
    log.info(f"Computing statistics from zarr for {len(variable_paths)} variables")
    iterator = tqdm(variable_paths, desc="Computing stats from zarr") if HAS_TQDM else variable_paths
    
    for var_path in iterator:
        try:
            # Open as xarray
            ds = xr.open_zarr(zarr_path, consolidated=False)
            
            # Navigate to variable
            parts = var_path.split('/')
            da = ds
            for part in parts:
                da = da[part]
            
            # Get semantic type from attrs
            semantic_type = da.attrs.get('semantic_type', 'continuous')
            
            # Compute stats
            stats = compute_variable_statistics(da, semantic_type, aoi)
            stats_dict[var_path] = stats
            
        except Exception as e:
            log.warning(f"Failed to compute stats for {var_path}: {e}")
            stats_dict[var_path] = {'error': str(e)}
    
    return stats_dict


# =============================================================================
# Zarr Writing
# =============================================================================

# =============================================================================
# Zarr Writing
# =============================================================================

def sanitize_attrs(attrs: dict) -> dict:
    """
    Convert attrs to JSON-serializable types.
    
    Zarr requires all attrs to be JSON serializable, but xarray/rioxarray
    often includes numpy types that aren't.
    """
    import numpy as np
    
    sanitized = {}
    for key, val in attrs.items():
        # Skip non-serializable complex objects
        if key in ('crs', 'transform', 'res', 'bounds', 'grid_mapping'):
            continue
        
        # Convert numpy types to Python types
        if isinstance(val, (np.integer, np.floating)):
            sanitized[key] = val.item()
        elif isinstance(val, np.ndarray):
            sanitized[key] = val.tolist()
        elif isinstance(val, (str, int, float, bool, type(None))):
            sanitized[key] = val
        elif isinstance(val, (list, tuple)):
            # Recursively sanitize lists
            sanitized[key] = [
                v.item() if isinstance(v, (np.integer, np.floating)) else v
                for v in val
            ]
        elif isinstance(val, dict):
            # Recursively sanitize dicts
            sanitized[key] = sanitize_attrs(val)
        # Skip anything else that might not be serializable
    
    return sanitized


def write_zarr_hierarchy(
    zarr_path: Path,
    variables: Dict[str, xr.DataArray],
    aoi: Optional[xr.DataArray] = None,
    strata: Optional[xr.DataArray] = None,
    compressor_cfg: Optional[Dict] = None,
    chunks: Optional[Dict[str, int]] = None
) -> None:
    """
    Write variables to hierarchical zarr structure.
    
    Structure:
        zarr_path/
            aoi
            strata
            annual/
                group1/
                    data/
                        var1
                        var2
                    quality/
                        var3
            irregular/
                ...
            static/
                ...
    """
    import numcodecs
    
    # Create root - force Zarr v2 format for compatibility
    p = Path(zarr_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode='w', zarr_format=2)
    
    # Configure compressor
    if compressor_cfg:
        compressor = numcodecs.Blosc(
            cname=compressor_cfg.get('cname', 'lz4'),
            clevel=compressor_cfg.get('clevel', 3),
            shuffle=compressor_cfg.get('shuffle', 2)
        )
    else:
        compressor = None
    
    # Write AOI and strata at root level
    if aoi is not None:
        log.info("Writing aoi to zarr root")
        aoi_array = root.create_array(
            'aoi',
            shape=aoi.shape,
            chunks=tuple(chunks.get(d, s) for d, s in zip(aoi.dims, aoi.shape)),
            dtype=aoi.dtype,
            compressor=compressor,
            overwrite=True
        )
        aoi_array[:] = aoi.values
        aoi_array.attrs.update(sanitize_attrs(aoi.attrs))
    
    if strata is not None:
        log.info("Writing strata to zarr root")
        strata_array = root.create_array(
            'strata',
            shape=strata.shape,
            chunks=tuple(chunks.get(d, s) for d, s in zip(strata.dims, strata.shape)),
            dtype=strata.dtype,
            compressor=compressor,
            overwrite=True
        )
        strata_array[:] = strata.values
        strata_array.attrs.update(sanitize_attrs(strata.attrs))
    
    # Write hierarchical variables
    log.info(f"Writing {len(variables)} variables to zarr")
    
    iterator = tqdm(variables.items(), desc="Writing zarr") if HAS_TQDM else variables.items()
    
    for var_path, da in iterator:
        # Parse path: category/group/subsection/varname
        parts = var_path.split('/')
        
        if len(parts) != 4:
            log.warning(f"Unexpected path structure: {var_path}")
            continue
        
        category, group_name, subsection, var_name = parts
        
        # Create group hierarchy
        cat_group = root.require_group(category)
        grp_group = cat_group.require_group(group_name)
        sub_group = grp_group.require_group(subsection)
        
        # Store group-level attrs
        if 'semantic_type' in da.attrs:
            sub_group.attrs['semantic_type'] = da.attrs['semantic_type']
        
        # Determine chunks
        var_chunks = tuple(
            chunks.get(d, s) for d, s in zip(da.dims, da.shape)
        )
        
        # Get fill_value from attrs if present
        fill_value = da.attrs.get('_FillValue', None)
        
        # Create array with fill_value
        ds = sub_group.create_array(
            var_name,
            shape=da.shape,
            chunks=var_chunks,
            dtype=da.dtype,
            compressor=compressor,
            fill_value=fill_value,  # Pass fill_value to Zarr
            overwrite=True
        )
        
        # Write data (compute if dask)
        log.debug(f"Writing {var_path}")
        if hasattr(da.data, 'compute'):
            # Dask array - write chunk by chunk to avoid loading all to memory
            import dask.array as dask_array
            
            # Use to_zarr for efficient dask writing
            # But since we're using zarr directly, we need to write chunks manually
            log.debug(f"  Writing dask array with shape {da.shape}")
            
            # Store_chunk writes individual chunks without loading full array
            dask_array.store(da.data, ds, lock=False, compute=True)
        else:
            # Already in memory (small array)
            ds[:] = da.values
        
        # Write attributes (sanitize for JSON serialization)
        ds.attrs.update(sanitize_attrs(da.attrs))
        
        # Write dimension coordinates
        for dim in da.dims:
            if dim in da.coords:
                coord_data = da.coords[dim].values
                ds.attrs[f'{dim}_coords'] = coord_data.tolist()
    
    # Consolidate metadata
    zarr.consolidate_metadata(str(zarr_path))
    
    log.info(f"Successfully wrote zarr to {zarr_path}")


def embed_statistics_in_zarr(
    zarr_path: Path,
    stats_dict: Dict[str, Dict[str, Any]]
) -> None:
    """Embed statistics in zarr variable attributes."""
    
    root = zarr.open_group(str(zarr_path), mode='a')
    
    for var_path, stats in stats_dict.items():
        parts = var_path.split('/')
        
        if len(parts) != 4:
            continue
        
        category, group_name, subsection, var_name = parts
        
        try:
            # Navigate to variable
            var = root[category][group_name][subsection][var_name]
            
            # Embed stats in attrs
            var.attrs['statistics'] = stats
            
        except KeyError:
            log.warning(f"Variable not found in zarr: {var_path}")
    
    # Consolidate metadata
    zarr.consolidate_metadata(str(zarr_path))
    
    log.info("Embedded statistics in zarr attributes")


def export_statistics_json(
    stats_dict: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """Export statistics to JSON sidecar file."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        json.dump(stats_dict, f, indent=2, sort_keys=True)
    
    log.info(f"Exported statistics to {output_path}")


def export_statistics_csv(
    stats_dict: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """Export statistics to CSV file."""
    import csv
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten stats for CSV format
    rows = []
    
    for var_path, stats in stats_dict.items():
        row = {'variable': var_path}
        
        # Add non-dict stats
        for key, val in stats.items():
            if not isinstance(val, dict):
                row[key] = val
        
        # Add dict stats (like histogram) as JSON strings
        if 'histogram' in stats:
            row['histogram'] = json.dumps(stats['histogram'])
        
        rows.append(row)
    
    if rows:
        # Get all unique keys
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)
        
        with output_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        log.info(f"Exported statistics to {output_path}")


# =============================================================================
# Early Validation
# =============================================================================

def validate_configuration(
    cfg: dict,
    spatial_cfg: SpatialConfig,
    group_specs: List[GroupSpec],
    global_window: Tuple[int, int]
) -> None:
    """
    Perform comprehensive validation before any data loading.
    
    Checks:
    - All files exist
    - All bands exist in files
    - CRS/resolution alignment (reports reprojection needs)
    
    Raises detailed errors if validation fails.
    """
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    
    log.info("=" * 80)
    log.info("VALIDATING CONFIGURATION")
    log.info("=" * 80)
    
    errors = []
    warnings = []
    reprojection_needed = []
    
    # Target CRS
    target_crs = CRS.from_wkt(spatial_cfg.crs_wkt)
    target_resolution = spatial_cfg.resolution
    target_transform = Affine(*spatial_cfg.transform)
    
    log.info(f"Target CRS: {target_crs.to_string()}")
    log.info(f"Target resolution: {target_resolution}m")
    log.info(f"Target shape: {spatial_cfg.shape}")
    
    # Validate AOI and strata first
    aoi_cfg = cfg.get('aoi', {})
    if aoi_cfg.get('path'):
        aoi_path = Path(aoi_cfg['path'])
        log.info(f"\nValidating AOI: {aoi_path}")
        
        if not aoi_path.exists():
            errors.append(f"AOI file not found: {aoi_path}")
        else:
            try:
                # Just open to check metadata - don't load data
                aoi_da = rioxarray.open_rasterio(aoi_path, masked=True)
                aoi_crs = aoi_da.rio.crs
                aoi_res = aoi_da.rio.resolution()
                aoi_shape = aoi_da.rio.shape
                
                log.info(f"  ✓ File exists")
                log.info(f"    CRS: {aoi_crs.to_string() if aoi_crs else 'None'}")
                log.info(f"    Resolution: {aoi_res}")
                log.info(f"    Shape: {aoi_shape}")
                
                if aoi_crs and not _crs_match(aoi_crs, target_crs):
                    reprojection_needed.append(
                        f"AOI: CRS mismatch\n"
                        f"    Source: {aoi_crs.to_string()}\n"
                        f"    Target: {target_crs.to_string()}"
                    )
                
                if not _resolution_match(aoi_res, (target_resolution, -target_resolution)):
                    reprojection_needed.append(
                        f"AOI: Resolution mismatch\n"
                        f"    Source: {aoi_res[0]:.1f}m × {abs(aoi_res[1]):.1f}m\n"
                        f"    Target: {target_resolution}m × {target_resolution}m"
                    )
                
                aoi_da.close()
            except Exception as e:
                errors.append(f"Failed to open AOI: {e}")
    
    strata_cfg = cfg.get('strata', {})
    if strata_cfg.get('path'):
        strata_path = Path(strata_cfg['path'])
        log.info(f"\nValidating strata: {strata_path}")
        
        if not strata_path.exists():
            errors.append(f"Strata file not found: {strata_path}")
        else:
            try:
                strata_da = rioxarray.open_rasterio(strata_path, masked=True)
                strata_crs = strata_da.rio.crs
                strata_res = strata_da.rio.resolution()
                strata_shape = strata_da.rio.shape
                
                log.info(f"  ✓ File exists")
                log.info(f"    CRS: {strata_crs.to_string() if strata_crs else 'None'}")
                log.info(f"    Resolution: {strata_res}")
                log.info(f"    Shape: {strata_shape}")
                
                if strata_crs and not _crs_match(strata_crs, target_crs):
                    reprojection_needed.append(
                        f"Strata: CRS mismatch\n"
                        f"    Source: {strata_crs.to_string()}\n"
                        f"    Target: {target_crs.to_string()}"
                    )
                
                if not _resolution_match(strata_res, (target_resolution, -target_resolution)):
                    reprojection_needed.append(
                        f"Strata: Resolution mismatch\n"
                        f"    Source: {strata_res[0]:.1f}m × {abs(strata_res[1]):.1f}m\n"
                        f"    Target: {target_resolution}m × {target_resolution}m"
                    )
                
                strata_da.close()
            except Exception as e:
                errors.append(f"Failed to open strata: {e}")
    
    # Validate each group
    total_bands = sum(len(spec.bands) for spec in group_specs)
    log.info(f"\nValidating {len(group_specs)} groups ({total_bands} total bands)")
    
    for spec in group_specs:
        log.info(f"\n{'─' * 80}")
        log.info(f"Group: {spec.category}/{spec.name}/{spec.subsection}")
        log.info(f"  Semantic type: {spec.semantic_type}")
        log.info(f"  Bands: {len(spec.bands)}")
        
        # Get years for this group
        if spec.category == 'static':
            years_to_check = []
        else:
            years_to_check = get_year_list(spec, global_window)
            log.info(f"  Years: {len(years_to_check)} ({years_to_check[0] if years_to_check else 'N/A'} - {years_to_check[-1] if years_to_check else 'N/A'})")
        
        # Determine file pattern
        pattern_type = detect_file_pattern(spec)
        log.info(f"  File pattern: {pattern_type}")
        
        # Collect unique files to validate
        files_to_check = set()
        
        if pattern_type == 'per_year':
            # Check files for each year
            for year in years_to_check:
                try:
                    file_path = resolve_file_path(spec.path, year)
                    files_to_check.add(file_path)
                except FileNotFoundError as e:
                    errors.append(f"{spec.name}/{spec.subsection}: {e}")
        
        elif pattern_type == 'multiband':
            # Single group-level file
            try:
                file_path = resolve_file_path(spec.path)
                files_to_check.add(file_path)
            except FileNotFoundError as e:
                errors.append(f"{spec.name}/{spec.subsection}: {e}")
        
        elif pattern_type == 'per_band':
            # Per-band files - validate year_range matches band count
            for band in spec.bands:
                if band.path:
                    try:
                        file_path = resolve_file_path(band.path)
                        files_to_check.add(file_path)
                        
                        # CRITICAL: Check if year_range matches band count
                        if spec.year_range:
                            da_check = rioxarray.open_rasterio(file_path, masked=True)
                            num_bands_in_file = len(da_check.band) if 'band' in da_check.dims else 1
                            expected_years = list(range(spec.year_range[0], spec.year_range[1] + 1))
                            
                            if len(expected_years) != num_bands_in_file:
                                errors.append(
                                    f"{spec.name}/{spec.subsection}/{band.id}: Year range mismatch!\n"
                                    f"    Config years: {spec.year_range[0]}-{spec.year_range[1]} ({len(expected_years)} years)\n"
                                    f"    File bands: {num_bands_in_file}\n"
                                    f"    File: {file_path.name}\n"
                                    f"    This will cause data loss - all values will become 0/NaN!"
                                )
                            da_check.close()
                            
                    except FileNotFoundError as e:
                        errors.append(f"{spec.name}/{spec.subsection}/{band.id}: {e}")
        
        # Validate files and bands
        for file_path in files_to_check:
            try:
                # Open and check - use masked=True to avoid loading all data
                da = rioxarray.open_rasterio(file_path, masked=True)
                file_crs = da.rio.crs
                file_res = da.rio.resolution()
                num_bands = len(da.band) if 'band' in da.dims else 1
                
                # Check CRS
                crs_matches = _crs_match(file_crs, target_crs) if file_crs else False
                res_matches = _resolution_match(file_res, (target_resolution, -target_resolution))
                
                if not crs_matches and file_crs:
                    reprojection_needed.append(
                        f"{spec.name}/{spec.subsection}: {file_path.name}\n"
                        f"    CRS: {file_crs.to_string()} → {target_crs.to_string()}"
                    )
                
                # Check resolution
                if not res_matches:
                    reprojection_needed.append(
                        f"{spec.name}/{spec.subsection}: {file_path.name}\n"
                        f"    Resolution: {file_res[0]:.1f}m × {abs(file_res[1]):.1f}m → "
                        f"{target_resolution}m × {target_resolution}m"
                    )
                
                # Validate band indices
                if pattern_type in ('multiband', 'per_year'):
                    for band in spec.bands:
                        if band.source_band is not None:
                            if band.source_band < 1 or band.source_band > num_bands:
                                errors.append(
                                    f"{spec.name}/{spec.subsection}/{band.id}: "
                                    f"source_band={band.source_band} out of range "
                                    f"(file has {num_bands} bands): {file_path.name}"
                                )
                
                status = "✓"
                if not crs_matches or not res_matches:
                    status = "⚠"
                
                log.info(f"  {status} {file_path.name}: {num_bands} bands, "
                        f"{file_crs.to_string() if file_crs else 'no CRS'}, "
                        f"{file_res[0]:.0f}m res")
                
                da.close()
                
            except Exception as e:
                errors.append(f"{spec.name}/{spec.subsection}: Failed to validate {file_path}: {e}")
    
    # Report results
    log.info(f"\n{'=' * 80}")
    log.info("VALIDATION SUMMARY")
    log.info(f"{'=' * 80}")
    
    if reprojection_needed:
        log.warning(f"\n⚠️  REPROJECTION REQUIRED FOR {len(reprojection_needed)} FILE(S):")
        for item in reprojection_needed:
            log.warning(f"  • {item}")
    else:
        log.info("\n✓ No reprojection needed - all files match target CRS and resolution")
    
    if warnings:
        log.warning(f"\n⚠️  {len(warnings)} WARNING(S):")
        for warning in warnings:
            log.warning(f"  • {warning}")
    
    if errors:
        log.error(f"\n❌ VALIDATION FAILED WITH {len(errors)} ERROR(S):")
        for error in errors:
            log.error(f"  • {error}")
        log.error(f"\n{'=' * 80}")
        raise ValueError(f"Configuration validation failed with {len(errors)} error(s). See log above.")
    
    log.info(f"\n✓ Validation passed!")
    log.info(f"  • {len(group_specs)} groups")
    log.info(f"  • {total_bands} total bands")
    log.info(f"  • {len(reprojection_needed)} files need reprojection")
    log.info(f"{'=' * 80}\n")


def _crs_match(crs1: CRS, crs2: CRS, tolerance: float = 1e-6) -> bool:
    """
    Check if two CRS match semantically.
    
    Uses rasterio's is_exact_same which compares the actual projection
    parameters rather than string representation.
    """
    if crs1 is None or crs2 is None:
        return False
    
    try:
        # Method 1: Check if they're exactly the same
        if crs1 == crs2:
            return True
        
        # Method 2: Compare EPSG codes if both have them
        if crs1.to_epsg() is not None and crs2.to_epsg() is not None:
            return crs1.to_epsg() == crs2.to_epsg()
        
        # Method 3: Compare WKT parameters semantically
        # Convert both to PROJ dictionaries and compare
        try:
            proj1 = crs1.to_dict()
            proj2 = crs2.to_dict()
            
            # Check projection type
            if proj1.get('proj') != proj2.get('proj'):
                return False
            
            # For Albers, check key parameters
            if proj1.get('proj') == 'aea':
                # Check critical parameters with tolerance
                params_to_check = [
                    'lat_0', 'lon_0', 'lat_1', 'lat_2',
                    'x_0', 'y_0'
                ]
                
                for param in params_to_check:
                    val1 = proj1.get(param, 0)
                    val2 = proj2.get(param, 0)
                    if abs(float(val1) - float(val2)) > tolerance:
                        return False
                
                return True
        except:
            pass
        
        # Fallback: if we can't determine, assume they don't match
        return False
        
    except Exception as e:
        log.debug(f"CRS comparison error: {e}")
        return False


def _resolution_match(res1: Tuple[float, float], res2: Tuple[float, float], tolerance: float = 1.0) -> bool:
    """Check if resolutions match within tolerance (in meters)."""
    return (abs(abs(res1[0]) - abs(res2[0])) < tolerance and 
            abs(abs(res1[1]) - abs(res2[1])) < tolerance)


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchical Zarr feature cube from raster data."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Override output zarr path from config"
    )
    parser.add_argument(
        "--ncore",
        type=int,
        default=1,
        help="Number of CPU cores for parallel processing"
    )
    parser.add_argument(
        "--no-stats",
        action='store_true',
        help="Skip statistics computation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="Enable debug logging"
    )
    parser.add_argument(
        "--validate-only",
        action='store_true',
        help="Only validate configuration and files, don't build zarr"
    )
    parser.add_argument(
        "--stats-from-zarr",
        action='store_true',
        help="Compute statistics from written zarr (safer for large datasets)"
    )
    
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main pipeline execution."""
    
    args = parse_args(argv)
    
    # Configure logging
    if args.verbose:
        log.setLevel(logging.DEBUG)
        # Also set root logger
        logging.getLogger().setLevel(logging.DEBUG)
        
        # But silence very verbose libraries
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        logging.getLogger('rasterio._io').setLevel(logging.WARNING)
        logging.getLogger('rasterio._env').setLevel(logging.WARNING)
    
    # Setup dask if multi-core
    if args.ncore > 1:
        from dask.distributed import Client, LocalCluster
        
        log.info(f"Starting Dask cluster with {args.ncore} workers")
        
        # Suppress distributed logging noise
        logging.getLogger('distributed').setLevel(logging.ERROR)
        logging.getLogger('distributed.nanny').setLevel(logging.ERROR)
        logging.getLogger('distributed.scheduler').setLevel(logging.ERROR)
        
        cluster = LocalCluster(
            n_workers=args.ncore,
            threads_per_worker=1,
            memory_limit='8GB',
            silence_logs=logging.ERROR,
            # Increase intervals to reduce "unresponsive" warnings during I/O
            heartbeat_interval='60s',  # default is 5s
            death_timeout='300s'  # default is 60s
        )
        client = Client(cluster)
        log.info(f"Dask dashboard: {client.dashboard_link}")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Parse spatial configuration
    spatial_cfg = parse_spatial_config(cfg)
    log.info(f"Spatial grid: {spatial_cfg.shape} pixels at {spatial_cfg.resolution}m resolution")
    
    # Get global time window
    global_window = get_global_time_window(cfg)
    log.info(f"Global time window: {global_window[0]} to {global_window[1]}")
    
    # Determine output path
    zarr_path = args.out or Path(cfg['dataset']['out_zarr']['path'])
    log.info(f"Output zarr: {zarr_path}")
    
    # Get chunk and compressor config
    chunk_cfg = cfg['dataset'].get('default_chunk', {})
    chunk_spec = chunk_cfg.get('annual', {})
    chunks = {
        'time': chunk_spec.get('time', 1),
        'y': chunk_spec.get('y', 256),
        'x': chunk_spec.get('x', 256),
        'snapshot': 1  # For irregular data
    }
    
    compressor_cfg = cfg['dataset'].get('compressor', {})
    
    # Create spatial template
    log.info("Creating spatial template")
    template = create_spatial_template(spatial_cfg, {'y': chunks['y'], 'x': chunks['x']})
    
    # Parse group specifications
    group_specs = parse_group_specs(cfg)
    
    # =========================================================================
    # EARLY VALIDATION - Fail fast before any data loading
    # =========================================================================
    validate_configuration(cfg, spatial_cfg, group_specs, global_window)
    
    # Exit early if only validating
    if args.validate_only:
        log.info("✓ Validation complete (--validate-only mode). Exiting without building zarr.")
        return
    
    # Load AOI and strata (only after validation passes)
    aoi = None
    strata = None
    
    aoi_cfg = cfg.get('aoi', {})
    if aoi_cfg.get('path'):
        log.info(f"Loading AOI from {aoi_cfg['path']}")
        aoi = open_raster_band(Path(aoi_cfg['path']), chunks={'y': chunks['y'], 'x': chunks['x']})
        aoi = align_to_template(aoi, template)
        aoi = (aoi > 0).astype(np.uint8)  # Convert to binary mask
    
    strata_cfg = cfg.get('strata', {})
    if strata_cfg.get('path'):
        log.info(f"Loading strata from {strata_cfg['path']}")
        strata = open_raster_band(Path(strata_cfg['path']), chunks={'y': chunks['y'], 'x': chunks['x']})
        strata = align_to_template(strata, template)
        strata = strata.astype(np.int16)
    
    # Build variable hierarchy
    variables = build_variable_hierarchy(
        group_specs,
        template,
        global_window,
        cfg['dataset'],
        chunks
    )
    
    # Write to zarr
    write_zarr_hierarchy(
        zarr_path,
        variables,
        aoi=aoi,
        strata=strata,
        compressor_cfg=compressor_cfg,
        chunks=chunks
    )
    
    # Compute statistics
    if not args.no_stats:
        log.info("Computing statistics")
        
        if args.stats_from_zarr:
            # Compute from written zarr (safer for very large datasets)
            log.info("Reading from zarr for statistics computation (memory-safe mode)")
            stats_dict = compute_statistics_from_zarr(zarr_path, aoi)
        else:
            # Compute from in-memory variables
            stats_dict = compute_all_statistics(variables, aoi)
        
        # Embed in zarr
        stats_cfg = cfg['dataset'].get('statistics', {})
        if stats_cfg.get('embed_in_zarr', True):
            embed_statistics_in_zarr(zarr_path, stats_dict)
        
        # Export JSON
        if stats_cfg.get('export_json', True):
            json_path = zarr_path.with_suffix('.stats.json')
            export_statistics_json(stats_dict, json_path)
        
        # Export CSV
        if stats_cfg.get('export_csv', True):
            csv_path = zarr_path.with_suffix('.stats.csv')
            export_statistics_csv(stats_dict, csv_path)
    
    log.info("✓ Zarr build complete!")
    
    # Clean up dask cluster if running
    if args.ncore > 1:
        log.info("Shutting down Dask cluster...")
        try:
            client.close()
            cluster.close()
        except:
            pass  # Already closed


if __name__ == "__main__":
    main()
