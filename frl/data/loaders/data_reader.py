"""
Data Reader for Forest Representation Model - UPDATED WITH SNAPSHOT SUPPORT

Changes from original:
- Now accepts BindingsRegistry and config dict instead of config path
- Single entry point for configuration parsing
- Better separation of concerns (parser vs reader)
- NEW: Support for snapshot input groups

Reads raw data from Zarr arrays based on spatial and temporal windows.
Handles padding for out-of-bounds requests and maintains band name metadata
for downstream registry calls.
"""

import zarr
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

from data.loaders.bindings.parser import InputGroup, BandConfig
from data.loaders.windows import SpatialWindow, TemporalWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GroupReadResult:
    """Result from reading a single input group"""
    data: np.ndarray  # Shape: [C_group, H, W] for temporal snapshot or [C_group, T, H, W] for full temporal
    band_names: List[str]  # Band names in same order as channels
    group_name: str
    category: str  # 'temporal', 'irregular', 'static', or 'snapshot'
    metadata: Dict[str, Any]  # Additional metadata (zarr path, actual bounds, etc.)


class DataReader:
    """
    Reads raw data from Zarr arrays for spatial and temporal windows.
    
    Handles:
    - Temporal groups: returns [C_group, T, H, W] or [C_group, H, W] for single timestep
    - Irregular groups: returns [C_group, H, W] (stub)
    - Static groups: returns [C_group, H, W]
    - Snapshot groups: returns [C_group, H, W] for specific year
    - NaN-padding for out-of-bounds requests
    - Missing value policy application (fill values → NaN)
    - Band name tracking for downstream processing
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        zarr_path: Optional[str] = None
    ):
        """
        Initialize DataReader.
        
        Args:
            config: Parsed bindings configuration dictionary
            zarr_path: Optional override for Zarr dataset path (uses config if not provided)
        """
        self.config = config
        
        # Open Zarr dataset
        if zarr_path is None:
            zarr_path = self.config['zarr']['path']
        
        self.zarr_path = Path(zarr_path)
        logger.info(f"Opening Zarr dataset: {self.zarr_path}")
        self.zarr_root = zarr.open(str(self.zarr_path), mode='r')
        
        # Cache zarr array metadata for bounds checking
        self._cache_array_metadata()
    
    def _cache_array_metadata(self):
        """Cache spatial/temporal bounds of zarr arrays for padding logic"""
        self.array_metadata = {}
        
        for category in ['temporal', 'irregular', 'static', 'snapshot']:
            for group_name, group in self.config['inputs'][category].items():
                zarr_group_path = group.zarr.group
                
                try:
                    zgroup = self.zarr_root[zarr_group_path]
                    
                    # Get first array to determine spatial shape
                    if group.bands and group.bands[0].array:
                        first_array_name = group.bands[0].array
                        first_array = zgroup[first_array_name]
                        
                        # Determine array shape
                        array_shape = first_array.shape
                        
                        # Store metadata
                        metadata = {
                            'zarr_group': zarr_group_path,
                            'array_shape': array_shape
                        }
                        
                        # For temporal groups, check if there's a temporal dimension
                        if category == 'temporal':
                            if len(array_shape) == 3:
                                # Shape is [T, H, W] - standard convention
                                metadata['temporal_shape'] = array_shape[0]
                                metadata['spatial_shape'] = array_shape[1:]  # (H, W)
                            elif len(array_shape) == 2:
                                # Shape is [H, W] - static temporal data
                                metadata['temporal_shape'] = 1
                                metadata['spatial_shape'] = array_shape  # (H, W)
                        else:
                            # Static/irregular/snapshot: just spatial dimensions
                            if len(array_shape) == 2:
                                metadata['spatial_shape'] = array_shape  # (H, W)
                            else:
                                logger.warning(f"Unexpected shape for {category}.{group_name}: {array_shape}")
                                metadata['spatial_shape'] = array_shape[-2:]  # Assume last 2 dims are spatial
                        
                        self.array_metadata[f"{category}.{group_name}"] = metadata
                        
                except Exception as e:
                    logger.warning(f"Could not cache metadata for {category}.{group_name}: {e}")
    
    def read_temporal_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow,
        return_full_temporal: bool = False
    ) -> GroupReadResult:
        """
        Read data from a temporal input group.
        
        Args:
            group_name: Name of temporal group (e.g., 'ls8day', 'lcms_chg')
            spatial_window: Spatial window to read
            temporal_window: Temporal window to read
            return_full_temporal: If True, return [C, T, H, W]; if False, return [C, H, W] for end year
            
        Returns:
            GroupReadResult with data shape [C_group, T, H, W] or [C_group, H, W]
        """
        # Get group configuration
        try:
            group = self.config['inputs']['temporal'][group_name]
        except KeyError:
            raise ValueError(f"Temporal group '{group_name}' not found")
        
        # Get zarr group
        zarr_group_path = group.zarr.group
        zgroup = self.zarr_root[zarr_group_path]
        
        # Get missing policy if specified
        missing_policy = group.missing_policy if hasattr(group, 'missing_policy') else None
        
        # Prepare to collect data for each band
        band_names = []
        band_data_list = []
        
        for band in group.bands:
            band_names.append(band.name)
            
            # Get array
            array_name = band.array if band.array else band.name
            try:
                zarray = zgroup[array_name]
            except KeyError:
                logger.warning(f"Array '{array_name}' not found in {zarr_group_path}, using NaN")
                # Create NaN array with expected shape
                if return_full_temporal:
                    band_data = np.full((temporal_window.window_length, 
                                         spatial_window.height, 
                                         spatial_window.width), np.nan, dtype=np.float32)
                else:
                    band_data = np.full((spatial_window.height, spatial_window.width), 
                                       np.nan, dtype=np.float32)
                band_data_list.append(band_data)
                continue
            
            # Read data with padding
            if return_full_temporal:
                # Read full temporal dimension [T, H, W]
                band_data = self._read_temporal_array_with_padding(
                    zarray, spatial_window, temporal_window, missing_policy
                )
            else:
                # Read only end year [H, W]
                band_data = self._read_temporal_snapshot_with_padding(
                    zarray, spatial_window, temporal_window, missing_policy
                )
            
            band_data_list.append(band_data)
        
        # Stack bands
        if return_full_temporal:
            # Stack to [C, T, H, W]
            data = np.stack(band_data_list, axis=0)
        else:
            # Stack to [C, H, W]
            data = np.stack(band_data_list, axis=0)
        
        return GroupReadResult(
            data=data,
            band_names=band_names,
            group_name=group_name,
            category='temporal',
            metadata={
                'zarr_group': zarr_group_path,
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'return_full_temporal': return_full_temporal,
                'missing_policy': missing_policy
            }
        )
    
    def read_snapshot_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        year: int
    ) -> GroupReadResult:
        """
        Read data from a snapshot input group for a specific year.
        
        Args:
            group_name: Name of snapshot group (e.g., 'ccdc_snapshot')
            spatial_window: Spatial window to read
            year: Snapshot year to read (e.g., 2024, 2022, 2020)
            
        Returns:
            GroupReadResult with data shape [C_group, H, W]
            
        Example:
            >>> reader = DataReader(config)
            >>> spatial_window = SpatialWindow.from_upper_left_and_hw((1000, 2000), (256, 256))
            >>> result = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2024)
            >>> print(result.data.shape)  # [20, 256, 256]
        """
        # Get group configuration
        try:
            group = self.config['inputs']['snapshot'][group_name]
        except KeyError:
            raise ValueError(f"Snapshot group '{group_name}' not found")
        
        # Get bands for this specific year using InputGroup method
        try:
            bands = group.get_bands_for_year(year)
        except ValueError as e:
            raise ValueError(f"Cannot read snapshot for year {year}: {e}")
        
        # Get zarr group
        zarr_group_path = group.zarr.group
        zgroup = self.zarr_root[zarr_group_path]
        
        # Get missing policy if specified
        missing_policy = group.missing_policy if hasattr(group, 'missing_policy') else None
        
        # Prepare to collect data for each band
        band_names = []
        band_data_list = []
        
        for band in bands:
            band_names.append(band.name)
            
            # Get array
            array_name = band.array if band.array else band.name
            try:
                zarray = zgroup[array_name]
            except KeyError:
                logger.warning(f"Array '{array_name}' not found in {zarr_group_path}, using NaN")
                band_data = np.full((spatial_window.height, spatial_window.width), 
                                   np.nan, dtype=np.float32)
                band_data_list.append(band_data)
                continue
            
            # Read data with padding and missing value handling
            # Snapshot arrays are [H, W] like static arrays
            band_data = self._read_static_array_with_padding(zarray, spatial_window, missing_policy)
            band_data_list.append(band_data)
        
        # Stack bands into [C, H, W] tensor
        data = np.stack(band_data_list, axis=0)
        
        return GroupReadResult(
            data=data,
            band_names=band_names,
            group_name=group_name,
            category='snapshot',
            metadata={
                'zarr_group': zarr_group_path,
                'spatial_window': spatial_window,
                'year': year,
                'zarr_prefix': group.get_zarr_prefix(year),
                'missing_policy': missing_policy
            }
        )
    
    def read_irregular_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> GroupReadResult:
        """
        Read data from an irregular input group.
        
        Irregular inputs have sparse temporal observations (e.g., NAIP imagery).
        Returns a multi-temporal array [C, T_obs, H, W] where T_obs is the number
        of observations within the temporal window.
        
        Args:
            group_name: Name of irregular group (e.g., 'naip')
            spatial_window: Spatial window to read
            temporal_window: Temporal window to filter observations
            
        Returns:
            GroupReadResult with data shape [C, T_obs, H, W] where T_obs is number
            of observations within the window
            
        Example:
            >>> reader = DataReader(config)
            >>> spatial_window = SpatialWindow.from_upper_left_and_hw((1000, 2000), (256, 256))
            >>> temporal_window = TemporalWindow(end_year=2020, window_length=10)
            >>> result = reader.read_irregular_group('naip', spatial_window, temporal_window)
            >>> print(result.data.shape)  # [6, 3, 256, 256] - 6 bands, 3 observations in window
        """
        # Get group configuration
        try:
            group = self.config['inputs']['irregular'][group_name]
        except KeyError:
            raise ValueError(f"Irregular group '{group_name}' not found")
        
        # Get available years for this group
        if not hasattr(group, 'years') or not group.years:
            raise ValueError(f"Irregular group '{group_name}' missing 'years' list")
        
        # Filter years to those within the temporal window
        window_start_year = temporal_window.end_year - temporal_window.window_length + 1
        window_end_year = temporal_window.end_year
        
        years_in_window = [
            year for year in group.years
            if window_start_year <= year <= window_end_year
        ]
        
        if not years_in_window:
            # No observations in this window - return empty array
            logger.warning(
                f"No observations for '{group_name}' in window "
                f"[{window_start_year}, {window_end_year}]"
            )
            # Return empty temporal dimension
            empty_data = np.full(
                (len(group.bands), 0, spatial_window.height, spatial_window.width),
                np.nan,
                dtype=np.float32
            )
            return GroupReadResult(
                data=empty_data,
                band_names=[band.name for band in group.bands],
                group_name=group_name,
                category='irregular',
                metadata={
                    'zarr_group': group.zarr.group,
                    'spatial_window': spatial_window,
                    'temporal_window': temporal_window,
                    'years_in_window': [],
                    'num_observations': 0
                }
            )
        
        # Get zarr group
        zarr_group_path = group.zarr.group
        zgroup = self.zarr_root[zarr_group_path]
        
        # Get missing policy if specified
        missing_policy = group.missing_policy if hasattr(group, 'missing_policy') else None
        
        # Read data for each band and each year in window
        # Structure: [C, T_obs, H, W]
        all_bands_data = []
        
        for band in group.bands:
            # Read this band for all years in window
            band_temporal_data = []
            
            for year in years_in_window:
                # Get array - irregular arrays are stored per year
                # Check if array name includes year or if there's a year subdirectory
                array_name = band.array if band.array else band.name
                
                # Try different naming conventions
                possible_paths = [
                    f"{year}/{array_name}",  # year/array
                    f"{array_name}_{year}",  # array_year
                    f"{array_name}/{year}",  # array/year
                    array_name,              # Just array name (all years in one)
                ]
                
                zarray = None
                for path in possible_paths:
                    try:
                        zarray = zgroup[path]
                        break
                    except KeyError:
                        continue
                
                if zarray is None:
                    logger.warning(
                        f"Array '{array_name}' for year {year} not found in {zarr_group_path}, using NaN"
                    )
                    year_data = np.full(
                        (spatial_window.height, spatial_window.width),
                        np.nan,
                        dtype=np.float32
                    )
                else:
                    # Read data with padding
                    # Check array shape to determine if it has years dimension
                    if len(zarray.shape) == 3:
                        # Array is [T, H, W] - all years in one array
                        # Find index for this year
                        try:
                            year_idx = group.years.index(year)
                            year_data = self._read_static_array_with_padding(
                                zarray[year_idx],
                                spatial_window,
                                missing_policy
                            )
                        except (ValueError, IndexError):
                            logger.warning(f"Year {year} not found in array, using NaN")
                            year_data = np.full(
                                (spatial_window.height, spatial_window.width),
                                np.nan,
                                dtype=np.float32
                            )
                    elif len(zarray.shape) == 2:
                        # Array is [H, W] - single year
                        year_data = self._read_static_array_with_padding(
                            zarray,
                            spatial_window,
                            missing_policy
                        )
                    else:
                        raise ValueError(
                            f"Unexpected array shape for irregular data: {zarray.shape}"
                        )
                
                band_temporal_data.append(year_data)
            
            # Stack this band's temporal data: [T_obs, H, W]
            band_temporal_stack = np.stack(band_temporal_data, axis=0)
            all_bands_data.append(band_temporal_stack)
        
        # Stack all bands: [C, T_obs, H, W]
        data = np.stack(all_bands_data, axis=0)
        
        return GroupReadResult(
            data=data,
            band_names=[band.name for band in group.bands],
            group_name=group_name,
            category='irregular',
            metadata={
                'zarr_group': zarr_group_path,
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'years_in_window': years_in_window,
                'num_observations': len(years_in_window),
                'missing_policy': missing_policy
            }
        )
    
    def read_static_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow
    ) -> GroupReadResult:
        """
        Read data from a static input group.
        
        Args:
            group_name: Name of static group (e.g., 'topo', 'soils')
            spatial_window: Spatial window to read
            
        Returns:
            GroupReadResult with data shape [C_group, H, W]
        """
        # Get group configuration
        try:
            group = self.config['inputs']['static'][group_name]
        except KeyError:
            raise ValueError(f"Static group '{group_name}' not found")
        
        # Get zarr group
        zarr_group_path = group.zarr.group
        zgroup = self.zarr_root[zarr_group_path]
        
        # Get missing policy if specified
        missing_policy = group.missing_policy if hasattr(group, 'missing_policy') else None
        
        # Prepare to collect data for each band
        band_names = []
        band_data_list = []
        
        for band in group.bands:
            band_names.append(band.name)
            
            # Get array
            array_name = band.array if band.array else band.name
            try:
                zarray = zgroup[array_name]
            except KeyError:
                logger.warning(f"Array '{array_name}' not found in {zarr_group_path}, using NaN")
                band_data = np.full((spatial_window.height, spatial_window.width), 
                                   np.nan, dtype=np.float32)
                band_data_list.append(band_data)
                continue
            
            # Read data with padding and missing value handling
            band_data = self._read_static_array_with_padding(zarray, spatial_window, missing_policy)
            band_data_list.append(band_data)
        
        # Stack bands into [C, H, W] tensor
        data = np.stack(band_data_list, axis=0)
        
        return GroupReadResult(
            data=data,
            band_names=band_names,
            group_name=group_name,
            category='static',
            metadata={
                'zarr_group': zarr_group_path,
                'spatial_window': spatial_window,
                'missing_policy': missing_policy
            }
        )
    
    def _read_temporal_array_with_padding(
        self,
        zarray: zarr.Array,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow,
        missing_policy: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Read temporal array with zero-padding for out-of-bounds regions.
        
        Returns full temporal dimension [T, H, W].
        
        Uses array's 'time_coords' attribute to map requested years to array indices.
        
        Args:
            zarray: Zarr array to read from
            spatial_window: Spatial window
            temporal_window: Temporal window
            missing_policy: Optional missing value policy
            
        Returns:
            Array with shape [T, H, W]
        """
        # Get array shape
        array_shape = zarray.shape
        
        if len(array_shape) == 2:
            # Array is [H, W] - no temporal dimension, replicate for each timestep
            H_available, W_available = array_shape
            static_data = self._read_static_array_with_padding(
                zarray, spatial_window, missing_policy
            )
            # Replicate across time
            return np.stack([static_data] * temporal_window.window_length, axis=0)
        
        elif len(array_shape) == 3:
            # Array is [T, H, W]
            T_available, H_available, W_available = array_shape
            
            # Get time coordinates from array attributes
            time_coords = zarray.attrs.get('time_coords', None)
            
            # Calculate requested year range
            window_start_year = temporal_window.end_year - temporal_window.window_length + 1
            window_end_year = temporal_window.end_year
            
            # Determine which timesteps to read
            if time_coords is not None:
                # Use time_coords to find exact indices for requested years
                try:
                    # Convert time_coords to years (handle various formats)
                    if isinstance(time_coords[0], str):
                        # Parse dates if stored as strings (e.g., "2020-01-01")
                        import pandas as pd
                        years = pd.to_datetime(time_coords).year.values
                    else:
                        # Assume numeric years
                        years = np.array(time_coords, dtype=int)
                    
                    # Find indices for requested years
                    year_mask = (years >= window_start_year) & (years <= window_end_year)
                    matching_indices = np.where(year_mask)[0]
                    
                    if len(matching_indices) == 0:
                        # No data in requested window - return all NaN
                        logger.warning(
                            f"No timesteps found for years {window_start_year}-{window_end_year}. "
                            f"Available years: {years[0]}-{years[-1]}"
                        )
                        return np.full(
                            (temporal_window.window_length, spatial_window.height, spatial_window.width),
                            np.nan,
                            dtype=np.float32
                        )
                    
                    # Determine temporal reading parameters
                    t_start_actual = matching_indices[0]
                    t_end_actual = matching_indices[-1] + 1
                    num_years_available = len(matching_indices)
                    
                    # Calculate padding
                    if num_years_available < temporal_window.window_length:
                        # Pad at beginning if we don't have all requested years
                        pad_temporal_start = temporal_window.window_length - num_years_available
                    else:
                        pad_temporal_start = 0
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to parse time_coords: {e}. "
                        f"Falling back to last-N-timesteps approach."
                    )
                    # Fall back to old behavior
                    time_coords = None
            
            if time_coords is None:
                # No time_coords available - use last N timesteps (old behavior)
                logger.warning(
                    f"Array missing 'time_coords' attribute. Using last {temporal_window.window_length} timesteps. "
                    f"This may not match requested years {window_start_year}-{window_end_year}."
                )
                T_requested = temporal_window.window_length
                if T_requested > T_available:
                    t_start_actual = 0
                    t_end_actual = T_available
                    pad_temporal_start = T_requested - T_available
                else:
                    t_start_actual = T_available - T_requested
                    t_end_actual = T_available
                    pad_temporal_start = 0
            
            # Check spatial bounds
            request_row_end = spatial_window.row_start + spatial_window.height
            request_col_end = spatial_window.col_start + spatial_window.width
            
            completely_oob_spatial = (
                spatial_window.row_start >= H_available or
                request_row_end <= 0 or
                spatial_window.col_start >= W_available or
                request_col_end <= 0
            )
            
            if completely_oob_spatial:
                # Return all NaN for completely out-of-bounds
                return np.full(
                    (temporal_window.window_length, spatial_window.height, spatial_window.width),
                    np.nan,
                    dtype=np.float32
                )
            
            # Calculate actual readable spatial region
            row_start_actual = max(spatial_window.row_start, 0)
            row_end_actual = min(request_row_end, H_available)
            col_start_actual = max(spatial_window.col_start, 0)
            col_end_actual = min(request_col_end, W_available)
            
            # Calculate spatial padding
            pad_top = max(0, -spatial_window.row_start)
            pad_bottom = max(0, request_row_end - H_available)
            pad_left = max(0, -spatial_window.col_start)
            pad_right = max(0, request_col_end - W_available)
            
            # Read actual data
            read_data = zarray[
                t_start_actual:t_end_actual,
                row_start_actual:row_end_actual,
                col_start_actual:col_end_actual
            ]
            
            # Convert to float32 first (required for NaN)
            read_data = read_data.astype(np.float32)
            
            # Apply spatial padding with NaN
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                read_data = np.pad(
                    read_data,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=np.nan
                )
            
            # Apply temporal padding if needed (with NaN for missing timesteps)
            if pad_temporal_start > 0:
                temporal_pad = np.full(
                    (pad_temporal_start, spatial_window.height, spatial_window.width),
                    np.nan,
                    dtype=np.float32
                )
                read_data = np.concatenate([temporal_pad, read_data], axis=0)
            
            # Apply missing value policy
            if missing_policy:
                # Apply to each timestep
                for t in range(read_data.shape[0]):
                    read_data[t] = self._apply_missing_policy(read_data[t], missing_policy)
            
            return read_data
        
        else:
            raise ValueError(f"Unexpected array shape: {array_shape}")
    
    def _read_temporal_snapshot_with_padding(
        self,
        zarray: zarr.Array,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow,
        missing_policy: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Read single timestep (end year) from temporal array.
        
        Returns [H, W].
        
        Uses array's 'time_coords' attribute to find the exact timestep for end_year.
        
        Args:
            zarray: Zarr array to read from
            spatial_window: Spatial window
            temporal_window: Temporal window
            missing_policy: Optional missing value policy
            
        Returns:
            Array with shape [H, W]
        """
        array_shape = zarray.shape
        
        if len(array_shape) == 2:
            # Static array, just read it
            return self._read_static_array_with_padding(zarray, spatial_window, missing_policy)
        
        elif len(array_shape) == 3:
            # Temporal array [T, H, W], read timestep for end_year
            T_available, H_available, W_available = array_shape
            
            # Get time coordinates from array attributes
            time_coords = zarray.attrs.get('time_coords', None)
            
            # Determine which timestep to read
            if time_coords is not None:
                try:
                    # Convert time_coords to years
                    if isinstance(time_coords[0], str):
                        import pandas as pd
                        years = pd.to_datetime(time_coords).year.values
                    else:
                        years = np.array(time_coords, dtype=int)
                    
                    # Find index for end_year
                    matching_indices = np.where(years == temporal_window.end_year)[0]
                    
                    if len(matching_indices) == 0:
                        logger.warning(
                            f"No timestep found for year {temporal_window.end_year}. "
                            f"Available years: {years[0]}-{years[-1]}. Using NaN."
                        )
                        return np.full(
                            (spatial_window.height, spatial_window.width),
                            np.nan,
                            dtype=np.float32
                        )
                    
                    # Use first matching index (should only be one)
                    t_idx = matching_indices[0]
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to parse time_coords: {e}. "
                        f"Falling back to last timestep."
                    )
                    time_coords = None
            
            if time_coords is None:
                # No time_coords - use last timestep (old behavior)
                logger.warning(
                    f"Array missing 'time_coords' attribute. Using last timestep. "
                    f"This may not match requested year {temporal_window.end_year}."
                )
                t_idx = T_available - 1
            
            # Read spatial region from last available timestep
            request_row_end = spatial_window.row_start + spatial_window.height
            request_col_end = spatial_window.col_start + spatial_window.width
            
            completely_oob_spatial = (
                spatial_window.row_start >= H_available or
                request_row_end <= 0 or
                spatial_window.col_start >= W_available or
                request_col_end <= 0
            )
            
            if completely_oob_spatial:
                return np.full((spatial_window.height, spatial_window.width), np.nan, dtype=np.float32)
            
            # Calculate actual readable region
            row_start_actual = max(spatial_window.row_start, 0)
            row_end_actual = min(request_row_end, H_available)
            col_start_actual = max(spatial_window.col_start, 0)
            col_end_actual = min(request_col_end, W_available)
            
            # Calculate padding
            pad_top = max(0, -spatial_window.row_start)
            pad_bottom = max(0, request_row_end - H_available)
            pad_left = max(0, -spatial_window.col_start)
            pad_right = max(0, request_col_end - W_available)
            
            # Read the timestep for end_year
            read_data = zarray[t_idx, row_start_actual:row_end_actual, col_start_actual:col_end_actual]
            
            # Convert to float32 first (required for NaN)
            read_data = read_data.astype(np.float32)
            
            # Apply padding with NaN
            padded_data = np.pad(
                read_data,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=np.nan
            )
            
            # Apply missing value policy
            if missing_policy:
                padded_data = self._apply_missing_policy(padded_data, missing_policy)
            
            return padded_data
        
        else:
            raise ValueError(f"Unexpected array shape: {array_shape}")
    
    def _read_static_array_with_padding(
        self,
        zarray: zarr.Array,
        spatial_window: SpatialWindow,
        missing_policy: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Read static array with NaN-padding for out-of-bounds regions.
        
        Assumes zarr array shape is [H, W].
        
        Args:
            zarray: Zarr array to read from
            spatial_window: Spatial region to read
            missing_policy: Optional missing value policy (e.g., {'nan_from_fill': [-9999]})
            
        Returns:
            Array with shape [H, W], padded with NaN where out-of-bounds
        """
        # Get array shape
        H_available, W_available = zarray.shape
        
        # Check if request is completely out of bounds
        request_row_end = spatial_window.row_start + spatial_window.height
        request_col_end = spatial_window.col_start + spatial_window.width
        
        completely_oob = (
            spatial_window.row_start >= H_available or
            request_row_end <= 0 or
            spatial_window.col_start >= W_available or
            request_col_end <= 0
        )
        
        if completely_oob:
            # Return all NaN for completely out-of-bounds
            return np.full((spatial_window.height, spatial_window.width), np.nan, dtype=np.float32)
        
        # Calculate actual readable region
        row_start_actual = max(spatial_window.row_start, 0)
        row_end_actual = min(request_row_end, H_available)
        col_start_actual = max(spatial_window.col_start, 0)
        col_end_actual = min(request_col_end, W_available)
        
        # Calculate padding needed
        pad_top = max(0, -spatial_window.row_start)
        pad_bottom = max(0, request_row_end - H_available)
        pad_left = max(0, -spatial_window.col_start)
        pad_right = max(0, request_col_end - W_available)
        
        # Read the actual data (what's available)
        read_data = zarray[row_start_actual:row_end_actual, col_start_actual:col_end_actual]
        
        # Convert to float32 first (required for NaN)
        read_data = read_data.astype(np.float32)
        
        # Apply padding with NaN
        padded_data = np.pad(
            read_data,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=np.nan
        )
        
        # Apply missing value policy
        if missing_policy:
            padded_data = self._apply_missing_policy(padded_data, missing_policy)
        
        return padded_data
    
    def read_all_temporal_groups(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow,
        return_full_temporal: bool = False
    ) -> Dict[str, GroupReadResult]:
        """
        Read all temporal input groups for given windows.
        
        Args:
            spatial_window: Spatial window to read
            temporal_window: Temporal window to read
            return_full_temporal: Whether to return full temporal dimension
            
        Returns:
            Dictionary mapping group_name -> GroupReadResult
        """
        results = {}
        
        for group_name in self.config['inputs']['temporal'].keys():
            results[group_name] = self.read_temporal_group(
                group_name,
                spatial_window,
                temporal_window,
                return_full_temporal
            )
        
        return results
    
    def read_all_static_groups(
        self,
        spatial_window: SpatialWindow
    ) -> Dict[str, GroupReadResult]:
        """
        Read all static input groups for given spatial window.
        
        Args:
            spatial_window: Spatial window to read
            
        Returns:
            Dictionary mapping group_name -> GroupReadResult
        """
        results = {}
        
        for group_name in self.config['inputs']['static'].keys():
            results[group_name] = self.read_static_group(group_name, spatial_window)
        
        return results
    
    def read_all_snapshot_groups(
        self,
        spatial_window: SpatialWindow,
        year: int
    ) -> Dict[str, GroupReadResult]:
        """
        Read all snapshot input groups for given spatial window and year.
        
        Args:
            spatial_window: Spatial window to read
            year: Snapshot year to read
            
        Returns:
            Dictionary mapping group_name -> GroupReadResult
        """
        results = {}
        
        for group_name in self.config['inputs']['snapshot'].keys():
            results[group_name] = self.read_snapshot_group(group_name, spatial_window, year)
        
        return results
    
    def read_all_irregular_groups(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, GroupReadResult]:
        """
        Read all irregular input groups for given windows.
        
        Args:
            spatial_window: Spatial window to read
            temporal_window: Temporal window to filter observations
            
        Returns:
            Dictionary mapping group_name -> GroupReadResult
        """
        results = {}
        
        for group_name in self.config['inputs']['irregular'].keys():
            results[group_name] = self.read_irregular_group(
                group_name,
                spatial_window,
                temporal_window
            )
        
        return results
    
    def read_snapshots_for_windows(
        self,
        spatial_window: SpatialWindow,
        anchor_end: int,
        offsets: List[int]
    ) -> Dict[str, Dict[str, GroupReadResult]]:
        """
        Read snapshot data for multiple training windows.
        
        Convenience method for training scenarios with multi-window bundles.
        
        Args:
            spatial_window: Spatial window to read
            anchor_end: Anchor end year (e.g., 2024)
            offsets: Window offsets (e.g., [0, -2, -4] for t0, t2, t4)
            
        Returns:
            Nested dict: {window_label: {group_name: GroupReadResult}}
            
        Example:
            >>> results = reader.read_snapshots_for_windows(
            ...     spatial_window,
            ...     anchor_end=2024,
            ...     offsets=[0, -2, -4]
            ... )
            >>> print(results.keys())  # dict_keys(['t0', 't2', 't4'])
            >>> print(results['t0']['ccdc_snapshot'].data.shape)  # [20, 256, 256]
        """
        results = {}
        
        for offset in offsets:
            year = anchor_end + offset
            window_label = f"t{abs(offset)}"
            
            # Read all snapshot groups for this year
            results[window_label] = self.read_all_snapshot_groups(spatial_window, year)
        
        return results
    
    def get_band_metadata(self, category: str, group_name: str, band_name: str) -> BandConfig:
        """
        Get band configuration for downstream registry calls.
        
        Args:
            category: 'temporal', 'irregular', 'static', or 'snapshot'
            group_name: Input group name
            band_name: Band name
            
        Returns:
            BandConfig object with normalization, mask, quality_weight, etc.
        """
        group = self.config['inputs'][category][group_name]
        
        for band in group.bands:
            if band.name == band_name:
                return band
        
        raise ValueError(f"Band '{band_name}' not found in {category}.{group_name}")
    
    def _apply_missing_policy(self, data: np.ndarray, missing_policy: Dict[str, Any]) -> np.ndarray:
        """
        Apply missing value policy to data.
        
        Handles policies like:
        - nan_from_fill: [value1, value2, ...] -> Replace these values with NaN
        - fill_value: value -> Replace NaN with this value
        
        Args:
            data: Input array
            missing_policy: Policy configuration dict
            
        Returns:
            Array with missing policy applied
        """
        # Handle nan_from_fill: replace fill values with NaN
        nan_from_fill = missing_policy.get('nan_from_fill', [])
        if nan_from_fill:
            for fill_value in nan_from_fill:
                data = np.where(data == fill_value, np.nan, data)
        
        # Handle fill_value: replace NaN with a specific value
        fill_value = missing_policy.get('fill_value')
        if fill_value is not None:
            data = np.nan_to_num(data, nan=fill_value)
        
        return data


if __name__ == '__main__':
    # Example usage showing recommended pattern
    import sys
    from data.loaders.bindings.parser import BindingsParser
    
    # Enable debug logging
    logging.basicConfig(level=logging.INFO, force=True)
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config/frl_bindings_v0.yaml'
    
    # Parse config once (single entry point)
    print("Parsing bindings configuration...")
    parser = BindingsParser(config_path)
    config = parser.parse()
    
    # Initialize reader with config
    reader = DataReader(config)
    
    # Define windows
    spatial_window = SpatialWindow.from_upper_left_and_hw(
        upper_left=(1000, 2000),
        hw=(256, 256)
    )
    
    temporal_window = TemporalWindow(
        end_year=2020,
        window_length=10
    )
    
    # Read temporal group
    print("\nReading temporal group 'ls8day'...")
    result = reader.read_temporal_group(
        'ls8day',
        spatial_window,
        temporal_window,
        return_full_temporal=False
    )
    
    print(f"Data shape: {result.data.shape}")
    print(f"Band names: {result.band_names}")
    print(f"Metadata: {result.metadata}")
    
    # Read static group
    print("\nReading static group 'topo'...")
    static_result = reader.read_static_group('topo', spatial_window)
    print(f"Data shape: {static_result.data.shape}")
    print(f"Band names: {static_result.band_names}")
    
    # NEW: Read snapshot group
    print("\nReading snapshot group 'ccdc_snapshot' for year 2024...")
    snapshot_result = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2024)
    print(f"Data shape: {snapshot_result.data.shape}")
    print(f"Band names (first 5): {snapshot_result.band_names[:5]}")
    print(f"Metadata: {snapshot_result.metadata}")
    
    # NEW: Read snapshots for training windows
    print("\nReading snapshots for training windows (t0, t2, t4)...")
    window_results = reader.read_snapshots_for_windows(
        spatial_window,
        anchor_end=2024,
        offsets=[0, -2, -4]
    )
    
    for window_label, groups in window_results.items():
        for group_name, result in groups.items():
            print(f"{window_label}.{group_name}: shape={result.data.shape}, year={result.metadata['year']}")
