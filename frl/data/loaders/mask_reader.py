"""
Mask Builder for Forest Representation Model

Builds mask and quality weight layers from Zarr arrays based on shared configuration.
Provides a similar interface to DataReader but focuses on generating boolean masks
and float quality/weight arrays for training.

Key Features:
- Reads mask arrays from shared.masks section
- Reads quality arrays from shared.quality section
- Computes derived masks (e.g., thresholds, expressions)
- Handles temporal windows for time-varying masks/weights
- Returns MaskResult and QualityResult objects similar to GroupReadResult
"""

import zarr
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

from data.loaders.bindings.parser import BandConfig
from data.loaders.bindings.utils import BindingsRegistry
from data.loaders.windows import SpatialWindow, TemporalWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MaskResult:
    """Result from reading a mask"""
    data: np.ndarray  # Shape: [H, W] for static or [T, H, W] for temporal, dtype bool
    mask_name: str
    mask_type: str  # 'boolean', 'boolean_or_uint8', 'threshold', 'expression'
    false_means_invalid: bool  # True if False values indicate invalid/masked pixels
    metadata: Dict[str, Any]


@dataclass
class QualityResult:
    """Result from reading a quality weight layer"""
    data: np.ndarray  # Shape: [H, W] or [T, H, W], dtype float32, values typically in [0, 1]
    quality_name: str
    quality_type: str  # 'float', 'expression'
    metadata: Dict[str, Any]



class MaskBuilder:
    """
    Builds mask and quality weight layers from Zarr arrays.
    
    Handles:
    - Boolean masks from Zarr arrays
    - Threshold-based masks (e.g., forest mask from probability)
    - Expression-based masks and weights
    - Temporal masks (e.g., interpolation quality over time)
    - Quality weight layers for loss weighting
    """
    
    def __init__(
        self, 
        registry: BindingsRegistry,
        config: Dict[str, Any],
        zarr_path: Optional[str] = None
    ):
        """
        Initialize MaskBuilder.
        
        Args:
            registry: BindingsRegistry instance for fast lookups
            config: Parsed bindings configuration dictionary
            zarr_path: Optional override for Zarr dataset path (uses config if not provided)
        """
        self.registry = registry
        self.config = config
        
        # Open Zarr dataset
        if zarr_path is None:
            zarr_path = self.config['zarr']['path']
        
        self.zarr_path = Path(zarr_path)
        logger.info(f"Opening Zarr dataset: {self.zarr_path}")
        self.zarr_root = zarr.open(str(self.zarr_path), mode='r')
        
        # Cache available masks and quality metrics
        self.available_masks = list(self.config['shared']['masks'].keys())
        self.available_quality = list(self.config['shared']['quality'].keys())
        
        logger.info(f"Available masks: {self.available_masks}")
        logger.info(f"Available quality metrics: {self.available_quality}")
    
    def _resolve_zarr_array(self, group_path: str, array_name: str):
        """
        Resolve and return a Zarr array from a group/array binding.

        Args:
            group_path: Path to a Zarr group or directly to a Zarr array.
            array_name: Name of the array within the group, if applicable.

        Returns:
            zarr.Array corresponding to the requested dataset.
        """
        obj = self.zarr_root[group_path]
        if isinstance(obj, zarr.Array):
            return obj
        return obj[array_name]


    def _enforce_dimension(self, arr, *, dimension: str, name: str):
        """
        Enforce the declared dimensionality of a mask or quality array.

        Args:
            arr: Array returned from the data source after slicing and time handling.
            dimension: Declared output dimension ('spatial', 'spatio-temporal', or 'temporal').
            name: Name of the mask or quality metric (for error reporting).

        Returns:
            Array normalized to the expected dimensionality.
        """
        if dimension == "spatial":
            # allow degenerate leading singleton time, but normalize it away
            if getattr(arr, "ndim", None) == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"{name}: dimension=spatial expects [H,W], got shape={arr.shape}")
            return arr

        if dimension == "spatio-temporal":
            if arr.ndim != 3:
                raise ValueError(f"{name}: dimension=spatio-temporal expects [T,H,W], got shape={arr.shape}")
            return arr

        if dimension == "temporal":
            # allow [T,1,1] etc? I'd say no—keep it strict.
            if arr.ndim != 1:
                raise ValueError(f"{name}: dimension=temporal expects [T], got shape={arr.shape}")
            return arr

        raise ValueError(f"{name}: unknown dimension={dimension!r}")  


    def read_mask(
        self,
        mask_name: str,
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow] = None
    ) -> MaskResult:
        """
        Read a mask layer from the configuration.
        
        Args:
            mask_name: Name of mask in shared.masks (e.g., 'forest', 'aoi')
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window (for time-varying masks)
            
        Returns:
            MaskResult with boolean mask data
        """
        # Get mask configuration
        try:
            mask_config = self.config['shared']['masks'][mask_name]
        except KeyError:
            raise ValueError(
                f"Mask '{mask_name}' not found. Available: {self.available_masks}"
            )
        
        mask_type = mask_config.get('type', 'boolean')
        behavior = mask_config.get('behavior', {})
        dimension = mask_config.get('dimension')
        false_means_invalid = behavior.get('false_means_invalid', True)
        missing_value = behavior.get('missing', False)
        
        # Handle different mask types
        if mask_type == 'boolean' or mask_type == 'boolean_or_uint8':
            mask_data = self._read_boolean_mask(
                mask_config, spatial_window, temporal_window, missing_value
            )
        elif mask_type == 'threshold':
            mask_data = self._read_threshold_mask(
                mask_config, spatial_window, temporal_window
            )
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")
        
        mask_data = self._enforce_dimension(
            mask_data,
            dimension=dimension,
            name=mask_name,
        )
        return MaskResult(
            data=mask_data,
            mask_name=mask_name,
            mask_type=mask_type,
            false_means_invalid=false_means_invalid,
            metadata={
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'behavior': behavior
            }
        )
    
    def _read_boolean_mask(
        self,
        mask_config: Dict[str, Any],
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow],
        missing_value: bool
    ) -> np.ndarray:
        """
        Read a boolean mask from Zarr.
        
        Args:
            mask_config: Mask configuration from shared.masks
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window
            missing_value: Value to use for missing/OOB pixels
            
        Returns:
            Boolean mask array [H, W] or [T, H, W]
        """
        zarr_ref = mask_config.get('zarr', {})
        group_path = zarr_ref.get('group')
        array_name = zarr_ref.get('array')
        
        if not group_path or not array_name:
            raise ValueError(f"Mask configuration missing zarr group/array: {mask_config}")
        
        # Get Zarr array
        try:
            zgroup = self.zarr_root[group_path]
            zarray = self._resolve_zarr_array(group_path, array_name)
        except KeyError as e:
            logger.warning(f"Zarr array not found: {group_path}/{array_name}, using missing value")
            # Return array filled with missing_value
            if temporal_window:
                return np.full(
                    (temporal_window.window_length, spatial_window.height, spatial_window.width),
                    missing_value,
                    dtype=bool
                )
            else:
                return np.full(
                    (spatial_window.height, spatial_window.width),
                    missing_value,
                    dtype=bool
                )
        
        # Determine if array has temporal dimension
        array_shape = zarray.shape
        has_temporal = len(array_shape) == 3
        
        if has_temporal and temporal_window:
            # Read temporal mask [T, H, W]
            mask_data = self._read_temporal_array_with_padding(
                zarray, spatial_window, temporal_window, missing_value
            )
        else:
            # Read static mask [H, W]
            mask_data = self._read_static_array_with_padding(
                zarray, spatial_window, missing_value
            )
        
        # Convert to boolean if needed
        if mask_data.dtype != bool:
            # Handle uint8 masks where 0 = False, non-zero = True
            ok_if = mask_config.get('ok_if', {})
            if ok_if:
                op = ok_if.get('op', '>=')
                value = ok_if.get('value', 1)
                mask_data = self._apply_threshold(mask_data, op, value)
            else:
                mask_data = mask_data.astype(bool)
        
        return mask_data

    def _read_threshold_mask(
        self,
        mask_config: Dict[str, Any],
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow]
    ) -> np.ndarray:
        """
        Read a threshold-based mask (e.g., forest mask from probability).
        
        Args:
            mask_config: Mask configuration with 'source' and 'ok_if' fields
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window
            
        Returns:
            Boolean mask array [H, W] or [T, H, W]
        """
        source = mask_config.get('source', {})
        zarr_ref = source.get('zarr', {})
        group_path = zarr_ref.get('group')
        array_name = zarr_ref.get('array')
        
        if not group_path or not array_name:
            raise ValueError(f"Threshold mask missing source zarr reference: {mask_config}")
        
        # Get Zarr array
        try:
            zgroup = self.zarr_root[group_path]
            zarray = self._resolve_zarr_array(group_path, array_name)
        except KeyError:
            logger.warning(f"Source array not found: {group_path}/{array_name}, returning False mask")
            if temporal_window:
                return np.zeros(
                    (temporal_window.window_length, spatial_window.height, spatial_window.width),
                    dtype=bool
                )
            else:
                return np.zeros((spatial_window.height, spatial_window.width), dtype=bool)
        
        # Handle temporal selection
        time_config = mask_config.get('time', {})
        use_time = time_config.get('use', 'all')
        
        # Determine if array has temporal dimension
        array_shape = zarray.shape
        has_temporal = len(array_shape) == 3
        
        if has_temporal and temporal_window:
            # Read temporal data
            data = self._read_temporal_array_with_padding(
                zarray, spatial_window, temporal_window, fill_value=0.0
            )
            
            # Apply temporal selection if needed
            if use_time == 'last_in_window':
                # Use only the last timestep
                data = data[-1:, :, :]  # Keep dims as [1, H, W]
            elif use_time == 'first_in_window':
                data = data[:1, :, :]
            # Otherwise keep all timesteps
        else:
            # Read static data
            data = self._read_static_array_with_padding(
                zarray, spatial_window, fill_value=0.0
            )
        
        # Apply threshold
        ok_if = mask_config.get('ok_if', {})
        if not ok_if:
            raise ValueError(f"Threshold mask missing 'ok_if' specification: {mask_config}")
        
        op = ok_if.get('op', '>=')
        value = ok_if.get('value', 0.5)
        
        mask_data = self._apply_threshold(data, op, value)
        
        return mask_data
    
    def read_quality(
        self,
        quality_name: str,
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow] = None
    ) -> QualityResult:
        """
        Read a quality weight layer from the configuration.
        
        Args:
            quality_name: Name of quality metric in shared.quality
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window (for time-varying quality)
            
        Returns:
            QualityResult with float quality/weight data
        """
        # Get quality configuration
        try:
            quality_config = self.config['shared']['quality'][quality_name]
        except KeyError:
            raise ValueError(
                f"Quality metric '{quality_name}' not found. Available: {self.available_quality}"
            )
        
        quality_type = quality_config.get('type', 'float')
        
        # Handle different quality types
        if quality_type == 'float':
            quality_data = self._read_float_quality(
                quality_config, spatial_window, temporal_window
            )
        elif quality_type == 'expression':
            quality_data = self._read_expression_quality(
                quality_config, spatial_window, temporal_window
            )
        else:
            raise ValueError(f"Unsupported quality type: {quality_type}")
        
        return QualityResult(
            data=quality_data,
            quality_name=quality_name,
            quality_type=quality_type,
            metadata={
                'spatial_window': spatial_window,
                'temporal_window': temporal_window
            }
        )
    
    def _read_float_quality(
        self,
        quality_config: Dict[str, Any],
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow]
    ) -> np.ndarray:
        """
        Read a float quality weight from Zarr.
        
        Args:
            quality_config: Quality configuration from shared.quality
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window
            
        Returns:
            Float array [H, W] or [T, H, W]
        """
        zarr_ref = quality_config.get('zarr', {})
        group_path = zarr_ref.get('group')
        array_name = zarr_ref.get('array')
        
        if not group_path or not array_name:
            raise ValueError(f"Quality metric missing zarr reference: {quality_config}")
        
        missing = quality_config.get('missing', {})
        fill_value = missing.get('fill', 0.0)
        
        # Get Zarr array
        try:
            zgroup = self.zarr_root[group_path]
            zarray = self._resolve_zarr_array(group_path, array_name)
        except KeyError:
            logger.warning(f"Quality array not found: {group_path}/{array_name}, using fill value")
            if temporal_window:
                return np.full(
                    (temporal_window.window_length, spatial_window.height, spatial_window.width),
                    fill_value,
                    dtype=np.float32
                )
            else:
                return np.full(
                    (spatial_window.height, spatial_window.width),
                    fill_value,
                    dtype=np.float32
                )
        
        # Handle temporal selection
        time_config = quality_config.get('time', {})
        use_time = time_config.get('use', 'all')
        
        # Determine if array has temporal dimension
        array_shape = zarray.shape
        has_temporal = len(array_shape) == 3
        
        if has_temporal and temporal_window:
            # Read temporal data
            data = self._read_temporal_array_with_padding(
                zarray, spatial_window, temporal_window, fill_value
            )
            
            # Apply temporal selection if needed
            if use_time == 'last_in_window':
                data = data[-1:, :, :]  # Keep dims as [1, H, W]
            elif use_time == 'first_in_window':
                data = data[:1, :, :]
        else:
            # Read static data
            data = self._read_static_array_with_padding(
                zarray, spatial_window, fill_value
            )
        
        # Apply any transforms
        transform = quality_config.get('transform', {})
        if transform:
            data = self._apply_transform(data, transform)
        
        return data.astype(np.float32)
    
    def _read_expression_quality(
        self,
        quality_config: Dict[str, Any],
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow]
    ) -> np.ndarray:
        """
        Read a quality metric computed from an expression.
        
        Args:
            quality_config: Quality configuration with 'expression' field
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window
            
        Returns:
            Float array [H, W] or [T, H, W]
        """
        expression = quality_config.get('expression', '')
        if not expression:
            raise ValueError(f"Expression quality missing 'expression' field: {quality_config}")
        
        # For now, handle simple cases like "pow(p_forest, 2)"
        # Extract variable references from expression
        # This is a simplified implementation - could be enhanced with proper parsing
        
        if 'p_forest' in expression and expression.startswith('pow(p_forest,'):
            # Handle pow(p_forest, 2) specifically
            import re
            match = re.search(r'pow\(p_forest,\s*(\d+(?:\.\d+)?)\)', expression)
            if match:
                exponent = float(match.group(1))
                
                # Read p_forest quality metric
                p_forest_result = self.read_quality('p_forest', spatial_window, temporal_window)
                data = np.power(p_forest_result.data, exponent)
                
                # Apply normalization if specified
                normalize = quality_config.get('normalize', {})
                if normalize:
                    mode = normalize.get('mode', 'none')
                    if mode == 'mean1':
                        # Normalize to mean=1 (per sample in batch, but here just per array)
                        mean_val = np.mean(data[data > 0]) if np.any(data > 0) else 1.0
                        if mean_val > 0:
                            data = data / mean_val
                
                return data.astype(np.float32)
        
        raise NotImplementedError(f"Expression not yet supported: {expression}")
    
    def _read_temporal_array_with_padding(
        self,
        zarray: zarr.Array,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow,
        fill_value: Union[bool, float] = 0.0
    ) -> np.ndarray:
        """
        Read temporal array with zero-padding for out-of-bounds regions.
        
        Assumes zarr array shape is [T, H, W].
        
        Args:
            zarray: Zarr array to read from
            spatial_window: Spatial region to read
            temporal_window: Temporal window to read
            fill_value: Value to use for missing/OOB data
            
        Returns:
            Array with shape [T, H, W]
        """
        T_available, H_available, W_available = zarray.shape
        
        # Calculate temporal bounds
        # Assuming years are 1985-based indexing (adjust as needed)
        # For now, just read all available timesteps or pad
        T_requested = temporal_window.window_length
        
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
            dtype = bool if isinstance(fill_value, bool) else np.float32
            return np.full(
                (T_requested, spatial_window.height, spatial_window.width),
                fill_value,
                dtype=dtype
            )
        
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
        
        # Handle temporal dimension
        # For simplicity, read all available timesteps or pad
        if T_requested > T_available:
            # Need to pad temporally (use zeros for missing timesteps)
            t_start_actual = 0
            t_end_actual = T_available
            pad_temporal = T_requested - T_available
        else:
            # Read the last T_requested timesteps
            t_start_actual = max(0, T_available - T_requested)
            t_end_actual = T_available
            pad_temporal = 0
        
        # Read the actual data
        read_data = zarray[
            t_start_actual:t_end_actual,
            row_start_actual:row_end_actual,
            col_start_actual:col_end_actual
        ]
        
        # Apply spatial padding
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_data = np.pad(
                read_data,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=fill_value
            )
        else:
            padded_data = read_data
        
        # Apply temporal padding if needed
        if pad_temporal > 0:
            dtype = bool if isinstance(fill_value, bool) else np.float32
            temporal_pad = np.full(
                (pad_temporal, spatial_window.height, spatial_window.width),
                fill_value,
                dtype=dtype
            )
            padded_data = np.concatenate([temporal_pad, padded_data], axis=0)
        
        dtype = bool if isinstance(fill_value, bool) else np.float32
        return padded_data.astype(dtype)
    
    def _read_static_array_with_padding(
        self,
        zarray: zarr.Array,
        spatial_window: SpatialWindow,
        fill_value: Union[bool, float] = 0.0
    ) -> np.ndarray:
        """
        Read static array with zero-padding for out-of-bounds regions.
        
        Assumes zarr array shape is [H, W].
        
        Args:
            zarray: Zarr array to read from
            spatial_window: Spatial region to read
            fill_value: Value to use for missing/OOB data
            
        Returns:
            Array with shape [H, W]
        """
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
            dtype = bool if isinstance(fill_value, bool) else np.float32
            return np.full((spatial_window.height, spatial_window.width), fill_value, dtype=dtype)
        
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
        
        # Read the actual data
        read_data = zarray[row_start_actual:row_end_actual, col_start_actual:col_end_actual]
        
        # Apply padding
        padded_data = np.pad(
            read_data,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=fill_value
        )
        
        dtype = bool if isinstance(fill_value, bool) else np.float32
        return padded_data.astype(dtype)
    
    def _apply_threshold(self, data: np.ndarray, op: str, value: float) -> np.ndarray:
        """
        Apply threshold operation to data to create boolean mask.
        
        Args:
            data: Input array
            op: Operator ('>=', '>', '<=', '<', '==', '!=')
            value: Threshold value
            
        Returns:
            Boolean mask array
        """
        if op == '>=':
            return data >= value
        elif op == '>':
            return data > value
        elif op == '<=':
            return data <= value
        elif op == '<':
            return data < value
        elif op == '==':
            return data == value
        elif op == '!=':
            return data != value
        else:
            raise ValueError(f"Unsupported operator: {op}")
    
    def _apply_transform(self, data: np.ndarray, transform: Dict[str, Any]) -> np.ndarray:
        """
        Apply a transform to quality data.
        
        Args:
            data: Input array
            transform: Transform configuration (e.g., linear_rescale)
            
        Returns:
            Transformed array
        """
        transform_type = transform.get('type')
        
        if transform_type == 'linear_rescale':
            in_min = transform.get('in_min', 0.0)
            in_max = transform.get('in_max', 1.0)
            out_min = transform.get('out_min', 0.0)
            out_max = transform.get('out_max', 1.0)
            clamp = transform.get('clamp', False)
            
            # Linear rescale: (data - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
            in_range = in_max - in_min
            out_range = out_max - out_min
            
            if in_range != 0:
                data = (data - in_min) / in_range * out_range + out_min
            
            if clamp:
                data = np.clip(data, out_min, out_max)
            
            return data
        
        else:
            logger.warning(f"Unsupported transform type: {transform_type}, returning data unchanged")
            return data
    
    def read_all_masks(
        self,
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow] = None
    ) -> Dict[str, MaskResult]:
        """
        Read all available masks for given windows.
        
        Args:
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window
            
        Returns:
            Dictionary mapping mask_name -> MaskResult
        """
        results = {}
        
        for mask_name in self.available_masks:
            try:
                results[mask_name] = self.read_mask(mask_name, spatial_window, temporal_window)
            except Exception as e:
                logger.warning(f"Failed to read mask '{mask_name}': {e}")
        
        return results
    
    def read_all_quality(
        self,
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow] = None
    ) -> Dict[str, QualityResult]:
        """
        Read all available quality metrics for given windows.
        
        Args:
            spatial_window: Spatial window to read
            temporal_window: Optional temporal window
            
        Returns:
            Dictionary mapping quality_name -> QualityResult
        """
        results = {}
        
        for quality_name in self.available_quality:
            try:
                results[quality_name] = self.read_quality(
                    quality_name, spatial_window, temporal_window
                )
            except Exception as e:
                logger.warning(f"Failed to read quality metric '{quality_name}': {e}")
        
        return results
    
    def combine_masks(
        self,
        mask_results: List[MaskResult],
        operation: str = 'and'
    ) -> np.ndarray:
        """
        Combine multiple masks using logical operations.
        
        Args:
            mask_results: List of MaskResult objects to combine
            operation: 'and' or 'or'
            
        Returns:
            Combined boolean mask
        """
        if not mask_results:
            raise ValueError("No masks provided to combine")
        
        combined = mask_results[0].data.copy()
        
        for mask_result in mask_results[1:]:
            if mask_result.data.shape != combined.shape:
                raise ValueError(
                    f"Mask shape mismatch: {mask_result.data.shape} vs {combined.shape}"
                )
            
            if operation == 'and':
                # Account for false_means_invalid
                if mask_result.false_means_invalid:
                    combined = combined & mask_result.data
                else:
                    combined = combined & (~mask_result.data)
            elif operation == 'or':
                if mask_result.false_means_invalid:
                    combined = combined | mask_result.data
                else:
                    combined = combined | (~mask_result.data)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        return combined
    
    def read_quality_for_group(
        self,
        category: str,
        group_name: str,
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow] = None
    ) -> QualityResult:
        """
        Read quality weights per band, respecting each band's specific quality_weight list.
        
        Each band can have different quality weights. This method:
        1. For each band, reads and combines its specific quality weights
        2. Stacks them into shape [C, T, H, W] or [C, H, W]
        3. Returns all-ones for bands with no quality weights
        
        This is the CORRECT way to handle per-band quality weights when bands
        have different quality specifications.
        
        Args:
            category: 'temporal', 'static', or 'irregular'
            group_name: Input group name (e.g., 'ls8day')
            spatial_window: Spatial window
            temporal_window: Temporal window (for temporal groups)
            
        Returns:
            QualityResult with shape [C, T, H, W] or [C, H, W] where each
            channel has its band-specific quality weights applied
            
        Example:
            >>> # ls8day has different quality per band:
            >>> # - NDVI_summer_p95: [ls_summer_obs_weight]
            >>> # - NDVI_winter_max: [ls_winter_obs_weight]
            >>> # - NDVI_amplitude: [ls_summer_obs_weight, ls_winter_obs_weight]
            >>> 
            >>> quality = builder.read_quality_per_band('temporal', 'ls8day', spatial, temporal)
            >>> # quality.data[0] = summer weight only
            >>> # quality.data[1] = winter weight only
            >>> # quality.data[2] = summer * winter
        """
        # Get group configuration
        try:
            group = self.config['inputs'][category][group_name]
        except KeyError:
            raise ValueError(
                f"Group '{group_name}' not found in category '{category}'"
            )
        
        # Determine base shape
        if category == 'temporal' and temporal_window:
            base_shape = (temporal_window.window_length, spatial_window.height, spatial_window.width)
        else:
            base_shape = (spatial_window.height, spatial_window.width)
        
        # Build quality for each band
        per_band_quality = []
        
        for band in group.bands:
            if not band.quality_weight:
                # No quality weights for this band - use all ones
                band_quality = np.ones(base_shape, dtype=np.float32)
            else:
                # Read and combine quality weights for this band
                quality_results = []
                for quality_ref in band.quality_weight:
                    quality_name = quality_ref.split('.')[-1]
                    try:
                        quality_result = self.read_quality(quality_name, spatial_window, temporal_window)
                        quality_results.append(quality_result)
                    except Exception as e:
                        logger.warning(f"Failed to read quality '{quality_name}' for band '{band.name}': {e}")
                        fallback = np.ones(base_shape, dtype=np.float32)
                        quality_results.append(QualityResult(
                            data=fallback,
                            quality_name=quality_name,
                            quality_type='fallback',
                            metadata={'error': str(e)}
                        ))
                
                # Combine quality weights for this band
                if len(quality_results) == 1:
                    band_quality = quality_results[0].data
                else:
                    band_quality = quality_results[0].data.copy()
                    for qr in quality_results[1:]:
                        band_quality = band_quality * qr.data  # Multiply quality weights
            
            per_band_quality.append(band_quality)
        
        # Stack into [C, T, H, W] or [C, H, W]
        stacked_quality = np.stack(per_band_quality, axis=0)
        
        return QualityResult(
            data=stacked_quality,
            quality_name=f"{group_name}_per_band",
            quality_type='per_band',
            metadata={
                'group_name': group_name,
                'category': category,
                'num_bands': len(group.bands),
                'band_names': [b.name for b in group.bands],
                'spatial_window': spatial_window,
                'temporal_window': temporal_window
            }
        )
    
    def read_mask_for_group(
        self,
        category: str,
        group_name: str,
        spatial_window: SpatialWindow,
        temporal_window: Optional[TemporalWindow] = None
    ) -> MaskResult:
        """
        Read masks per band, respecting each band's specific mask list.
        
        Each band can have different masks. This method:
        1. For each band, reads and combines its specific masks
        2. Stacks them into shape [C, T, H, W] or [C, H, W]
        3. Returns all-True for bands with no masks
        
        This is the CORRECT way to handle per-band masks when bands
        have different mask specifications.
        
        Args:
            category: 'temporal', 'static', or 'irregular'
            group_name: Input group name
            spatial_window: Spatial window
            temporal_window: Temporal window (for temporal groups)
            
        Returns:
            MaskResult with shape [C, T, H, W] or [C, H, W] where each
            channel has its band-specific masks applied
        """
        # Get group configuration
        try:
            group = self.config['inputs'][category][group_name]
        except KeyError:
            raise ValueError(
                f"Group '{group_name}' not found in category '{category}'"
            )
        
        # Determine base shape
        if category == 'temporal' and temporal_window:
            base_shape = (temporal_window.window_length, spatial_window.height, spatial_window.width)
        else:
            base_shape = (spatial_window.height, spatial_window.width)
        
        # Build mask for each band
        per_band_masks = []
        
        for band in group.bands:
            if not band.mask:
                # No masks for this band - use all True
                band_mask = np.ones(base_shape, dtype=bool)
            else:
                # Read and combine masks for this band
                mask_results = []
                for mask_ref in band.mask:
                    mask_name = mask_ref.split('.')[-1]
                    try:
                        mask_result = self.read_mask(mask_name, spatial_window, temporal_window)
                        mask_results.append(mask_result)
                    except Exception as e:
                        logger.warning(f"Failed to read mask '{mask_name}' for band '{band.name}': {e}")
                        fallback = np.ones(base_shape, dtype=bool)
                        mask_results.append(MaskResult(
                            data=fallback,
                            mask_name=mask_name,
                            mask_type='fallback',
                            false_means_invalid=True,
                            metadata={'error': str(e)}
                        ))
                
                # Combine masks for this band
                if len(mask_results) == 1:
                    band_mask = mask_results[0].data
                else:
                    band_mask = self.combine_masks(mask_results, operation='and')
            
            per_band_masks.append(band_mask)
        
        # Stack into [C, T, H, W] or [C, H, W]
        stacked_masks = np.stack(per_band_masks, axis=0)
        
        return MaskResult(
            data=stacked_masks,
            mask_name=f"{group_name}_per_band",
            mask_type='per_band',
            false_means_invalid=True,
            metadata={
                'group_name': group_name,
                'category': category,
                'num_bands': len(group.bands),
                'band_names': [b.name for b in group.bands],
                'spatial_window': spatial_window,
                'temporal_window': temporal_window
            }
        )
    
    def read_mask_for_snapshot_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        year: int
    ) -> MaskResult:
        """
        Read masks for a snapshot group at a specific year.
        
        Each band can have different masks. This method:
        1. Gets bands for the specified year
        2. For each band, reads and combines its specific masks
        3. Stacks them into shape [C, H, W]
        4. Returns all-True for bands with no masks
        
        Args:
            group_name: Snapshot group name
            spatial_window: Spatial window
            year: Snapshot year to read
            
        Returns:
            MaskResult with shape [C, H, W] where each channel has its band-specific masks
        """
        # Get group configuration
        try:
            group = self.config['inputs']['snapshot'][group_name]
        except KeyError:
            raise ValueError(
                f"Snapshot group '{group_name}' not found"
            )
        
        # Get bands for this year
        try:
            bands = group.get_bands_for_year(year)
        except ValueError as e:
            raise ValueError(f"Cannot read masks for year {year}: {e}")
        
        # Determine base shape (no temporal dimension for snapshots)
        base_shape = (spatial_window.height, spatial_window.width)
        
        # Build mask for each band
        per_band_masks = []
        
        for band in bands:
            if not band.mask:
                # No masks for this band - use all True
                band_mask = np.ones(base_shape, dtype=bool)
            else:
                # Read and combine masks for this band
                mask_results = []
                for mask_ref in band.mask:
                    mask_name = mask_ref.split('.')[-1]
                    try:
                        # Snapshots are static (no temporal window)
                        mask_result = self.read_mask(mask_name, spatial_window, None)
                        mask_results.append(mask_result)
                    except Exception as e:
                        logger.warning(f"Failed to read mask '{mask_name}' for band '{band.name}': {e}")
                        fallback = np.ones(base_shape, dtype=bool)
                        mask_results.append(MaskResult(
                            data=fallback,
                            mask_name=mask_name,
                            mask_type='fallback',
                            false_means_invalid=True,
                            metadata={'error': str(e)}
                        ))
                
                # Combine masks for this band
                if len(mask_results) == 1:
                    band_mask = mask_results[0].data
                else:
                    band_mask = self.combine_masks(mask_results, operation='and')
            
            per_band_masks.append(band_mask)
        
        # Stack into [C, H, W]
        stacked_masks = np.stack(per_band_masks, axis=0)
        
        return MaskResult(
            data=stacked_masks,
            mask_name=f"{group_name}_snapshot_y{year}",
            mask_type='per_band',
            false_means_invalid=True,
            metadata={
                'group_name': group_name,
                'category': 'snapshot',
                'year': year,
                'num_bands': len(bands),
                'band_names': [b.name for b in bands],
                'spatial_window': spatial_window
            }
        )
    
    def read_quality_for_snapshot_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        year: int
    ) -> QualityResult:
        """
        Read quality weights for a snapshot group at a specific year.
        
        Args:
            group_name: Snapshot group name
            spatial_window: Spatial window
            year: Snapshot year to read
            
        Returns:
            QualityResult with shape [C, H, W]
        """
        # Get group configuration
        try:
            group = self.config['inputs']['snapshot'][group_name]
        except KeyError:
            raise ValueError(
                f"Snapshot group '{group_name}' not found"
            )
        
        # Get bands for this year
        try:
            bands = group.get_bands_for_year(year)
        except ValueError as e:
            raise ValueError(f"Cannot read quality for year {year}: {e}")
        
        # Determine base shape
        base_shape = (spatial_window.height, spatial_window.width)
        
        # Build quality for each band
        per_band_quality = []
        
        for band in bands:
            if not band.quality_weight:
                # No quality weights for this band - use all ones
                band_quality = np.ones(base_shape, dtype=np.float32)
            else:
                # Read and combine quality weights for this band
                quality_results = []
                for quality_ref in band.quality_weight:
                    quality_name = quality_ref.split('.')[-1]
                    try:
                        # Snapshots are static (no temporal window)
                        quality_result = self.read_quality(quality_name, spatial_window, None)
                        quality_results.append(quality_result)
                    except Exception as e:
                        logger.warning(f"Failed to read quality '{quality_name}' for band '{band.name}': {e}")
                        fallback = np.ones(base_shape, dtype=np.float32)
                        quality_results.append(QualityResult(
                            data=fallback,
                            quality_name=quality_name,
                            quality_type='fallback',
                            metadata={'error': str(e)}
                        ))
                
                # Combine quality weights for this band
                if len(quality_results) == 1:
                    band_quality = quality_results[0].data
                else:
                    band_quality = quality_results[0].data.copy()
                    for qr in quality_results[1:]:
                        band_quality = band_quality * qr.data  # Multiply quality weights
            
            per_band_quality.append(band_quality)
        
        # Stack into [C, H, W]
        stacked_quality = np.stack(per_band_quality, axis=0)
        
        return QualityResult(
            data=stacked_quality,
            quality_name=f"{group_name}_snapshot_y{year}",
            quality_type='per_band',
            metadata={
                'group_name': group_name,
                'category': 'snapshot',
                'year': year,
                'num_bands': len(bands),
                'band_names': [b.name for b in bands],
                'spatial_window': spatial_window
            }
        )
    
    def read_mask_for_irregular_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> MaskResult:
        """
        Read masks for an irregular group.
        
        Each band can have different masks. This method:
        1. Filters years within the temporal window
        2. For each band, reads and combines its specific masks
        3. Stacks them into shape [C, T_obs, H, W]
        4. Returns all-True for bands with no masks
        
        Args:
            group_name: Irregular group name
            spatial_window: Spatial window
            temporal_window: Temporal window to filter observations
            
        Returns:
            MaskResult with shape [C, T_obs, H, W]
        """
        # Get group configuration
        try:
            group = self.config['inputs']['irregular'][group_name]
        except KeyError:
            raise ValueError(
                f"Irregular group '{group_name}' not found"
            )
        
        # Filter years to those within the temporal window
        window_start_year = temporal_window.end_year - temporal_window.window_length + 1
        window_end_year = temporal_window.end_year
        
        years_in_window = [
            year for year in group.years
            if window_start_year <= year <= window_end_year
        ]
        
        if not years_in_window:
            # No observations - return empty [C, 0, H, W]
            empty_data = np.zeros(
                (len(group.bands), 0, spatial_window.height, spatial_window.width),
                dtype=bool
            )
            return MaskResult(
                data=empty_data,
                mask_name=f"{group_name}_irregular_empty",
                mask_type='per_band',
                false_means_invalid=True,
                metadata={
                    'group_name': group_name,
                    'category': 'irregular',
                    'num_bands': len(group.bands),
                    'years_in_window': [],
                    'num_observations': 0
                }
            )
        
        # Determine base shape (per observation)
        base_shape = (spatial_window.height, spatial_window.width)
        
        # Build mask for each band
        all_bands_masks = []
        
        for band in group.bands:
            # For irregular data, masks are typically static (same for all observations)
            if not band.mask:
                # No masks - use all True for all observations
                band_mask_all_obs = np.ones(
                    (len(years_in_window), spatial_window.height, spatial_window.width),
                    dtype=bool
                )
            else:
                # Read and combine masks for this band
                mask_results = []
                for mask_ref in band.mask:
                    mask_name = mask_ref.split('.')[-1]
                    try:
                        # Read static mask (no temporal window for mask itself)
                        mask_result = self.read_mask(mask_name, spatial_window, None)
                        mask_results.append(mask_result)
                    except Exception as e:
                        logger.warning(f"Failed to read mask '{mask_name}' for band '{band.name}': {e}")
                        fallback = np.ones(base_shape, dtype=bool)
                        mask_results.append(MaskResult(
                            data=fallback,
                            mask_name=mask_name,
                            mask_type='fallback',
                            false_means_invalid=True,
                            metadata={'error': str(e)}
                        ))
                
                # Combine masks for this band
                if len(mask_results) == 1:
                    band_mask_single = mask_results[0].data
                else:
                    band_mask_single = self.combine_masks(mask_results, operation='and')
                
                # Replicate mask across all observations [T_obs, H, W]
                band_mask_all_obs = np.stack([band_mask_single] * len(years_in_window), axis=0)
            
            all_bands_masks.append(band_mask_all_obs)
        
        # Stack into [C, T_obs, H, W]
        stacked_masks = np.stack(all_bands_masks, axis=0)
        
        return MaskResult(
            data=stacked_masks,
            mask_name=f"{group_name}_irregular",
            mask_type='per_band',
            false_means_invalid=True,
            metadata={
                'group_name': group_name,
                'category': 'irregular',
                'num_bands': len(group.bands),
                'band_names': [b.name for b in group.bands],
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'years_in_window': years_in_window,
                'num_observations': len(years_in_window)
            }
        )
    
    def read_quality_for_irregular_group(
        self,
        group_name: str,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> QualityResult:
        """
        Read quality weights for an irregular group.
        
        Args:
            group_name: Irregular group name
            spatial_window: Spatial window
            temporal_window: Temporal window to filter observations
            
        Returns:
            QualityResult with shape [C, T_obs, H, W]
        """
        # Get group configuration
        try:
            group = self.config['inputs']['irregular'][group_name]
        except KeyError:
            raise ValueError(
                f"Irregular group '{group_name}' not found"
            )
        
        # Filter years
        window_start_year = temporal_window.end_year - temporal_window.window_length + 1
        window_end_year = temporal_window.end_year
        
        years_in_window = [
            year for year in group.years
            if window_start_year <= year <= window_end_year
        ]
        
        if not years_in_window:
            # No observations - return empty [C, 0, H, W]
            empty_data = np.ones(
                (len(group.bands), 0, spatial_window.height, spatial_window.width),
                dtype=np.float32
            )
            return QualityResult(
                data=empty_data,
                quality_name=f"{group_name}_irregular_empty",
                quality_type='per_band',
                metadata={
                    'group_name': group_name,
                    'category': 'irregular',
                    'num_bands': len(group.bands),
                    'years_in_window': [],
                    'num_observations': 0
                }
            )
        
        # Determine base shape
        base_shape = (spatial_window.height, spatial_window.width)
        
        # Build quality for each band
        all_bands_quality = []
        
        for band in group.bands:
            if not band.quality_weight:
                # No quality weights - use all ones for all observations
                band_quality_all_obs = np.ones(
                    (len(years_in_window), spatial_window.height, spatial_window.width),
                    dtype=np.float32
                )
            else:
                # Read and combine quality weights for this band
                quality_results = []
                for quality_ref in band.quality_weight:
                    quality_name = quality_ref.split('.')[-1]
                    try:
                        # Read static quality (no temporal window for quality itself)
                        quality_result = self.read_quality(quality_name, spatial_window, None)
                        quality_results.append(quality_result)
                    except Exception as e:
                        logger.warning(f"Failed to read quality '{quality_name}' for band '{band.name}': {e}")
                        fallback = np.ones(base_shape, dtype=np.float32)
                        quality_results.append(QualityResult(
                            data=fallback,
                            quality_name=quality_name,
                            quality_type='fallback',
                            metadata={'error': str(e)}
                        ))
                
                # Combine quality weights for this band
                if len(quality_results) == 1:
                    band_quality_single = quality_results[0].data
                else:
                    band_quality_single = quality_results[0].data.copy()
                    for qr in quality_results[1:]:
                        band_quality_single = band_quality_single * qr.data
                
                # Replicate quality across all observations [T_obs, H, W]
                band_quality_all_obs = np.stack([band_quality_single] * len(years_in_window), axis=0)
            
            all_bands_quality.append(band_quality_all_obs)
        
        # Stack into [C, T_obs, H, W]
        stacked_quality = np.stack(all_bands_quality, axis=0)
        
        return QualityResult(
            data=stacked_quality,
            quality_name=f"{group_name}_irregular",
            quality_type='per_band',
            metadata={
                'group_name': group_name,
                'category': 'irregular',
                'num_bands': len(group.bands),
                'band_names': [b.name for b in group.bands],
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'years_in_window': years_in_window,
                'num_observations': len(years_in_window)
            }
        )


if __name__ == '__main__':
    # Example usage
    import sys
    import numpy as np

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
    registry = BindingsRegistry(config)
    
    # Initialize builder with registry
    builder = MaskBuilder(registry, config)
    
    # Define windows
    spatial_window = SpatialWindow.from_upper_left_and_hw(
        upper_left=(0, 0),
        hw=(256, 256)
    )
    
    temporal_window = TemporalWindow(
        end_year=2024,
        window_length=10
    )
    
    # Read a static mask
    print("\nReading AOI mask...")
    aoi_result = builder.read_mask('aoi', spatial_window)
    print(f"AOI mask shape: {aoi_result.data.shape}")
    print(f"AOI mask type: {aoi_result.mask_type}")
    print(f"False means invalid: {aoi_result.false_means_invalid}")
    
    # Read a threshold-based mask
    print("\nReading forest mask...")
    forest_result = builder.read_mask('forest', spatial_window, temporal_window)
    print(f"Forest mask shape: {forest_result.data.shape}")
    print(f"Forest mask type: {forest_result.mask_type}")
    
    # Read quality weights
    print("\nReading quality weights...")
    ls_summer_result = builder.read_quality('ls_summer_obs_weight', spatial_window, temporal_window)
    arr = ls_summer_result.data
    print(f"LS summer obs weight shape: {arr.shape}")
    print(f"LS summer obs weight range: [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")
    print(f"LS summer obs weight mean: {np.nanmean(arr):.3f}")
    
    # Read all masks
    print("\nReading all masks...")
    all_masks = builder.read_all_masks(spatial_window, temporal_window)
    print(f"Total masks read: {len(all_masks)}")
    for name, result in all_masks.items():
        print(f"  {name}: shape={result.data.shape}, type={result.mask_type}")
