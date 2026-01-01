"""
Derived Feature Builder for Forest Representation Model

Generates derived features from raw inputs according to the bindings configuration.
Returns unnormalized derived data with corresponding masks and quality weights.

Handles:
- Temporal position encoding: p_t, c_t channels
- Temporal differences: year-to-year changes (ls8_delta)
- Spatial gradients: Sobel operators at multiple scales (TODO)
- Mask and quality weight inheritance from source bands
- Integration with DataReader and MaskBuilder
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)

@dataclass
class DerivedFeatureResult:
    """Result from building a derived feature"""
    data: np.ndarray              # Raw derived data (unnormalized)
    band_names: List[str]         # Names of derived bands
    feature_name: str             # Name of derived feature
    source_category: Optional[str] # Source category if applicable
    source_group: Optional[str]   # Source group if applicable
    metadata: Dict[str, Any]      # Feature-specific metadata
    mask: Optional[np.ndarray] = None      # Boolean mask (same shape as data)
    quality: Optional[np.ndarray] = None   # Float quality weights (same shape as data)


class DerivedFeatureBuilder:
    """
    Build derived features from raw inputs.
    
    Handles:
    - Temporal position encoding: p_t, c_t channels
    - Temporal differences: year-to-year changes (ls8_delta)
    - Spatial gradients: directional Sobel at multiple scales (TODO)
    - Mask and quality weight inheritance from source bands
    - On-the-fly computation (not stored in Zarr)
    """
    
    def __init__(self, config: Dict[str, Any], data_reader, mask_builder):
        """
        Initialize DerivedFeatureBuilder.
        
        Args:
            config: Parsed bindings configuration dictionary
            data_reader: DataReader instance for reading source data
            mask_builder: MaskBuilder instance for reading masks/quality
        """
        self.config = config
        self.reader = data_reader
        self.mask_builder = mask_builder
        
        # Get derived config
        self.derived_config = config.get('derived', {})
        
        # Track enabled features
        self.enabled_features = self._get_enabled_features()
        
        logger.info(f"Initialized DerivedFeatureBuilder with {len(self.enabled_features)} enabled features")
    
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled derived features"""
        enabled = []
        for feature_name, feature_config in self.derived_config.items():
            # Skip YAML anchors (start with _)
            if feature_name.startswith('_'):
                continue
            
            # Check if enabled (default: True if 'enabled' key not present)
            if feature_config.get('enabled', True):
                enabled.append(feature_name)
        
        return enabled
    
    def build_derived_feature(
        self,
        feature_name: str,
        spatial_window,
        temporal_window=None
    ) -> DerivedFeatureResult:
        """
        Build a single derived feature.
        
        Args:
            feature_name: Name of derived feature (e.g., 'temporal_position')
            spatial_window: SpatialWindow for reading source data
            temporal_window: Optional TemporalWindow for temporal features
            
        Returns:
            DerivedFeatureResult with data, masks, quality weights
        """
        if feature_name not in self.derived_config:
            raise ValueError(f"Derived feature '{feature_name}' not found in config")
        
        feature_config = self.derived_config[feature_name]
        
        # Check if enabled
        if not feature_config.get('enabled', True):
            raise ValueError(f"Derived feature '{feature_name}' is disabled")
        
        logger.debug(f"Building derived feature: {feature_name}")
        
        # Dispatch to appropriate builder
        if feature_name == 'temporal_position':
            return self._build_temporal_position(feature_config, spatial_window, temporal_window)
        
        elif feature_name == 'ls8_delta':
            return self._build_temporal_delta(feature_config, spatial_window, temporal_window)
        
        elif feature_name in ['spectral_grads_window', 'spectral_grads_last', 'topo_grads']:
            return self._build_gradients(feature_name, feature_config, spatial_window, temporal_window)
        
        else:
            raise NotImplementedError(f"Derived feature '{feature_name}' not implemented yet")
    
    def build_all_derived_features(
        self,
        spatial_window,
        temporal_window=None
    ) -> Dict[str, DerivedFeatureResult]:
        """
        Build all enabled derived features.
        
        Args:
            spatial_window: SpatialWindow for reading source data
            temporal_window: Optional TemporalWindow for temporal features
            
        Returns:
            Dict mapping feature_name -> DerivedFeatureResult
        """
        results = {}
        
        for feature_name in self.enabled_features:
            try:
                result = self.build_derived_feature(
                    feature_name,
                    spatial_window,
                    temporal_window
                )
                results[feature_name] = result
                
            except Exception as e:
                logger.warning(f"Failed to build derived feature '{feature_name}': {e}")
                continue
        
        return results
    
    def _build_temporal_position(
        self,
        feature_config: Dict[str, Any],
        spatial_window,
        temporal_window
    ) -> DerivedFeatureResult:
        """
        Build temporal position encoding channels.
        
        Creates two channels per timestep:
        - p_t: Normalized position in [0, 1]  (t / (T-1))
        - c_t: Centered position in [-1, 1]   ((t - (T-1)/2) / ((T-1)/2))
        
        Shape: [2, T, H, W] where T = window_length
        """
        if temporal_window is None:
            raise ValueError("temporal_position requires a temporal window")
        
        T = temporal_window.window_length
        H = spatial_window.height
        W = spatial_window.width
        
        # Parse channel configs
        channels = feature_config['channels']
        channel_names = [ch['name'] for ch in channels]
        
        # Compute position values
        t = np.arange(T, dtype=np.float32)
        
        # p_t: normalized position [0, 1]
        if T > 1:
            p_t = t / (T - 1)
        else:
            p_t = np.array([0.0], dtype=np.float32)
        
        # c_t: centered position [-1, 1]
        if T > 1:
            c_t = (t - (T - 1) / 2) / ((T - 1) / 2)
        else:
            c_t = np.array([0.0], dtype=np.float32)
        
        # Expand to spatial dimensions [T, H, W]
        p_t_spatial = np.broadcast_to(p_t[:, None, None], (T, H, W)).copy()
        c_t_spatial = np.broadcast_to(c_t[:, None, None], (T, H, W)).copy()
        
        # Stack channels [2, T, H, W]
        data = np.stack([p_t_spatial, c_t_spatial], axis=0)
        
        # No masks or quality weights for temporal position
        # (all pixels equally valid)
        mask = np.ones((2, T, H, W), dtype=bool)
        quality = np.ones((2, T, H, W), dtype=np.float32)
        
        return DerivedFeatureResult(
            data=data,
            band_names=channel_names,
            feature_name='temporal_position',
            source_category=None,
            source_group=None,
            metadata={
                'window_length': T,
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'formulas': {
                    'p_t': 't / (T - 1)',
                    'c_t': '(t - (T-1)/2) / ((T-1)/2)'
                }
            },
            mask=mask,
            quality=quality
        )
    
    def _build_temporal_delta(
        self,
        feature_config: Dict[str, Any],
        spatial_window,
        temporal_window
    ) -> DerivedFeatureResult:
        """
        Build temporal differences: x[t] - x[t-1]
        
        Computes first differences along the time axis for selected bands.
        
        Shape: [C, T-1, H, W] where C = number of delta bands, T-1 = number of differences
        """
        if temporal_window is None:
            raise ValueError("ls8_delta requires a temporal window")
        
        # Parse source reference (e.g., 'inputs.temporal.ls8day')
        source_ref = feature_config['source']
        category, group_name = self._parse_source_reference(source_ref)
        
        # Read source data
        source_result = self.reader.read_temporal_group(
            group_name,
            spatial_window,
            temporal_window,
            return_full_temporal=True  # Need full time series
        )
        
        # Get source band indices
        delta_bands = feature_config['bands']
        source_band_names = source_result.band_names
        
        delta_data_list = []
        delta_mask_list = []
        delta_quality_list = []
        delta_names = []
        
        for delta_band in delta_bands:
            from_band = delta_band['from']
            delta_name = delta_band['name']
            
            # Find source band index
            try:
                source_idx = source_band_names.index(from_band)
            except ValueError:
                logger.warning(f"Source band '{from_band}' not found, skipping delta")
                continue
            
            # Get source data [T, H, W]
            source_data = source_result.data[source_idx]
            
            # Compute difference along time axis: x[t] - x[t-1] for t=1..T-1
            # Results in [T-1, H, W]
            delta_data = np.diff(source_data, axis=0)
            
            delta_data_list.append(delta_data)
            delta_names.append(delta_name)
            
            # Inherit mask and quality from source band
            # Also slice to match delta shape [T-1, H, W]
            source_mask = self.mask_builder.read_mask_for_group(
                category, group_name, spatial_window, temporal_window
            )
            source_quality = self.mask_builder.read_quality_for_group(
                category, group_name, spatial_window, temporal_window
            )
            
            # Source shapes assumed: [C, T, H, W]
            m = source_mask.data[source_idx]        # [T, H, W] (bool)
            q = source_quality.data[source_idx]     # [T, H, W] (float)
            
            # Delta at t uses x[t] and x[t-1], so require both valid:
            # delta timesteps correspond to t = 1..T-1
            delta_m = m[1:] & m[:-1]                # [T-1, H, W]
            delta_q = q[1:] * q[:-1]                # [T-1, H, W]
            
            delta_mask_list.append(delta_m)
            delta_quality_list.append(delta_q)
        
        # Stack all delta bands [C, T-1, H, W]
        data = np.stack(delta_data_list, axis=0)
        mask = np.stack(delta_mask_list, axis=0)
        quality = np.stack(delta_quality_list, axis=0)
        
        return DerivedFeatureResult(
            data=data,
            band_names=delta_names,
            feature_name='ls8_delta',
            source_category=category,
            source_group=group_name,
            metadata={
                'operation': 'diff',
                'axis': 'time',
                'spatial_window': spatial_window,
                'temporal_window': temporal_window,
                'source_bands': [b['from'] for b in delta_bands],
                'num_timesteps': data.shape[1]  # T-1
            },
            mask=mask,
            quality=quality
        )
    
    def _build_gradients(
        self,
        feature_name: str,
        feature_config: Dict[str, Any],
        spatial_window,
        temporal_window
    ) -> DerivedFeatureResult:
        """
        Build spatial gradient features.
        
        Supports:
        - spectral_grads_window: Gradients from 10-year mean
        - spectral_grads_last: Gradients from last timestep
        - topo_grads: Gradients from topography
        
        Returns directional gradients at multiple scales.
        """
        # This is a complex multi-step pipeline
        # For now, return a placeholder that shows the structure
        
        raise NotImplementedError(
            f"Gradient feature '{feature_name}' not yet implemented. "
            "This requires implementing the full gradient pipeline with:"
            "\n  1. Temporal aggregation (mean/select)"
            "\n  2. Per-band Sobel gradients"
            "\n  3. Multi-scale computation"
            "\n  4. Cross-band compositing"
            "\n  5. Scale selection and export"
        )
    
    def _parse_source_reference(self, source_ref: str) -> Tuple[str, str]:
        """
        Parse source reference like 'inputs.temporal.ls8day'
        
        Returns:
            (category, group_name) tuple
        """
        parts = source_ref.split('.')
        
        if len(parts) < 3 or parts[0] != 'inputs':
            raise ValueError(f"Invalid source reference: {source_ref}")
        
        category = parts[1]  # 'temporal', 'static', etc.
        group_name = parts[2]  # 'ls8day', 'topo', etc.
        
        return category, group_name
    
    def _compute_sobel_gradients(
        self,
        data: np.ndarray,
        directions: List[int] = [0, 45, 90, 135],
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Compute Sobel gradients in multiple directions.
        
        Args:
            data: Input array [H, W]
            directions: List of angles in degrees
            kernel_size: Size of Sobel kernel (3 or 5)
            
        Returns:
            Gradients [D, H, W] where D = len(directions)
        """
        H, W = data.shape
        gradients = []
        
        for angle in directions:
            if kernel_size == 3:
                # Standard 3x3 Sobel
                if angle == 0:  # Vertical edges
                    gx = ndimage.sobel(data, axis=1)
                    gradients.append(np.abs(gx))
                elif angle == 90:  # Horizontal edges
                    gy = ndimage.sobel(data, axis=0)
                    gradients.append(np.abs(gy))
                elif angle == 45 or angle == 135:
                    # Diagonal edges (approximate)
                    gx = ndimage.sobel(data, axis=1)
                    gy = ndimage.sobel(data, axis=0)
                    if angle == 45:
                        grad = (gx + gy) / np.sqrt(2)
                    else:
                        grad = (gx - gy) / np.sqrt(2)
                    gradients.append(np.abs(grad))
            else:
                raise NotImplementedError(f"Kernel size {kernel_size} not implemented")
        
        return np.stack(gradients, axis=0)
    
    def get_derived_band_metadata(
        self,
        feature_name: str,
        band_name: str
    ) -> Dict[str, Any]:
        """
        Get metadata for a specific derived band.
        
        Args:
            feature_name: Name of derived feature
            band_name: Name of band within feature
            
        Returns:
            Dict with band configuration
        """
        if feature_name not in self.derived_config:
            raise ValueError(f"Derived feature '{feature_name}' not found")
        
        feature_config = self.derived_config[feature_name]
        
        # For temporal_position
        if feature_name == 'temporal_position':
            channels = feature_config['channels']
            for ch in channels:
                if ch['name'] == band_name:
                    return {
                        'name': ch['name'],
                        'formula': ch['formula'],
                        'dtype': ch['dtype'],
                        'norm': feature_config['injection']['norm']
                    }
        
        # For ls8_delta
        elif feature_name == 'ls8_delta':
            bands = feature_config['bands']
            for band in bands:
                if band['name'] == band_name:
                    return {
                        'name': band['name'],
                        'from': band['from'],
                        'norm': band['norm'],
                        'operation': 'diff'
                    }
        
        raise ValueError(f"Band '{band_name}' not found in feature '{feature_name}'")


class DerivedFeatureIntegrator:
    """
    Helper class to integrate derived features with raw inputs.
    
    Handles:
    - Concatenating derived features to raw inputs
    - Tracking band names and metadata
    - Combining masks and quality weights
    """
    
    def __init__(
        self,
        data_reader,
        mask_builder,
        derived_builder: DerivedFeatureBuilder
    ):
        self.reader = data_reader
        self.mask_builder = mask_builder
        self.derived_builder = derived_builder
    
    def read_with_derived(
        self,
        category: str,
        group_name: str,
        spatial_window,
        temporal_window=None,
        include_derived: bool = True
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Read raw data and optionally concatenate derived features.
        
        Args:
            category: Input category ('temporal', 'static', etc.)
            group_name: Group name ('ls8day', 'topo', etc.)
            spatial_window: SpatialWindow
            temporal_window: Optional TemporalWindow
            include_derived: Whether to include derived features
            
        Returns:
            (data, band_names, mask, quality) where derived features are
            concatenated along channel dimension
        """
        # Read raw data
        if category == 'temporal':
            raw_result = self.reader.read_temporal_group(
                group_name, spatial_window, temporal_window, True
            )
        elif category == 'static':
            raw_result = self.reader.read_static_group(
                group_name, spatial_window
            )
        else:
            raise ValueError(f"Unsupported category: {category}")
        
        # Read masks and quality
        if category == 'temporal':
            mask_result = self.mask_builder.read_mask_for_group(
                category, group_name, spatial_window, temporal_window
            )
            quality_result = self.mask_builder.read_quality_for_group(
                category, group_name, spatial_window, temporal_window
            )
        else:
            mask_result = self.mask_builder.read_mask_for_group(
                category, group_name, spatial_window
            )
            quality_result = self.mask_builder.read_quality_for_group(
                category, group_name, spatial_window
            )
        
        data = raw_result.data
        band_names = raw_result.band_names.copy()
        mask = mask_result.data
        quality = quality_result.data
        
        if not include_derived:
            return data, band_names, mask, quality
        
        # Check if this group has derived features to inject
        derived_to_inject = self._get_derived_for_group(category, group_name)
        
        for derived_name in derived_to_inject:
            try:
                derived_result = self.derived_builder.build_derived_feature(
                    derived_name, spatial_window, temporal_window
                )
                
                # Concatenate along channel dimension
                data = np.concatenate([data, derived_result.data], axis=0)
                band_names.extend(derived_result.band_names)
                mask = np.concatenate([mask, derived_result.mask], axis=0)
                quality = np.concatenate([quality, derived_result.quality], axis=0)
                
            except Exception as e:
                logger.warning(f"Failed to inject derived feature '{derived_name}': {e}")
        
        return data, band_names, mask, quality
    
    def _get_derived_for_group(
        self,
        category: str,
        group_name: str
    ) -> List[str]:
        """
        Get list of derived features that should be injected into this group.
        
        For temporal_position, checks the 'injection.apply_to' list.
        """
        derived_features = []
        
        # Check temporal_position
        tp_config = self.derived_builder.derived_config.get('temporal_position', {})
        if tp_config.get('enabled', True):
            apply_to = tp_config.get('injection', {}).get('apply_to', [])
            group_ref = f'inputs.{category}.{group_name}'
            
            if group_ref in apply_to:
                derived_features.append('temporal_position')
        
        return derived_features


if __name__ == '__main__':
    """Example usage of DerivedFeatureBuilder"""
    import sys
    from pathlib import Path
    
    print("DerivedFeatureBuilder - Example Usage")
    print("=" * 80)
    print()
    print("This module provides:")
    print("  1. Temporal position encoding (p_t, c_t)")
    print("  2. Temporal differences (ls8_delta)")
    print("  3. Spatial gradients (when enabled)")
    print()
    print("Usage in your pipeline:")
    print()
    print("  from derived_features import DerivedFeatureBuilder")
    print("  from data_reader import DataReader")
    print("  from mask_builder import MaskBuilder")
    print()
    print("  # Initialize")
    print("  reader = DataReader(config)")
    print("  mask_builder = MaskBuilder(registry, config)")
    print("  derived_builder = DerivedFeatureBuilder(config, reader, mask_builder)")
    print()
    print("  # Build single feature")
    print("  result = derived_builder.build_derived_feature(")
    print("      'temporal_position',")
    print("      spatial_window,")
    print("      temporal_window")
    print("  )")
    print()
    print("  # Build all enabled features")
    print("  all_derived = derived_builder.build_all_derived_features(")
    print("      spatial_window,")
    print("      temporal_window")
    print("  )")
    print()
    print("  # Integrate with raw inputs")
    print("  integrator = DerivedFeatureIntegrator(reader, mask_builder, derived_builder)")
    print("  data, names, mask, quality = integrator.read_with_derived(")
    print("      'temporal', 'ls8day', spatial_window, temporal_window")
    print("  )")
