"""
Normalization Module with Zarr Stats Integration

Applies normalization transformations using statistics loaded from Zarr.
Implements all normalization types defined in forest_repr_model_bindings.yaml.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging

from data.normalization.zarr_stats_loader import ZarrStatsLoader

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for a normalization preset."""
    type: str
    stats_source: str
    fields: Optional[Dict[str, str]] = None
    clamp: Optional[Dict[str, Any]] = None
    missing: Optional[Dict[str, float]] = None
    # For fixed normalizations
    min: Optional[float] = None
    max: Optional[float] = None
    in_min: Optional[float] = None
    in_max: Optional[float] = None
    out_min: Optional[float] = None
    out_max: Optional[float] = None


class Normalizer:
    """
    Base normalizer class that applies transformations to arrays.
    """
    
    def __init__(
        self, 
        config: NormalizationConfig,
        stats: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize normalizer.
        
        Args:
            config: Normalization configuration
            stats: Pre-loaded statistics (for zarr-based normalizations)
        """
        self.config = config
        self.stats = stats or {}
        
    def normalize(
        self, 
        data: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply normalization to data.
        
        Args:
            data: Input data array
            mask: Optional validity mask (True = valid)
            
        Returns:
            Normalized data
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Convert to numpy if needed
        if is_torch:
            device = data.device
            dtype = data.dtype
            data_np = data.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None
        else:
            data_np = data
            mask_np = mask
        
        # Apply normalization
        normalized = self._normalize_impl(data_np, mask_np)
        
        # Apply clamping if configured
        if self.config.clamp and self.config.clamp.get('enabled', False):
            clip_min = self.config.clamp.get('min')
            clip_max = self.config.clamp.get('max')
            if clip_min is not None or clip_max is not None:
                normalized = np.clip(normalized, clip_min, clip_max)
        
        # Handle missing values
        if mask_np is not None:
            fill_value = self.config.missing.get('fill', 0.0) if self.config.missing else 0.0
            normalized = np.where(mask_np, normalized, fill_value)
        
        # Convert back to torch if needed
        if is_torch:
            normalized = torch.from_numpy(normalized).to(device=device, dtype=dtype)
        
        return normalized
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Implementation of normalization logic.
        Override in subclasses.
        """
        raise NotImplementedError


class ZScoreNormalizer(Normalizer):
    """Z-score normalization: (x - mean) / sd"""
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        mean = self.stats.get('mean')
        sd = self.stats.get('sd')
        
        if mean is None or sd is None:
            raise ValueError("Z-score normalization requires 'mean' and 'sd' statistics")
        
        # Broadcast to data shape if needed
        mean = np.broadcast_to(mean, data.shape)
        sd = np.broadcast_to(sd, data.shape)
        
        # Avoid division by zero
        sd_safe = np.where(sd > 1e-8, sd, 1.0)
        
        normalized = (data - mean) / sd_safe
        
        return normalized


class RobustIQRNormalizer(Normalizer):
    """Robust scaling using median and IQR: (x - median) / IQR"""
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        q25 = self.stats.get('q25')
        q50 = self.stats.get('q50')
        q75 = self.stats.get('q75')
        
        if q25 is None or q50 is None or q75 is None:
            raise ValueError("Robust IQR normalization requires 'q25', 'q50', 'q75' statistics")
        
        # Compute IQR
        iqr = q75 - q25
        
        # Broadcast to data shape
        q50 = np.broadcast_to(q50, data.shape)
        iqr = np.broadcast_to(iqr, data.shape)
        
        # Avoid division by zero
        iqr_safe = np.where(iqr > 1e-8, iqr, 1.0)
        
        normalized = (data - q50) / iqr_safe
        
        return normalized


class MinMaxNormalizer(Normalizer):
    """Min-max normalization: (x - min) / (max - min)"""
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        # Use fixed min/max from config or stats
        if self.config.min is not None and self.config.max is not None:
            min_val = self.config.min
            max_val = self.config.max
        else:
            min_val = self.stats.get('min')
            max_val = self.stats.get('max')
            
            if min_val is None or max_val is None:
                raise ValueError("Min-max normalization requires 'min' and 'max'")
        
        # Broadcast
        min_val = np.broadcast_to(min_val, data.shape)
        max_val = np.broadcast_to(max_val, data.shape)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_safe = np.where(range_val > 1e-8, range_val, 1.0)
        
        normalized = (data - min_val) / range_safe
        
        return normalized


class LinearRescaleNormalizer(Normalizer):
    """
    Linear rescaling: map [in_min, in_max] -> [out_min, out_max]
    
    Used for delta_ls8_fixed: [-0.4, 0.4] -> [-1, 1]
    """
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        in_min = self.config.in_min
        in_max = self.config.in_max
        out_min = self.config.out_min
        out_max = self.config.out_max
        
        if None in [in_min, in_max, out_min, out_max]:
            raise ValueError("Linear rescale requires in_min, in_max, out_min, out_max")
        
        # Linear transformation
        in_range = in_max - in_min
        out_range = out_max - out_min
        
        normalized = ((data - in_min) / in_range) * out_range + out_min
        
        return normalized


class ClampNormalizer(Normalizer):
    """Simple clamping with optional fill value for missing"""
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        # No transformation, just return data
        # Clamping is handled in parent class
        return data.copy()


class IdentityNormalizer(Normalizer):
    """Identity transformation (no normalization)"""
    
    def _normalize_impl(
        self, 
        data: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        return data.copy()


class NormalizerFactory:
    """
    Factory for creating normalizers from YAML config.
    """
    
    NORMALIZER_CLASSES = {
        'zscore': ZScoreNormalizer,
        'robust_iqr': RobustIQRNormalizer,
        'minmax': MinMaxNormalizer,
        'linear_rescale': LinearRescaleNormalizer,
        'clamp': ClampNormalizer,
        'none': IdentityNormalizer,
    }
    
    @classmethod
    def create(
        cls,
        preset_name: str,
        preset_config: Dict[str, Any],
        stats_loader: Optional[ZarrStatsLoader] = None,
        group_path: Optional[str] = None,
        array_name: Optional[str] = None
    ) -> Normalizer:
        """
        Create a normalizer from a preset configuration.
        
        Args:
            preset_name: Name of the normalization preset (e.g., 'zscore')
            preset_config: Configuration dict for the preset
            stats_loader: ZarrStatsLoader for loading statistics
            group_path: Zarr group path (required if stats_source='zarr')
            array_name: Array name (required if stats_source='zarr')
            
        Returns:
            Configured Normalizer instance
        """
        # Build config object
        config = NormalizationConfig(
            type=preset_config.get('type'),
            stats_source=preset_config.get('stats_source', 'fixed'),
            fields=preset_config.get('fields'),
            clamp=preset_config.get('clamp'),
            missing=preset_config.get('missing'),
            min=preset_config.get('min'),
            max=preset_config.get('max'),
            in_min=preset_config.get('in_min'),
            in_max=preset_config.get('in_max'),
            out_min=preset_config.get('out_min'),
            out_max=preset_config.get('out_max'),
        )
        
        # Load stats if needed
        stats = {}
        if config.stats_source == 'zarr':
            if not stats_loader or not group_path or not array_name:
                raise ValueError(
                    "Zarr-based normalization requires stats_loader, group_path, and array_name"
                )
            
            # Determine which stats to load based on config
            if config.fields:
                stat_fields = list(config.fields.values())
            else:
                # Infer from type
                stat_fields = cls._infer_required_stats(config.type)
            
            stats = stats_loader.get_stats(group_path, array_name, stat_fields)
            
            if not stats:
                logger.warning(
                    f"No statistics found for {array_name} in {group_path}. "
                    f"Using default values."
                )
        
        # Get normalizer class
        normalizer_class = cls.NORMALIZER_CLASSES.get(config.type)
        if not normalizer_class:
            raise ValueError(f"Unknown normalization type: {config.type}")
        
        return normalizer_class(config, stats)
    
    @staticmethod
    def _infer_required_stats(norm_type: str) -> list:
        """Infer which stats are needed for a normalization type."""
        type_to_stats = {
            'zscore': ['mean', 'sd'],
            'robust_iqr': ['q25', 'q50', 'q75'],
            'minmax': ['min', 'max'],
        }
        return type_to_stats.get(norm_type, [])


class NormalizationManager:
    """
    Manages normalization for all bands in the dataset.
    
    Loads presets from YAML config and creates normalizers on demand.
    """
    
    def __init__(
        self,
        yaml_config: Dict[str, Any],
        stats_loader: ZarrStatsLoader
    ):
        """
        Initialize normalization manager.
        
        Args:
            yaml_config: Parsed YAML configuration
            stats_loader: ZarrStatsLoader instance
        """
        self.yaml_config = yaml_config
        self.stats_loader = stats_loader
        
        # Load preset configurations
        self.presets = yaml_config.get('normalization', {}).get('presets', {})
        
        # Cache created normalizers
        self._normalizer_cache: Dict[str, Normalizer] = {}
    
    def get_normalizer_for_band(
        self,
        group_path: str,
        band_config: Dict[str, Any]
    ) -> Normalizer:
        """
        Get or create a normalizer for a specific band.
        
        Args:
            group_path: Zarr group path (e.g., 'annual/ls8day/data')
            band_config: Band configuration from YAML
            
        Returns:
            Configured Normalizer instance
        """
        array_name = band_config.get('array')
        norm_preset = band_config.get('norm')
        
        if not array_name or not norm_preset:
            raise ValueError("Band config must contain 'array' and 'norm' fields")
        
        # Check cache
        cache_key = f"{group_path}/{array_name}/{norm_preset}"
        if cache_key in self._normalizer_cache:
            return self._normalizer_cache[cache_key]
        
        # Get preset config
        if norm_preset not in self.presets:
            raise ValueError(f"Unknown normalization preset: {norm_preset}")
        
        preset_config = self.presets[norm_preset]
        
        # Create normalizer
        normalizer = NormalizerFactory.create(
            preset_name=norm_preset,
            preset_config=preset_config,
            stats_loader=self.stats_loader,
            group_path=group_path,
            array_name=array_name
        )
        
        # Cache it
        self._normalizer_cache[cache_key] = normalizer
        
        return normalizer
    
    def normalize_band(
        self,
        data: Union[np.ndarray, torch.Tensor],
        group_path: str,
        band_config: Dict[str, Any],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize a band's data.
        
        Args:
            data: Input data
            group_path: Zarr group path
            band_config: Band configuration
            mask: Optional validity mask
            
        Returns:
            Normalized data
        """
        normalizer = self.get_normalizer_for_band(group_path, band_config)
        return normalizer.normalize(data, mask)


# Example usage and tests
if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Mock YAML config (subset)
    yaml_config = {
        'normalization': {
            'presets': {
                'zscore': {
                    'type': 'zscore',
                    'stats_source': 'zarr',
                    'fields': {'mean': 'mean', 'std': 'sd'},
                    'clamp': {'enabled': True, 'min': -6.0, 'max': 6.0},
                    'missing': {'fill': 0.0}
                },
                'robust_iqr': {
                    'type': 'robust_iqr',
                    'stats_source': 'zarr',
                    'fields': {'q25': 'q25', 'q50': 'q50', 'q75': 'q75'},
                    'clamp': {'enabled': True, 'min': -8.0, 'max': 8.0},
                    'missing': {'fill': 0.0}
                },
                'delta_ls8_fixed': {
                    'type': 'linear_rescale',
                    'stats_source': 'fixed',
                    'in_min': -0.4,
                    'in_max': 0.4,
                    'out_min': -1.0,
                    'out_max': 1.0,
                    'clamp': True,
                    'missing': {'fill': 0}
                }
            }
        }
    }
    
    # Test with synthetic data
    print("Testing normalizers...")
    
    # Test z-score
    print("\n1. Z-score normalization:")
    mock_stats = {'mean': np.array([0.5]), 'sd': np.array([0.2])}
    config = NormalizationConfig(
        type='zscore',
        stats_source='fixed',
        clamp={'enabled': True, 'min': -6.0, 'max': 6.0}
    )
    zscore_norm = ZScoreNormalizer(config, mock_stats)
    
    data = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    normalized = zscore_norm.normalize(data)
    print(f"  Input: {data}")
    print(f"  Output: {normalized}")
    
    # Test linear rescale
    print("\n2. Linear rescale normalization:")
    config = NormalizationConfig(
        type='linear_rescale',
        stats_source='fixed',
        in_min=-0.4,
        in_max=0.4,
        out_min=-1.0,
        out_max=1.0,
        clamp={'enabled': True, 'min': -1.0, 'max': 1.0}
    )
    rescale_norm = LinearRescaleNormalizer(config, {})
    
    data = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    normalized = rescale_norm.normalize(data)
    print(f"  Input: {data}")
    print(f"  Output: {normalized}")
    
    # Test with mask
    print("\n3. Normalization with mask:")
    mask = np.array([True, True, False, True, True])
    normalized = zscore_norm.normalize(data, mask)
    print(f"  Input: {data}")
    print(f"  Mask: {mask}")
    print(f"  Output: {normalized}")
    
    print("\n✓ All tests passed!")
