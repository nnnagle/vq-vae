"""
Bindings Utilities

Helper classes and functions for working with parsed bindings configuration,
including reference resolution, band lookup, and configuration queries.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class ResolvedBand:
    """A fully resolved band with all references expanded"""
    name: str
    array_path: str  # Full path: group/array
    norm_config: Dict[str, Any]
    masks: List[Dict[str, Any]]
    quality_weights: List[Dict[str, Any]]
    loss_weight: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class BindingsRegistry:
    """
    Registry for quick lookups of bindings configuration elements.
    
    Built on top of parsed bindings configuration, provides O(1) lookups
    for common queries.
    """
    
    def __init__(self, parsed_config: Dict[str, Any]):
        """
        Initialize registry from parsed configuration.
        
        Args:
            parsed_config: Output from BindingsParser.parse()
        """
        self.config = parsed_config
        self._build_indices()
    
    def _build_indices(self):
        """Build lookup indices for fast access"""
        # Map from full reference path to config object
        self.ref_index: Dict[str, Any] = {}
        
        # Map from (group, array) to list of bands using that array
        self.array_to_bands: Dict[tuple, List[str]] = defaultdict(list)
        
        # Map from band name to full reference path
        self.band_name_index: Dict[str, str] = {}
        
        # Map from normalization preset name to list of bands using it
        self.norm_to_bands: Dict[str, List[str]] = defaultdict(list)
        
        # Build indices
        self._index_shared()
        self._index_inputs()
        self._index_derived()
    
    def _index_shared(self):
        """Index shared masks and quality metrics"""
        for mask_name, mask_config in self.config['shared']['masks'].items():
            ref = f"shared.masks.{mask_name}"
            self.ref_index[ref] = mask_config
        
        for quality_name, quality_config in self.config['shared']['quality'].items():
            ref = f"shared.quality.{quality_name}"
            self.ref_index[ref] = quality_config
    
    def _index_inputs(self):
        """Index all input groups and bands"""
        for category in ['temporal', 'irregular', 'static']:
            for group_name, group in self.config['inputs'][category].items():
                group_ref = f"inputs.{category}.{group_name}"
                self.ref_index[group_ref] = group
                
                # Index each band
                for band in group.bands:
                    band_ref = f"{group_ref}.{band.name}"
                    self.ref_index[band_ref] = band
                    self.band_name_index[band.name] = band_ref
                    
                    # Index by array
                    if band.array:
                        array_key = (group.zarr.group, band.array)
                        self.array_to_bands[array_key].append(band_ref)
                    
                    # Index by normalization preset
                    if band.norm:
                        self.norm_to_bands[band.norm].append(band_ref)
    
    def _index_derived(self):
        """Index derived features"""
        for feature_name, feature_config in self.config['derived'].items():
            if not feature_name.startswith('_'):
                ref = f"derived.{feature_name}"
                self.ref_index[ref] = feature_config
    
    def lookup(self, ref_path: str) -> Any:
        """
        Fast lookup of reference.
        
        Args:
            ref_path: Reference like 'shared.masks.forest'
            
        Returns:
            Configuration object
            
        Raises:
            KeyError: If reference not found
        """
        if ref_path not in self.ref_index:
            raise KeyError(f"Reference not found: {ref_path}")
        return self.ref_index[ref_path]
    
    def get_bands_using_array(self, group: str, array: str) -> List[str]:
        """
        Find all bands that use a specific Zarr array.
        
        Args:
            group: Zarr group path (e.g., 'annual/ls8day/data')
            array: Array name (e.g., 'NDVI_summer_p95')
            
        Returns:
            List of band reference paths
        """
        return self.array_to_bands.get((group, array), [])
    
    def get_bands_using_norm(self, preset_name: str) -> List[str]:
        """
        Find all bands that use a specific normalization preset.
        
        Args:
            preset_name: Preset name (e.g., 'zscore')
            
        Returns:
            List of band reference paths
        """
        return self.norm_to_bands.get(preset_name, [])
    
    def find_band_by_name(self, name: str) -> Optional[str]:
        """
        Find band reference by name.
        
        Args:
            name: Band name (e.g., 'NDVI_summer_p95')
            
        Returns:
            Full reference path or None if not found
        """
        return self.band_name_index.get(name)


class ReferenceResolver:
    """
    Resolves configuration references to their actual values.
    
    Handles dotted paths, nested lookups, and circular dependency prevention.
    """
    
    def __init__(self, config: Dict[str, Any], registry: Optional[BindingsRegistry] = None):
        """
        Initialize resolver.
        
        Args:
            config: Parsed bindings bindings configuration dictionary
            registry: Optional BindingsRegistry for fast lookups
        """
        self.config = config
        self.registry = registry or BindingsRegistry(config)
        self._resolution_stack = []
    
    def resolve(self, ref_path: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resolve a reference path to its value.
        
        Args:
            ref_path: Dotted reference path
            context: Optional context dict for local variable resolution
            
        Returns:
            Resolved value
            
        Raises:
            ValueError: If circular reference detected
            KeyError: If reference not found
        """
        # Check for circular references
        if ref_path in self._resolution_stack:
            cycle = ' -> '.join(self._resolution_stack + [ref_path])
            raise ValueError(f"Circular reference detected: {cycle}")
        
        self._resolution_stack.append(ref_path)
        
        try:
            # Try fast lookup first
            try:
                result = self.registry.lookup(ref_path)
                return result
            except KeyError:
                pass
            
            # Fall back to manual traversal
            parts = ref_path.split('.')
            current = self.config
            
            for part in parts:
                if isinstance(current, dict):
                    if part in current:
                        current = current[part]
                    else:
                        raise KeyError(f"Key '{part}' not found in path '{ref_path}'")
                else:
                    raise KeyError(f"Cannot traverse non-dict at '{part}' in '{ref_path}'")
            
            return current
            
        finally:
            self._resolution_stack.pop()
    
    def resolve_band_fully(
        self, 
        category: str, 
        group_name: str, 
        band_name: str
    ) -> ResolvedBand:
        """
        Resolve all references for a band, returning a complete specification.
        
        Args:
            category: 'temporal', 'irregular', or 'static'
            group_name: Input group name
            band_name: Band name
            
        Returns:
            ResolvedBand with all references expanded
        """
        # Get band config
        group_ref = f"inputs.{category}.{group_name}"
        group = self.resolve(group_ref)
        
        band = None
        for b in group.bands:
            if b.name == band_name:
                band = b
                break
        
        if band is None:
            raise KeyError(f"Band '{band_name}' not found in {group_ref}")
        
        # Resolve normalization
        norm_config = None
        if band.norm:
            norm_ref = f"normalization.presets.{band.norm}"
            norm_config = self.resolve(norm_ref)
        
        # Resolve masks
        masks = []
        for mask_ref in band.mask:
            masks.append(self.resolve(mask_ref))
        
        # Resolve quality weights
        quality_weights = []
        for qw_ref in band.quality_weight:
            quality_weights.append(self.resolve(qw_ref))
        
        # Resolve loss weight
        loss_weight = None
        if band.loss_weight:
            loss_weight = self.resolve(band.loss_weight)
        
        # Build array path
        array_path = f"{group.zarr.group}/{band.array}" if band.array else None
        
        return ResolvedBand(
            name=band.name,
            array_path=array_path,
            norm_config=norm_config,
            masks=masks,
            quality_weights=quality_weights,
            loss_weight=loss_weight,
            metadata={
                'category': category,
                'group': group_name,
                'kind': group.kind,
                'time_window_years': group.time_window_years,
                'num_classes': band.num_classes
            }
        )


class BandSelector:
    """
    Helper for selecting and filtering bands based on criteria.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize selector.
        
        Args:
            config: Parsed bindings bindings configuration dictionary
        """
        self.config = config
        self.registry = BindingsRegistry(config)
    
    def select_by_category(self, category: str) -> List[tuple]:
        """
        Get all bands in a category.
        
        Args:
            category: 'temporal', 'irregular', or 'static'
            
        Returns:
            List of (group_name, band) tuples
        """
        results = []
        for group_name, group in self.config['inputs'][category].items():
            for band in group.bands:
                results.append((group_name, band))
        return results
    
    def select_by_norm(self, preset_name: str) -> List[tuple]:
        """
        Get all bands using a specific normalization preset.
        
        Args:
            preset_name: Preset name (e.g., 'zscore')
            
        Returns:
            List of (category, group_name, band) tuples
        """
        results = []
        band_refs = self.registry.get_bands_using_norm(preset_name)
        
        for ref in band_refs:
            # Parse reference: inputs.temporal.ls8day.NDVI_summer_p95
            parts = ref.split('.')
            if len(parts) >= 4:
                category = parts[1]
                group_name = parts[2]
                band_name = parts[3]
                
                group = self.config['inputs'][category][group_name]
                for band in group.bands:
                    if band.name == band_name:
                        results.append((category, group_name, band))
                        break
        
        return results
    
    def select_by_mask(self, mask_name: str) -> List[tuple]:
        """
        Get all bands that use a specific mask.
        
        Args:
            mask_name: Mask reference (e.g., 'shared.masks.forest')
            
        Returns:
            List of (category, group_name, band) tuples
        """
        results = []
        
        for category in ['temporal', 'irregular', 'static']:
            for group_name, group in self.config['inputs'][category].items():
                for band in group.bands:
                    if mask_name in band.mask:
                        results.append((category, group_name, band))
        
        return results
    
    def select_temporal_with_window(self, window_years: int) -> List[tuple]:
        """
        Get all temporal bands with a specific time window.
        
        Args:
            window_years: Time window length in years
            
        Returns:
            List of (group_name, band) tuples
        """
        results = []
        
        for group_name, group in self.config['inputs']['temporal'].items():
            if group.time_window_years == window_years:
                for band in group.bands:
                    results.append((group_name, band))
        
        return results


class BindingsQuery:
    """
    High-level query interface for bindings configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with parsed bindings configuration"""
        self.config = config
        self.registry = BindingsRegistry(config)
        self.resolver = ReferenceResolver(config, self.registry)
        self.selector = BandSelector(config)
    
    def count_bands(self) -> Dict[str, int]:
        """Count bands by category"""
        return {
            category: sum(
                len(group.bands) 
                for group in self.config['inputs'][category].values()
            )
            for category in ['temporal', 'irregular', 'static']
        }
    
    def count_total_channels(self) -> int:
        """
        Count total number of channels/bands across all inputs.
        
        For temporal inputs, this counts bands * timesteps.
        """
        total = 0
        
        # Temporal: bands * timesteps
        for group in self.config['inputs']['temporal'].values():
            timesteps = group.time_window_years or 1
            total += len(group.bands) * timesteps
        
        # Irregular: just count bands (sparse)
        for group in self.config['inputs']['irregular'].values():
            total += len(group.bands)
        
        # Static: just count bands
        for group in self.config['inputs']['static'].values():
            total += len(group.bands)
        
        return total
    
    def list_normalization_presets(self) -> List[str]:
        """Get list of all normalization preset names"""
        return list(self.config['normalization']['presets'].keys())
    
    def list_shared_masks(self) -> List[str]:
        """Get list of all shared mask names"""
        return list(self.config['shared']['masks'].keys())
    
    def list_shared_quality(self) -> List[str]:
        """Get list of all shared quality metric names"""
        return list(self.config['shared']['quality'].keys())
    
    def list_derived_features(self) -> List[str]:
        """Get list of all derived feature names"""
        return [
            k for k in self.config['derived'].keys() 
            if not k.startswith('_')
        ]
    
    def get_window_config(self) -> Dict[str, Any]:
        """Get training windowing configuration"""
        return self.config.get('training', {}).get('windowing', {})
    
    def get_sampling_config(self) -> Dict[str, Any]:
        """Get sampling configuration"""
        return self.config.get('sampling', {})
    
    def get_loss_config(self, loss_name: str) -> Dict[str, Any]:
        """Get configuration for a specific loss"""
        losses = self.config.get('losses', {})
        if loss_name not in losses:
            raise KeyError(f"Loss '{loss_name}' not found")
        return losses[loss_name]
    
    def get_model_encoder_inputs(self, encoder_name: str) -> Dict[str, List[str]]:
        """
        Get input specification for a model encoder.
        
        Args:
            encoder_name: 'type_encoder' or 'phase_encoder'
            
        Returns:
            Dict with keys like 'temporal', 'static', 'labels'
        """
        model_inputs = self.config.get('model_inputs', {})
        if encoder_name not in model_inputs:
            raise KeyError(f"Encoder '{encoder_name}' not found")
        return model_inputs[encoder_name]
    
    def validate_reference_exists(self, ref_path: str) -> bool:
        """
        Check if a reference exists.
        
        Args:
            ref_path: Reference to check
            
        Returns:
            True if reference exists, False otherwise
        """
        try:
            self.resolver.resolve(ref_path)
            return True
        except (KeyError, ValueError):
            return False


def print_bindings_tree(bindings: Dict[str, Any], indent: int = 0):
    """
    Print bindings configuration as a tree structure.
    
    Args:
        bindings: Bindings configuration dictionary
        indent: Current indentation level
    """
    indent_str = "  " * indent
    
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_bindings_tree(value, indent + 1)
        elif isinstance(value, list):
            print(f"{indent_str}{key}: [{len(value)} items]")
        else:
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"{indent_str}{key}: {value_str}")


if __name__ == '__main__':
    # Example usage
    from data.loaders.bindings.parser import BindingsParser
    
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    config = parser.parse()
    
    # Create query interface
    query = BindingsQuery(config)
    
    print("Bindings Configuration Query Examples")
    print("=" * 70)
    
    print(f"\nBand counts: {query.count_bands()}")
    print(f"Total channels: {query.count_total_channels()}")
    
    print(f"\nNormalization presets: {query.list_normalization_presets()}")
    print(f"Shared masks: {query.list_shared_masks()}")
    print(f"Derived features: {query.list_derived_features()}")
    
    # Resolve a band fully
    resolver = ReferenceResolver(config)
    resolved = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
    print(f"\nResolved band: {resolved.name}")
    print(f"  Array path: {resolved.array_path}")
    print(f"  Norm: {resolved.norm_config['type'] if resolved.norm_config else 'none'}")
    print(f"  Masks: {len(resolved.masks)}")
    print(f"  Quality weights: {len(resolved.quality_weights)}")
