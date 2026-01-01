"""
Zarr Statistics Loader

Loads pre-computed statistics from Zarr variable attributes for normalization.
Handles hierarchical Zarr structure with stats embedded in variable .attrs['statistics'].

Expected Zarr structure:
    category/
        group/
            subsection/
                variable_name
                    .attrs['statistics'] = {
                        'mean': float,
                        'sd': float,
                        'q25': float,
                        'q50': float,
                        'q75': float,
                        ...
                    }

Statistics are computed and embedded by build_zarr.py during dataset creation.
"""

import zarr
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union, Any
import logging

logger = logging.getLogger(__name__)


class ZarrStatsLoader:
    """
    Load and cache statistics from Zarr variable attributes.
    
    Supports the hierarchical structure defined in forest_repr_model_bindings.yaml
    where stats are embedded in each variable's .attrs['statistics'] dict.
    """
    
    def __init__(self, zarr_path: Union[str, Path], cache_stats: bool = True):
        """
        Initialize the stats loader.
        
        Args:
            zarr_path: Path to the root Zarr dataset
            cache_stats: Whether to cache loaded statistics in memory
        """
        self.zarr_path = Path(zarr_path)
        self.cache_stats = cache_stats
        self._stats_cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Open the Zarr store
        try:
            self.zarr_root = zarr.open(str(self.zarr_path), mode='r')
            logger.info(f"Opened Zarr dataset at {self.zarr_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to open Zarr dataset at {self.zarr_path}: {e}")
    
    def get_stats(
        self, 
        group_path: str, 
        array_name: str,
        stat_fields: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load statistics for a specific array from its zarr attributes.
        
        Args:
            group_path: Path to the data group (e.g., 'annual/ls8day/data')
            array_name: Name of the array (e.g., 'NDVI_summer_p95')
            stat_fields: List of stat fields to load (e.g., ['mean', 'sd', 'q25', 'q50', 'q75'])
                        If None, loads all available stats
        
        Returns:
            Dictionary mapping stat names to numpy arrays (scalar values as 0-d arrays)
            
        Example:
            >>> loader.get_stats('annual/ls8day/data', 'NDVI_summer_p95', ['mean', 'sd'])
            {'mean': array(0.45), 'sd': array(0.12)}
        """
        cache_key = f"{group_path}/{array_name}"
        
        # Check cache first
        if self.cache_stats and cache_key in self._stats_cache:
            cached_stats = self._stats_cache[cache_key]
            if stat_fields is None:
                return cached_stats
            # Return filtered view of cached stats (still returns dict, but from cache)
            return {k: v for k, v in cached_stats.items() if k in stat_fields}
        
        # Navigate to the variable in zarr hierarchy
        # group_path is like 'annual/ls8day/data'
        full_path = f"{group_path}/{array_name}"
        
        try:
            variable = self.zarr_root[full_path]
        except KeyError:
            logger.warning(f"Variable not found: {full_path}")
            return {}
        
        # Get statistics from attrs
        if 'statistics' not in variable.attrs:
            logger.warning(f"No statistics found in attrs for {full_path}")
            return {}
        
        stats_dict = dict(variable.attrs['statistics'])
        
        # Convert to numpy arrays and standardize names
        stats = {}
        
        # Define stat name mappings
        stat_mapping = {
            'mean': 'mean',
            'std': 'sd',  # Map std to sd
            'sd': 'sd',
            'q25': 'q25',
            'q50': 'q50',
            'median': 'q50',  # Alternative name
            'q75': 'q75',
            'min': 'min',
            'max': 'max',
            'count': 'count',
            'q02': 'q02',
            'q98': 'q98',
        }
        
        # Extract all available stats (not filtered by stat_fields yet)
        for stat_key in stat_mapping.keys():
            standard_name = stat_mapping.get(stat_key, stat_key)
            
            # Try to find this stat in the dict
            for search_key in [stat_key, standard_name]:
                if search_key in stats_dict:
                    value = stats_dict[search_key]
                    # Convert to numpy array (scalar becomes 0-d array)
                    if isinstance(value, (list, tuple)):
                        stats[standard_name] = np.array(value)
                    else:
                        stats[standard_name] = np.array(value, dtype=np.float32)
                    break
        
        # Cache the full stats dict
        if self.cache_stats:
            self._stats_cache[cache_key] = stats
        
        # Return filtered stats if requested
        if stat_fields is not None:
            return {k: v for k, v in stats.items() if k in stat_fields}
        
        return stats    


    def get_stats_for_band(
        self,
        band_config: Dict[str, Any],
        group_path: str,
        required_stats: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load statistics for a band defined in the bindings YAML.
        
        Args:
            band_config: Band configuration dictionary containing 'array' key
            group_path: Zarr group path (e.g., 'annual/ls8day/data')
            required_stats: Specific stats needed for the normalization preset
        
        Returns:
            Dictionary of statistics
        """
        array_name = band_config.get('array')
        if not array_name:
            raise ValueError("Band config missing 'array' field")
        
        return self.get_stats(group_path, array_name, required_stats)
    
    def compute_iqr(self, q25: np.ndarray, q75: np.ndarray) -> np.ndarray:
        """
        Compute IQR from quartiles.
        
        Args:
            q25: 25th percentile array
            q75: 75th percentile array
            
        Returns:
            IQR array (q75 - q25)
        """
        return q75 - q25
    
    def validate_stats(
        self,
        stats: Dict[str, np.ndarray],
        norm_type: str
    ) -> bool:
        """
        Validate that required statistics are present for a normalization type.
        
        Args:
            stats: Dictionary of loaded statistics
            norm_type: Type of normalization ('zscore', 'robust_iqr', etc.)
            
        Returns:
            True if all required stats are present
            
        Raises:
            ValueError: If required stats are missing
        """
        required_by_type = {
            'zscore': ['mean', 'sd'],
            'robust_iqr': ['q25', 'q50', 'q75'],
            'minmax': ['min', 'max'],
        }
        
        required = required_by_type.get(norm_type, [])
        missing = [s for s in required if s not in stats]
        
        if missing:
            raise ValueError(
                f"Missing required stats for {norm_type} normalization: {missing}"
            )
        
        return True
    
    def get_group_arrays(self, group_path: str) -> List[str]:
        """
        List all arrays in a data group.
        
        Args:
            group_path: Path to the group (e.g., 'annual/ls8day/data')
            
        Returns:
            List of array names
        """
        try:
            group = self.zarr_root[group_path]
            if isinstance(group, zarr.Group):
                return list(group.array_keys())
            else:
                logger.warning(f"{group_path} is not a group")
                return []
        except KeyError:
            logger.warning(f"Group not found: {group_path}")
            return []
    
    def get_stats_summary(self, group_path: str, array_name: str) -> str:
        """
        Get a human-readable summary of available statistics.
        
        Args:
            group_path: Path to the data group
            array_name: Name of the array
            
        Returns:
            Formatted string summarizing available stats
        """
        stats = self.get_stats(group_path, array_name)
        
        if not stats:
            return f"No statistics found for {array_name}"
        
        lines = [f"Statistics for {array_name} (from zarr attrs):"]
        for stat_name, stat_value in stats.items():
            if stat_value.ndim == 0:
                # Scalar value
                lines.append(f"  {stat_name}: {stat_value.item():.6f}")
            elif stat_value.size < 10:
                lines.append(f"  {stat_name}: {stat_value}")
            else:
                lines.append(f"  {stat_name}: shape={stat_value.shape}, dtype={stat_value.dtype}")
        
        return '\n'.join(lines)
    
    def clear_cache(self):
        """Clear the statistics cache."""
        self._stats_cache.clear()
        logger.info("Statistics cache cleared")
    
    def __repr__(self) -> str:
        return f"ZarrStatsLoader(zarr_path={self.zarr_path}, cached_stats={len(self._stats_cache)})"


class StatsRegistry:
    """
    Registry for managing statistics across multiple arrays and groups.
    
    Useful for batch loading and caching stats for an entire training run.
    """
    
    def __init__(self, loader: ZarrStatsLoader):
        """
        Initialize the registry.
        
        Args:
            loader: ZarrStatsLoader instance
        """
        self.loader = loader
        self.registry: Dict[str, Dict[str, np.ndarray]] = {}
    
    def register_from_yaml(self, yaml_config: Dict[str, Any]):
        """
        Pre-load all statistics referenced in the bindings YAML.
        
        Args:
            yaml_config: Parsed YAML configuration dictionary
        """
        # This would parse the YAML and pre-load all required stats
        # Implementation depends on your YAML structure
        
        # Example for temporal inputs:
        if 'inputs' in yaml_config and 'temporal' in yaml_config['inputs']:
            for group_name, group_config in yaml_config['inputs']['temporal'].items():
                zarr_config = group_config.get('zarr', {})
                group_path = zarr_config.get('group')
                
                if not group_path:
                    continue
                
                for band in group_config.get('bands', []):
                    array_name = band.get('array')
                    norm_preset = band.get('norm')
                    
                    if array_name and norm_preset:
                        # Determine required stats based on norm preset
                        required_stats = self._get_required_stats_for_norm(
                            yaml_config, norm_preset
                        )
                        
                        key = f"{group_path}/{array_name}"
                        self.registry[key] = self.loader.get_stats(
                            group_path, array_name, required_stats
                        )
    
    def _get_required_stats_for_norm(
        self, 
        yaml_config: Dict[str, Any], 
        norm_preset: str
    ) -> List[str]:
        """
        Determine which stats are needed for a normalization preset.
        
        Args:
            yaml_config: Full YAML configuration
            norm_preset: Name of the normalization preset
            
        Returns:
            List of required stat field names
        """
        norm_config = yaml_config.get('normalization', {}).get('presets', {}).get(norm_preset, {})
        norm_type = norm_config.get('type', '')
        
        type_to_stats = {
            'zscore': ['mean', 'sd'],
            'robust_iqr': ['q25', 'q50', 'q75'],
            'minmax': ['min', 'max'],
        }
        
        return type_to_stats.get(norm_type, [])
    
    def get(self, group_path: str, array_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get statistics from the registry.
        
        Args:
            group_path: Path to the data group
            array_name: Name of the array
            
        Returns:
            Dictionary of statistics or None if not found
        """
        key = f"{group_path}/{array_name}"
        return self.registry.get(key)
    
    def __len__(self) -> int:
        return len(self.registry)
    
    def __repr__(self) -> str:
        return f"StatsRegistry(registered_arrays={len(self.registry)})"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load stats for a specific array
    zarr_path = "/data/VA/zarr/va_vae_dataset_test.zarr"
    
    try:
        loader = ZarrStatsLoader(zarr_path)
        
        # Get stats for a Landsat band
        # Stats are loaded from the variable's .attrs['statistics']
        stats = loader.get_stats(
            group_path="annual/ls8day/data",
            array_name="NDVI_summer_p95",
            stat_fields=['mean', 'sd', 'q25', 'q50', 'q75']
        )
        
        print("\nLoaded statistics from zarr attrs:")
        for stat_name, stat_value in stats.items():
            if stat_value.ndim == 0:
                print(f"  {stat_name}: {stat_value.item():.6f}")
            else:
                print(f"  {stat_name}: {stat_value.shape}, {stat_value.dtype}")
        
        # Validate for z-score normalization
        loader.validate_stats(stats, 'zscore')
        print("\n✓ Stats validated for z-score normalization")
        
        # Print summary
        print("\n" + loader.get_stats_summary("annual/ls8day/data", "NDVI_summer_p95"))
        
    except Exception as e:
        print(f"Error: {e}")
