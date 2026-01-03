"""
Derived Statistics Loader

Loads pre-computed statistics and covariance matrices from Zarr.
Provides easy access for training code and loss functions.

Usage:
    loader = DerivedStatsLoader(bindings_config, zarr_path)
    
    # Get feature stats
    stats = loader.get_feature_stats('ls8_delta')
    mean = stats['mean']  # [C] array
    sd = stats['sd']      # [C] array
    
    # Get covariance matrix
    cov_inv = loader.get_covariance_inverse('topo_cov')  # [N, N] array
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DerivedStatsLoader:
    """
    Load pre-computed derived statistics from Zarr.
    
    Provides access to:
    - Feature statistics (mean, sd, quantiles, etc.) for normalization
    - Covariance matrices and their inverses for loss functions
    
    Usage:
        loader = DerivedStatsLoader(bindings_config, zarr_path)
        
        # Load feature stats for normalization
        ls8_delta_stats = loader.get_feature_stats('ls8_delta')
        mean = ls8_delta_stats['mean']
        sd = ls8_delta_stats['sd']
        
        # Load covariance inverse for Mahalanobis distance
        topo_cov_inv = loader.get_covariance_inverse('topo_cov')
    """
    
    def __init__(
        self,
        bindings_config: Dict[str, Any],
        zarr_path: Optional[str] = None
    ):
        """
        Initialize DerivedStatsLoader.
        
        Args:
            bindings_config: Parsed bindings configuration
            zarr_path: Path to Zarr dataset (overrides config if provided)
        """
        self.config = bindings_config
        
        # Get zarr location for derived stats
        ds_config = self.config.get('derived_statistics', {})
        self.zarr_location = ds_config.get('zarr_location', 'derived_stats')
        
        # Open zarr dataset
        self.zarr_path = zarr_path or self.config['zarr']['path']
        self.zarr_root = zarr.open(self.zarr_path, mode='r')
        
        # Check if derived stats exist
        if self.zarr_location not in self.zarr_root:
            raise ValueError(
                f"Derived statistics not found at '{self.zarr_location}' in Zarr dataset. "
                f"Run DerivedStatsComputer.compute_and_save() first."
            )
        
        self.stats_group = self.zarr_root[self.zarr_location]
        
        # Cache loaded stats for efficiency
        self._feature_stats_cache = {}
        self._cov_cache = {}
        
        logger.info(f"Initialized DerivedStatsLoader from {self.zarr_path}")
        logger.info(f"  Stats location: {self.zarr_location}")
        
        # List available features and covariances
        if 'features' in self.stats_group:
            features = list(self.stats_group['features'].keys())
            logger.info(f"  Available features: {features}")
        
        if 'covariance_matrices' in self.stats_group:
            covs = list(self.stats_group['covariance_matrices'].keys())
            logger.info(f"  Available covariances: {covs}")
    
    def get_feature_stats(self, feature_name: str) -> Dict[str, np.ndarray]:
        """
        Get all statistics for a derived feature.
        
        Args:
            feature_name: Name of derived feature (e.g., 'ls8_delta')
        
        Returns:
            Dictionary mapping stat_name -> array
            Example: {
                'mean': array([...]),  # [C]
                'sd': array([...]),    # [C]
                'q25': array([...]),   # [C]
                ...
            }
        
        Raises:
            KeyError: If feature not found
        """
        # Check cache first
        if feature_name in self._feature_stats_cache:
            return self._feature_stats_cache[feature_name]
        
        # Load from Zarr
        if 'features' not in self.stats_group:
            raise KeyError(f"No features found in {self.zarr_location}")
        
        features_group = self.stats_group['features']
        
        if feature_name not in features_group:
            available = list(features_group.keys())
            raise KeyError(
                f"Feature '{feature_name}' not found. Available: {available}"
            )
        
        feature_group = features_group[feature_name]
        
        # Load all statistics
        stats = {}
        for stat_name in feature_group.keys():
            stats[stat_name] = np.array(feature_group[stat_name])
        
        # Cache and return
        self._feature_stats_cache[feature_name] = stats
        
        logger.debug(f"Loaded stats for '{feature_name}': {list(stats.keys())}")
        return stats
    
    def get_feature_stat(
        self,
        feature_name: str,
        stat_name: str
    ) -> np.ndarray:
        """
        Get a specific statistic for a feature.
        
        Args:
            feature_name: Name of derived feature
            stat_name: Name of statistic (e.g., 'mean', 'sd', 'q25')
        
        Returns:
            Statistic array [C]
        
        Raises:
            KeyError: If feature or stat not found
        """
        stats = self.get_feature_stats(feature_name)
        
        if stat_name not in stats:
            available = list(stats.keys())
            raise KeyError(
                f"Stat '{stat_name}' not found for '{feature_name}'. "
                f"Available: {available}"
            )
        
        return stats[stat_name]
    
    def get_covariance(self, cov_name: str) -> np.ndarray:
        """
        Get covariance matrix.
        
        Args:
            cov_name: Name of covariance matrix (e.g., 'topo_cov')
        
        Returns:
            Covariance matrix [N, N]
        
        Raises:
            KeyError: If covariance not found
        """
        cache_key = f"{cov_name}_covariance"
        
        # Check cache
        if cache_key in self._cov_cache:
            return self._cov_cache[cache_key]
        
        # Load from Zarr
        if 'covariance_matrices' not in self.stats_group:
            raise KeyError(f"No covariance matrices found in {self.zarr_location}")
        
        cov_group = self.stats_group['covariance_matrices']
        
        if cov_name not in cov_group:
            available = list(cov_group.keys())
            raise KeyError(
                f"Covariance '{cov_name}' not found. Available: {available}"
            )
        
        cov_matrix_group = cov_group[cov_name]
        
        if 'covariance' not in cov_matrix_group:
            raise KeyError(f"Covariance matrix not found for '{cov_name}'")
        
        cov = np.array(cov_matrix_group['covariance'])
        
        # Cache and return
        self._cov_cache[cache_key] = cov
        
        logger.debug(f"Loaded covariance '{cov_name}': shape {cov.shape}")
        return cov
    
    def get_covariance_inverse(self, cov_name: str) -> np.ndarray:
        """
        Get inverse covariance matrix.
        
        Args:
            cov_name: Name of covariance matrix (e.g., 'topo_cov')
        
        Returns:
            Inverse covariance matrix [N, N]
        
        Raises:
            KeyError: If covariance or inverse not found
        """
        cache_key = f"{cov_name}_inverse"
        
        # Check cache
        if cache_key in self._cov_cache:
            return self._cov_cache[cache_key]
        
        # Load from Zarr
        if 'covariance_matrices' not in self.stats_group:
            raise KeyError(f"No covariance matrices found in {self.zarr_location}")
        
        cov_group = self.stats_group['covariance_matrices']
        
        if cov_name not in cov_group:
            available = list(cov_group.keys())
            raise KeyError(
                f"Covariance '{cov_name}' not found. Available: {available}"
            )
        
        cov_matrix_group = cov_group[cov_name]
        
        # Get inverse_id from config
        ds_config = self.config.get('derived_statistics', {})
        cov_matrices_config = ds_config.get('covariance_matrices', {})
        
        if cov_name not in cov_matrices_config:
            raise KeyError(f"Covariance '{cov_name}' not in config")
        
        inverse_config = cov_matrices_config[cov_name].get('inverse', {})
        inverse_id = inverse_config.get('id', f"{cov_name}_inv")
        
        if inverse_id not in cov_matrix_group:
            raise KeyError(
                f"Inverse not found for '{cov_name}'. "
                f"Expected key '{inverse_id}' in Zarr group. "
                f"Available: {list(cov_matrix_group.keys())}"
            )
        
        inv = np.array(cov_matrix_group[inverse_id])
        
        # Cache and return
        self._cov_cache[cache_key] = inv
        
        logger.debug(f"Loaded inverse covariance '{cov_name}': shape {inv.shape}")
        return inv
    
    def get_covariance_mean(self, cov_name: str) -> np.ndarray:
        """
        Get mean vector used for covariance computation.
        
        Args:
            cov_name: Name of covariance matrix
        
        Returns:
            Mean vector [N]
        """
        cache_key = f"{cov_name}_mean"
        
        if cache_key in self._cov_cache:
            return self._cov_cache[cache_key]
        
        cov_group = self.stats_group['covariance_matrices']
        cov_matrix_group = cov_group[cov_name]
        
        if 'mean' not in cov_matrix_group:
            raise KeyError(f"Mean not found for '{cov_name}'")
        
        mean = np.array(cov_matrix_group['mean'])
        
        self._cov_cache[cache_key] = mean
        return mean
    
    def get_covariance_metadata(self, cov_name: str) -> Dict[str, Any]:
        """
        Get metadata for covariance matrix.
        
        Args:
            cov_name: Name of covariance matrix
        
        Returns:
            Dictionary with metadata (n_features, n_samples, domain, etc.)
        """
        cov_group = self.stats_group['covariance_matrices']
        cov_matrix_group = cov_group[cov_name]
        
        return dict(cov_matrix_group.attrs)
    
    def list_features(self) -> list:
        """Get list of available feature names."""
        if 'features' not in self.stats_group:
            return []
        return list(self.stats_group['features'].keys())
    
    def list_covariances(self) -> list:
        """Get list of available covariance matrix names."""
        if 'covariance_matrices' not in self.stats_group:
            return []
        return list(self.stats_group['covariance_matrices'].keys())
    
    def clear_cache(self):
        """Clear cached statistics (useful if zarr is updated)."""
        self._feature_stats_cache = {}
        self._cov_cache = {}
        logger.debug("Cleared stats cache")


if __name__ == '__main__':
    # Example usage
    import sys
    from data.loaders.bindings.parser import BindingsParser
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '/mnt/project/frl_bindings_v0.yaml'
    
    print("Example: Loading Derived Statistics")
    print("=" * 70)
    
    # Parse config
    print("\n1. Parsing configuration...")
    parser = BindingsParser(config_path)
    config = parser.parse()
    print(f"   ✓ Loaded config: {config['name']}")
    
    # Initialize loader
    print("\n2. Initializing DerivedStatsLoader...")
    try:
        loader = DerivedStatsLoader(config)
        print("   ✓ Successfully loaded stats from Zarr")
    except ValueError as e:
        print(f"   ✗ Error: {e}")
        print("\n   Run DerivedStatsComputer first to generate statistics!")
        sys.exit(0)
    
    # List available features
    print("\n3. Available features:")
    features = loader.list_features()
    if features:
        for feature in features:
            print(f"   - {feature}")
    else:
        print("   (none)")
    
    # List available covariances
    print("\n4. Available covariance matrices:")
    covs = loader.list_covariances()
    if covs:
        for cov in covs:
            metadata = loader.get_covariance_metadata(cov)
            print(f"   - {cov}: {metadata.get('n_features')} features, "
                  f"{metadata.get('n_samples')} samples")
    else:
        print("   (none)")
    
    # Example: Load feature stats
    if features:
        print(f"\n5. Loading stats for '{features[0]}'...")
        stats = loader.get_feature_stats(features[0])
        print(f"   Available stats: {list(stats.keys())}")
        print(f"   Mean shape: {stats['mean'].shape}")
        print(f"   Mean values: {stats['mean']}")
    
    # Example: Load covariance
    if covs:
        print(f"\n6. Loading covariance matrix '{covs[0]}'...")
        cov = loader.get_covariance(covs[0])
        print(f"   Covariance shape: {cov.shape}")
        print(f"   Covariance matrix:")
        print(cov)
        
        try:
            inv = loader.get_covariance_inverse(covs[0])
            print(f"\n   Inverse shape: {inv.shape}")
            print(f"   Inverse matrix:")
            print(inv)
            
            # Verify: cov @ inv ≈ I
            identity = cov @ inv
            print(f"\n   Verification (cov @ inv):")
            print(identity)
            print(f"   Identity error: {np.abs(identity - np.eye(cov.shape[0])).max():.2e}")
        except KeyError as e:
            print(f"\n   Inverse not available: {e}")
    
    print("\n✓ Example complete!")
