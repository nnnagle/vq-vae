"""
Integration Example: Zarr Stats Loader + Normalization

Demonstrates how to:
1. Load statistics from Zarr dataset
2. Create normalizers from YAML config
3. Apply normalization to data batches
4. Cache and reuse normalizers efficiently
"""

import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any

from data.normalization.zarr_stats_loader import ZarrStatsLoader, StatsRegistry
from data.normalization.normalization import NormalizationManager, NormalizerFactory


class DataNormalizationPipeline:
    """
    Complete pipeline for loading and normalizing data using Zarr stats.
    
    This class coordinates:
    - Loading YAML configuration
    - Loading statistics from Zarr
    - Creating and caching normalizers
    - Applying normalization to data batches
    """
    
    def __init__(self, config_path: str, zarr_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to forest_repr_model_bindings.yaml
            zarr_path: Path to Zarr dataset root
        """
        self.config_path = Path(config_path)
        self.zarr_path = Path(zarr_path)
        
        # Load YAML config
        print(f"Loading config from {self.config_path}...")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize stats loader
        print(f"Opening Zarr dataset at {self.zarr_path}...")
        self.stats_loader = ZarrStatsLoader(self.zarr_path, cache_stats=True)
        
        # Initialize normalization manager
        print("Initializing normalization manager...")
        self.norm_manager = NormalizationManager(self.config, self.stats_loader)
        
        # Pre-load commonly used stats
        self.stats_registry = StatsRegistry(self.stats_loader)
        
        print("✓ Pipeline initialized successfully")
    
    def preload_stats(self, verbose: bool = True):
        """
        Pre-load all statistics needed for training.
        
        This caches statistics in memory for faster access during training.
        """
        print("\nPre-loading statistics...")
        
        # Count total bands
        total_bands = 0
        loaded_bands = 0
        
        # Iterate through all input groups
        for input_category in ['temporal', 'static', 'irregular']:
            if input_category not in self.config.get('inputs', {}):
                continue
                
            for group_name, group_config in self.config['inputs'][input_category].items():
                zarr_config = group_config.get('zarr', {})
                group_path = zarr_config.get('group')
                
                if not group_path:
                    continue
                
                bands = group_config.get('bands', [])
                total_bands += len(bands)
                
                for band in bands:
                    array_name = band.get('array')
                    norm_preset = band.get('norm')
                    
                    if not array_name or not norm_preset:
                        continue
                    
                    try:
                        # This will load and cache the stats
                        normalizer = self.norm_manager.get_normalizer_for_band(
                            group_path, band
                        )
                        loaded_bands += 1
                        
                        if verbose:
                            print(f"  ✓ {group_name}/{array_name} ({norm_preset})")
                    
                    except Exception as e:
                        print(f"  ✗ {group_name}/{array_name}: {e}")
        
        print(f"\nLoaded stats for {loaded_bands}/{total_bands} bands")
    
    def normalize_batch(
        self,
        data: Dict[str, torch.Tensor],
        input_spec: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize a batch of data.
        
        Args:
            data: Dictionary mapping band names to data tensors
            input_spec: Specification of which group this data belongs to
            
        Returns:
            Dictionary with normalized tensors
        """
        normalized = {}
        
        group_path = input_spec['group_path']
        bands_config = input_spec['bands']
        
        for band_config in bands_config:
            band_name = band_config['name']
            
            if band_name not in data:
                continue
            
            # Get data and optional mask
            band_data = data[band_name]
            band_mask = data.get(f"{band_name}_mask")
            
            # Normalize
            normalized[band_name] = self.norm_manager.normalize_band(
                band_data,
                group_path,
                band_config,
                mask=band_mask
            )
            
            # Pass through mask unchanged
            if band_mask is not None:
                normalized[f"{band_name}_mask"] = band_mask
        
        return normalized
    
    def get_stats_summary(self, group_name: str = None) -> str:
        """
        Get a summary of loaded statistics.
        
        Args:
            group_name: Optional group to filter by
            
        Returns:
            Formatted summary string
        """
        lines = ["Statistics Summary", "=" * 50]
        
        for cache_key, normalizer in self.norm_manager._normalizer_cache.items():
            if group_name and group_name not in cache_key:
                continue
            
            parts = cache_key.split('/')
            lines.append(f"\n{cache_key}:")
            lines.append(f"  Type: {normalizer.config.type}")
            
            if normalizer.stats:
                lines.append(f"  Stats: {list(normalizer.stats.keys())}")
                for stat_name, stat_array in normalizer.stats.items():
                    if stat_array.size == 1:
                        lines.append(f"    {stat_name}: {stat_array.item():.6f}")
        
        return '\n'.join(lines)


def example_1_basic_usage():
    """Example 1: Basic usage of stats loader and normalizer."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Paths (adjust these for your system)
    zarr_path = "/data/VA/zarr/va_vae_dataset_test.zarr"
    
    # Initialize stats loader
    loader = ZarrStatsLoader(zarr_path)
    
    # Load stats for a specific band
    stats = loader.get_stats(
        group_path="annual/ls8day/data",
        array_name="NDVI_summer_p95",
        stat_fields=['mean', 'sd']
    )
    
    print("\nLoaded statistics:")
    for name, value in stats.items():
        print(f"  {name}: {value}")
    
    # Create a normalizer
    from data.normalization.normalization import NormalizationConfig, ZScoreNormalizer
    
    config = NormalizationConfig(
        type='zscore',
        stats_source='zarr',
        clamp={'enabled': True, 'min': -6.0, 'max': 6.0},
        missing={'fill': 0.0}
    )
    
    normalizer = ZScoreNormalizer(config, stats)
    
    # Normalize some sample data
    sample_data = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
    normalized = normalizer.normalize(sample_data)
    
    print("\nNormalization result:")
    print(f"  Input:  {sample_data.numpy()}")
    print(f"  Output: {normalized.numpy()}")


def example_2_full_pipeline():
    """Example 2: Full pipeline with YAML config."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Pipeline")
    print("="*70)
    
    # Paths (adjust these for your system)
    config_path = "config/frl_bindings_v0.yaml"
    zarr_path = "/data/VA/zarr/va_vae_dataset_test.zarr"
    
    # Initialize pipeline
    pipeline = DataNormalizationPipeline(config_path, zarr_path)
    
    # Pre-load all stats
    pipeline.preload_stats(verbose=False)
    
    # Simulate a data batch (temporal group: ls8day)
    batch_data = {
        'NDVI_summer_p95': torch.randn(4, 10, 256, 256),  # [B, T, H, W]
        'NDVI_winter_max': torch.randn(4, 10, 256, 256),
        'NDVI_amplitude': torch.randn(4, 10, 256, 256),
    }
    
    # Define the input spec
    input_spec = {
        'group_path': 'annual/ls8day/data',
        'bands': [
            {'name': 'NDVI_summer_p95', 'array': 'NDVI_summer_p95', 'norm': 'zscore'},
            {'name': 'NDVI_winter_max', 'array': 'NDVI_winter_max', 'norm': 'zscore'},
            {'name': 'NDVI_amplitude', 'array': 'NDVI_amplitude', 'norm': 'zscore'},
        ]
    }
    
    # Normalize the batch
    print("\nNormalizing batch...")
    normalized_batch = pipeline.normalize_batch(batch_data, input_spec)
    
    print("Batch normalized successfully!")
    for name, tensor in normalized_batch.items():
        print(f"  {name}: shape={tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")
    
    # Print stats summary
    print("\n" + pipeline.get_stats_summary(group_name='ls8day'))


def example_3_custom_normalization():
    """Example 3: Creating custom normalizers."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Normalization")
    print("="*70)
    
    from data.normalization.normalization import NormalizationConfig, LinearRescaleNormalizer
    
    # Create a custom linear rescale normalizer
    # This maps [-0.4, 0.4] -> [-1, 1] (for delta values)
    config = NormalizationConfig(
        type='linear_rescale',
        stats_source='fixed',
        in_min=-0.4,
        in_max=0.4,
        out_min=-1.0,
        out_max=1.0,
        clamp={'enabled': True, 'min': -1.0, 'max': 1.0}
    )
    
    normalizer = LinearRescaleNormalizer(config, {})
    
    # Test data: delta values
    delta_data = torch.tensor([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6], dtype=torch.float32)
    normalized = normalizer.normalize(delta_data)
    
    print("\nLinear rescale normalization:")
    print(f"  Input range: [-0.4, 0.4]")
    print(f"  Output range: [-1.0, 1.0]")
    print(f"  Input:  {delta_data.numpy()}")
    print(f"  Output: {normalized.numpy()}")
    print(f"  Note: 0.6 clamped to 1.0")


def example_4_batch_with_masks():
    """Example 4: Normalization with validity masks."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Normalization with Masks")
    print("="*70)
    
    zarr_path = "/data/VA/zarr/va_vae_dataset_test.zarr"
    loader = ZarrStatsLoader(zarr_path)
    
    # Load stats
    stats = loader.get_stats(
        "annual/ls8day/data",
        "NBR_annual_min",
        ['mean', 'sd']
    )
    
    # Create normalizer
    from data.normalization.normalization import NormalizationConfig, ZScoreNormalizer
    
    config = NormalizationConfig(
        type='zscore',
        stats_source='zarr',
        clamp={'enabled': True, 'min': -6.0, 'max': 6.0},
        missing={'fill': 0.0}
    )
    
    normalizer = ZScoreNormalizer(config, stats)
    
    # Sample data with some invalid pixels
    data = torch.tensor([
        [0.3, 0.5, 0.7],
        [0.9, 1.1, 1.3],
        [1.5, 1.7, 1.9]
    ], dtype=torch.float32)
    
    # Mask: True = valid, False = invalid
    mask = torch.tensor([
        [True, True, False],
        [True, False, True],
        [False, True, True]
    ], dtype=torch.bool)
    
    # Normalize with mask
    normalized = normalizer.normalize(data, mask)
    
    print("\nNormalization with mask:")
    print(f"  Input:\n{data.numpy()}")
    print(f"  Mask:\n{mask.numpy()}")
    print(f"  Output:\n{normalized.numpy()}")
    print(f"  Note: Invalid pixels filled with 0.0")


def example_5_stats_validation():
    """Example 5: Validating statistics for different norm types."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Statistics Validation")
    print("="*70)
    
    zarr_path = "/data/VA/zarr/va_vae_dataset_test.zarr"
    loader = ZarrStatsLoader(zarr_path)
    
    # Test z-score normalization requirements
    print("\n1. Z-score normalization:")
    stats = loader.get_stats(
        "annual/ls8day/data",
        "NDVI_summer_p95",
        ['mean', 'sd']
    )
    
    try:
        loader.validate_stats(stats, 'zscore')
        print("   ✓ Valid for z-score normalization")
        print(f"   Loaded: {list(stats.keys())}")
    except ValueError as e:
        print(f"   ✗ Invalid: {e}")
    
    # Test robust IQR normalization requirements
    print("\n2. Robust IQR normalization:")
    stats = loader.get_stats(
        "static/topo/data",
        "elevation",
        ['q25', 'q50', 'q75']
    )
    
    try:
        loader.validate_stats(stats, 'robust_iqr')
        print("   ✓ Valid for robust IQR normalization")
        print(f"   Loaded: {list(stats.keys())}")
        # Print percentile values
        print(f"   25th percentile (q25): {stats['q25']}")
        print(f"   50th percentile (q50): {stats['q50']}")
        print(f"   75th percentile (q75): {stats['q75']}")

        # Compute IQR
        iqr = loader.compute_iqr(stats['q25'], stats['q75'])
        print(f"   Computed IQR: {iqr}")
    except ValueError as e:
        print(f"   ✗ Invalid: {e}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("ZARR STATS LOADER + NORMALIZATION EXAMPLES")
    print("="*70)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Full Pipeline", example_2_full_pipeline),
        ("Custom Normalization", example_3_custom_normalization),
        ("Masks", example_4_batch_with_masks),
        ("Stats Validation", example_5_stats_validation),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_usage,
            example_2_full_pipeline,
            example_3_custom_normalization,
            example_4_batch_with_masks,
            example_5_stats_validation,
        ]
        
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run all examples
        run_all_examples()
