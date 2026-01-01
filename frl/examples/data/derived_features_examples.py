"""
Derived Features - Complete Usage Example

Demonstrates:
1. Building individual derived features
2. Building all derived features
3. Integrating with DataReader and MaskBuilder
4. Combining raw + derived inputs
5. Handling masks and quality weights

This example shows the complete workflow for using derived features
in your training pipeline.
"""

import numpy as np
from pathlib import Path
import sys


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_1_temporal_position():
    """Example 1: Temporal Position Encoding"""
    print_section("Example 1: Temporal Position Encoding")
    
    print("Temporal position encoding creates two channels that encode")
    print("the position within a time window:")
    print("  - p_t: Normalized position [0, 1]")
    print("  - c_t: Centered position [-1, 1]")
    print()
    
    # Simulate the computation
    T = 10  # 10-year window
    t = np.arange(T, dtype=np.float32)
    
    # Normalized position
    p_t = t / (T - 1)
    
    # Centered position
    c_t = (t - (T - 1) / 2) / ((T - 1) / 2)
    
    print(f"Window length: {T} years")
    print()
    print("Timestep | p_t    | c_t")
    print("-" * 30)
    for i in range(T):
        print(f"{i:8d} | {p_t[i]:6.3f} | {c_t[i]:6.3f}")
    
    print()
    print("These channels are concatenated to temporal inputs like ls8day,")
    print("providing the model with explicit temporal context.")
    print()
    print("Example usage:")
    print("  result = builder.build_derived_feature(")
    print("      'temporal_position',")
    print("      spatial_window,")
    print("      temporal_window")
    print("  )")
    print("  # Returns: [2, T, H, W] array with p_t and c_t")


def example_2_temporal_delta():
    """Example 2: Temporal Differences (ls8_delta)"""
    print_section("Example 2: Temporal Differences (ls8_delta)")
    
    print("Temporal differences compute year-to-year changes:")
    print("  delta[t] = value[t+1] - value[t]  for t = 0..T-2")
    print()
    print("Returns T-1 differences (no prepend needed)")
    print()
    
    # Simulate NDVI time series
    np.random.seed(42)
    T = 10
    
    # Create synthetic NDVI with trend + noise
    t = np.arange(T)
    ndvi = 0.6 + 0.02 * t + 0.05 * np.random.randn(T)
    
    # Compute delta (T-1 values)
    delta = np.diff(ndvi)
    
    print("Year | NDVI   | Delta")
    print("-" * 30)
    for i in range(T):
        if i < T - 1:
            print(f"{2011+i:4d} | {ndvi[i]:6.3f} | {delta[i]:6.3f}")
        else:
            print(f"{2011+i:4d} | {ndvi[i]:6.3f} | (no delta)")
    
    print()
    print("Key features:")
    print("  - Shape: [C, T-1, H, W] where T=10 -> 9 differences")
    print("  - Captures year-to-year variability")
    print("  - Sensitive to disturbances and recovery")
    print("  - No artificial first timestep")
    print()
    print("Example usage:")
    print("  result = builder.build_derived_feature(")
    print("      'ls8_delta',")
    print("      spatial_window,")
    print("      temporal_window")
    print("  )")
    print("  # Returns: [4, 9, H, W] array with 4 delta bands, 9 timesteps")


def example_3_full_workflow():
    """Example 3: Complete Workflow"""
    print_section("Example 3: Complete Integration Workflow")
    
    print("This example shows how to use derived features in a training pipeline.")
    print()
    print("Step 1: Initialize Components")
    print("-" * 40)
    print("""
    from bindings.parser import BindingsParser
    from data_reader import DataReader
    from mask_builder import MaskBuilder
    from derived_features import DerivedFeatureBuilder, DerivedFeatureIntegrator
    
    # Parse config
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    config = parser.parse()
    registry = BindingsRegistry(config)
    
    # Initialize readers
    reader = DataReader(config)
    mask_builder = MaskBuilder(registry, config)
    derived_builder = DerivedFeatureBuilder(config, reader, mask_builder)
    integrator = DerivedFeatureIntegrator(reader, mask_builder, derived_builder)
    """)
    
    print()
    print("Step 2: Read Data with Derived Features")
    print("-" * 40)
    print("""
    # Define windows
    spatial = SpatialWindow.from_upper_left_and_hw((0, 0), (256, 256))
    temporal = TemporalWindow(end_year=2024, window_length=10)
    
    # Read ls8day with temporal_position automatically injected
    data, names, mask, quality = integrator.read_with_derived(
        'temporal',
        'ls8day',
        spatial,
        temporal,
        include_derived=True
    )
    
    # data.shape: [9, 10, 256, 256]
    #   - 7 original ls8day bands
    #   - 2 temporal_position channels (p_t, c_t)
    
    print(f"Bands: {names}")
    # ['NDVI_summer_p95', 'NDVI_winter_max', ..., 'p_t', 'c_t']
    """)
    
    print()
    print("Step 3: Read Standalone Derived Features")
    print("-" * 40)
    print("""
    # Build ls8_delta separately
    delta_result = derived_builder.build_derived_feature(
        'ls8_delta',
        spatial,
        temporal
    )
    
    # delta_result.data.shape: [4, 10, 256, 256]
    # delta_result.band_names: ['dNDVI_summer_p95', 'dNDVI_winter_max', ...]
    # delta_result.mask.shape: [4, 10, 256, 256]
    # delta_result.quality.shape: [4, 10, 256, 256]
    
    # Access individual components
    delta_data = delta_result.data
    delta_mask = delta_result.mask
    delta_quality = delta_result.quality
    """)
    
    print()
    print("Step 4: Build All Enabled Derived Features")
    print("-" * 40)
    print("""
    # Build everything at once
    all_derived = derived_builder.build_all_derived_features(
        spatial,
        temporal
    )
    
    # all_derived is a dict: {'temporal_position': result1, 'ls8_delta': result2}
    
    for name, result in all_derived.items():
        print(f"{name}:")
        print(f"  Shape: {result.data.shape}")
        print(f"  Bands: {result.band_names}")
    """)


def example_4_pytorch_integration():
    """Example 4: PyTorch Dataset Integration"""
    print_section("Example 4: PyTorch Dataset Integration")
    
    print("Integrating derived features into a PyTorch Dataset:")
    print()
    print("""
import torch
from torch.utils.data import Dataset

class ForestDataset(Dataset):
    def __init__(self, config_path, zarr_path):
        # Parse config
        parser = BindingsParser(config_path)
        self.config = parser.parse()
        registry = BindingsRegistry(self.config)
        
        # Initialize readers
        self.reader = DataReader(self.config, zarr_path)
        self.mask_builder = MaskBuilder(registry, self.config)
        self.derived_builder = DerivedFeatureBuilder(
            self.config,
            self.reader,
            self.mask_builder
        )
        self.integrator = DerivedFeatureIntegrator(
            self.reader,
            self.mask_builder,
            self.derived_builder
        )
        
        # Pre-compute sample locations
        self.samples = self._generate_samples()
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Define windows
        spatial = SpatialWindow.from_center_and_hw(
            sample['center'],
            (256, 256)
        )
        temporal = TemporalWindow(
            end_year=sample['year'],
            window_length=10
        )
        
        # Read ls8day + temporal_position
        ls8_data, ls8_names, ls8_mask, ls8_quality = self.integrator.read_with_derived(
            'temporal', 'ls8day', spatial, temporal
        )
        
        # Read ls8_delta separately
        delta_result = self.derived_builder.build_derived_feature(
            'ls8_delta', spatial, temporal
        )
        
        # Read static data
        topo_data, topo_names, topo_mask, topo_quality = self.integrator.read_with_derived(
            'static', 'topo', spatial
        )
        
        # Convert to tensors
        return {
            'ls8': torch.from_numpy(ls8_data),
            'ls8_delta': torch.from_numpy(delta_result.data),
            'topo': torch.from_numpy(topo_data),
            'masks': {
                'ls8': torch.from_numpy(ls8_mask),
                'ls8_delta': torch.from_numpy(delta_result.mask),
                'topo': torch.from_numpy(topo_mask)
            },
            'quality': {
                'ls8': torch.from_numpy(ls8_quality),
                'ls8_delta': torch.from_numpy(delta_result.quality),
                'topo': torch.from_numpy(topo_quality)
            }
        }
    """)


def example_5_normalization():
    """Example 5: Normalization After Deriving"""
    print_section("Example 5: Normalization Pipeline")
    
    print("Derived features are returned UNNORMALIZED.")
    print("Normalize them using the normalization preset from the config:")
    print()
    print("""
from normalization import NormalizationManager
from zarr_stats_loader import ZarrStatsLoader

# Initialize normalization
stats_loader = ZarrStatsLoader(zarr_path)
norm_manager = NormalizationManager(config, stats_loader)

# Build derived feature (unnormalized)
delta_result = derived_builder.build_derived_feature(
    'ls8_delta',
    spatial,
    temporal
)

# Normalize each band according to its config
normalized_deltas = []
for i, band_name in enumerate(delta_result.band_names):
    # Get normalization preset for this band
    band_config = derived_builder.get_derived_band_metadata(
        'ls8_delta',
        band_name
    )
    
    # Normalize (will use 'zscore_computed' preset)
    # Note: 'computed' stats may need to be accumulated during training
    normalized = norm_manager.normalize_band(
        data=delta_result.data[i],
        group_path='derived/ls8_delta',  # Virtual path
        band_config={
            'array': band_name,
            'norm': band_config['norm']
        },
        mask=delta_result.mask[i]
    )
    
    normalized_deltas.append(normalized)

# Stack normalized bands
normalized_delta_data = np.stack(normalized_deltas, axis=0)
    """)
    
    print()
    print("Key points:")
    print("  - Derived features are ALWAYS returned unnormalized")
    print("  - Each band has its own normalization preset in the config")
    print("  - 'zscore_computed' means stats are computed during training")
    print("  - Masks are used to exclude invalid pixels from normalization")


def example_6_quality_masks():
    """Example 6: Masks and Quality Weights"""
    print_section("Example 6: Masks and Quality Weights")
    
    print("Derived features inherit masks and quality from source bands.")
    print()
    print("Example: ls8_delta")
    print("-" * 40)
    print("""
# Build ls8_delta
delta_result = derived_builder.build_derived_feature(
    'ls8_delta',
    spatial,
    temporal
)

# delta_result has:
#   - data: [4, 9, 256, 256] - The actual delta values
#   - mask: [4, 9, 256, 256] - Inherited from source bands
#   - quality: [4, 9, 256, 256] - Inherited from source bands

# Each delta band inherits from its source:
#   dNDVI_summer_p95 <- NDVI_summer_p95 (includes summer_obs_weight)
#   dNDVI_winter_max <- NDVI_winter_max (includes winter_obs_weight)

# Use masks and quality in training
valid = delta_result.mask  # [4, 9, 256, 256]
weights = delta_result.quality  # [4, 9, 256, 256]

# Apply to data
masked_data = delta_result.data * valid
weighted_data = masked_data * weights

# Or combine masks
combined_mask = valid.all(axis=0)  # [10, 256, 256] - all bands valid
    """)
    
    print()
    print("Example: temporal_position")
    print("-" * 40)
    print("""
# Build temporal_position
pos_result = derived_builder.build_derived_feature(
    'temporal_position',
    spatial,
    temporal
)

# Temporal position has:
#   - mask: all-True (no source to inherit from)
#   - quality: all-ones (equal weighting)

# This makes sense because temporal position is deterministic
# and has no quality variation
    """)


def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("  DERIVED FEATURES - Complete Usage Examples")
    print("=" * 80)
    
    examples = [
        example_1_temporal_position,
        example_2_temporal_delta,
        example_3_full_workflow,
        example_4_pytorch_integration,
        example_5_normalization,
        example_6_quality_masks
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\nError in example {i}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print()
    print("Derived features provide:")
    print("  ✓ Temporal position encoding for time-aware models")
    print("  ✓ Temporal differences to capture year-to-year changes")
    print("  ✓ Spatial gradients for edge/boundary detection (when enabled)")
    print("  ✓ Automatic mask and quality inheritance")
    print("  ✓ Seamless integration with DataReader and normalization")
    print()
    print("Key design principles:")
    print("  • Features are computed on-the-fly (not stored in Zarr)")
    print("  • Always returned unnormalized")
    print("  • Inherit masks/quality from source bands")
    print("  • Can be injected into existing groups or used standalone")
    print()
    print("Next steps:")
    print("  1. Enable desired features in your bindings YAML")
    print("  2. Initialize DerivedFeatureBuilder in your pipeline")
    print("  3. Use DerivedFeatureIntegrator for automatic injection")
    print("  4. Or build features manually for more control")
    print()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific example
        example_num = int(sys.argv[1])
        examples = [
            example_1_temporal_position,
            example_2_temporal_delta,
            example_3_full_workflow,
            example_4_pytorch_integration,
            example_5_normalization,
            example_6_quality_masks
        ]
        
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Example {example_num} not found. Available: 1-{len(examples)}")
    else:
        # Run all examples
        run_all_examples()
