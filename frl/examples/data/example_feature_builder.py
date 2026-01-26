"""
Example script demonstrating how to use the FeatureBuilder.

This script shows:
1. Creating a FeatureBuilder from configuration
2. Building features from dataset samples
3. Inspecting feature configurations
4. Understanding masks and normalization
5. Working with covariance/Mahalanobis transforms

The FeatureBuilder takes raw tensors from ForestDatasetV2 samples and:
- Extracts and stacks channels according to feature definitions
- Applies masks (global feature masks, channel masks, NaN masks)
- Broadcasts spatial masks across time for temporal features
- Applies normalization (zscore, robust_iqr, etc.)
- For covariance features: centers and applies Mahalanobis transform
"""

import numpy as np
from pathlib import Path
import json

from data.loaders.config import DatasetBindingsParser
from data.loaders.dataset import ForestDatasetV2
from data.loaders.builders import FeatureBuilder, create_feature_builder_from_config


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_1_basic_usage():
    """Example 1: Basic FeatureBuilder usage."""
    print_section("Example 1: Basic FeatureBuilder Usage")

    config_path = 'config/frl_binding_v1.yaml'

    print(f"\n1. Loading config from: {config_path}")
    parser = DatasetBindingsParser(config_path)
    config = parser.parse()

    print("\n2. Creating dataset and feature builder...")
    dataset = ForestDatasetV2(config, split='train', min_aoi_fraction=0.3)
    builder = FeatureBuilder(config)

    print(f"\n   Dataset has {len(dataset)} patches")
    print(f"   Available features: {list(config.features.keys())}")

    print("\n3. Loading a sample and building features...")
    sample = dataset[0]

    # Build a static feature
    feature_result = builder.build_feature('topo', sample)

    print(f"\n   Feature: {feature_result.feature_name}")
    print(f"   Data shape: {feature_result.data.shape}")
    print(f"   Mask shape: {feature_result.mask.shape}")
    print(f"   Channels: {feature_result.channel_names}")
    print(f"   Is temporal: {feature_result.is_temporal}")

    # Check mask statistics
    valid_pixels = feature_result.mask.sum()
    total_pixels = feature_result.mask.size
    print(f"   Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")

    return builder, dataset, sample


def example_2_inspect_features():
    """Example 2: Inspect feature configurations."""
    print_section("Example 2: Feature Configuration Inspection")

    builder = create_feature_builder_from_config('config/frl_binding_v1.yaml')

    # List all features
    print("\nAvailable features and their properties:")

    for feature_name in builder.config.features.keys():
        info = builder.get_feature_info(feature_name)

        print(f"\n  {feature_name}:")
        print(f"    Dimensions: {info['dim']}")
        print(f"    Channels: {info['n_channels']}")
        print(f"    Has covariance: {info['has_covariance']}")
        if info['global_masks']:
            print(f"    Global masks: {info['global_masks']}")

        # Show channel details for first feature
        if feature_name == 'topo':
            print("    Channel details:")
            for ch in info['channels']:
                print(f"      - {ch['reference']}")
                print(f"        Source: {ch['dataset_group']}.{ch['channel_name']}")
                print(f"        Norm: {ch['normalization']}")
                if ch['mask']:
                    print(f"        Mask: {ch['mask']}")


def example_3_temporal_features():
    """Example 3: Working with temporal features."""
    print_section("Example 3: Temporal Features")

    config_path = 'config/frl_binding_v1.yaml'
    parser = DatasetBindingsParser(config_path)
    config = parser.parse()

    dataset = ForestDatasetV2(config, split='train', min_aoi_fraction=0.3)
    builder = FeatureBuilder(config)

    sample = dataset[0]

    # Build the phase_ls8 temporal feature
    print("\nBuilding temporal feature 'phase_ls8'...")
    feature_result = builder.build_feature('phase_ls8', sample)

    print(f"\n  Data shape: {feature_result.data.shape}")
    print(f"  Expected: [C={len(feature_result.channel_names)}, T=15, H=256, W=256]")
    print(f"  Mask shape: {feature_result.mask.shape}")
    print(f"  Expected: [T=15, H=256, W=256]")

    print("\n  Channels and their ranges (after normalization):")
    for c_idx, channel_name in enumerate(feature_result.channel_names):
        channel_data = feature_result.data[c_idx]
        valid_data = channel_data[feature_result.mask]
        if len(valid_data) > 0:
            print(f"    {channel_name}: [{valid_data.min():.3f}, {valid_data.max():.3f}], "
                  f"mean={valid_data.mean():.3f}")


def example_4_covariance_features():
    """Example 4: Features with covariance (Mahalanobis transform)."""
    print_section("Example 4: Covariance Features")

    config_path = 'config/frl_binding_v1.yaml'
    parser = DatasetBindingsParser(config_path)
    config = parser.parse()

    # Check stats file
    stats_path = Path(config.stats.file)
    if not stats_path.exists():
        print(f"\n  Stats file not found: {stats_path}")
        print("  Run example_compute_stats.py first to generate statistics.")
        return

    dataset = ForestDatasetV2(config, split='train', min_aoi_fraction=0.3)
    builder = FeatureBuilder(config)

    sample = dataset[0]

    # Build infonce_type_spectral which has covariance
    print("\nBuilding 'infonce_type_spectral' with Mahalanobis transform...")

    # First without Mahalanobis
    result_no_mah = builder.build_feature(
        'infonce_type_spectral', sample,
        apply_mahalanobis=False
    )

    # Then with Mahalanobis
    result_with_mah = builder.build_feature(
        'infonce_type_spectral', sample,
        apply_mahalanobis=True
    )

    print(f"\n  Channels: {result_with_mah.channel_names}")
    print(f"\n  Comparing with and without Mahalanobis transform:")

    # Compare covariance matrices
    mask = result_with_mah.mask
    n_channels = result_with_mah.data.shape[0]

    print("\n  Covariance of normalized data (without Mahalanobis):")
    flat_no_mah = result_no_mah.data[:, mask].reshape(n_channels, -1)
    cov_no_mah = np.cov(flat_no_mah)
    print(f"    Diagonal: {np.diag(cov_no_mah)}")

    print("\n  Covariance of transformed data (with Mahalanobis):")
    flat_with_mah = result_with_mah.data[:, mask].reshape(n_channels, -1)
    cov_with_mah = np.cov(flat_with_mah)
    print(f"    Diagonal: {np.diag(cov_with_mah)}")
    print(f"    (Should be close to identity matrix)")


def example_5_mask_details():
    """Example 5: Understanding mask application."""
    print_section("Example 5: Mask Details")

    config_path = 'config/frl_binding_v1.yaml'
    parser = DatasetBindingsParser(config_path)
    config = parser.parse()

    dataset = ForestDatasetV2(config, split='train', min_aoi_fraction=0.3)
    builder = FeatureBuilder(config)

    sample = dataset[0]

    # Build feature with global masks
    print("\nAnalyzing masks for 'infonce_type_topo'...")
    print("  This feature has global masks: aoi, forest, dem_mask")

    feature_result = builder.build_feature('infonce_type_topo', sample)

    # Also get individual masks for comparison
    static_mask = sample['static_mask']
    channel_names = sample['metadata']['channel_names']['static_mask']

    aoi_idx = channel_names.index('aoi')
    forest_idx = channel_names.index('forest')
    dem_mask_idx = channel_names.index('dem_mask')

    aoi = static_mask[aoi_idx] > 0
    forest = static_mask[forest_idx] > 0
    dem_mask = static_mask[dem_mask_idx] > 0

    print(f"\n  Individual mask coverage:")
    print(f"    AOI: {100*aoi.sum()/aoi.size:.1f}%")
    print(f"    Forest: {100*forest.sum()/forest.size:.1f}%")
    print(f"    DEM mask: {100*dem_mask.sum()/dem_mask.size:.1f}%")

    # Combined mask (AND of all)
    combined_manual = aoi & forest & dem_mask
    print(f"\n  Combined (AND): {100*combined_manual.sum()/combined_manual.size:.1f}%")

    # Feature builder result
    print(f"  Feature mask: {100*feature_result.mask.sum()/feature_result.mask.size:.1f}%")
    print(f"  (May differ due to NaN handling)")


def example_6_build_all_features():
    """Example 6: Build all features at once."""
    print_section("Example 6: Building All Features")

    config_path = 'config/frl_binding_v1.yaml'
    parser = DatasetBindingsParser(config_path)
    config = parser.parse()

    dataset = ForestDatasetV2(config, split='train', min_aoi_fraction=0.3)
    builder = FeatureBuilder(config)

    sample = dataset[0]

    print("\nBuilding all features...")
    all_features = builder.build_all_features(sample)

    print(f"\nSuccessfully built {len(all_features)} features:")
    for name, result in all_features.items():
        print(f"\n  {name}:")
        print(f"    Shape: {result.data.shape}")
        print(f"    Valid: {100*result.mask.sum()/result.mask.size:.1f}%")
        print(f"    Temporal: {result.is_temporal}")


def main():
    """Run all examples."""
    print("\n" + "#" * 70)
    print("#  FeatureBuilder Examples")
    print("#" * 70)

    try:
        example_1_basic_usage()
    except FileNotFoundError as e:
        print(f"\n  Skipped: {e}")

    example_2_inspect_features()

    try:
        example_3_temporal_features()
    except FileNotFoundError as e:
        print(f"\n  Skipped: {e}")

    try:
        example_4_covariance_features()
    except FileNotFoundError as e:
        print(f"\n  Skipped: {e}")

    try:
        example_5_mask_details()
    except FileNotFoundError as e:
        print(f"\n  Skipped: {e}")

    try:
        example_6_build_all_features()
    except FileNotFoundError as e:
        print(f"\n  Skipped: {e}")

    print("\n" + "=" * 70)
    print("  Examples complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
