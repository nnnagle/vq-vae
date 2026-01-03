#!/usr/bin/env python3
"""
Demonstration of derived feature normalization in ForestDataset.

Shows statistics for static layers and derived features before and after normalization.
"""

import sys
import numpy as np
from pathlib import Path

# Add frl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.loaders.config import BindingsParser
from data.loaders.dataset import ForestDataset
import yaml


def compute_nan_safe_stats(data: np.ndarray, name: str) -> dict:
    """
    Compute NaN-safe statistics for an array.

    Args:
        data: Input array of any shape
        name: Name for logging

    Returns:
        Dictionary of statistics
    """
    # Flatten for statistics
    flat = data.flatten()

    # Remove NaNs and Infs
    valid = flat[np.isfinite(flat)]

    if len(valid) == 0:
        return {
            'name': name,
            'shape': data.shape,
            'n_valid': 0,
            'n_nan': np.isnan(flat).sum(),
            'n_inf': np.isinf(flat).sum(),
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q50': np.nan,
            'q75': np.nan,
        }

    return {
        'name': name,
        'shape': data.shape,
        'n_valid': len(valid),
        'n_nan': np.isnan(flat).sum(),
        'n_inf': np.isinf(flat).sum(),
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid),
        'q25': np.percentile(valid, 25),
        'q50': np.percentile(valid, 50),
        'q75': np.percentile(valid, 75),
    }


def print_stats(stats: dict, indent: str = "  "):
    """Pretty print statistics."""
    print(f"{indent}Shape: {stats['shape']}")
    print(f"{indent}Valid pixels: {stats['n_valid']:,} "
          f"(NaN: {stats['n_nan']}, Inf: {stats['n_inf']})")

    if stats['n_valid'] > 0:
        print(f"{indent}Mean:   {stats['mean']:8.4f}")
        print(f"{indent}Std:    {stats['std']:8.4f}")
        print(f"{indent}Min:    {stats['min']:8.4f}")
        print(f"{indent}Q25:    {stats['q25']:8.4f}")
        print(f"{indent}Median: {stats['q50']:8.4f}")
        print(f"{indent}Q75:    {stats['q75']:8.4f}")
        print(f"{indent}Max:    {stats['max']:8.4f}")
    else:
        print(f"{indent}(all NaN/Inf)")


def main():
    print("=" * 80)
    print("ForestDataset Normalization Demonstration")
    print("=" * 80)

    # Parse configs
    print("\n1. Loading configurations...")
    config_dir = Path(__file__).parent.parent.parent / 'config'

    bindings_path = config_dir / 'frl_bindings_v0.yaml'
    training_path = config_dir / 'frl_training_v1.yaml'

    if not bindings_path.exists():
        print(f"   ✗ Bindings config not found: {bindings_path}")
        return

    if not training_path.exists():
        print(f"   ✗ Training config not found: {training_path}")
        return

    print(f"   ✓ Loading bindings from: {bindings_path}")
    bindings_parser = BindingsParser(str(bindings_path))
    bindings_config = bindings_parser.parse()

    print(f"   ✓ Loading training config from: {training_path}")
    with open(training_path) as f:
        training_config = yaml.safe_load(f)

    # Create datasets
    print("\n2. Creating datasets...")

    print("   Creating dataset WITHOUT normalization...")
    try:
        dataset_raw = ForestDataset(
            bindings_config,
            training_config,
            split='train',
            normalize_derived=False
        )
        print(f"   ✓ Raw dataset created: {len(dataset_raw)} samples")
    except Exception as e:
        print(f"   ✗ Failed to create raw dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n   Creating dataset WITH normalization...")
    try:
        dataset_norm = ForestDataset(
            bindings_config,
            training_config,
            split='train',
            normalize_derived=True
        )
        print(f"   ✓ Normalized dataset created: {len(dataset_norm)} samples")
    except Exception as e:
        print(f"   ✗ Failed to create normalized dataset: {e}")
        print(f"   Note: You may need to run DerivedStatsComputer first.")
        import traceback
        traceback.print_exc()
        # Continue with raw dataset only
        dataset_norm = None

    # Get a sample
    print("\n3. Loading sample data...")
    try:
        print("   Loading sample from raw dataset...")
        bundle_raw = dataset_raw[0]
        print(f"   ✓ Raw sample loaded (anchor: {bundle_raw.anchor_id})")

        if dataset_norm is not None:
            print("   Loading sample from normalized dataset...")
            bundle_norm = dataset_norm[0]
            print(f"   ✓ Normalized sample loaded (anchor: {bundle_norm.anchor_id})")
        else:
            bundle_norm = None
    except Exception as e:
        print(f"   ✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze static layer
    print("\n" + "=" * 80)
    print("STATIC LAYER ANALYSIS: elevation (from topo group)")
    print("=" * 80)

    # Get elevation from static topo group
    if 'topo' in bundle_raw.static:
        elevation_raw = bundle_raw.static['topo'].data[0]  # First channel is elevation

        print("\nBEFORE normalization (raw data):")
        stats_raw = compute_nan_safe_stats(elevation_raw, 'elevation')
        print_stats(stats_raw)

        if bundle_norm is not None and 'topo' in bundle_norm.static:
            # Note: Static layers are not normalized by this implementation
            # This just shows they remain the same
            elevation_norm = bundle_norm.static['topo'].data[0]

            print("\nAFTER normalization attempt:")
            print("  (Note: Static layers are NOT normalized by normalize_derived flag)")
            stats_norm = compute_nan_safe_stats(elevation_norm, 'elevation')
            print_stats(stats_norm)
    else:
        print("  ✗ 'topo' group not found in bundle")

    # Analyze derived ls8_delta
    print("\n" + "=" * 80)
    print("DERIVED FEATURE ANALYSIS: ls8_delta")
    print("=" * 80)

    # Check if ls8_delta exists in any window
    window_with_derived = None
    for window_label in ['t0', 't2', 't4']:
        if window_label in bundle_raw.windows:
            if 'ls8_delta' in bundle_raw.windows[window_label].derived:
                window_with_derived = window_label
                break

    if window_with_derived is not None:
        print(f"\nAnalyzing ls8_delta in window '{window_with_derived}'...")

        ls8_delta_raw = bundle_raw.windows[window_with_derived].derived['ls8_delta']

        # Show first channel statistics
        print("\nChannel 0 (dNDVI_summer_p95) BEFORE normalization:")
        stats_raw = compute_nan_safe_stats(ls8_delta_raw[0], 'ls8_delta[0]')
        print_stats(stats_raw)

        if bundle_norm is not None:
            if 'ls8_delta' in bundle_norm.windows[window_with_derived].derived:
                ls8_delta_norm = bundle_norm.windows[window_with_derived].derived['ls8_delta']

                print("\nChannel 0 (dNDVI_summer_p95) AFTER normalization:")
                stats_norm = compute_nan_safe_stats(ls8_delta_norm[0], 'ls8_delta[0]')
                print_stats(stats_norm)

                # Show normalization effect
                print("\n" + "-" * 80)
                print("NORMALIZATION EFFECT:")
                print("-" * 80)
                if stats_raw['n_valid'] > 0 and stats_norm['n_valid'] > 0:
                    print(f"  Mean:  {stats_raw['mean']:8.4f} → {stats_norm['mean']:8.4f} "
                          f"(delta: {stats_norm['mean'] - stats_raw['mean']:+.4f})")
                    print(f"  Std:   {stats_raw['std']:8.4f} → {stats_norm['std']:8.4f} "
                          f"(delta: {stats_norm['std'] - stats_raw['std']:+.4f})")
                    print(f"  Range: [{stats_raw['min']:.4f}, {stats_raw['max']:.4f}] → "
                          f"[{stats_norm['min']:.4f}, {stats_norm['max']:.4f}]")

                    # Check if normalized to ~N(0,1)
                    if abs(stats_norm['mean']) < 0.1 and abs(stats_norm['std'] - 1.0) < 0.2:
                        print(f"\n  ✓ Successfully normalized to ~N(0, 1)")
                    else:
                        print(f"\n  ⚠ Normalization applied but may not be N(0, 1)")
                        print(f"    Expected: mean ≈ 0, std ≈ 1")
                        print(f"    Got:      mean = {stats_norm['mean']:.4f}, std = {stats_norm['std']:.4f}")
            else:
                print("\n  ✗ ls8_delta not found in normalized bundle")
        else:
            print("\n  (Normalized dataset not available)")

        # Show all channels summary
        print("\n" + "-" * 80)
        print("ALL CHANNELS SUMMARY:")
        print("-" * 80)

        n_channels = ls8_delta_raw.shape[0]
        channel_names = [
            'dNDVI_summer_p95', 'dNDVI_winter_max', 'dNDVI_amplitude', 'dNBR_annual_min'
        ]

        for i in range(min(n_channels, len(channel_names))):
            channel_name = channel_names[i] if i < len(channel_names) else f'Channel {i}'

            raw_stats = compute_nan_safe_stats(ls8_delta_raw[i], f'ch{i}')

            if bundle_norm is not None and 'ls8_delta' in bundle_norm.windows[window_with_derived].derived:
                ls8_delta_norm = bundle_norm.windows[window_with_derived].derived['ls8_delta']
                norm_stats = compute_nan_safe_stats(ls8_delta_norm[i], f'ch{i}')

                print(f"\n  {channel_name}:")
                print(f"    Raw:        mean={raw_stats['mean']:7.4f}, std={raw_stats['std']:7.4f}")
                print(f"    Normalized: mean={norm_stats['mean']:7.4f}, std={norm_stats['std']:7.4f}")
            else:
                print(f"\n  {channel_name}:")
                print(f"    Raw:        mean={raw_stats['mean']:7.4f}, std={raw_stats['std']:7.4f}")
                print(f"    Normalized: (not available)")
    else:
        print("\n  ✗ ls8_delta not found in any window")
        print("\n  Available derived features:")
        for window_label in ['t0', 't2', 't4']:
            if window_label in bundle_raw.windows:
                derived_features = list(bundle_raw.windows[window_label].derived.keys())
                if derived_features:
                    print(f"    {window_label}: {', '.join(derived_features)}")
                else:
                    print(f"    {window_label}: (none)")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
