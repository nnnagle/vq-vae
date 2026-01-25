#!/usr/bin/env python3
"""
Example script to inspect computed statistics.

This script loads and displays the computed statistics JSON file.
"""

import json
import numpy as np
from pathlib import Path


def main():
    """Load and display statistics."""

    # Path to stats file
    stats_path = "/data/VA/zarr/va_vae_dataset_test_stats.json"

    if not Path(stats_path).exists():
        print(f"Stats file not found: {stats_path}")
        print("Run example_compute_stats.py first!")
        return

    # Load stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    print(f"Loaded stats from: {stats_path}\n")

    # Display stats for each feature
    for feature_name, feature_stats in stats.items():
        print(f"Feature: {feature_name}")
        print("=" * 60)

        # Check if this is a dict or has covariance
        has_covariance = 'covariance' in feature_stats

        if has_covariance:
            # Print channel stats
            for channel_name, channel_stats in feature_stats.items():
                if channel_name == 'covariance':
                    continue

                print(f"\n  Channel: {channel_name}")
                for stat_name, stat_value in channel_stats.items():
                    print(f"    {stat_name}: {stat_value:.6f}")

            # Print covariance
            cov_matrix = np.array(feature_stats['covariance'])
            print(f"\n  Covariance matrix: {cov_matrix.shape}")
            print("  " + str(cov_matrix).replace('\n', '\n  '))
        else:
            # Just channel stats
            for channel_name, channel_stats in feature_stats.items():
                print(f"\n  Channel: {channel_name}")
                for stat_name, stat_value in channel_stats.items():
                    print(f"    {stat_name}: {stat_value:.6f}")

        print("\n")


if __name__ == "__main__":
    main()
