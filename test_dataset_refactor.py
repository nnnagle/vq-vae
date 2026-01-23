"""
Test script for the refactored dataset loader.
"""

import numpy as np
from pathlib import Path

from frl.data.loaders.config import DatasetBindingsParser
from frl.data.loaders.dataset import ForestDatasetV2, collate_fn
from torch.utils.data import DataLoader


def test_parser():
    """Test that the parser can load the new YAML."""
    print("=" * 60)
    print("Testing DatasetBindingsParser...")
    print("=" * 60)

    yaml_path = Path("frl/config/forest_repr_model_bindings.yaml")
    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        return None

    parser = DatasetBindingsParser(yaml_path)
    config = parser.parse()

    print(f"\n✓ Parsed config successfully!")
    print(f"  - Version: {config.version}")
    print(f"  - Name: {config.name}")
    print(f"  - Zarr path: {config.zarr.path}")
    print(f"  - Time window: {config.time_window.start} - {config.time_window.end}")
    print(f"  - Number of years: {config.time_window.n_years}")

    print(f"\nDataset groups:")
    for name, group in config.dataset_groups.items():
        print(f"  - {name}:")
        print(f"      type: {group.dtype}")
        print(f"      dim: {group.dim}")
        print(f"      channels: {len(group.channels)}")
        print(f"      channel names: {', '.join(group.get_channel_names())}")

    print(f"\nAll source paths ({len(config.get_all_source_paths())} total):")
    for path in config.get_all_source_paths()[:5]:  # Show first 5
        print(f"  - {path}")
    if len(config.get_all_source_paths()) > 5:
        print(f"  ... and {len(config.get_all_source_paths()) - 5} more")

    return config


def test_dataset(config):
    """Test that the dataset can load data."""
    print("\n" + "=" * 60)
    print("Testing ForestDatasetV2...")
    print("=" * 60)

    # Check if zarr exists
    if not Path(config.zarr.path).exists():
        print(f"\nWARNING: Zarr dataset not found at {config.zarr.path}")
        print("Skipping dataset loading test.")
        return

    try:
        # Create dataset with debug window for fast testing
        print("\nCreating dataset with debug window...")
        dataset = ForestDatasetV2(
            config,
            split='train',
            patch_size=256,
            debug_window=((0, 0), (512, 512)),  # Small debug region
            epoch_mode='number',
            sample_number=2,  # Just 2 samples for testing
        )

        print(f"✓ Dataset created successfully!")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Split: {dataset.split}")

        # Load first sample
        print("\nLoading first sample...")
        sample = dataset[0]

        print(f"✓ Sample loaded successfully!")
        print(f"\nSample contents:")
        for key, value in sample.items():
            if key == 'metadata':
                print(f"  - metadata:")
                for mk, mv in value.items():
                    if mk == 'channel_names':
                        print(f"      - channel_names:")
                        for gname, cnames in mv.items():
                            print(f"          {gname}: {cnames}")
                    elif mk == 'spatial_window':
                        print(f"      - spatial_window: {mv.bounds}")
                    else:
                        print(f"      - {mk}: {mv}")
            else:
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"      min={np.nanmin(value):.3f}, max={np.nanmax(value):.3f}, "
                      f"nan_count={np.isnan(value).sum()}")

        # Test DataLoader
        print("\n" + "-" * 60)
        print("Testing DataLoader with collate_fn...")
        print("-" * 60)

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )

        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully!")
        print(f"\nBatch contents:")
        for key, value in batch.items():
            if key == 'metadata':
                print(f"  - metadata: list of {len(value)} dicts")
            else:
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during dataset testing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DATASET REFACTOR TEST SUITE")
    print("=" * 60 + "\n")

    # Test parser
    config = test_parser()

    if config is None:
        print("\n✗ Parser test failed, stopping.")
        return

    # Test dataset
    test_dataset(config)


if __name__ == "__main__":
    main()
