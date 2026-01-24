"""
Example script demonstrating how to use ForestDatasetV2 with the new bindings format.

This script shows:
1. Loading configuration from YAML
2. Creating a dataset with train/val/test splits
3. Loading individual samples
4. Using the dataset with PyTorch DataLoader
5. Accessing data by channel name (not just index)
"""

import numpy as np
from pathlib import Path

from frl.data.loaders.config import DatasetBindingsParser
from frl.data.loaders.dataset import ForestDatasetV2, collate_fn


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_1_load_config():
    """Example 1: Load and inspect the bindings configuration."""
    print_section("Example 1: Loading Configuration")

    yaml_path = "frl/config/forest_repr_model_bindings.yaml"

    print(f"\n📄 Loading config from: {yaml_path}")
    parser = DatasetBindingsParser(yaml_path)
    config = parser.parse()

    print(f"\n✓ Configuration loaded successfully!")
    print(f"  Version: {config.version}")
    print(f"  Name: {config.name}")
    print(f"  Zarr path: {config.zarr.path}")
    print(f"  Time window: {config.time_window.start}-{config.time_window.end} ({config.time_window.n_years} years)")

    print(f"\n📊 Dataset groups:")
    for group_name, group in config.dataset_groups.items():
        print(f"\n  {group_name}:")
        print(f"    - Type: {group.dtype}")
        print(f"    - Dimensions: {group.dim}")
        print(f"    - Channels ({len(group.channels)}):")
        for i, ch in enumerate(group.channels[:3]):  # Show first 3
            ch_type = "formula" if ch.is_formula_based() else "source"
            print(f"      {i}. {ch.name} ({ch_type})")
        if len(group.channels) > 3:
            print(f"      ... and {len(group.channels) - 3} more")

    return config


def example_2_create_dataset(config):
    """Example 2: Create a dataset instance."""
    print_section("Example 2: Creating Dataset")

    # Check if zarr exists
    if not Path(config.zarr.path).exists():
        print(f"\n⚠️  Zarr dataset not found at: {config.zarr.path}")
        print("    This example requires the zarr dataset to be available.")
        print("    Skipping remaining examples.")
        return None

    print(f"\n📦 Creating training dataset...")
    print(f"  - Split: train")
    print(f"  - Patch size: 256")
    print(f"  - Using debug window for fast testing")

    try:
        dataset = ForestDatasetV2(
            config,
            split='train',           # Use training split
            patch_size=256,          # 256x256 patches
            min_aoi_fraction=0.3,    # Require 30% valid AOI
            epoch_mode='number',     # Fixed number per epoch
            sample_number=10,        # Just 10 samples for demo
            debug_window=((0, 0), (1024, 1024)),  # Small spatial region
        )

        print(f"\n✓ Dataset created successfully!")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Split: {dataset.split}")
        print(f"  - Patch size: {dataset.patch_size}")

        return dataset

    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_3_load_sample(dataset):
    """Example 3: Load and inspect a single sample."""
    print_section("Example 3: Loading a Single Sample")

    print(f"\n📥 Loading sample 0...")
    sample = dataset[0]

    print(f"\n✓ Sample loaded! Keys: {list(sample.keys())}")

    print(f"\n📊 Data shapes and statistics:")
    for key, value in sample.items():
        if key == 'metadata':
            continue
        print(f"\n  {key}:")
        print(f"    - Shape: {value.shape}")
        print(f"    - Dtype: {value.dtype}")
        print(f"    - Min: {np.nanmin(value):.4f}")
        print(f"    - Max: {np.nanmax(value):.4f}")
        print(f"    - Mean: {np.nanmean(value):.4f}")
        print(f"    - NaN count: {np.isnan(value).sum()} / {value.size} pixels")

    return sample


def example_4_access_by_name(sample):
    """Example 4: Access channels by name instead of index."""
    print_section("Example 4: Accessing Channels by Name")

    metadata = sample['metadata']

    print("\n📋 Channel names for each group:")
    for group_name, channel_names in metadata['channel_names'].items():
        print(f"\n  {group_name}:")
        for i, name in enumerate(channel_names):
            print(f"    [{i}] {name}")

    # Example: Extract specific channels by name
    print("\n🎯 Extracting specific channels:")

    # Static channels
    static_data = sample['static']
    static_names = metadata['channel_names']['static']

    if 'elevation' in static_names:
        elevation_idx = static_names.index('elevation')
        elevation = static_data[elevation_idx]
        print(f"\n  Elevation (index {elevation_idx}):")
        print(f"    - Shape: {elevation.shape}")
        print(f"    - Range: [{np.nanmin(elevation):.1f}, {np.nanmax(elevation):.1f}]")

    # Annual temporal channels
    annual_data = sample['annual']
    annual_names = metadata['channel_names']['annual']

    if 'temporal_position' in annual_names:
        temp_pos_idx = annual_names.index('temporal_position')
        temp_pos = annual_data[temp_pos_idx]
        print(f"\n  Temporal position (index {temp_pos_idx}):")
        print(f"    - Shape: {temp_pos.shape}")
        print(f"    - Range: [{np.nanmin(temp_pos):.3f}, {np.nanmax(temp_pos):.3f}]")
        print(f"    - First timestep: {temp_pos[0, 0, 0]:.3f}")
        print(f"    - Last timestep: {temp_pos[-1, 0, 0]:.3f}")

    # Masks
    static_mask = sample['static_mask']
    mask_names = metadata['channel_names']['static_mask']

    if 'aoi' in mask_names:
        aoi_idx = mask_names.index('aoi')
        aoi = static_mask[aoi_idx]
        valid_pixels = aoi.sum()
        total_pixels = aoi.size
        print(f"\n  AOI mask (index {aoi_idx}):")
        print(f"    - Shape: {aoi.shape}")
        print(f"    - Valid pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")

    if 'forest' in mask_names:
        forest_idx = mask_names.index('forest')
        forest = static_mask[forest_idx]
        forest_pixels = forest.sum()
        print(f"\n  Forest mask (index {forest_idx}):")
        print(f"    - Shape: {forest.shape}")
        print(f"    - Forest pixels: {forest_pixels} / {total_pixels} ({100*forest_pixels/total_pixels:.1f}%)")


def example_5_dataloader(dataset):
    """Example 5: Use with PyTorch DataLoader."""
    print_section("Example 5: Using with DataLoader")

    try:
        from torch.utils.data import DataLoader
    except ImportError:
        print("\n⚠️  PyTorch not available, skipping DataLoader example")
        return

    print(f"\n🔄 Creating DataLoader...")
    print(f"  - Batch size: 4")
    print(f"  - Shuffle: True")
    print(f"  - Num workers: 0 (single process)")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for debugging, increase for production
    )

    print(f"\n✓ DataLoader created!")
    print(f"  - Total batches: {len(dataloader)}")

    # Load first batch
    print(f"\n📥 Loading first batch...")
    batch = next(iter(dataloader))

    print(f"\n✓ Batch loaded!")
    print(f"\n📊 Batch structure:")
    for key, value in batch.items():
        if key == 'metadata':
            print(f"  {key}: list of {len(value)} dicts")
        else:
            print(f"  {key}:")
            print(f"    - Shape: {value.shape}")
            print(f"    - Dtype: {value.dtype}")

            # Show dimension meaning
            if len(value.shape) == 4:
                print(f"    - Dimensions: [Batch={value.shape[0]}, Channels={value.shape[1]}, Height={value.shape[2]}, Width={value.shape[3]}]")
            elif len(value.shape) == 5:
                print(f"    - Dimensions: [Batch={value.shape[0]}, Channels={value.shape[1]}, Time={value.shape[2]}, Height={value.shape[3]}, Width={value.shape[4]}]")

    print(f"\n💡 Accessing batched data:")
    print(f"  - batch['static'] has shape [B, C, H, W]")
    print(f"  - batch['annual'] has shape [B, C, T, H, W]")
    print(f"  - batch['metadata'] is a list of {len(batch['metadata'])} metadata dicts")


def example_6_multiple_splits():
    """Example 6: Create datasets for train/val/test splits."""
    print_section("Example 6: Train/Val/Test Splits")

    yaml_path = "frl/config/forest_repr_model_bindings.yaml"
    config = DatasetBindingsParser(yaml_path).parse()

    if not Path(config.zarr.path).exists():
        print(f"\n⚠️  Zarr dataset not found, skipping split example")
        return

    print(f"\n📊 Creating datasets for all splits...")

    splits = ['train', 'val', 'test']
    datasets = {}

    for split in splits:
        try:
            ds = ForestDatasetV2(
                config,
                split=split,
                patch_size=256,
                epoch_mode='full',  # Use all patches
                debug_window=((0, 0), (2048, 2048)),
            )
            datasets[split] = ds
            print(f"  ✓ {split:5s}: {len(ds):4d} samples")
        except Exception as e:
            print(f"  ✗ {split:5s}: Error - {e}")

    if len(datasets) == 3:
        total = sum(len(ds) for ds in datasets.values())
        print(f"\n  Total: {total} samples across all splits")
        print(f"  Train: {len(datasets['train'])/total*100:.1f}%")
        print(f"  Val:   {len(datasets['val'])/total*100:.1f}%")
        print(f"  Test:  {len(datasets['test'])/total*100:.1f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  ForestDatasetV2 Usage Examples")
    print("=" * 70)

    # Example 1: Load config
    config = example_1_load_config()

    # Example 2: Create dataset
    dataset = example_2_create_dataset(config)

    if dataset is None:
        print("\n⚠️  Cannot continue examples without dataset")
        return

    # Example 3: Load sample
    sample = example_3_load_sample(dataset)

    # Example 4: Access by name
    example_4_access_by_name(sample)

    # Example 5: DataLoader
    example_5_dataloader(dataset)

    # Example 6: Multiple splits
    example_6_multiple_splits()

    print("\n" + "=" * 70)
    print("  ✓ All examples completed!")
    print("=" * 70)

    print("\n💡 Next steps:")
    print("  1. Modify the YAML config to add/remove channels")
    print("  2. Adjust patch_size, epoch_mode, sample_number for your use case")
    print("  3. Increase num_workers in DataLoader for faster loading")
    print("  4. Implement FeatureProcessor for normalization and masking")
    print()


if __name__ == "__main__":
    main()
