"""
Example usage of ForestDataset and complete data loading pipeline.

Demonstrates:
- Config parsing
- Dataset creation
- DataLoader setup
- Batch inspection
- Training loop structure
"""

import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data.loaders.bindings.parser import BindingsParser
from data.loaders.forest_dataset import ForestDataset, forest_collate_fn


def example_basic_usage():
    """
    Basic example: Create dataset and inspect a single sample.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Parse bindings config
    bindings_parser = BindingsParser('config/frl_bindings_v0.yaml')
    bindings_config = bindings_parser.parse()
    print("✓ Bindings config parsed")
    
    # Parse training config
    with open('config/frl_training_v1.yaml') as f:
        training_config = yaml.safe_load(f)
    print("✓ Training config loaded")
    
    # Create dataset
    dataset = ForestDataset(
        bindings_config,
        training_config,
        split='train'
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Get a single sample
    bundle = dataset[0]
    print(f"\n✓ Loaded sample 0:")
    print(f"  Anchor: {bundle.anchor_id}")
    print(f"  Spatial window: {bundle.spatial_window}")
    
    # Inspect data shapes
    print(f"\n  Window t0 data:")
    for group_name, result in bundle.windows['t0'].temporal.items():
        print(f"    temporal.{group_name}: {result.data.shape}")
    
    for group_name, result in bundle.windows['t0'].snapshot.items():
        print(f"    snapshot.{group_name}: {result.data.shape}")
    
    print(f"\n  Static data:")
    for group_name, result in bundle.static.items():
        print(f"    static.{group_name}: {result.data.shape}")
    
    # Inspect masks
    print(f"\n  Window t0 masks:")
    for group_name, mask in bundle.windows['t0'].temporal_masks.items():
        print(f"    temporal_masks.{group_name}: {mask.data.shape}, dtype={mask.data.dtype}")
    
    # Inspect quality
    print(f"\n  Window t0 quality:")
    for group_name, quality in bundle.windows['t0'].temporal_quality.items():
        print(f"    temporal_quality.{group_name}: {quality.data.shape}")
    
    # Inspect derived features
    if bundle.windows['t0'].derived:
        print(f"\n  Derived features:")
        for feature_name, data in bundle.windows['t0'].derived.items():
            print(f"    {feature_name}: {data.shape}")
    
    print("\n" + "=" * 80 + "\n")


def example_dataloader():
    """
    Example: Create DataLoader and inspect batched data.
    """
    print("=" * 80)
    print("EXAMPLE 2: DataLoader with Batching")
    print("=" * 80)
    
    # Parse configs
    bindings_parser = BindingsParser('config/frl_bindings_v0.yaml')
    bindings_config = bindings_parser.parse()
    
    with open('config/frl_training_v1.yaml') as f:
        training_config = yaml.safe_load(f)
    
    # Create dataset
    dataset = ForestDataset(bindings_config, training_config, split='train')
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Create DataLoader
    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=forest_collate_fn,
        num_workers=0,  # Use 0 for debugging, increase for training
        pin_memory=True,
        shuffle=False  # Sampler handles shuffling
    )
    print(f"✓ DataLoader created: batch_size={batch_size}")
    
    # Get first batch
    batch = next(iter(dataloader))
    print(f"\n✓ Loaded batch:")
    print(f"  Batch size: {len(batch['anchor_ids'])}")
    print(f"  Anchor IDs: {batch['anchor_ids']}")
    
    # Inspect batched data shapes
    print(f"\n  Batched window t0 temporal data:")
    for group_name, tensor in batch['windows']['t0']['temporal'].items():
        print(f"    {group_name}: {tensor.shape}, dtype={tensor.dtype}")
    
    print(f"\n  Batched window t0 snapshot data:")
    for group_name, tensor in batch['windows']['t0']['snapshot'].items():
        print(f"    {group_name}: {tensor.shape}, dtype={tensor.dtype}")
    
    print(f"\n  Batched static data:")
    for group_name, tensor in batch['static'].items():
        print(f"    {group_name}: {tensor.shape}, dtype={tensor.dtype}")
    
    # Show that all windows are batched
    print(f"\n  All windows:")
    for window_label in ['t0', 't2', 't4']:
        n_temporal = len(batch['windows'][window_label]['temporal'])
        n_snapshot = len(batch['windows'][window_label]['snapshot'])
        print(f"    {window_label}: {n_temporal} temporal groups, {n_snapshot} snapshot groups")
    
    print("\n" + "=" * 80 + "\n")


def example_training_loop():
    import numpy as np
    """
    Example: Simulate a training loop structure.
    """
    print("=" * 80)
    print("EXAMPLE 3: Training Loop Structure")
    print("=" * 80)
    
    # Parse configs
    bindings_parser = BindingsParser('config/frl_bindings_v0.yaml')
    bindings_config = bindings_parser.parse()
    
    with open('config/frl_training_v1.yaml') as f:
        training_config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = ForestDataset(bindings_config, training_config, split='train')
    val_dataset = ForestDataset(bindings_config, training_config, split='val')
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=forest_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=forest_collate_fn,
        num_workers=0
    )
    
    # Simulate training loop
    num_epochs = 2
    
    for epoch in range(num_epochs):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 40}")
        
        # IMPORTANT: Regenerate samples at start of epoch
        train_dataset.on_epoch_start()
        
        # Training phase
        print(f"\nTraining:")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # Just show first 2 batches
                print(f"  ... ({len(train_loader)} total batches)")
                break
            
            # Access data
            ls8_t0 = batch['windows']['t0']['temporal']['ls8day']
            anchor_ids = batch['anchor_ids']
            
            print(f"  Batch {batch_idx + 1}: ls8_t0 shape={ls8_t0.shape}, anchors={anchor_ids}")
            
            # This is where you would:
            # - Move to device: batch = move_to_device(batch, device)
            # - Forward pass: outputs = model(batch)
            # - Compute loss: loss = criterion(outputs, targets)
            # - Backward: loss.backward()
            # - Optimizer step: optimizer.step()
        
        # Validation phase
        print(f"\nValidation:")
        val_dataset.on_epoch_start()  # Also regenerate val samples
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 2:
                print(f"  ... ({len(val_loader)} total batches)")
                break
            
            ls8_t0 = batch['windows']['t0']['temporal']['ls8day']
            print(f"  Val Batch {batch_idx + 1}: ls8_t0 shape={ls8_t0.shape}")
            
            # This is where you would:
            # - Forward pass (no grad): with torch.no_grad(): outputs = model(batch)
            # - Compute metrics: metrics.update(outputs, targets)
    
    print("\n" + "=" * 80 + "\n")


def example_data_inspection():
    """
    Example: Deep dive into data structure and values.
    """
    print("=" * 80)
    print("EXAMPLE 4: Data Inspection")
    print("=" * 80)
    
    # Parse configs
    bindings_parser = BindingsParser('config/frl_bindings_v0.yaml')
    bindings_config = bindings_parser.parse()
    
    with open('config/frl_training_v1.yaml') as f:
        training_config = yaml.safe_load(f)
    
    # Create dataset
    dataset = ForestDataset(bindings_config, training_config, split='train')
    
    # Get sample
    bundle = dataset[0]
    
    # Inspect ls8day data
    ls8_result = bundle.windows['t0'].temporal['ls8day']
    ls8_data = ls8_result.data  # [C, T, H, W]
    
    print(f"LS8day temporal data:")
    print(f"  Shape: {ls8_data.shape}")
    print(f"  Bands: {ls8_result.band_names}")
    print(f"  Dtype: {ls8_data.dtype}")
    print(f"  Min/Max: {np.nanmin(ls8_data):.4f} / {np.nanmax(ls8_data):.4f}")
    print(f"  Has NaN: {np.isnan(ls8_data).any()}")
    print(f"  NaN count: {np.isnan(ls8_data).sum()}")
    
    # Inspect mask
    ls8_mask = bundle.windows['t0'].temporal_masks['ls8day']
    print(f"\nLS8day mask:")
    print(f"  Shape: {ls8_mask.data.shape}")
    print(f"  Dtype: {ls8_mask.data.dtype}")
    print(f"  True pixels: {ls8_mask.data.sum()} / {ls8_mask.data.size}")
    print(f"  False means invalid: {ls8_mask.false_means_invalid}")
    
    # Inspect quality
    ls8_quality = bundle.windows['t0'].temporal_quality['ls8day']
    print(f"\nLS8day quality weights:")
    print(f"  Shape: {ls8_quality.data.shape}")
    print(f"  Dtype: {ls8_quality.data.dtype}")
    print(f"  Min/Max: {np.nanmin(ls8_quality.data):.4f} / {np.nanmax(ls8_quality.data):.4f}")
    print(f"  Mean: {np.nanmean(ls8_quality.data):.4f}")
    
    # Compare across windows
    print(f"\nComparing ls8day across windows:")
    for window_label in ['t0', 't2', 't4']:
        window_data = bundle.windows[window_label]
        ls8 = window_data.temporal['ls8day']
        year = window_data.year
        print(f"  {window_label} (year={year}): shape={ls8.data.shape}, mean={np.nanmean(ls8.data):.4f}")
    
    # Inspect static data
    topo_result = bundle.static['topo']
    print(f"\nTopography static data:")
    print(f"  Shape: {topo_result.data.shape}")
    print(f"  Bands: {topo_result.band_names}")
    print(f"  Min/Max: {np.nanmin(topo_result.data):.4f} / {np.nanmax(topo_result.data):.4f}")
    
    print("\n" + "=" * 80 + "\n")


def example_all():
    """Run all examples."""
    example_basic_usage()
    example_dataloader()
    example_training_loop()
    example_data_inspection()


if __name__ == '__main__':
    import sys
    import numpy as np
    
    # Allow running specific examples
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_basic_usage,
            example_dataloader,
            example_training_loop,
            example_data_inspection
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run all examples
        example_all()
