# Data Loading Refactor: Window-Stacked Arrays

## Overview

Refactored the data loading pipeline to return window-stacked arrays instead of nested dictionaries. This optimization enables:
- Efficient gradient masking for selective backpropagation
- Better compatibility with distributed training
- Cleaner encoder architecture with batch processing

## Changes

### 1. TrainingBundle Structure (data_bundle.py)

**Before:**
```python
bundle.windows = {
    't0': WindowData(temporal={'ls8': (C,T,H,W)}, ...),
    't2': WindowData(temporal={'ls8': (C,T,H,W)}, ...),
    't4': WindowData(temporal={'ls8': (C,T,H,W)}, ...)
}
bundle.static = {'topo': (C,H,W)}
```

**After:**
```python
bundle.temporal = {'ls8': (Win, C, T, H, W)}  # Stacked across windows
bundle.snapshot = {'ccdc': (Win, C, H, W)}    # Stacked across windows
bundle.static = {'topo': (C, H, W)}           # No change
bundle.anchor_mask = [0., 1., 0.]             # NEW: Binary mask (Win,)
```

### 2. Collate Function (forest_dataset.py)

**Before:** Complex nested dict collation across windows

**After:** Simple array stacking
```python
# Input: List of TrainingBundles, each with (Win, C, T, H, W) arrays
# Output: Batched dict with (B, Win, C, T, H, W) arrays

batch = {
    'temporal': {'ls8': Tensor(B, Win, C, T, H, W)},
    'snapshot': {'ccdc': Tensor(B, Win, C, H, W)},
    'static': {'topo': Tensor(B, C, H, W)},
    'anchor_mask': Tensor(B, Win),  # Stacked anchor masks
}
```

### 3. BundleBuilder (data_bundle.py)

- Reads data for each window
- Stacks arrays across Win dimension using `np.stack(arrays, axis=0)`
- Creates binary anchor_mask indicating which window is the anchor
- All stacking happens during bundle creation (dataset.__getitem__)

## Benefits

### 1. Gradient Masking
```python
# Efficient per-sample gradient control
mask = anchor_mask.view(B, Win, 1, 1, 1, 1)
x_masked = mask * x + (1 - mask) * x.detach()
```

### 2. Efficient Batching
- Single `np.stack()` operation in collate instead of nested loops
- All windows processed in one forward pass: `(B*Win, C, T, H, W)`

### 3. Distributed Training Ready
- Works with `torch.utils.data.distributed.DistributedSampler`
- Each GPU process gets its own samples with correct anchor masks
- No special handling needed for multi-GPU

## Usage Example

```python
# Dataset
dataset = ForestDataset(bindings_config, training_config, normalize=True)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=forest_collate_fn,
    num_workers=4
)

# Training
for batch in dataloader:
    # Extract data
    ls8 = batch['temporal']['ls8day']  # (4, 3, 7, 10, 256, 256)
    anchor_mask = batch['anchor_mask']  # (4, 3)

    # Apply gradient masking
    mask = anchor_mask.view(B, Win, 1, 1, 1, 1)
    ls8_masked = mask * ls8 + (1 - mask) * ls8.detach()

    # Flatten for processing
    ls8_flat = ls8_masked.view(B * Win, C, T, H, W)  # (12, 7, 10, 256, 256)

    # Forward pass
    outputs = model(ls8_flat)  # Process all windows together
```

## Distributed Training

Works seamlessly with DistributedDataParallel:

```python
from torch.utils.data.distributed import DistributedSampler

# Each GPU process
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=forest_collate_fn)

# Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for proper shuffling
    for batch in dataloader:
        # Each GPU gets different samples with correct anchor masks
        outputs = model(batch)
```

## Migration Notes

### Code that needs updating:

1. **Encoder forward pass**: Now receives `batch` dict with stacked arrays
2. **Loss computation**: Use `anchor_mask` to select anchor windows
3. **Data access**: Change from `batch['windows']['t0']['temporal']['ls8']` to `batch['temporal']['ls8']`

### Backward compatibility:

The old nested dict structure is **not** compatible. All code using TrainingBundle must be updated.

## Files Modified

- `frl/data/loaders/builders/data_bundle.py` - New TrainingBundle with stacked arrays
- `frl/data/loaders/dataset/forest_dataset.py` - Simplified collate function
- Branch: `refactor/stacked-window-arrays`

## Testing Checklist

- [ ] Single sample shape validation
- [ ] Batch collation correctness
- [ ] Anchor mask correctness
- [ ] Gradient masking works
- [ ] Distributed training compatibility
- [ ] Memory usage comparison

## Performance Impact

**Expected improvements:**
- Faster collation (fewer Python loops)
- Efficient gradient masking (no per-sample loops)
- Better GPU utilization (process B*Win samples together)

**Memory:**
- Similar total memory (same data, different organization)
- Slightly higher peak during collation (all windows stacked at once)
