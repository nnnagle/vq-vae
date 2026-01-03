# Forest Representation Learning - Data Loader Documentation

Complete documentation for the multi-window forest representation data loading pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Components](#components)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The data loading pipeline provides multi-window temporal training data for forest representation learning. Each training sample contains:

- **3 temporal windows** (t0, t2, t4) for years 2024, 2022, 2020
- **Per-window data**: Temporal sequences, snapshot metrics, irregular observations
- **Shared static data**: Topography, land cover types, historical metrics
- **Masks and quality weights**: Per-group validity masks and quality scores

### Key Features

- ✅ **Multi-window bundling**: Automatic loading of 3 temporal windows
- ✅ **Spatial splits**: Deterministic train/val/test using checkerboard pattern
- ✅ **Weighted sampling**: Configurable anchor year weights
- ✅ **Flexible epochs**: Full coverage or sampled batches
- ✅ **Raw data**: No normalization (compute stats during training)
- ✅ **Type-safe**: Strong typing with dataclasses

---

## Quick Start

### Installation

```bash
# Core dependencies
pip install numpy zarr torch pyyaml

# Optional: for xarray support
pip install xarray
```

### Minimal Example

```python
from data.loaders.bindings.parser import BindingsParser
from data.loaders.forest_dataset import ForestDataset, forest_collate_fn
from torch.utils.data import DataLoader
import yaml

# 1. Parse configurations
bindings_parser = BindingsParser('config/frl_bindings_v0.yaml')
bindings_config = bindings_parser.parse()

with open('config/frl_training_v1.yaml') as f:
    training_config = yaml.safe_load(f)

# 2. Create dataset
dataset = ForestDataset(bindings_config, training_config, split='train')

# 3. Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=forest_collate_fn,
    num_workers=4
)

# 4. Training loop
for epoch in range(num_epochs):
    dataset.on_epoch_start()  # Regenerate samples
    
    for batch in loader:
        # batch['windows']['t0']['temporal']['ls8day'] -> [B, C, T, H, W]
        # batch['anchor_ids'] -> ['t0', 't2', ...]
        train_step(batch)
```

---

## Architecture

### Data Flow

```
Configuration Files
    ↓
[BindingsParser] → Bindings Config (data structure)
[YAML Loader]    → Training Config (training params)
    ↓
[ForestPatchSampler] → (spatial_window, anchor_year) pairs
    ↓
[DataReader]         → Raw data from Zarr
[MaskBuilder]        → Masks and quality weights
    ↓
[BundleBuilder]      → TrainingBundle (complete sample)
    ↓
[ForestDataset]      → PyTorch Dataset interface
    ↓
[forest_collate_fn]  → Batched tensors
    ↓
Training Loop
```

### Component Hierarchy

```
ForestDataset
├── ForestPatchSampler
│   └── (generates sample specs)
├── DataReader
│   └── (reads raw Zarr data)
├── MaskBuilder
│   └── (reads masks/quality)
└── BundleBuilder
    └── (combines into TrainingBundle)
```

---

## Components

### 1. Configuration Parsers

**Purpose**: Parse and validate YAML configurations

**Files**:
- `data/loaders/bindings/parser.py` - Bindings parser
- Config files: `config/frl_bindings_v0.yaml`, `config/frl_training_v1.yaml`

**What they do**:
- Define data structure (groups, bands, masks, normalization)
- Define training parameters (spatial domain, sampling, losses)

### 2. ForestPatchSampler

**Purpose**: Generate (spatial_window, anchor_year) sample specifications

**File**: `data/loaders/forest_patch_sampler.py`

**What it does**:
- Loads AOI mask and computes valid patches
- Applies checkerboard spatial split (train/val/test)
- Samples anchor years with configurable weights
- Supports full/fractional/fixed epoch sizes

### 3. DataReader

**Purpose**: Read raw data arrays from Zarr

**File**: `data/loaders/data_reader.py`

**What it does**:
- Reads temporal, static, irregular, and snapshot data
- Handles spatial/temporal padding (NaN for out-of-bounds)
- Applies missing value policies
- Returns per-group results with band metadata

### 4. MaskBuilder

**Purpose**: Read validity masks and quality weights

**File**: `data/loaders/mask_builder.py`

**What it does**:
- Reads masks for each data group (shape-matched)
- Reads quality weights for each data group
- Combines multiple masks per band
- Always returns valid results (never None)

### 5. BundleBuilder

**Purpose**: Construct complete TrainingBundle objects

**File**: `data/loaders/data_bundle.py`

**What it does**:
- Orchestrates DataReader and MaskBuilder
- Builds WindowData for each temporal window (t0, t2, t4)
- Loads shared static data once
- Computes derived features (optional)
- Returns TrainingBundle with all data

### 6. ForestDataset

**Purpose**: PyTorch Dataset interface

**File**: `data/loaders/forest_dataset.py`

**What it does**:
- Wraps all components into Dataset API
- Handles epoch management
- Returns TrainingBundle objects
- Integrates with PyTorch DataLoader

### 7. Collate Function

**Purpose**: Batch TrainingBundle objects into tensors

**File**: `data/loaders/forest_dataset.py` (function `forest_collate_fn`)

**What it does**:
- Stacks individual bundles along batch dimension
- Preserves nested structure
- Converts numpy arrays to PyTorch tensors

---

## Configuration

### Bindings Configuration (`frl_bindings_v0.yaml`)

Defines the data structure:

```yaml
zarr:
  path: /path/to/dataset.zarr

inputs:
  temporal:      # Time series data
    ls8day:      # Landsat annual metrics
      bands: [...]
  
  static:        # Unchanging data
    topo:        # Topography
      bands: [...]
  
  snapshot:      # Year-specific data
    ccdc_snapshot:
      years: [2020, 2022, 2024]
      bands: [...]
  
  irregular:     # Sparse temporal
    naip:        # Aerial imagery
      years: [2011, 2014, ...]
      bands: [...]

shared:
  masks:         # Validity masks
    aoi: {...}
    forest: {...}
  
  quality:       # Quality weights
    forest_weight: {...}

normalization:   # Normalization presets
  zscore: {...}
  robust_iqr: {...}

derived:         # Derived features
  temporal_position:
    enabled: true
  ls8_delta:
    enabled: true
```

### Training Configuration (`frl_training_v1.yaml`)

Defines training parameters:

```yaml
spatial_domain:
  debug_mode: true
  debug_window:
    origin: [8192, 8192]  # [row, col]
    size: [1024, 1024]    # [H, W]
    block_grid: [7, 7]    # Checkerboard blocks
  
  full_domain:
    origin: [0, 0]
    size: [13056, 23552]
    block_grid: [7, 7]

temporal_domain:
  end_years: [2020, 2022, 2024]  # Fixed endpoints
  window_length: 10               # Years per window
  sampling:
    mode: weighted
    weights:
      2024: 0.4
      2022: 0.3
      2020: 0.3

training:
  epoch:
    num_epochs: 100
    mode: full          # full/frac/number
    sample_frac: 0.1    # If mode=frac
    sample_number: 1000 # If mode=number
    batch_size: 4

sampling:
  patch_size: 256
```

---

## API Reference

### ForestDataset

Main entry point for data loading.

```python
class ForestDataset(Dataset):
    def __init__(
        self,
        bindings_config: Dict[str, Any],
        training_config: Dict[str, Any],
        split: Optional[str] = None  # 'train', 'val', 'test', or None
    )
    
    def __len__(self) -> int
    
    def __getitem__(self, idx: int) -> TrainingBundle
    
    def on_epoch_start(self):
        """Call at start of each epoch to regenerate samples."""
```

**Example**:
```python
dataset = ForestDataset(bindings_config, training_config, split='train')
print(len(dataset))  # Number of samples this epoch
bundle = dataset[0]  # Get first sample
```

---

### TrainingBundle

Complete training sample with all data.

```python
@dataclass
class TrainingBundle:
    windows: Dict[str, WindowData]           # {'t0': ..., 't2': ..., 't4': ...}
    static: Dict[str, GroupReadResult]       # Static data (shared)
    static_masks: Dict[str, MaskResult]
    static_quality: Dict[str, QualityResult]
    anchor_id: str                           # 't0', 't2', or 't4'
    spatial_window: SpatialWindow
    metadata: Dict[str, Any]
```

**Access patterns**:
```python
# Get data for t0 window
ls8_data = bundle.windows['t0'].temporal['ls8day'].data  # [C, T, H, W]
ls8_mask = bundle.windows['t0'].temporal_masks['ls8day'].data
ls8_quality = bundle.windows['t0'].temporal_quality['ls8day'].data

# Get static data
topo = bundle.static['topo'].data  # [C, H, W]

# Check which window is anchor
if bundle.anchor_id == 't0':
    # This sample focuses on 2024
    pass
```

---

### WindowData

Data for one temporal window (t0, t2, or t4).

```python
@dataclass
class WindowData:
    temporal: Dict[str, GroupReadResult]     # [C, T, H, W] per group
    snapshot: Dict[str, GroupReadResult]     # [C, H, W] per group
    irregular: Dict[str, GroupReadResult]    # [C, T_obs, H, W] per group
    derived: Dict[str, np.ndarray]           # Derived features
    
    temporal_masks: Dict[str, MaskResult]    # Masks (match data shapes)
    snapshot_masks: Dict[str, MaskResult]
    irregular_masks: Dict[str, MaskResult]
    
    temporal_quality: Dict[str, QualityResult]  # Quality weights
    snapshot_quality: Dict[str, QualityResult]
    irregular_quality: Dict[str, QualityResult]
    
    year: int                                # End year (2024, 2022, 2020)
    window_label: str                        # 't0', 't2', 't4'
```

---

### forest_collate_fn

Collate function for DataLoader.

```python
def forest_collate_fn(batch: List[TrainingBundle]) -> Dict[str, Any]:
    """
    Collate list of bundles into batched dictionary.
    
    Args:
        batch: List of TrainingBundle objects
    
    Returns:
        Dictionary with batched structure:
        {
            'windows': {
                't0': {
                    'temporal': {'ls8day': Tensor[B,C,T,H,W], ...},
                    'snapshot': {...},
                    'temporal_masks': {...},
                    'temporal_quality': {...},
                    ...
                },
                't2': {...},
                't4': {...}
            },
            'static': {'topo': Tensor[B,C,H,W], ...},
            'static_masks': {...},
            'static_quality': {...},
            'anchor_ids': ['t0', 't2', ...],
            'spatial_windows': [...],
            'metadata': [...]
        }
    """
```

**Example**:
```python
loader = DataLoader(dataset, batch_size=4, collate_fn=forest_collate_fn)
batch = next(iter(loader))

# Access batched data
ls8_batch = batch['windows']['t0']['temporal']['ls8day']
print(ls8_batch.shape)  # [4, 7, 10, 256, 256]

# Get anchors
print(batch['anchor_ids'])  # ['t0', 't2', 't0', 't4']
```

---

### ForestPatchSampler

Generates sample specifications.

```python
class ForestPatchSampler:
    def __init__(
        self,
        bindings_config: Dict[str, Any],
        training_config: Dict[str, Any],
        split: Optional[str] = None
    )
    
    def __len__(self) -> int
    
    def __getitem__(self, idx: int) -> Tuple[SpatialWindow, int]:
        """Returns (spatial_window, anchor_year)"""
    
    def new_epoch(self):
        """Regenerate sample indices for new epoch."""
```

**Example**:
```python
sampler = ForestPatchSampler(bindings_config, training_config, 'train')
spatial_window, anchor_year = sampler[0]
print(spatial_window)  # SpatialWindow(row_start=1024, col_start=2048, ...)
print(anchor_year)     # 2024
```

---

### SpatialWindow

Defines a rectangular spatial region.

```python
@dataclass
class SpatialWindow:
    row_start: int
    col_start: int
    height: int
    width: int
    
    @classmethod
    def from_upper_left_and_hw(
        cls,
        upper_left: Tuple[int, int],
        hw: Tuple[int, int]
    ) -> SpatialWindow
```

**Example**:
```python
window = SpatialWindow.from_upper_left_and_hw(
    upper_left=(1000, 2000),
    hw=(256, 256)
)
```

---

### TemporalWindow

Defines a temporal range.

```python
@dataclass
class TemporalWindow:
    end_year: int         # Last year in window
    window_length: int    # Number of years
```

**Example**:
```python
window = TemporalWindow(end_year=2024, window_length=10)
# Covers years 2015-2024
```

---

## Usage Examples

### Example 1: Basic Training Loop

```python
from data.loaders.forest_dataset import ForestDataset, forest_collate_fn
from torch.utils.data import DataLoader

# Create datasets
train_dataset = ForestDataset(bindings_config, training_config, 'train')
val_dataset = ForestDataset(bindings_config, training_config, 'val')

# Create loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=forest_collate_fn,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    collate_fn=forest_collate_fn,
    num_workers=4
)

# Training loop
for epoch in range(num_epochs):
    # Regenerate samples
    train_dataset.on_epoch_start()
    val_dataset.on_epoch_start()
    
    # Train
    model.train()
    for batch in train_loader:
        # Move to device
        ls8_t0 = batch['windows']['t0']['temporal']['ls8day'].to(device)
        anchor_ids = batch['anchor_ids']
        
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            # Compute metrics...
```

---

### Example 2: Accessing Different Windows

```python
bundle = dataset[0]

# All three windows are available
for window_label in ['t0', 't2', 't4']:
    window = bundle.windows[window_label]
    print(f"\n{window_label} (year={window.year}):")
    
    # Temporal data
    ls8 = window.temporal['ls8day']
    print(f"  LS8: {ls8.data.shape}")
    
    # Snapshot data
    ccdc = window.snapshot['ccdc_snapshot']
    print(f"  CCDC: {ccdc.data.shape}")
    
    # Masks
    mask = window.temporal_masks['ls8day']
    print(f"  Mask: {mask.data.shape}, valid={mask.data.sum()}")
```

---

### Example 3: Working with Masks and Quality

```python
batch = next(iter(dataloader))

# Get data, mask, and quality for ls8day at t0
ls8_data = batch['windows']['t0']['temporal']['ls8day']      # [B, C, T, H, W]
ls8_mask = batch['windows']['t0']['temporal_masks']['ls8day'] # [B, C, T, H, W]
ls8_quality = batch['windows']['t0']['temporal_quality']['ls8day'] # [B, C, T, H, W]

# Apply mask (invalid pixels set to 0)
masked_data = ls8_data * ls8_mask

# Apply quality weighting
weighted_data = masked_data * ls8_quality

# Compute loss only on valid pixels
loss = criterion(predictions, weighted_data)
loss = (loss * ls8_mask).sum() / ls8_mask.sum()  # Mean over valid pixels
```

---

### Example 4: Anchor-Specific Processing

```python
batch = next(iter(dataloader))

# Process based on which window is anchor
for i, anchor_id in enumerate(batch['anchor_ids']):
    if anchor_id == 't0':
        # Sample focuses on most recent data (2024)
        recent_data = batch['windows']['t0']['temporal']['ls8day'][i]
        # ... process recent data
    
    elif anchor_id == 't2':
        # Sample focuses on middle period (2022)
        middle_data = batch['windows']['t2']['temporal']['ls8day'][i]
        # ... process middle data
    
    elif anchor_id == 't4':
        # Sample focuses on older data (2020)
        older_data = batch['windows']['t4']['temporal']['ls8day'][i]
        # ... process older data
```

---

### Example 5: Debug Mode

```python
# Use debug window for fast iteration
training_config['spatial_domain']['debug_mode'] = True
training_config['spatial_domain']['debug_window'] = {
    'origin': [8192, 8192],
    'size': [1024, 1024],
    'block_grid': [7, 7]
}

# Smaller epoch size for debugging
training_config['training']['epoch']['mode'] = 'number'
training_config['training']['epoch']['sample_number'] = 100

dataset = ForestDataset(bindings_config, training_config, 'train')
print(f"Debug dataset: {len(dataset)} samples")  # 100 samples
```

---

## Troubleshooting

### Issue: Dataset has 0 samples

**Cause**: No patches pass AOI threshold in the specified window.

**Solution**:
1. Check debug window is in valid region with AOI coverage
2. Lower `min_aoi_fraction` threshold (default 0.3)
3. Use full domain instead of debug window
4. Check AOI mask is loaded correctly

```python
# Lower threshold
# In sampler config (hardcoded for now):
# min_aoi_fraction: float = 0.1  # Instead of 0.3
```

---

### Issue: All windows have identical data

**Cause**: DataReader not respecting temporal window end years.

**Solution**: Ensure temporal arrays have proper year mapping. Check that:
- Zarr arrays have time dimension
- Years map correctly to array indices
- `_read_temporal_array_with_padding()` uses correct year slicing

---

### Issue: Shape mismatch errors

**Cause**: Masks/quality don't match data shapes.

**Solution**: 
- MaskBuilder should always return matching shapes
- Check `read_mask_for_group()` logic
- Verify band configurations in bindings YAML

---

### Issue: Out of memory

**Cause**: Too many workers or large batch size.

**Solution**:
```python
# Reduce batch size
loader = DataLoader(dataset, batch_size=2, ...)  # Instead of 4

# Reduce workers
loader = DataLoader(dataset, num_workers=2, ...)  # Instead of 4

# Disable pin_memory
loader = DataLoader(dataset, pin_memory=False, ...)
```

---

### Issue: Slow data loading

**Cause**: Not enough workers or inefficient Zarr chunking.

**Solution**:
```python
# Increase workers
loader = DataLoader(dataset, num_workers=8, ...)

# Enable persistent workers
loader = DataLoader(dataset, persistent_workers=True, ...)

# Prefetch batches
loader = DataLoader(dataset, prefetch_factor=2, ...)
```

---

## Performance Tips

### 1. Use Multiple Workers

```python
loader = DataLoader(
    dataset,
    num_workers=4,        # Parallel data loading
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2     # Prefetch 2 batches per worker
)
```

### 2. Enable Pin Memory (GPU training)

```python
loader = DataLoader(
    dataset,
    pin_memory=True  # Faster CPU->GPU transfer
)
```

### 3. Optimize Epoch Size

```python
# For large datasets, sample subset per epoch
training_config['training']['epoch']['mode'] = 'frac'
training_config['training']['epoch']['sample_frac'] = 0.1  # 10% per epoch
```

### 4. Cache Frequently Used Data

Consider pre-loading static data that doesn't change:

```python
# In model __init__
self.static_data_cache = {}

# In forward pass
if 'topo' not in self.static_data_cache:
    self.static_data_cache['topo'] = batch['static']['topo']
```

---

## File Structure

```
data/loaders/
├── bindings/
│   ├── __init__.py
│   ├── parser.py              # BindingsParser
│   └── utils.py               # BindingsRegistry, etc.
├── __init__.py
├── windows.py                 # SpatialWindow, TemporalWindow
├── data_reader.py             # DataReader
├── mask_builder.py            # MaskBuilder
├── data_bundle.py             # BundleBuilder, TrainingBundle, WindowData
├── forest_patch_sampler.py    # ForestPatchSampler
└── forest_dataset.py          # ForestDataset, forest_collate_fn

examples/data/
└── forest_dataset_examples.py # Example usage

config/
├── frl_bindings_v0.yaml       # Data structure config
└── frl_training_v1.yaml       # Training config
```

---

## Next Steps

1. **Normalization**: Add normalization wrapper after computing statistics
2. **Data augmentation**: Add spatial augmentation (flip, rotate)
3. **Caching**: Implement optional bundle caching for repeated epochs
4. **Profiling**: Profile data loading to identify bottlenecks
5. **Derived features**: Expand derived feature computation

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Run example scripts: `python examples/data/forest_dataset_examples.py`
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Check Zarr structure: `zarr.open('/path/to/data.zarr').tree()`

---

## Summary

The data loader provides:
- ✅ Multi-window training samples (t0, t2, t4)
- ✅ Deterministic spatial splits (checkerboard)
- ✅ Weighted temporal sampling
- ✅ Complete masks and quality weights
- ✅ PyTorch integration
- ✅ Raw data (normalize in training loop)

**Ready to train!**