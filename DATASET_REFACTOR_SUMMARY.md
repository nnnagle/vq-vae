# Dataset Refactor Summary

## Overview

I've successfully implemented a new dataset loader system on the `dataset-refactor` branch. This refactor simplifies the data loading pipeline and aligns with your new YAML bindings format.

## What Was Implemented

### 1. Configuration Data Structures (`dataset_config.py`)

**New dataclasses:**
- `ChannelConfig`: Represents a single channel (source-based or formula-based)
- `DatasetGroupConfig`: Represents a dataset group (static_mask, annual, etc.)
- `TimeWindowConfig`: Temporal window configuration
- `ZarrConfig`: Zarr dataset path and structure
- `BindingsConfig`: Top-level configuration container

**Key features:**
- Type-safe configuration with validation
- Distinguishes source-based vs formula-based channels
- Supports temporal groups `[C,T,H,W]` and static groups `[C,H,W]`
- Helper methods for accessing channels by name

### 2. YAML Parser (`dataset_bindings_parser.py`)

**`DatasetBindingsParser` class:**
- Loads YAML and creates structured `BindingsConfig` objects
- Validates required fields and structure
- Handles all dataset group types (static_mask, annual_mask, annual, static)
- Parses channel specifications including:
  - `source`: Zarr path to load
  - `formula`: Computed channels (e.g., temporal_position)
  - `year`: Extract specific year from temporal array
  - `ok_if`: Threshold operations for binary masks
  - `fill_value`: Values to replace with NaN

**Example usage:**
```python
from frl.data.loaders.config import DatasetBindingsParser

parser = DatasetBindingsParser('frl/config/forest_repr_model_bindings.yaml')
config = parser.parse()

# Access groups
static_group = config.get_group('static')
print(f"Static channels: {static_group.get_channel_names()}")
```

### 3. Dataset Loader (`forest_dataset_v2.py`)

**`ForestDatasetV2` class:**
- PyTorch `Dataset` that loads data according to bindings config
- Reuses existing `ForestPatchSampler` for train/val/test splitting
- Returns raw data (no normalization - delegated to FeatureProcessor)

**Key capabilities:**
1. **Source loading:** Reads zarr arrays by path
2. **Temporal slicing:** Clips arrays to time_window with NaN padding for missing years
3. **Year extraction:** Extracts specific years from temporal arrays for static outputs
4. **Formula computation:** Computes derived channels (e.g., temporal_position)
5. **Thresholding:** Applies `ok_if` operations to create binary masks
6. **Fill value handling:** Replaces fill_values with NaN
7. **Validation:** Checks that all source paths exist during initialization

**Return format:**
```python
{
    'static_mask': np.ndarray,  # [C=8, H=256, W=256] uint8
    'annual_mask': np.ndarray,  # [C=1, T=15, H=256, W=256] uint8
    'annual': np.ndarray,       # [C=8, T=15, H=256, W=256] float16
    'static': np.ndarray,       # [C=23, H=256, W=256] float16
    'metadata': {
        'spatial_window': SpatialWindow,
        'channel_names': {
            'static_mask': ['aoi', 'dem_mask', ...],
            'annual': ['evi2_summer_p95', ..., 'temporal_position'],
            ...
        },
        'patch_idx': 0
    }
}
```

**Example usage:**
```python
from frl.data.loaders.config import DatasetBindingsParser
from frl.data.loaders.dataset import ForestDatasetV2, collate_fn
from torch.utils.data import DataLoader

# Parse config
config = DatasetBindingsParser('frl/config/forest_repr_model_bindings.yaml').parse()

# Create dataset
dataset = ForestDatasetV2(
    config,
    split='train',
    patch_size=256,
    epoch_mode='full',
)

# Use with DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

for batch in dataloader:
    # batch['static'] shape: [B=4, C=23, H=256, W=256]
    # batch['annual'] shape: [B=4, C=8, T=15, H=256, W=256]
    # batch['metadata'] is a list of metadata dicts
    pass
```

### 4. Updated YAML Format

**New bindings file:** `frl/config/forest_repr_model_bindings.yaml`

Key sections implemented:
```yaml
zarr:
  path: /data/VA/zarr/va_vae_dataset_test.zarr
  structure: hierarchical

time_window:
  start: 2010
  end: 2024

dataset:
  static_mask:
    type: uint8
    dim: [C,H,W]
    channels:
      - {name: aoi, source: aoi}
      - name: forest
        source: annual/lcms_lu_p/data/lcms_lu_p_forest
        year: 2024
        ok_if: {op: ">=", value: 0.25}

  annual:
    type: float16
    dim: [C,T,H,W]
    channels:
      - {name: evi2_summer_p95, source: annual/ls8day/data/EVI2_summer_p95}
      - name: temporal_position
        formula: "t / (T - 1)"  # Computed channel
```

## Key Design Decisions

### 1. **Simplified Architecture**
- **Removed:** Multi-window bundling (Win dimension), anchor years, TrainingBundle complexity
- **Kept:** Train/val/test spatial splitting via ForestPatchSampler
- **Result:** Single temporal window per sample, simpler batch structure

### 2. **Separation of Concerns**
- **ForestDatasetV2:** Loads raw data only
- **FeatureProcessor** (to be implemented): Handles normalization, masking, feature creation
- **Stats calculator** (to be implemented): Computes statistics for normalization

### 3. **Temporal Handling**
- Arrays are sliced to `time_window` range
- Missing years are NaN-padded (e.g., if zarr has 2015-2024 but config needs 2010-2024)
- Year extraction uses `year: 2024` syntax for static outputs from temporal sources

### 4. **Formula Channels**
- Computed inline during loading (e.g., temporal_position)
- Broadcast to match group dimensions
- Currently supports: `t / (T - 1)` for temporal position

### 5. **Metadata for Slicing**
- Returns channel names in order for each group
- FeatureProcessor can use this to slice by name rather than index
- Example: `batch['static'][:, channel_names['static'].index('elevation')]`

## Testing

### Syntax Check (Works Now)
```bash
python check_dataset_syntax.py
```
✓ Validates YAML structure
✗ Import checks fail (zarr/numpy not installed in current environment)

### Full Integration Test (Requires Dependencies)
```bash
python test_dataset_refactor.py
```
Tests:
- Parser loading
- Dataset initialization
- Sample loading
- DataLoader batching

**Note:** This requires `numpy`, `zarr`, `torch`, and `pyyaml` to be installed.

## Files Changed

**New files:**
```
frl/config/forest_repr_model_bindings.yaml       # Example bindings
frl/data/loaders/config/dataset_config.py        # Config dataclasses
frl/data/loaders/config/dataset_bindings_parser.py  # YAML parser
frl/data/loaders/dataset/forest_dataset_v2.py    # Dataset loader
test_dataset_refactor.py                         # Integration test
check_dataset_syntax.py                          # Syntax validator
```

**Modified files:**
```
frl/data/loaders/config/__init__.py    # Added new imports
frl/data/loaders/dataset/__init__.py   # Added ForestDatasetV2, collate_fn
```

## What's NOT Implemented (Future Work)

Based on your YAML, these sections are deferred:

1. **Normalization section** - Will be handled by FeatureProcessor
2. **Features section** - Will be handled by FeatureProcessor
3. **Sampling-strategy section** - For loss computation
4. **Losses section** - For training loop
5. **Stats computation** - Separate stats calculator needed

## Next Steps

You mentioned you want to implement these in order:
1. ✅ **Bindings parser** (DONE)
2. ✅ **Dataloader** (DONE)
3. ⏭️ **Stats calculator** - Compute univariate stats and covariance
4. ⏭️ **FeatureProcessor** - Apply normalization, masks, create features

Would you like me to implement the stats calculator next?

## Questions Resolved

During implementation, these design questions were resolved:

1. **Q:** Should derived channels be in separate section or inline?
   **A:** Inline in dataset groups (e.g., `annual.temporal_position`)

2. **Q:** How to handle year extraction from temporal arrays?
   **A:** Use `year: 2024` syntax, fail if year doesn't exist

3. **Q:** What to do with fill_values?
   **A:** Replace with NaN during loading

4. **Q:** How to handle missing years in time_window?
   **A:** Pad with NaN

5. **Q:** Return structure from __getitem__?
   **A:** Simple dict with group names as keys, metadata for channel name slicing

6. **Q:** File organization?
   **A:** Replace existing code (we're on new branch)

7. **Q:** When to validate sources?
   **A:** During dataset.__init__ (fail fast)

8. **Q:** Reuse spatial sampler?
   **A:** Yes, reuse ForestPatchSampler for train/val/test splits

## Git Status

Branch: `dataset-refactor`
Commit: `b4c641e`
Status: Ready for review

All changes committed with message linking to Claude session.
