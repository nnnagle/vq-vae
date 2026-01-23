# ForestDatasetV2 Examples

This directory contains example scripts demonstrating how to use the refactored dataset loader.

## Quick Start

### Simple Example (Minimal)

The fastest way to test if the dataset is working:

```bash
python example_dataset_simple.py
```

This script:
- Loads the configuration
- Creates a small dataset (5 samples, debug window)
- Loads one sample
- Prints shapes and channel names

**Expected output:**
```
Dataset created with 5 samples

Sample keys: ['static_mask', 'annual_mask', 'annual', 'static', 'metadata']

Data shapes:
  static_mask: (8, 256, 256), dtype=uint8
  annual_mask: (1, 15, 256, 256), dtype=uint8
  annual: (8, 15, 256, 256), dtype=float16
  static: (23, 256, 256), dtype=float16

Channel names:
  static_mask: ['aoi', 'dem_mask', 'tpi_mask', ...]
  annual: ['evi2_summer_p95', ..., 'temporal_position']
  ...

✓ Dataset is working!
```

### Comprehensive Examples

For detailed usage examples:

```bash
python example_dataset_usage.py
```

This script demonstrates:

1. **Loading Configuration** - Parse YAML bindings
2. **Creating Dataset** - Initialize with various options
3. **Loading Samples** - Access individual data samples
4. **Channel Name Indexing** - Slice data by channel name (not just index)
5. **DataLoader Integration** - Use with PyTorch DataLoader
6. **Train/Val/Test Splits** - Create datasets for each split

## Usage Patterns

### Basic Usage

```python
from frl.data.loaders.config import DatasetBindingsParser
from frl.data.loaders.dataset import ForestDatasetV2

# 1. Load config
config = DatasetBindingsParser('frl/config/forest_repr_model_bindings.yaml').parse()

# 2. Create dataset
dataset = ForestDatasetV2(
    config,
    split='train',        # 'train', 'val', 'test', or None for all
    patch_size=256,       # Spatial patch size
    epoch_mode='full',    # 'full', 'frac', or 'number'
)

# 3. Load sample
sample = dataset[0]
# Returns dict with keys: static_mask, annual_mask, annual, static, metadata
```

### Accessing Data by Channel Name

Instead of remembering channel indices, use channel names:

```python
sample = dataset[0]
metadata = sample['metadata']

# Get channel names
static_names = metadata['channel_names']['static']
# ['elevation', 'slope', 'northness', ...]

# Find index and extract
elevation_idx = static_names.index('elevation')
elevation = sample['static'][elevation_idx]  # Shape: [H, W]

# Or for temporal data
annual_names = metadata['channel_names']['annual']
ndvi_idx = annual_names.index('ndvi_summer_p95')
ndvi = sample['annual'][ndvi_idx]  # Shape: [T, H, W]
```

### Using with DataLoader

```python
from torch.utils.data import DataLoader
from frl.data.loaders.dataset import collate_fn

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

for batch in dataloader:
    # batch['static'] shape: [B, C, H, W]
    # batch['annual'] shape: [B, C, T, H, W]
    # batch['metadata'] is a list of B metadata dicts
    pass
```

### Creating Train/Val/Test Datasets

```python
splits = {}
for split in ['train', 'val', 'test']:
    splits[split] = ForestDatasetV2(
        config,
        split=split,
        patch_size=256,
    )

print(f"Train: {len(splits['train'])} samples")
print(f"Val:   {len(splits['val'])} samples")
print(f"Test:  {len(splits['test'])} samples")
```

## Dataset Parameters

### Required
- `config`: BindingsConfig object from parser

### Optional
- `split`: `'train'`, `'val'`, `'test'`, or `None` (default: `None`)
- `patch_size`: Spatial patch size in pixels (default: `256`)
- `min_aoi_fraction`: Minimum fraction of valid AOI pixels (default: `0.3`)
- `epoch_mode`: How to sample patches per epoch (default: `'full'`)
  - `'full'`: Use all patches, shuffle each epoch
  - `'frac'`: Random sample of `sample_frac * n_patches`
  - `'number'`: Fixed `sample_number` samples per epoch
- `sample_frac`: Fraction for `'frac'` mode (default: `None`)
- `sample_number`: Number for `'number'` mode (default: `None`)
- `debug_window`: Spatial region `((row, col), (height, width))` for testing (default: `None`)

## Sample Structure

Each sample is a dictionary with:

```python
{
    'static_mask': np.ndarray,   # [C, H, W] - uint8 masks
    'annual_mask': np.ndarray,   # [C, T, H, W] - temporal masks
    'annual': np.ndarray,        # [C, T, H, W] - temporal features
    'static': np.ndarray,        # [C, H, W] - static features
    'metadata': {
        'channel_names': {
            'static_mask': ['aoi', 'dem_mask', ...],
            'annual': ['evi2_summer_p95', ..., 'temporal_position'],
            'static': ['elevation', 'slope', ...],
        },
        'spatial_window': SpatialWindow,
        'patch_idx': int,
    }
}
```

## Configuration

The dataset reads from `frl/config/forest_repr_model_bindings.yaml`.

Key sections:
- `zarr`: Path to zarr dataset
- `time_window`: Temporal range (start/end years)
- `dataset`: Groups with channels (source or formula-based)

Example channel types:
```yaml
# Source-based (load from zarr)
- {name: elevation, source: static/topo/data/elevation}

# Formula-based (computed)
- name: temporal_position
  formula: "t / (T - 1)"

# With year extraction
- name: forest
  source: annual/lcms_lu_p/data/lcms_lu_p_forest
  year: 2024
  ok_if: {op: ">=", value: 0.25}

# With fill value
- name: mean_green
  source: static/ccdc-metric_history/data/mean_green
  fill_value: -9999
```

## Troubleshooting

### "Zarr dataset not found"
Make sure the zarr path in the YAML matches your actual data location:
```yaml
zarr:
  path: /data/VA/zarr/va_vae_dataset_test.zarr
```

### "Missing zarr arrays"
The dataset validates all source paths during initialization. If you see missing arrays, check:
1. The source path in the YAML is correct
2. The zarr was built with all required arrays

### "Module not found: zarr/numpy"
The dataset requires these dependencies:
```bash
pip install zarr numpy pyyaml torch
```

### Slow loading
- Use `debug_window` for testing with small spatial regions
- Use `epoch_mode='number'` with small `sample_number` for quick tests
- Increase `num_workers` in DataLoader for production

## Next Steps

After loading raw data with ForestDatasetV2:

1. **Stats Calculator** - Compute statistics for normalization
2. **FeatureProcessor** - Apply normalization, masking, feature extraction
3. **Training Loop** - Use processed features for model training

See `DATASET_REFACTOR_SUMMARY.md` for the full refactoring plan.
