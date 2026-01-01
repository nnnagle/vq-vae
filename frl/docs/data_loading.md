# Zarr Statistics Loader & Normalization System

A complete system for loading pre-computed statistics from Zarr datasets and applying normalization transformations defined in YAML configuration files.

## Overview

This system consists of three main components:

1. **`zarr_stats_loader.py`** - Loads statistics from hierarchical Zarr datasets
2. **`normalization.py`** - Applies normalization transformations using loaded statistics
3. **`integration_example.py`** - Example usage patterns and integration guide

## Quick Start

```python
from zarr_stats_loader import ZarrStatsLoader
from normalization import NormalizationManager
import yaml

# 1. Load your YAML config
with open('forest_repr_model_bindings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Initialize stats loader
loader = ZarrStatsLoader('/path/to/dataset.zarr')

# 3. Initialize normalization manager
norm_manager = NormalizationManager(config, loader)

# 4. Normalize data
band_config = {'array': 'NDVI_summer_p95', 'norm': 'zscore'}
normalized_data = norm_manager.normalize_band(
    data=my_data,
    group_path='annual/ls8day/data',
    band_config=band_config
)
```

## Zarr Dataset Structure

The system expects statistics to be embedded in variable attributes in Zarr:

```
dataset.zarr/
├── annual/
│   └── ls8day/
│       └── data/
│           ├── NDVI_summer_p95
│           │   .attrs['statistics'] = {
│           │       'mean': 0.45,
│           │       'sd': 0.12,
│           │       'q25': 0.38,
│           │       'q50': 0.46,
│           │       'q75': 0.54,
│           │       ...
│           │   }
│           ├── NDVI_winter_max
│           │   .attrs['statistics'] = {...}
│           └── ...
├── static/
│   └── topo/
│       └── data/
│           ├── elevation
│           │   .attrs['statistics'] = {...}
│           └── ...
└── ...
```

Statistics are computed and embedded by `build_zarr.py` during dataset creation using the `embed_statistics_in_zarr()` function.

## Normalization Types

The system supports all normalization types from your YAML config:

### 1. Z-Score Normalization

```yaml
zscore:
  type: zscore
  stats_source: zarr
  fields: {mean: mean, std: sd}
  clamp: {enabled: true, min: -6.0, max: 6.0}
  missing: {fill: 0.0}
```

**Formula:** `(x - mean) / sd`

**Required stats:** `mean`, `sd`

### 2. Robust IQR Normalization

```yaml
robust_iqr:
  type: robust_iqr
  stats_source: zarr
  fields: {q25: q25, q50: q50, q75: q75}
  clamp: {enabled: true, min: -8.0, max: 8.0}
  missing: {fill: 0.0}
```

**Formula:** `(x - median) / IQR`

**Required stats:** `q25`, `q50`, `q75`

### 3. Min-Max Normalization

```yaml
minmax_0_20:
  type: minmax
  stats_source: fixed
  min: 0.0
  max: 20.0
  clamp: {enabled: true, min: 0.0, max: 1.0}
  missing: {fill: 0.0}
```

**Formula:** `(x - min) / (max - min)`

**Required stats:** `min`, `max` (or from config)

### 4. Linear Rescale

```yaml
delta_ls8_fixed:
  type: linear_rescale
  stats_source: fixed
  in_min: -0.4
  in_max: 0.4
  out_min: -1.0
  out_max: 1.0
  clamp: true
  missing: {fill: 0}
```

**Formula:** Maps `[in_min, in_max]` → `[out_min, out_max]`

**Required config:** All four range parameters

### 5. Clamp Only

```yaml
prob01:
  type: clamp
  clamp: {enabled: true, min: 0.0, max: 1.0}
  missing: {fill: 0.0}
```

**Formula:** Just clamps values to range

### 6. Identity (No Normalization)

```yaml
identity:
  type: none
  missing: {fill: 0}
```

**Formula:** Returns input unchanged

## API Reference

### ZarrStatsLoader

Main class for loading statistics from Zarr datasets.

#### Constructor

```python
loader = ZarrStatsLoader(
    zarr_path: str | Path,    # Path to Zarr dataset root
    cache_stats: bool = True   # Cache loaded stats in memory
)
```

#### Methods

##### get_stats()

```python
stats = loader.get_stats(
    group_path: str,              # e.g., 'annual/ls8day/data'
    array_name: str,              # e.g., 'NDVI_summer_p95'
    stat_fields: List[str] = None # e.g., ['mean', 'sd']
) -> Dict[str, np.ndarray]
```

Returns dictionary mapping stat names to numpy arrays.

##### validate_stats()

```python
loader.validate_stats(
    stats: Dict[str, np.ndarray],
    norm_type: str  # 'zscore', 'robust_iqr', 'minmax'
) -> bool
```

Validates that required statistics are present for a normalization type.

##### compute_iqr()

```python
iqr = loader.compute_iqr(
    q25: np.ndarray,
    q75: np.ndarray
) -> np.ndarray
```

Helper to compute IQR from quartiles.

##### get_stats_summary()

```python
summary = loader.get_stats_summary(
    group_path: str,
    array_name: str
) -> str
```

Returns human-readable summary of available statistics.

### NormalizationManager

High-level manager for normalizing data according to YAML config.

#### Constructor

```python
manager = NormalizationManager(
    yaml_config: Dict[str, Any],     # Parsed YAML config
    stats_loader: ZarrStatsLoader    # Stats loader instance
)
```

#### Methods

##### normalize_band()

```python
normalized = manager.normalize_band(
    data: np.ndarray | torch.Tensor,     # Input data
    group_path: str,                     # Zarr group path
    band_config: Dict[str, Any],         # Band config from YAML
    mask: np.ndarray | torch.Tensor = None  # Optional validity mask
) -> np.ndarray | torch.Tensor
```

Main method for normalizing a band. Automatically:
- Loads appropriate normalizer (cached)
- Applies transformation
- Handles clamping
- Fills masked values

##### get_normalizer_for_band()

```python
normalizer = manager.get_normalizer_for_band(
    group_path: str,
    band_config: Dict[str, Any]
) -> Normalizer
```

Gets or creates a normalizer for a specific band (cached).

### Normalizer Classes

All normalizers inherit from base `Normalizer` class and implement:

```python
normalized = normalizer.normalize(
    data: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor = None
) -> np.ndarray | torch.Tensor
```

**Available normalizers:**
- `ZScoreNormalizer`
- `RobustIQRNormalizer`
- `MinMaxNormalizer`
- `LinearRescaleNormalizer`
- `ClampNormalizer`
- `IdentityNormalizer`

## Usage Patterns

### Pattern 1: Direct Stats Loading

For simple cases where you just need stats:

```python
loader = ZarrStatsLoader('/path/to/data.zarr')

# Load specific stats
stats = loader.get_stats(
    'annual/ls8day/data',
    'NDVI_summer_p95',
    ['mean', 'sd', 'q25', 'q50', 'q75']
)

print(f"Mean: {stats['mean']}")
print(f"SD: {stats['sd']}")
```

### Pattern 2: Manual Normalization

When you need fine control:

```python
from normalization import NormalizationConfig, ZScoreNormalizer

# Create config
config = NormalizationConfig(
    type='zscore',
    stats_source='zarr',
    clamp={'enabled': True, 'min': -6.0, 'max': 6.0}
)

# Load stats
stats = loader.get_stats('annual/ls8day/data', 'NDVI_summer_p95', ['mean', 'sd'])

# Create normalizer
normalizer = ZScoreNormalizer(config, stats)

# Normalize
normalized = normalizer.normalize(my_data)
```

### Pattern 3: YAML-Driven Pipeline (Recommended)

For production use with full YAML config:

```python
from integration_example import DataNormalizationPipeline

# Initialize pipeline
pipeline = DataNormalizationPipeline(
    config_path='forest_repr_model_bindings.yaml',
    zarr_path='/path/to/data.zarr'
)

# Pre-load all stats (optional but recommended)
pipeline.preload_stats()

# Normalize a batch
normalized_batch = pipeline.normalize_batch(
    data=batch_data,
    input_spec=input_specification
)
```

### Pattern 4: Integration with PyTorch DataLoader

```python
class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, config_path, zarr_path):
        self.pipeline = DataNormalizationPipeline(config_path, zarr_path)
        self.pipeline.preload_stats(verbose=False)
    
    def __getitem__(self, idx):
        # Load raw data
        raw_data = self.load_raw_data(idx)
        
        # Normalize
        normalized = self.pipeline.normalize_batch(
            raw_data,
            input_spec=self.input_spec
        )
        
        return normalized
```

## Performance Considerations

### Statistics Caching

By default, statistics are cached in memory after first load:

```python
# Enable caching (default)
loader = ZarrStatsLoader(zarr_path, cache_stats=True)

# Disable for memory-constrained environments
loader = ZarrStatsLoader(zarr_path, cache_stats=False)

# Clear cache manually
loader.clear_cache()
```

### Normalizer Caching

Normalizers are cached by the `NormalizationManager`:

```python
# First call: creates and caches normalizer
norm1 = manager.normalize_band(data1, group_path, band_config)

# Subsequent calls: reuses cached normalizer (fast!)
norm2 = manager.normalize_band(data2, group_path, band_config)
```

### Pre-loading Stats

For training loops, pre-load all statistics at startup:

```python
pipeline = DataNormalizationPipeline(config_path, zarr_path)
pipeline.preload_stats()  # Loads all stats once

# Now all normalizations use cached stats
for batch in dataloader:
    normalized = pipeline.normalize_batch(batch, spec)
```

## Error Handling

### Missing Statistics

```python
try:
    stats = loader.get_stats('group/path', 'array_name', ['mean', 'sd'])
except KeyError:
    print("Stats group not found")
```

The system logs warnings for missing stats but continues:

```
WARNING: Stat 'sd' not found for NDVI_summer_p95 in annual/ls8day/stats
```

### Validation Errors

```python
try:
    loader.validate_stats(stats, 'zscore')
except ValueError as e:
    print(f"Invalid stats: {e}")
```

### Missing Config

```python
try:
    normalizer = manager.get_normalizer_for_band(group_path, band_config)
except ValueError as e:
    print(f"Config error: {e}")
```

## Masking and Missing Values

### Validity Masks

Masks indicate which pixels/values are valid:

```python
# True = valid, False = invalid
mask = torch.tensor([True, True, False, True], dtype=torch.bool)

normalized = normalizer.normalize(data, mask)
# Invalid pixels are filled with missing.fill value (default: 0.0)
```

### Missing Value Fill

Configured per normalization preset:

```yaml
zscore:
  type: zscore
  # ...
  missing: {fill: 0.0}  # Fill invalid pixels with 0.0
```

## Testing

Run the included examples:

```bash
# Run all examples
python integration_example.py

# Run specific example
python integration_example.py 1  # Basic usage
python integration_example.py 2  # Full pipeline
python integration_example.py 3  # Custom normalization
python integration_example.py 4  # With masks
python integration_example.py 5  # Stats validation
```

## Troubleshooting

### "Variable not found"

**Cause:** Variable path doesn't exist in Zarr.

**Solution:** Check your Zarr structure. Variables should follow the pattern:

```
category/group/subsection/variable_name
```

For example: `annual/ls8day/data/NDVI_summer_p95`

### "No statistics found in attrs"

**Cause:** Statistics haven't been embedded in the variable's attributes.

**Solution:** Ensure `build_zarr.py` was run with statistics computation enabled (not `--no-stats`). Statistics should be in the variable's `.attrs['statistics']` dict.

### "Missing required stats for normalization"

**Cause:** Required statistics are missing from the attrs dict.

**Solution:** Ensure all required stats are computed and stored:
- Z-score needs: `mean`, `sd`
- Robust IQR needs: `q25`, `q50`, `q75`
- Min-max needs: `min`, `max`

Check the `compute_variable_statistics()` function in `build_zarr.py` to ensure these stats are being computed for your semantic type.

### Memory issues with large datasets

**Solution:** Disable stats caching:

```python
loader = ZarrStatsLoader(zarr_path, cache_stats=False)
```

Or clear cache periodically:

```python
loader.clear_cache()
```

## Advanced Usage

### Custom Normalizers

Create your own normalizer by extending the base class:

```python
from normalization import Normalizer

class MyCustomNormalizer(Normalizer):
    def _normalize_impl(self, data, mask):
        # Your custom normalization logic
        return custom_transform(data)

# Register it
NormalizerFactory.NORMALIZER_CLASSES['my_custom'] = MyCustomNormalizer
```

### Batch Stats Loading

Load stats for multiple arrays at once:

```python
arrays = ['NDVI_summer_p95', 'NDVI_winter_max', 'NDVI_amplitude']
group_path = 'annual/ls8day/data'

all_stats = {}
for array_name in arrays:
    all_stats[array_name] = loader.get_stats(
        group_path, array_name, ['mean', 'sd']
    )
```

### Stats Registry

For managing stats across multiple groups:

```python
from zarr_stats_loader import StatsRegistry

registry = StatsRegistry(loader)
registry.register_from_yaml(yaml_config)

# Access stats by key
stats = registry.get('annual/ls8day/data', 'NDVI_summer_p95')
```

## Integration with Your Training Pipeline

### Minimal Integration

```python
# In your dataset __init__:
self.norm_manager = NormalizationManager(yaml_config, stats_loader)
self.norm_manager.preload_stats()

# In your __getitem__:
normalized = self.norm_manager.normalize_band(
    data=raw_data,
    group_path='annual/ls8day/data',
    band_config=band_config,
    mask=quality_mask
)
```

### Full Integration

See `integration_example.py` for complete examples including:
- Pipeline initialization
- Batch processing
- Mask handling
- Error recovery
- Performance optimization

## License

This code is part of the forest representation model training pipeline.

## Support

For issues or questions, refer to the examples in `integration_example.py` or check the inline documentation in the source files.