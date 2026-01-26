# FeatureBuilder

The `FeatureBuilder` transforms raw tensors from `ForestDatasetV2` into normalized features ready for model training. It handles masking, normalization, and optional Mahalanobis transforms according to the bindings configuration.

## Overview

```
Raw Sample (from ForestDatasetV2)
         │
         ▼
┌─────────────────────────────────────┐
│          FeatureBuilder             │
│                                     │
│  1. Extract channels by reference   │
│  2. Build combined mask             │
│  3. Apply normalization             │
│  4. Apply Mahalanobis transform     │
│  5. Zero out masked values          │
│                                     │
└─────────────────────────────────────┘
         │
         ▼
FeatureResult(data, mask, ...)
```

## Quick Start

```python
from data.loaders.config import DatasetBindingsParser
from data.loaders.dataset import ForestDatasetV2
from data.loaders.builders import FeatureBuilder

# Load config and create dataset
config = DatasetBindingsParser('config/frl_binding_v1.yaml').parse()
dataset = ForestDatasetV2(config, split='train')

# Create feature builder (loads stats from config.stats.file)
builder = FeatureBuilder(config)

# Build a feature from a sample
sample = dataset[0]
result = builder.build_feature('infonce_type_spectral', sample)

# Access results
print(result.data.shape)   # [C, H, W] normalized data
print(result.mask.shape)   # [H, W] boolean mask (True = valid)
```

## API Reference

### FeatureBuilder

```python
class FeatureBuilder:
    def __init__(
        self,
        config: BindingsConfig,
        stats_path: Optional[str] = None,
    )
```

**Parameters:**
- `config`: Parsed bindings configuration from `DatasetBindingsParser`
- `stats_path`: Optional path to stats JSON file. If not provided, uses `config.stats.file`

### build_feature

```python
def build_feature(
    self,
    feature_name: str,
    sample: Dict[str, Any],
    apply_normalization: bool = True,
    apply_mahalanobis: bool = True,
) -> FeatureResult
```

Build a single feature from a dataset sample.

**Parameters:**
- `feature_name`: Name of the feature (e.g., `'topo'`, `'phase_ls8'`)
- `sample`: Dataset sample dictionary from `ForestDatasetV2[idx]`
- `apply_normalization`: Whether to apply normalization presets (default: True)
- `apply_mahalanobis`: Whether to apply Mahalanobis transform for covariance features (default: True)

**Returns:** `FeatureResult` dataclass

### FeatureResult

```python
@dataclass
class FeatureResult:
    data: np.ndarray        # Normalized feature data [C, H, W] or [C, T, H, W]
    mask: np.ndarray        # Boolean mask [H, W] or [T, H, W], True = valid
    feature_name: str       # Name of the feature
    channel_names: List[str]  # Names of channels in order
    is_temporal: bool       # Whether this is a temporal feature
```

### build_all_features

```python
def build_all_features(
    self,
    sample: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    apply_normalization: bool = True,
    apply_mahalanobis: bool = True,
) -> Dict[str, FeatureResult]
```

Build multiple features at once.

**Parameters:**
- `sample`: Dataset sample dictionary
- `feature_names`: List of features to build (None = all features in config)

**Returns:** Dictionary mapping feature names to `FeatureResult` objects

### get_feature_info

```python
def get_feature_info(self, feature_name: str) -> Dict[str, Any]
```

Get information about a feature's configuration without building it.

**Returns:**
```python
{
    'name': 'infonce_type_spectral',
    'dim': ['C', 'H', 'W'],
    'n_channels': 7,
    'channels': [
        {
            'reference': 'static.mean_red',
            'dataset_group': 'static',
            'channel_name': 'mean_red',
            'mask': None,
            'normalization': 'zscore'
        },
        ...
    ],
    'global_masks': ['static_mask.aoi', 'static_mask.forest'],
    'has_covariance': True
}
```

## Masking

The FeatureBuilder combines masks from three sources using logical AND:

### 1. Global Feature Masks

Defined in the feature's `masks` field in the YAML:

```yaml
features:
  infonce_type_topo:
    masks:
      - static_mask.aoi
      - static_mask.forest
      - static_mask.dem_mask
```

### 2. Channel-Level Masks

Defined per-channel in the feature configuration:

```yaml
features:
  my_feature:
    channels:
      - static.elevation:
          mask: static_mask.dem_mask
          norm: zscore
```

### 3. NaN Masks

Any NaN values in the feature data automatically invalidate those pixels. NaNs are then zeroed out in the final output.

### Spatial Broadcast for Temporal Features

For temporal features (shape `[C, T, H, W]`), spatial-only masks (shape `[H, W]`) are automatically broadcast across all timesteps.

## Normalization

Each channel can specify a normalization preset. The FeatureBuilder applies normalization using precomputed statistics from the stats JSON file.

### Supported Normalization Types

| Type | Formula | Stats Used |
|------|---------|------------|
| `zscore` | `(x - mean) / sd` | mean, sd |
| `robust_iqr` | `(x - q50) / (q75 - q25)` | q25, q50, q75 |
| `linear_rescale` | Scale from `[in_min, in_max]` to `[out_min, out_max]` | Config params |
| `clamp` | Clip to `[min, max]` | Config params |
| `identity` | No transformation | None |

### Clamping

All normalization types can have optional clamping applied after the transform:

```yaml
normalization:
  presets:
    zscore:
      type: zscore
      clamp:
        enabled: true
        min: -6.0
        max: 6.0
```

## Mahalanobis Transform

For features with covariance configured (`covariance.calculate: true`), the FeatureBuilder can apply a Mahalanobis (whitening) transform:

1. **Center**: Subtract channel means
2. **Whiten**: Multiply by whitening matrix `W` where `W @ W.T = Σ⁻¹`

This transforms the data such that Euclidean distance in the transformed space equals Mahalanobis distance in the original space.

```yaml
features:
  infonce_type_spectral:
    covariance:
      calculate: true
      stat_domain: patch
```

### Disabling Mahalanobis

```python
# Build without Mahalanobis transform
result = builder.build_feature('infonce_type_spectral', sample, apply_mahalanobis=False)
```

## Configuration Reference

Features are defined in the bindings YAML under the `features` section:

```yaml
features:
  topo:
    dim: [C, H, W]
    channels:
      - static.elevation:
          mask: static_mask.dem_mask
          norm: zscore
      - static.slope:
          mask: static_mask.dem_mask
          norm: zscore
      - static.northness:
          mask: static_mask.dem_mask
          norm: trig
      - static.eastness:
          mask: static_mask.dem_mask
          norm: trig
    masks:
      - static_mask.aoi
      - static_mask.dem_mask

  phase_ls8:
    dim: [C, T, H, W]
    channels:
      - annual.temporal_position:
          norm: identity
      - annual.evi2_summer_p95:
          norm: zscore
      # ... more channels
    masks:
      - static_mask.aoi
      - static_mask.forest

  infonce_type_spectral:
    dim: [C, H, W]
    channels:
      - static.mean_red:
          norm: zscore
      # ... more channels
    masks:
      - static_mask.aoi
      - static_mask.forest
    covariance:
      calculate: true
      stat_domain: patch
```

## Statistics File

The FeatureBuilder requires precomputed statistics. Generate them using:

```python
from data.stats import compute_stats_from_config
compute_stats_from_config('config/frl_binding_v1.yaml')
```

The stats JSON has this structure:

```json
{
  "feature_name": {
    "static.elevation": {
      "mean": 450.2,
      "sd": 120.5,
      "q02": 200.0,
      "q25": 380.0,
      "q50": 450.0,
      "q75": 520.0,
      "q98": 700.0
    },
    "static.slope": { ... },
    "covariance": [
      [1.0, 0.2, ...],
      [0.2, 1.0, ...],
      ...
    ]
  }
}
```

## Example: Training Loop Integration

```python
from data.loaders.config import DatasetBindingsParser
from data.loaders.dataset import ForestDatasetV2, collate_fn
from data.loaders.builders import FeatureBuilder
from torch.utils.data import DataLoader
import torch

# Setup
config = DatasetBindingsParser('config/frl_binding_v1.yaml').parse()
dataset = ForestDatasetV2(config, split='train')
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
builder = FeatureBuilder(config)

# Training loop
for batch in loader:
    # batch is a dict with group tensors and metadata
    # Process each sample in batch
    batch_features = []
    batch_masks = []

    for i in range(len(batch['metadata'])):
        # Extract single sample from batch
        sample = {
            key: val[i].numpy() if torch.is_tensor(val) else val[i]
            for key, val in batch.items()
        }

        result = builder.build_feature('infonce_type_spectral', sample)
        batch_features.append(torch.from_numpy(result.data))
        batch_masks.append(torch.from_numpy(result.mask))

    features = torch.stack(batch_features)  # [B, C, H, W]
    masks = torch.stack(batch_masks)        # [B, H, W]

    # Use in model...
```

## Convenience Function

For quick setup:

```python
from data.loaders.builders import create_feature_builder_from_config

builder = create_feature_builder_from_config('config/frl_binding_v1.yaml')
```
