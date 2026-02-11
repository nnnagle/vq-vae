# Pre-Normalization Transforms

The `transforms` module provides element-wise mathematical transforms (log, sqrt, etc.) that are applied to raw channel data **before** normalization. This lets you reshape skewed distributions so that downstream normalization (z-score, robust IQR, etc.) is more effective.

## Overview

```
Raw Sample (from ForestDatasetV2)
         |
         v
+-----------------------------------------+
|            FeatureBuilder                |
|                                          |
|  1. Extract channels by reference        |
|  2. Build combined mask                  |
|  3. Apply pre-normalization TRANSFORM  <-- NEW
|  4. Apply normalization                  |
|  5. Apply Mahalanobis transform          |
|  6. Zero out masked values               |
|                                          |
+-----------------------------------------+
         |
         v
 FeatureResult(data, mask, ...)
```

The same transforms are applied inside `StatsCalculator` before computing statistics, so that stats (mean, sd, quantiles, covariance) describe the **transformed** distribution.

## Quick Start

### YAML Configuration

Transforms are specified per-channel in the `features` section:

```yaml
features:
  ccdc_history:
    dim: [C, H, W]
    channels:
      static.spectral_distance_per_decade: {transform: log1p, norm: robust_iqr}
      static.variance_ndvi: {transform: sqrt, norm: robust_iqr}
      static.num_segments: {norm: robust_iqr}   # no transform
```

### Python API

```python
from data.loaders.transforms import apply_transform, TRANSFORMS

# Apply a named transform
transformed = apply_transform(raw_data, 'log1p')

# No-op when transform is None
same_data = apply_transform(raw_data, None)  # returns raw_data unchanged

# List available transforms
print(list(TRANSFORMS.keys()))
# ['cbrt', 'log', 'log1p', 'log10', 'sqrt']
```

## Available Transforms

| Name    | NumPy Function  | Formula         | Domain          | Typical Use Case                               |
|---------|-----------------|-----------------|-----------------|------------------------------------------------|
| `log`   | `np.log(x)`     | ln(x)           | x > 0           | Multiplicative data, heavy right skew          |
| `log1p` | `np.log1p(x)`   | ln(1 + x)       | x > -1          | Non-negative data with zeros (counts, rates)   |
| `log10` | `np.log10(x)`   | log10(x)        | x > 0           | Data spanning orders of magnitude              |
| `sqrt`  | `np.sqrt(x)`    | sqrt(x)         | x >= 0          | Moderate right skew, variance stabilization    |
| `cbrt`  | `np.cbrt(x)`    | x^(1/3)         | all reals        | Mild skew, works with negatives                |

### Domain Violations

Values outside a transform's domain produce `NaN`:

- `log(-1)` -> `NaN`
- `sqrt(-1)` -> `NaN`
- `log(0)` -> `-inf` (treated as invalid)

These are caught by the existing NaN masking in both `StatsCalculator` and `FeatureBuilder`. If unexpected NaN counts increase after adding a transform, the raw data likely contains values outside the domain. Use `log1p` for data with zeros, or `cbrt` for data that can be negative.

## API Reference

### Module: `data.loaders.transforms`

#### Constants

##### `TRANSFORMS`

```python
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]]
```

Registry mapping transform names to NumPy element-wise functions. Each function accepts and returns an `np.ndarray`.

```python
from data.loaders.transforms import TRANSFORMS

fn = TRANSFORMS['log1p']
result = fn(my_array)
```

#### Functions

##### `apply_transform`

```python
def apply_transform(
    data: np.ndarray,
    transform_name: Optional[str],
) -> np.ndarray
```

Apply a named transform to an array.

**Parameters:**
- `data`: Input array of any shape
- `transform_name`: Name of a registered transform, or `None` to skip

**Returns:** Transformed array (new allocation when a transform is applied). When `transform_name` is `None`, returns `data` unchanged (no copy).

**Raises:** `ValueError` if `transform_name` is not `None` and not in the registry.

##### `get_transform_names`

```python
def get_transform_names() -> List[str]
```

Return the sorted list of available transform names.

**Returns:** `['cbrt', 'log', 'log1p', 'log10', 'sqrt']`

##### `validate_transform`

```python
def validate_transform(name: str) -> None
```

Validate that a name is a registered transform.

**Parameters:**
- `name`: Transform name to check

**Raises:** `ValueError` with a message listing available transforms if the name is invalid.

### Dataclass: `FeatureChannelConfig`

The `transform` field is an optional string on `FeatureChannelConfig`:

```python
@dataclass
class FeatureChannelConfig:
    dataset_group: str
    channel_name: str
    mask: Optional[str] = None
    quality: Optional[str] = None
    norm: Optional[str] = None
    transform: Optional[str] = None  # 'log', 'log1p', 'sqrt', etc.
```

The transform name is validated at construction time via `__post_init__`.

## How Stats and Transforms Interact

Statistics must be computed on the **transformed** distribution for normalization to be meaningful. For example, z-scoring `log1p(spectral_distance)` requires the mean and standard deviation of the log-transformed values, not the raw values.

The `StatsCalculator` applies transforms to each channel immediately after extracting raw data, before building the NaN mask and computing statistics. This means:

1. Transform is applied to raw channel data
2. Domain violations become `NaN` and are excluded by the mask
3. Stats (mean, sd, quantiles, covariance) are computed on transformed values
4. At training time, `FeatureBuilder` applies the same transform then normalizes using these stats

**If you change a channel's transform, you must recompute statistics.** Set `stats.compute: always` or delete the stats JSON file to force recomputation.

## Pipeline Integration

### StatsCalculator

In `_extract_feature_data()`, after raw channels are stacked into `[C, H, W]`:

```python
# Applied automatically - no user action needed
for c_idx, (channel_ref, channel_config) in enumerate(feature_config.channels.items()):
    if channel_config.transform:
        feature_data[c_idx] = apply_transform(feature_data[c_idx], channel_config.transform)
```

### FeatureBuilder

In `_apply_normalization()`, per-channel before the normalization preset:

```python
# Applied automatically - no user action needed
if channel_config.transform:
    normalized_data[c_idx] = apply_transform(normalized_data[c_idx], channel_config.transform)
```

In `_apply_mahalanobis_transform()`, before centering and whitening:

```python
# Applied automatically - no user action needed
if channel_config.transform:
    transformed_data[c_idx] = apply_transform(transformed_data[c_idx], channel_config.transform)
```

## Configuration Examples

### Right-skewed CCDC metrics

```yaml
features:
  ccdc_history:
    dim: [C, H, W]
    channels:
      static.spectral_distance_per_decade: {transform: log1p, norm: robust_iqr}
      static.variance_ndvi: {transform: sqrt, norm: robust_iqr}
      static.num_segments: {norm: robust_iqr}       # count data, no transform needed
      static.mean_ndvi: {norm: robust_iqr}           # already well-behaved
```

### Mixed transforms with Mahalanobis

Transforms are applied before centering and whitening, so covariance is computed in the transformed space:

```yaml
features:
  infonce_type_spectral:
    dim: [C, H, W]
    channels:
      static.mean_red: {transform: sqrt, norm: zscore}
      static.mean_nir: {transform: sqrt, norm: zscore}
      static.mean_swir1: {transform: sqrt, norm: zscore}
    covariance:
      calculate: true
      stat_domain: patch
```

### Temporal features

Transforms work identically for temporal `[C, T, H, W]` features:

```yaml
features:
  phase_ccdc:
    dim: [C, T, H, W]
    channels:
      annual.temporal_position: {norm: identity}
      annual.spectral_velocity: {transform: log1p, norm: zscore}
      annual.seas_amp_swir1: {transform: sqrt, norm: zscore}
```

## Choosing a Transform

| Data characteristic               | Recommended transform |
|------------------------------------|-----------------------|
| Strictly positive, heavy right skew | `log`                |
| Non-negative with zeros             | `log1p`              |
| Spans orders of magnitude           | `log10`              |
| Non-negative, moderate skew         | `sqrt`               |
| Can be negative, mild skew          | `cbrt`               |
| Already symmetric / well-behaved    | (none)               |
