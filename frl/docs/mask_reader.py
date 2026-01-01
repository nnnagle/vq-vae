# MaskBuilder Documentation

## Quick API Reference

### Initialization
```python
builder = MaskBuilder(registry, config, zarr_path=None)
```

### Individual Masks & Quality
```python
# Read individual mask
mask = builder.read_mask(mask_name, spatial_window, temporal_window=None)
# Returns: MaskResult with shape [H, W] or [T, H, W]

# Read individual quality weight
quality = builder.read_quality(quality_name, spatial_window, temporal_window=None)
# Returns: QualityResult with shape [H, W] or [T, H, W]
```

### Group-Based (Recommended)
```python
# Temporal/Static groups
mask = builder.read_mask_for_group(category, group_name, spatial_window, temporal_window=None)
quality = builder.read_quality_for_group(category, group_name, spatial_window, temporal_window=None)
# Returns: Shape [C, T, H, W] or [C, H, W] matching data group

# Snapshot groups (year-specific)
mask = builder.read_mask_for_snapshot_group(group_name, spatial_window, year)
quality = builder.read_quality_for_snapshot_group(group_name, spatial_window, year)
# Returns: Shape [C, H, W] for specific year

# Irregular groups (sparse temporal)
mask = builder.read_mask_for_irregular_group(group_name, spatial_window, temporal_window)
quality = builder.read_quality_for_irregular_group(group_name, spatial_window, temporal_window)
# Returns: Shape [C, T_obs, H, W] where T_obs varies
```

### Utilities
```python
# Combine multiple masks
combined = builder.combine_masks(mask_results, operation='and')  # or 'or'

# Read all available masks/quality
all_masks = builder.read_all_masks(spatial_window, temporal_window=None)
all_quality = builder.read_all_quality(spatial_window, temporal_window=None)
```

---

## Overview

**MaskBuilder** reads mask and quality weight layers from Zarr arrays based on the `shared.masks` and `shared.quality` sections of your bindings configuration. It provides a parallel interface to DataReader, ensuring masks and quality weights always match the shape of your input data.

### Key Features

- ✅ **Per-band mask/quality** - Each band gets its specific masks/weights
- ✅ **Shape guarantees** - Output always matches corresponding DataReader output
- ✅ **Multiple mask types** - Boolean, threshold-based, expression-based
- ✅ **Temporal support** - Time-varying masks and quality weights
- ✅ **Dimension enforcement** - Validates spatial/spatio-temporal/temporal dimensions
- ✅ **Graceful defaults** - All-True/all-ones for missing masks/quality

---

## Data Structures

### MaskResult
```python
@dataclass
class MaskResult:
    data: np.ndarray              # Boolean array [H,W] or [T,H,W] or [C,T,H,W]
    mask_name: str                # Name from config
    mask_type: str                # 'boolean', 'threshold', 'expression', 'per_band'
    false_means_invalid: bool     # True if False = invalid/masked pixels
    metadata: Dict[str, Any]      # Windows, behavior, etc.
```

### QualityResult
```python
@dataclass
class QualityResult:
    data: np.ndarray              # Float32 array [H,W] or [T,H,W] or [C,T,H,W]
    quality_name: str             # Name from config
    quality_type: str             # 'float', 'expression', 'per_band'
    metadata: Dict[str, Any]      # Windows, normalization, etc.
```

---

## Individual Masks & Quality

### Reading Individual Masks

```python
mask_result = builder.read_mask(
    mask_name='aoi',              # Name from shared.masks
    spatial_window=spatial,
    temporal_window=None          # Optional, for time-varying masks
)
```

**Supported Mask Types:**

#### 1. Boolean Masks
```yaml
# In bindings YAML:
shared:
  masks:
    aoi:
      type: boolean
      dimension: spatial
      zarr: {group: static/masks, array: aoi}
      behavior:
        false_means_invalid: true
        missing: false
```

```python
aoi_mask = builder.read_mask('aoi', spatial)
# Shape: [H, W], dtype: bool
# True = valid pixels, False = invalid/masked
```

#### 2. Boolean or Uint8 Masks
```yaml
lcms_interp_ok:
  type: boolean_or_uint8
  dimension: spatio-temporal
  zarr: {group: annual/lcms_chg/data, array: qa}
  ok_if: {op: '>=', value: 1}
```

```python
lcms_qa = builder.read_mask('lcms_interp_ok', spatial, temporal)
# Shape: [T, H, W], dtype: bool
# Pixels with qa >= 1 → True, else False
```

#### 3. Threshold Masks
```yaml
forest:
  type: threshold
  dimension: spatial
  source:
    zarr: {group: static/predictions, array: p_forest}
  ok_if: {op: '>=', value: 0.5}
  time: {use: last_in_window}
```

```python
forest_mask = builder.read_mask('forest', spatial, temporal)
# Reads p_forest, applies threshold >= 0.5
# If temporal: uses last timestep only
# Shape: [1, H, W] (or [H, W] if dimension=spatial)
```

### Reading Individual Quality Weights

```python
quality_result = builder.read_quality(
    quality_name='ls_summer_obs_weight',
    spatial_window=spatial,
    temporal_window=temporal
)
```

**Supported Quality Types:**

#### 1. Float Quality (Direct from Zarr)
```yaml
shared:
  quality:
    ls_summer_obs_weight:
      type: float
      dimension: spatio-temporal
      zarr: {group: annual/ls8day/data, array: summer_n_obs}
      transform:
        type: linear_rescale
        in_min: 0
        in_max: 20
        out_min: 0.0
        out_max: 1.0
        clamp: true
      time: {use: all}
```

```python
quality = builder.read_quality('ls_summer_obs_weight', spatial, temporal)
# Shape: [T, H, W], dtype: float32
# Values in [0.0, 1.0] after transform
```

#### 2. Expression Quality (Computed)
```yaml
forest_weight:
  type: expression
  dimension: spatial
  expression: "pow(p_forest, 2)"
  normalize: {mode: mean1}
```

```python
quality = builder.read_quality('forest_weight', spatial)
# Computes p_forest^2, normalizes to mean=1
# Shape: [H, W], dtype: float32
```

---

## Group-Based Methods (Recommended)

### Why Use Group-Based Methods?

✅ **Automatic shape matching** - Guaranteed to match DataReader output  
✅ **Per-band handling** - Each band gets its specific masks/quality  
✅ **No manual broadcasting** - Handles all dimensionality automatically  
✅ **Consistent API** - Same pattern for all group types  

### Temporal & Static Groups

```python
# Read data
data = reader.read_temporal_group('ls8day', spatial, temporal, True)
# Shape: [7, 10, 256, 256] - 7 bands, 10 years

# Read matching mask
mask = builder.read_mask_for_group(
    category='temporal',
    group_name='ls8day',
    spatial_window=spatial,
    temporal_window=temporal
)
# Shape: [7, 10, 256, 256] - guaranteed match!

# Read matching quality
quality = builder.read_quality_for_group(
    category='temporal',
    group_name='ls8day',
    spatial_window=spatial,
    temporal_window=temporal
)
# Shape: [7, 10, 256, 256] - guaranteed match!

# Apply directly - shapes match perfectly
masked_data = data.data * mask.data
weighted_data = masked_data * quality.data
```

**How it works:**

1. Looks up `ls8day` group configuration
2. For each of 7 bands:
   - Reads masks from `band.mask` (e.g., `['shared.masks.aoi', 'shared.masks.dem_ok']`)
   - Combines masks with AND operation
   - Result: `[T, H, W]` mask for this band
3. Stacks 7 bands → `[7, T, H, W]`
4. If band has no masks → uses all-True

**Per-band quality weights:**
```yaml
# Different bands can have different quality!
ls8day:
  bands:
    - name: NDVI_summer_p95
      quality_weight: [shared.quality.ls_summer_obs_weight]  # Summer only
    - name: NDVI_winter_max
      quality_weight: [shared.quality.ls_winter_obs_weight]  # Winter only
    - name: NDVI_amplitude
      quality_weight: [shared.quality.ls_summer_obs_weight,
                       shared.quality.ls_winter_obs_weight]  # Both!
```

```python
quality = builder.read_quality_for_group('temporal', 'ls8day', spatial, temporal)
# quality.data[0] = summer weight only
# quality.data[1] = winter weight only
# quality.data[2] = summer * winter (multiplied together)
```

### Snapshot Groups (Year-Specific)

```python
# Read snapshot data for 2024
data = reader.read_snapshot_group('ccdc_snapshot', spatial, year=2024)
# Shape: [20, 256, 256] - 20 CCDC bands for 2024

# Read matching mask
mask = builder.read_mask_for_snapshot_group(
    group_name='ccdc_snapshot',
    spatial_window=spatial,
    year=2024
)
# Shape: [20, 256, 256] - matches perfectly!

# Read matching quality
quality = builder.read_quality_for_snapshot_group(
    group_name='ccdc_snapshot',
    spatial_window=spatial,
    year=2024
)
# Shape: [20, 256, 256]
```

**Key differences from temporal:**
- No temporal window (snapshots are static for a specific year)
- Calls `group.get_bands_for_year(2024)` to get correct bands
- Output is `[C, H, W]` not `[C, T, H, W]`

### Irregular Groups (Sparse Temporal)

```python
# Read irregular data (e.g., NAIP with 3 observations in window)
data = reader.read_irregular_group('naip', spatial, temporal)
# Shape: [4, 3, 256, 256] - 4 bands, 3 observations

# Read matching mask
mask = builder.read_mask_for_irregular_group(
    group_name='naip',
    spatial_window=spatial,
    temporal_window=temporal
)
# Shape: [4, 3, 256, 256] - matches perfectly!

# Read matching quality
quality = builder.read_quality_for_irregular_group(
    group_name='naip',
    spatial_window=spatial,
    temporal_window=temporal
)
# Shape: [4, 3, 256, 256]
```

**How it works:**
1. Filters `group.years` to those in temporal window
2. For each band, reads static mask/quality
3. Replicates across all observations in window
4. Returns `[C, T_obs, H, W]` where T_obs varies

**Edge case - no observations:**
```python
# If no NAIP observations in this window
mask = builder.read_mask_for_irregular_group('naip', spatial, temporal)
# Shape: [4, 0, 256, 256] - empty temporal dimension
# Can still process without errors
```

---

## Dimension Enforcement

MaskBuilder enforces dimension specifications from your config:

```yaml
mask_name:
  dimension: spatial          # Expects [H, W]
  dimension: spatio-temporal  # Expects [T, H, W]
  dimension: temporal         # Expects [T]
```

**Validation:**
```python
# If dimension=spatial but array is [1, H, W]
# → Automatically squeezes to [H, W]

# If dimension=spatial but array is [T, H, W] where T > 1
# → Raises ValueError

# If dimension=spatio-temporal but array is [H, W]
# → Raises ValueError
```

**Why this matters:**
- Catches configuration errors early
- Ensures consistent behavior across different mask types
- Prevents silent bugs from unexpected shapes

---

## Combining Masks

```python
# Read multiple masks
aoi = builder.read_mask('aoi', spatial)
dem = builder.read_mask('dem_ok', spatial)
tpi = builder.read_mask('tpi_ok', spatial)

# Combine with AND (all must be True)
combined = builder.combine_masks([aoi, dem, tpi], operation='and')
# Result: [H, W], True only where all three are True

# Combine with OR (any can be True)
combined = builder.combine_masks([mask1, mask2], operation='or')
# Result: [H, W], True where either is True
```

**Handles `false_means_invalid`:**
```python
# If mask has false_means_invalid=False (unusual)
# → Inverts the mask before combining
# This ensures consistent behavior
```

**Shape validation:**
```python
# All masks must have same shape
combined = builder.combine_masks([mask_2d, mask_3d])  # ❌ Error!
# ValueError: Mask shape mismatch: (10, 256, 256) vs (256, 256)
```

---

## Padding & Missing Values

### Spatial Padding
Out-of-bounds pixels are padded with fill values:

```python
# Window extends beyond Zarr array bounds
spatial = SpatialWindow.from_upper_left_and_hw((-10, -10), (256, 256))

mask = builder.read_mask('aoi', spatial)
# First 10 rows/cols are padded with missing value
# Default: False for masks, 0.0 for quality
```

### Temporal Padding
Missing timesteps are padded:

```python
# Request 15 years but only 10 available
temporal = TemporalWindow(end_year=2020, window_length=15)

mask = builder.read_mask('forest', spatial, temporal)
# Shape: [15, H, W]
# First 5 timesteps are padded
```

### Missing Arrays
If Zarr array doesn't exist:

```python
# Array 'nonexistent' not found in Zarr
mask = builder.read_mask('nonexistent', spatial)
# Logs warning, returns array filled with missing value
# Mask: filled with False
# Quality: filled with 0.0
```

**Configure missing behavior:**
```yaml
mask_name:
  behavior:
    missing: true  # Use True instead of False for missing pixels

quality_name:
  missing:
    fill: 1.0      # Use 1.0 instead of 0.0 for missing pixels
```

---

## Temporal Selection

Control which timesteps are used:

```yaml
quality_name:
  time:
    use: all               # Use all timesteps (default)
    use: last_in_window    # Use only the last timestep
    use: first_in_window   # Use only the first timestep
```

```python
# Config has time: {use: last_in_window}
quality = builder.read_quality('snapshot_quality', spatial, temporal)
# Reads [T, H, W] then takes last → [1, H, W]
```

**Use cases:**
- `last_in_window` - Current state (e.g., latest forest probability)
- `first_in_window` - Initial state
- `all` - Full time series for temporal weighting

---

## Transforms

Apply transformations to quality weights:

### Linear Rescale
```yaml
quality_name:
  transform:
    type: linear_rescale
    in_min: 0
    in_max: 100
    out_min: 0.0
    out_max: 1.0
    clamp: true
```

```python
# Input values [0-100] → Output values [0.0-1.0]
# If clamp=true: values outside input range are clamped to output range
quality = builder.read_quality('rescaled_quality', spatial)
```

**Formula:**
```
output = (input - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
```

### Expression Normalization
```yaml
forest_weight:
  expression: "pow(p_forest, 2)"
  normalize:
    mode: mean1  # Normalize to mean=1
```

```python
# Computes p_forest^2
# Then divides by mean (excluding zeros)
# Result has mean=1.0
```

---

## Complete Example

```python
from data.loaders.bindings.parser import BindingsParser
from data.loaders.bindings.utils import BindingsRegistry
from data.loaders.windows import SpatialWindow, TemporalWindow
from data.loaders.data_reader import DataReader
from data.loaders.mask_builder import MaskBuilder

# Parse config
parser = BindingsParser('config/frl_bindings_v0.yaml')
config = parser.parse()
registry = BindingsRegistry(config)

# Initialize readers
reader = DataReader(config)
builder = MaskBuilder(registry, config)

# Define windows
spatial = SpatialWindow.from_upper_left_and_hw((1000, 2000), (256, 256))
temporal = TemporalWindow(end_year=2020, window_length=10)

# ============================================================================
# APPROACH 1: Individual masks/quality (manual handling)
# ============================================================================

# Read individual masks
aoi_mask = builder.read_mask('aoi', spatial)
forest_mask = builder.read_mask('forest', spatial, temporal)

# Read individual quality
ls_summer_quality = builder.read_quality('ls_summer_obs_weight', spatial, temporal)

# Manually broadcast and combine as needed
# ... complex manual logic ...

# ============================================================================
# APPROACH 2: Group-based (recommended - automatic)
# ============================================================================

# TEMPORAL GROUPS
ls8day_data = reader.read_temporal_group('ls8day', spatial, temporal, True)
ls8day_mask = builder.read_mask_for_group('temporal', 'ls8day', spatial, temporal)
ls8day_quality = builder.read_quality_for_group('temporal', 'ls8day', spatial, temporal)

assert ls8day_data.data.shape == ls8day_mask.data.shape == ls8day_quality.data.shape
# ✓ Guaranteed: all shapes are [7, 10, 256, 256]

# Apply
masked = ls8day_data.data * ls8day_mask.data
weighted = masked * ls8day_quality.data

# STATIC GROUPS
topo_data = reader.read_static_group('topo', spatial)
topo_mask = builder.read_mask_for_group('static', 'topo', spatial)
topo_quality = builder.read_quality_for_group('static', 'topo', spatial)

assert topo_data.data.shape == topo_mask.data.shape == topo_quality.data.shape
# ✓ Guaranteed: all shapes are [10, 256, 256]

# SNAPSHOT GROUPS
ccdc_data = reader.read_snapshot_group('ccdc_snapshot', spatial, 2024)
ccdc_mask = builder.read_mask_for_snapshot_group('ccdc_snapshot', spatial, 2024)
ccdc_quality = builder.read_quality_for_snapshot_group('ccdc_snapshot', spatial, 2024)

assert ccdc_data.data.shape == ccdc_mask.data.shape == ccdc_quality.data.shape
# ✓ Guaranteed: all shapes are [20, 256, 256]

# IRREGULAR GROUPS
naip_data = reader.read_irregular_group('naip', spatial, temporal)
naip_mask = builder.read_mask_for_irregular_group('naip', spatial, temporal)
naip_quality = builder.read_quality_for_irregular_group('naip', spatial, temporal)

assert naip_data.data.shape == naip_mask.data.shape == naip_quality.data.shape
# ✓ Guaranteed: all shapes are [4, T_obs, 256, 256]

print("All masks and quality weights perfectly aligned with data!")
```

---

## Design Philosophy

### 1. Per-Band Concatenation (Not Broadcasting)

**❌ Wrong Approach:**
```python
# Read all masks, combine, broadcast to all bands
combined_mask = combine_all_masks()  # [T, H, W]
broadcast_mask = np.broadcast_to(combined_mask[None], [C, T, H, W])  # ❌
# Problem: Band 0 might not need all masks!
```

**✅ Correct Approach:**
```python
# For each band, read its specific masks, combine, then stack
for band in bands:
    band_masks = [read_mask(m) for m in band.mask]
    band_combined = combine(band_masks)  # [T, H, W]
    all_bands.append(band_combined)
stack(all_bands)  # [C, T, H, W] ✓
```

### 2. Shape Guarantees

**Principle:** Output shape ALWAYS matches DataReader output shape.

| Group Type | Data Shape | Mask/Quality Shape | Match? |
|-----------|------------|-------------------|--------|
| Temporal | `[C, T, H, W]` | `[C, T, H, W]` | ✓ |
| Static | `[C, H, W]` | `[C, H, W]` | ✓ |
| Snapshot | `[C, H, W]` | `[C, H, W]` | ✓ |
| Irregular | `[C, T_obs, H, W]` | `[C, T_obs, H, W]` | ✓ |

**This enables:**
```python
# Element-wise operations without any reshaping
final = data * mask * quality  # Always works!
```

### 3. Graceful Degradation

**Missing masks/quality → sensible defaults:**
- Missing masks → all-True (no masking)
- Missing quality → all-ones (equal weighting)
- Missing arrays → filled with default values
- Empty windows → empty arrays with correct shape

**Never crashes, always returns valid data.**

---

## Troubleshooting

### Shape Mismatches

**Problem:**
```python
ValueError: Mask shape mismatch: (10, 256, 256) vs (256, 256)
```

**Solution:** All masks being combined must have same dimensionality. Check:
- Are you combining temporal mask `[T,H,W]` with static mask `[H,W]`?
- Did you specify correct `temporal_window` parameter?

### Dimension Errors

**Problem:**
```python
ValueError: forest: dimension=spatial expects [H,W], got shape=(10, 256, 256)
```

**Solution:** Dimension in config doesn't match array shape:
- If mask should be temporal: change `dimension: spatial` → `dimension: spatio-temporal`
- If mask should be static but has singleton time dim: array will auto-squeeze if `shape[0]==1`

### Missing Arrays

**Problem:**
```
WARNING: Quality array not found: annual/ls8day/data/summer_n_obs, using fill value
```

**Solution:**
- Check Zarr path is correct
- Verify array name matches Zarr dataset
- Array might be optional - warning is informational, returns default value

### Per-Band Quality Not Working

**Problem:** All bands getting same quality weights.

**Solution:** Use correct method:
```python
# ✓ Correct - per-band
quality = builder.read_quality_for_group('temporal', 'ls8day', ...)

# ❌ Wrong - would broadcast (but this method doesn't exist anymore!)
```

---

## Advanced: Custom Expressions

Current expression support is limited. To extend:

```python
def _read_expression_quality(self, quality_config, spatial, temporal):
    expression = quality_config['expression']
    
    # Add your custom expressions here
    if 'my_function' in expression:
        # Parse and evaluate
        result = your_custom_logic()
        return result
    
    # Fall back to existing logic
    ...
```

**Currently supported:**
- `pow(p_forest, N)` - Power function
- Variables from other quality metrics

**To add:**
- Arithmetic operations: `quality1 * quality2 + 0.5`
- Conditionals: `quality1 if mask else quality2`
- NumPy functions: `np.clip(quality, 0, 1)`

---

## See Also

- **DataReader** - Companion reader for actual data
- **BindingsRegistry** - Configuration management
- **Windows** - Spatial and temporal window utilities
- **Bindings YAML Reference** - Complete configuration format