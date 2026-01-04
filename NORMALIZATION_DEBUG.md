# Normalization Debugging Guide

## Problem Summary

Elevation layer normalization is producing unexpected results:
- **Before normalization (raw data)**: mean=535.89, std=157.43
- **After normalization**: mean=696.97, std=135.33
- **Expected**: Should normalize to approximately mean≈0, std≈1 (or at least decrease, not increase!)

## Zarr Statistics

The elevation statistics embedded in Zarr (`static/topo/data/elevation`):
```json
{
  "mean": 479.19,
  "sd": 142.58,
  "q02": 183.82,
  "q25": 378.23,
  "q50": 456.96,  // median
  "q75": 561.99,
  "q98": 776.66,
  "min": 0.0,
  "max": 1102.21,
  "count": 9072
}
```

## Expected Normalization

Elevation uses `robust_iqr` normalization (not zscore):
```yaml
robust_iqr:
  type: robust_iqr
  stats_source: zarr
  fields: {q25: q25, q50: q50, q75: q75}
  clamp: {enabled: true, min: -8.0, max: 8.0}
```

Formula: `normalized = (x - q50) / IQR` where `IQR = q75 - q25`

**Expected calculation for this patch:**
- IQR = 561.99 - 378.23 = 183.76
- q50 (median) = 456.96
- Raw patch mean = 535.89
- Expected normalized mean = (535.89 - 456.96) / 183.76 ≈ **0.43**

**Actual result:** mean = 696.97 ❌

## Why the Raw Mean Differs from Zarr Mean

Note that the raw patch mean (535.89) differs from the global Zarr mean (479.19). This is expected because:
- Zarr stats are **global** statistics computed over the entire dataset
- Raw data shown is from a **single patch**
- Different patches will have different local statistics
- Normalization uses global stats to make all patches comparable

## Possible Causes

1. **Stats not loading correctly** - Stats may be None, zeros, or malformed arrays
2. **Wrong statistics being used** - Might be using stats from a different band
3. **Inverse operation** - Something is applying denormalization instead
4. **Array shape mismatch** - Stats might have wrong shape for broadcasting
5. **Exception being silently caught** - Error in normalization returning wrong data
6. **Channel ordering issue** - Data channel 0 might not actually be elevation

## Debug Logging Added

I've added comprehensive logging to `forest_dataset.py` that will show:

```python
logger.info(f"Normalizing {group_name}.{band_name} (array={array_name}, preset={norm_preset})")
logger.info(f"  Data shape: {shape}, dtype: {dtype}")
logger.info(f"  Stats from Zarr:")
for k, v in stats.items():
    logger.info(f"    {k}: shape={v.shape}, dtype={v.dtype}, value={v}")
logger.info(f"  Before: mean={mean:.4f}, std={std:.4f}")
logger.info(f"  Normalizer: {normalizer_class_name}")
logger.info(f"  Normalizer config: type={type}, stats_source={source}")
logger.info(f"  After:  mean={mean:.4f}, std={std:.4f}")
```

## How to Debug

### Step 1: Run the demo with logging

```bash
# Set logging to INFO level
export PYTHONPATH=/home/user/vq-vae/frl:$PYTHONPATH

python frl/examples/data/normalization_demo.py 2>&1 | tee normalization_debug.log
```

### Step 2: Check the log output

Look for the elevation normalization section in the logs. You should see something like:

```
INFO: Normalizing topo.elevation (array=elevation, preset=robust_iqr)
INFO:   Data shape: (256, 256), dtype: float32
INFO:   Stats from Zarr:
INFO:     q25: shape=(), dtype=float32, value=378.23
INFO:     q50: shape=(), dtype=float32, value=456.96
INFO:     q75: shape=(), dtype=float32, value=561.99
INFO:   Before: mean=535.8868, std=157.4312
INFO:   Normalizer: RobustIQRNormalizer
INFO:   Normalizer config: type=robust_iqr, stats_source=zarr
INFO:   After:  mean=0.4298, std=0.8567
```

### Step 3: Analyze the output

Check:
1. Are the stats loading correctly? (Should see q25=378.23, q50=456.96, q75=561.99)
2. Are the stats scalar (shape=())? (Should be 0-dimensional arrays)
3. Is the correct normalizer being used? (Should be RobustIQRNormalizer)
4. What's the actual "After" mean? (Should be around 0.43, not 696.97)

### Step 4: Manual verification

You can manually verify the normalization in Python:

```python
import numpy as np

# From the log output
raw_mean = 535.8868
q25 = 378.23
q50 = 456.96
q75 = 561.99

# Calculate IQR
iqr = q75 - q25  # Should be 183.76

# Expected normalized mean
expected = (raw_mean - q50) / iqr
print(f"Expected normalized mean: {expected:.4f}")  # Should be ~0.43
```

## Code Locations

- **Normalization implementation**: `frl/data/normalization/normalization.py:142` (RobustIQRNormalizer)
- **Stats loading**: `frl/data/normalization/zarr_stats_loader.py:60` (get_stats method)
- **Dataset normalization**: `frl/data/loaders/dataset/forest_dataset.py:261` (_normalize_regular_bands)
- **Normalization manager**: `frl/data/normalization/normalization.py:423` (normalize_band)

## Next Steps

1. Run the demo with the new debug logging
2. Share the log output showing the elevation normalization section
3. Based on the logs, we can identify whether:
   - Stats are loading incorrectly
   - Wrong normalizer is being used
   - There's an exception being caught
   - Some other transformation is interfering

## Quick Test

If you want to quickly test if the RobustIQRNormalizer works correctly:

```python
import numpy as np
import sys
sys.path.insert(0, 'frl')

from data.normalization.normalization import RobustIQRNormalizer, NormalizationConfig

# Create config
config = NormalizationConfig(
    type='robust_iqr',
    stats_source='fixed',
    clamp={'enabled': True, 'min': -8.0, 'max': 8.0},
    missing={'fill': 0.0}
)

# Create stats dict
stats = {
    'q25': np.array(378.23, dtype=np.float32),
    'q50': np.array(456.96, dtype=np.float32),
    'q75': np.array(561.99, dtype=np.float32),
}

# Create normalizer
normalizer = RobustIQRNormalizer(config, stats)

# Test data
test_data = np.array([[535.89]], dtype=np.float32)

# Normalize
result = normalizer.normalize(test_data)

print(f"Input: {test_data[0,0]:.2f}")
print(f"Output: {result[0,0]:.4f}")
print(f"Expected: ~0.43")
```

This will verify if the normalizer itself is working correctly in isolation.
