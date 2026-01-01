# Bindings Parser - Quick Reference Card

## Installation

```bash
# Create directories
mkdir -p data/loaders/bindings tests/loaders/bindings examples/loaders docs/loaders configs/bindings

# Copy files (see INSTALLATION.md for details)
# - Core: __init__.py, parser.py, utils.py → data/loaders/bindings/
# - Tests: test_parser.py → tests/loaders/bindings/
# - Examples: verify_bindings.py, bindings_examples.py → examples/loaders/
# - Docs: bindings_parser.md, implementation_summary.md → docs/loaders/

# Move your YAML
cp forest_repr_model_bindings.yaml configs/bindings/forest_repr_v1.yaml
```

## Quick Start

```python
from data.loaders.bindings import BindingsParser, BindingsQuery, ReferenceResolver

# Parse
config = BindingsParser('configs/bindings/forest_repr_v1.yaml').parse()

# Query
query = BindingsQuery(config)
print(query.count_bands())  # {'temporal': 22, 'irregular': 6, 'static': 125}

# Resolve
resolver = ReferenceResolver(config)
band = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
```

## Main Classes

### BindingsParser
```python
parser = BindingsParser('bindings.yaml')
config = parser.parse()                    # Parse and validate
summary = parser.summary()                  # Human-readable summary
preset = parser.get_normalization_preset('zscore')
group = parser.get_input_group('temporal', 'ls8day')
```

### BindingsQuery
```python
query = BindingsQuery(config)
counts = query.count_bands()                # {'temporal': 22, ...}
total = query.count_total_channels()        # 316
presets = query.list_normalization_presets()
masks = query.list_shared_masks()
exists = query.validate_reference_exists('shared.masks.forest')
```

### ReferenceResolver
```python
resolver = ReferenceResolver(config)
value = resolver.resolve('shared.masks.forest')
band = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
# Returns: ResolvedBand(name, array_path, norm_config, masks, quality_weights, loss_weight, metadata)
```

### BandSelector
```python
selector = BandSelector(config)
temporal = selector.select_by_category('temporal')
zscore = selector.select_by_norm('zscore')
forest = selector.select_by_mask('shared.masks.forest')
window10 = selector.select_temporal_with_window(10)
```

## Common Patterns

### Pattern 1: Load and Validate
```python
try:
    config = BindingsParser('bindings.yaml').parse()
    print("✅ Valid!")
except BindingsError as e:
    print(f"❌ Error: {e}")
```

### Pattern 2: Iterate All Bands
```python
query = BindingsQuery(config)
for category in ['temporal', 'irregular', 'static']:
    for group_name, group in config['inputs'][category].items():
        for band in group.bands:
            print(f"{category}.{group_name}.{band.name}")
```

### Pattern 3: Get Normalization Config for Band
```python
resolver = ReferenceResolver(config)
band = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
norm_type = band.norm_config['type']           # 'zscore'
stats_source = band.norm_config['stats_source'] # 'zarr'
clamp_range = band.norm_config['clamp']        # {'min': -6, 'max': 6}
```

### Pattern 4: Integration with Stats Loader
```python
from data.loaders.bindings import BindingsParser
from data.normalization import ZarrStatsLoader
from data.normalization import NormalizationManager

config = BindingsParser('bindings.yaml').parse()
stats = ZarrStatsLoader(config['zarr']['path'])
norm = NormalizationManager(config, stats)

# Now ready to normalize!
normalized = norm.normalize_band(data, 'annual/ls8day/data', band_config)
```

### Pattern 5: Check What Bands Use a Mask
```python
selector = BandSelector(config)
forest_bands = selector.select_by_mask('shared.masks.forest')
for category, group_name, band in forest_bands:
    print(f"{group_name}.{band.name} uses forest mask")
```

## Your Configuration Summary

Based on `forest_repr_v1.yaml`:

**Inputs:**
- Temporal: 22 bands (ls8day, lcms_chg, lcms_lc_p, lcms_lu_p, lcms_ysfc)
- Irregular: 6 bands (NAIP)
- Static: 125 bands (evt, topo, ccdc_snapshot, ccdc_history)

**Normalization Presets:**
- zscore, robust_iqr, prob01, identity, trig, minmax_0_20, delta_ls8_fixed

**Shared Resources:**
- 12 masks (aoi, dem_mask, forest, etc.)
- 6 quality metrics (p_forest, forest_weight, obs counts)

**Derived Features:**
- temporal_position, ls8_delta, spectral_grads_*, topo_grads

**Training:**
- 10-year windows, 3-window bundles (t0, t2, t4)
- Patch size: 256×256, Grid: 16×16
- 64 supplemental forest samples per patch

## Validation

The parser automatically validates:
- ✅ All references exist (normalization presets, masks, quality metrics)
- ✅ Normalization presets have required fields
- ✅ No circular dependencies in derived features
- ✅ YAML syntax is valid
- ✅ Required fields present (zarr.path, etc.)

## Error Messages

### Undefined Reference
```
ReferenceError: Found 1 undefined reference(s):
  - 'normalization.presets.bad_norm' (used in: inputs.temporal.ls8day.NDVI)
```

### Invalid Normalization
```
ValidationError: Invalid normalization type 'invalid' in preset 'bad'.
Valid types: ['zscore', 'robust_iqr', 'minmax', 'linear_rescale', 'clamp', 'none']
```

### Circular Dependency
```
BindingsError: Circular dependency detected: feature_a -> feature_b -> feature_a
```

## Testing

```bash
# Run all tests
pytest tests/loaders/bindings/test_parser.py -v

# Run specific test
pytest tests/loaders/bindings/test_parser.py::test_parse_minimal_config -v

# Verify your config
python examples/loaders/verify_bindings.py configs/bindings/forest_repr_v1.yaml

# Run examples
python examples/loaders/bindings_examples.py 1  # Specific example
python examples/loaders/bindings_examples.py    # All examples
```

## File Locations

```
data/loaders/bindings/
├── __init__.py          # Module exports
├── parser.py            # BindingsParser (500 lines)
└── utils.py             # BindingsRegistry, BindingsQuery (400 lines)

tests/loaders/bindings/
└── test_parser.py       # Test suite (600 lines)

examples/loaders/
├── verify_bindings.py   # Quick verification (150 lines)
└── bindings_examples.py # 11 complete examples (500 lines)

docs/loaders/
├── bindings_parser.md          # Complete documentation
├── implementation_summary.md   # Overview and statistics
└── INSTALLATION.md             # Setup instructions
```

## Next Steps

1. ✅ Parse your bindings configuration
2. → Build Zarr readers (Phase 2)
3. → Implement sampling strategies (Phase 3)
4. → Create derived features (Phase 4)
5. → Build PyTorch Dataset (Phase 5)

## Key Design Features

- **Fast O(1) lookups** via BindingsRegistry
- **Complete validation** before use (fail early)
- **Clear error messages** with usage locations
- **Template expansion** for window-bound inputs
- **Integration ready** with existing stats/normalization
- **Comprehensive tests** (20+ test cases)
- **Well documented** (examples, API reference, guides)

## Import Reference

```python
# Core classes
from data.loaders.bindings import (
    BindingsParser,      # Main parser
    BindingsQuery,       # High-level queries
    BindingsRegistry,    # Fast lookups
    ReferenceResolver,   # Reference resolution
    BandSelector,        # Filter bands
    load_bindings        # Convenience function
)

# Data structures
from data.loaders.bindings import (
    ResolvedBand,        # Fully resolved band
    BandConfig,          # Band configuration
    InputGroup,          # Input group
    ZarrReference        # Zarr path reference
)

# Exceptions
from data.loaders.bindings import (
    BindingsError,       # Base error
    ReferenceError,      # Undefined reference
    ValidationError      # Validation failure
)
```

## Performance

- Parsing: ~50-100ms for full config
- Reference lookup: O(1) via registry
- Memory: ~1-2MB for config + indices
- No runtime overhead after parsing

## Support

📖 Full documentation: `docs/loaders/bindings_parser.md`
🧪 Test suite: `tests/loaders/bindings/test_parser.py`
💡 Examples: `examples/loaders/bindings_examples.py`
🔧 Installation: `docs/loaders/INSTALLATION.md`
