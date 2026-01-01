# InputGroup Helper Methods for Snapshots

## Problem Solved

Functions that receive a parsed `config` dict (without the `parser` object) can now work with snapshot inputs directly.

## Solution

Added three methods to the `InputGroup` dataclass:

### 1. `get_bands_for_year(year)`

Get bands for a specific year from a snapshot input.

```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
bands_2024 = snapshot.get_bands_for_year(2024)

for band in bands_2024:
    print(f"{band.name}: {band.array}")
# green_2024: snap_2024_0831_green
# red_2024: snap_2024_0831_red
# ...
```

### 2. `get_zarr_prefix(year)`

Get the zarr prefix for a specific year.

```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
prefix = snapshot.get_zarr_prefix(2024)
# Returns: 'snap_2024_0831'
```

### 3. `is_snapshot` (property)

Check if an InputGroup is a snapshot input.

```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
topo = config['inputs']['static']['topo']

print(snapshot.is_snapshot)  # True
print(topo.is_snapshot)      # False
```

## Before vs After

### BEFORE: Needed parser object

```python
def load_snapshot_data(config, parser, year):
    """Function needs both config AND parser"""
    bands = parser.get_snapshot_bands('ccdc_snapshot', year)
    prefix = parser.get_zarr_prefix_for_year('ccdc_snapshot', year)
    return bands, prefix
```

**Problems:**
- ❌ Must pass `parser` everywhere
- ❌ Parser is heavyweight (not serializable)
- ❌ Awkward function signatures

### AFTER: Self-contained config

```python
def load_snapshot_data(config, year):
    """Function only needs config"""
    snapshot = config['inputs']['snapshot']['ccdc_snapshot']
    bands = snapshot.get_bands_for_year(year)
    prefix = snapshot.get_zarr_prefix(year)
    return bands, prefix
```

**Benefits:**
- ✅ Only need lightweight `config` dict
- ✅ Config is self-contained
- ✅ Cleaner function signatures
- ✅ Easier to serialize/cache config

## Use Case: PyTorch DataLoader

### The Problem

DataLoaders often receive config in `__init__` but not the parser:

```python
class ForestDataset(Dataset):
    def __init__(self, config, zarr_path, anchor_end, offsets):
        # We have config, but not parser!
        self.config = config
        # How do we get snapshot bands???
```

### The Solution

```python
class ForestDataset(Dataset):
    def __init__(self, config, zarr_path, anchor_end, offsets):
        self.config = config
        self.zarr_root = zarr.open(zarr_path)
        
        # Get snapshot directly from config
        self.snapshot = config['inputs']['snapshot']['ccdc_snapshot']
        
        # Pre-compute window configurations
        self.windows = {}
        for offset in offsets:
            year = anchor_end + offset
            window_label = f"t{abs(offset)}"
            
            # Use InputGroup methods - NO PARSER NEEDED!
            self.windows[window_label] = {
                'year': year,
                'bands': self.snapshot.get_bands_for_year(year),
                'zarr_prefix': self.snapshot.get_zarr_prefix(year)
            }
    
    def __getitem__(self, idx):
        data = {}
        
        for window_label, window_config in self.windows.items():
            # Load data for this window
            window_data = self._load_window(
                window_config['bands'],
                window_config['zarr_prefix']
            )
            data[window_label] = window_data
        
        return data
```

## Use Case: Distributed Training

### The Problem

When using `DistributedDataParallel`, you often serialize/pickle config:

```python
# Main process
parser = BindingsParser('bindings.yaml')
config = parser.parse()

# Send to workers - parser is NOT serializable!
dataset = ForestDataset(config, ...)  # Can't include parser!
```

### The Solution

```python
# Main process
parser = BindingsParser('bindings.yaml')
config = parser.parse()

# Config is self-contained - workers don't need parser!
dataset = ForestDataset(config, ...)  # ✅ Works!

# In worker process
def worker_fn(config):
    snapshot = config['inputs']['snapshot']['ccdc_snapshot']
    bands = snapshot.get_bands_for_year(2024)  # ✅ Works!
```

## API Reference

### `InputGroup.get_bands_for_year(year: int) -> List[BandConfig]`

**Args:**
- `year` (int): Year to get bands for

**Returns:**
- `List[BandConfig]`: Bands for the specified year

**Raises:**
- `ValueError`: If not a snapshot input or year not available

**Example:**
```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
bands = snapshot.get_bands_for_year(2024)
# Returns list of 20 BandConfig objects
```

### `InputGroup.get_zarr_prefix(year: int) -> str`

**Args:**
- `year` (int): Year to get prefix for

**Returns:**
- `str`: Instantiated zarr prefix (e.g., 'snap_2024_0831')

**Raises:**
- `ValueError`: If not a snapshot input or year not available
- `AttributeError`: If zarr_pattern not set

**Example:**
```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
prefix = snapshot.get_zarr_prefix(2024)
# Returns: 'snap_2024_0831'
```

### `InputGroup.is_snapshot` (property)

**Returns:**
- `bool`: True if this is a snapshot input

**Example:**
```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
if snapshot.is_snapshot:
    print(f"Available years: {snapshot.years}")
```

## Complete Example

```python
from data.loaders.bindings import BindingsParser
import zarr

# Parse config once
parser = BindingsParser('bindings.yaml')
config = parser.parse()

# Now config is self-contained - can be passed anywhere
def load_training_data(config, zarr_path, anchor_end, offsets):
    """
    Load snapshot data for training.
    Notice: only needs config, not parser!
    """
    # Open zarr
    zarr_root = zarr.open(zarr_path, mode='r')
    
    # Get snapshot from config
    snapshot = config['inputs']['snapshot']['ccdc_snapshot']
    
    # Check it's actually a snapshot
    if not snapshot.is_snapshot:
        raise ValueError("Expected snapshot input!")
    
    # Load data for each window
    window_data = {}
    for offset in offsets:
        year = anchor_end + offset
        window_label = f"t{abs(offset)}"
        
        # Get bands for this year (no parser needed!)
        bands = snapshot.get_bands_for_year(year)
        
        # Get zarr group
        zarr_group = zarr_root[snapshot.zarr.group]
        
        # Load data
        data = {}
        for band in bands:
            data[band.name] = zarr_group[band.array][...]
        
        window_data[window_label] = data
    
    return window_data

# Use it
data = load_training_data(
    config,
    zarr_path='/data/VA/zarr/dataset.zarr',
    anchor_end=2024,
    offsets=[0, -2, -4]
)

print(f"Loaded windows: {list(data.keys())}")  # ['t0', 't2', 't4']
print(f"Bands in t0: {len(data['t0'])}")       # 20
```

## Error Handling

```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']

# Invalid year
try:
    bands = snapshot.get_bands_for_year(2026)
except ValueError as e:
    print(e)
    # "Year 2026 not available for snapshot 'ccdc_snapshot'.
    #  Available years: [2020, 2022, 2024]"

# Not a snapshot
topo = config['inputs']['static']['topo']
try:
    bands = topo.get_bands_for_year(2024)
except ValueError as e:
    print(e)
    # "get_bands_for_year() only works on snapshot inputs.
    #  Input 'topo' has no years."
```

## Migration Guide

If you have existing code that passes parser around:

### Before
```python
class MyDataset(Dataset):
    def __init__(self, config, parser, ...):
        self.config = config
        self.parser = parser  # ← storing parser
    
    def __getitem__(self, idx):
        # Using parser methods
        bands = self.parser.get_snapshot_bands('ccdc_snapshot', 2024)
        prefix = self.parser.get_zarr_prefix_for_year('ccdc_snapshot', 2024)
```

### After
```python
class MyDataset(Dataset):
    def __init__(self, config, ...):  # ← no parser parameter
        self.config = config
        # Get snapshot once
        self.snapshot = config['inputs']['snapshot']['ccdc_snapshot']
    
    def __getitem__(self, idx):
        # Using InputGroup methods
        bands = self.snapshot.get_bands_for_year(2024)
        prefix = self.snapshot.get_zarr_prefix(2024)
```

## Summary

**Key Insight:** Config should be self-contained. Functions that receive config shouldn't also need the parser.

**Solution:** Add helper methods to `InputGroup` so it can answer snapshot-related queries directly.

**Benefits:**
- ✅ Cleaner function signatures
- ✅ Easier serialization
- ✅ Better encapsulation
- ✅ Simpler distributed training
- ✅ Config is truly self-contained

# get_snapshot_bands() Method Documentation

## Overview

The `get_snapshot_bands()` method returns only the bands for a specific year from a snapshot input. This is the recommended way to get bands when loading data for a specific training window.

## Method Signature

```python
def get_snapshot_bands(self, snapshot_name: str, year: int) -> List[BandConfig]
```

## Parameters

- **snapshot_name** (str): Name of the snapshot input (e.g., `'ccdc_snapshot'`)
- **year** (int): Year to get bands for (e.g., `2024`)

## Returns

- **List[BandConfig]**: List of band configurations for the specified year

## Raises

- **KeyError**: If snapshot not found
- **ValueError**: If year not available in snapshot

## Usage

### Basic Usage

```python
parser = BindingsParser('bindings.yaml')
config = parser.parse()

# Get bands for 2024
bands_2024 = parser.get_snapshot_bands('ccdc_snapshot', 2024)

print(f"Bands for 2024: {len(bands_2024)}")
for band in bands_2024:
    print(f"  {band.name}: {band.array}")

# Output:
# Bands for 2024: 20
#   green_2024: snap_2024_0831_green
#   red_2024: snap_2024_0831_red
#   nir_2024: snap_2024_0831_nir
#   ...
```

### Training Window Example

```python
# Training configuration
anchor_end = 2024
offsets = [0, -2, -4]  # t0, t2, t4

for i, offset in enumerate(offsets):
    window_year = anchor_end + offset
    window_label = f"t{abs(offset)}"
    
    # Get bands for this window
    bands = parser.get_snapshot_bands('ccdc_snapshot', window_year)
    
    print(f"Window {window_label} ({window_year}): {len(bands)} bands")

# Output:
# Window t0 (2024): 20 bands
# Window t2 (2022): 20 bands
# Window t4 (2020): 20 bands
```

### Data Loading Example

```python
import zarr

def load_snapshot_for_window(zarr_root, parser, snapshot_name, window_year):
    """Load snapshot data for a specific training window."""
    
    # Get bands for this year
    bands = parser.get_snapshot_bands(snapshot_name, window_year)
    
    # Get snapshot configuration
    snapshot_config = parser.get_input_group('snapshot', snapshot_name)
    zarr_group = zarr_root[snapshot_config.zarr.group]
    
    # Load data
    data = {}
    for band in bands:
        data[band.name] = zarr_group[band.array][...]
    
    return data

# Usage
zarr_root = zarr.open('/data/VA/zarr/dataset.zarr', mode='r')

data_t0 = load_snapshot_for_window(zarr_root, parser, 'ccdc_snapshot', 2024)
data_t2 = load_snapshot_for_window(zarr_root, parser, 'ccdc_snapshot', 2022)
data_t4 = load_snapshot_for_window(zarr_root, parser, 'ccdc_snapshot', 2020)

print(f"Loaded {len(data_t0)} bands for t0")
print(f"Loaded {len(data_t2)} bands for t2")
print(f"Loaded {len(data_t4)} bands for t4")
```

### Error Handling

```python
try:
    bands = parser.get_snapshot_bands('ccdc_snapshot', 2026)
except ValueError as e:
    print(f"Invalid year: {e}")
    # Output: Invalid year: Year 2026 not available for snapshot 'ccdc_snapshot'.
    #         Available years: [2020, 2022, 2024]

try:
    bands = parser.get_snapshot_bands('nonexistent', 2024)
except KeyError as e:
    print(f"Snapshot not found: {e}")
    # Output: Snapshot not found: Input group 'nonexistent' not found in 'snapshot'.
    #         Available: ['ccdc_snapshot']
```

### Validation Pattern

```python
# Validate that all required years are available
def validate_training_config(parser, snapshot_name, anchor_end, offsets):
    """Validate that snapshot has all required years."""
    
    metadata = parser.get_snapshot_metadata(snapshot_name)
    available_years = metadata['years']
    
    required_years = [anchor_end + offset for offset in offsets]
    missing_years = [y for y in required_years if y not in available_years]
    
    if missing_years:
        raise ValueError(
            f"Training configuration requires missing snapshots: {missing_years}\n"
            f"Available years: {available_years}"
        )
    
    print(f"✓ All required years available for anchor_end={anchor_end}")
    return True

# Usage
validate_training_config(parser, 'ccdc_snapshot', 2024, [0, -2, -4])
```

## Comparison with Alternatives

### Method 1: get_snapshot_bands() (Recommended)

```python
bands = parser.get_snapshot_bands('ccdc_snapshot', 2024)
```

**Pros:**
- ✅ Clean and explicit
- ✅ Validates year is available
- ✅ Returns only relevant bands
- ✅ Clear error messages

**Cons:**
- None

### Method 2: Manual filtering

```python
snapshot = parser.get_input_group('snapshot', 'ccdc_snapshot')
bands = [b for b in snapshot.bands if b.name.endswith('_2024')]
```

**Pros:**
- Works

**Cons:**
- ❌ More verbose
- ❌ No validation
- ❌ Fragile (depends on naming convention)
- ❌ Easy to make mistakes

### Method 3: Direct access

```python
snapshot = config['inputs']['snapshot']['ccdc_snapshot']
bands = [b for b in snapshot.bands if '_2024' in b.name]
```

**Pros:**
- Works

**Cons:**
- ❌ No validation
- ❌ No error checking
- ❌ Could match wrong bands (e.g., `test_2024_data` would match)

## Best Practices

### 1. Always validate years before training

```python
# At training initialization
metadata = parser.get_snapshot_metadata('ccdc_snapshot')
for year in required_years:
    if year not in metadata['years']:
        raise ValueError(f"Missing snapshot for year {year}")
```

### 2. Use get_snapshot_bands() in your dataloader

```python
class ForestDataset(Dataset):
    def __init__(self, parser, zarr_root, anchor_end, offsets):
        self.parser = parser
        self.zarr_root = zarr_root
        
        # Pre-compute window years
        self.window_years = [anchor_end + offset for offset in offsets]
        
        # Pre-fetch band lists
        self.window_bands = {
            f"t{abs(offset)}": parser.get_snapshot_bands('ccdc_snapshot', year)
            for offset, year in zip(offsets, self.window_years)
        }
    
    def __getitem__(self, idx):
        data = {}
        for window_label, bands in self.window_bands.items():
            data[window_label] = self._load_bands(bands)
        return data
```

### 3. Cache band lists, not data

```python
# Good: Cache band lists (cheap)
self.bands_t0 = parser.get_snapshot_bands('ccdc_snapshot', 2024)
self.bands_t2 = parser.get_snapshot_bands('ccdc_snapshot', 2022)

# Bad: Loading data in __init__ (expensive)
# self.data_t0 = load_all_data(...)  # Don't do this!
```

## Related Methods

```python
# Get all bands (all years)
all_bands = parser.get_input_group('snapshot', 'ccdc_snapshot').bands
# Returns: 60 bands (20 per year × 3 years)

# Get bands for one year
year_bands = parser.get_snapshot_bands('ccdc_snapshot', 2024)
# Returns: 20 bands (just for 2024)

# Get metadata
metadata = parser.get_snapshot_metadata('ccdc_snapshot')
# Returns: {'years': [2020, 2022, 2024], 'zarr_pattern': '...', ...}

# Get zarr prefix
prefix = parser.get_zarr_prefix_for_year('ccdc_snapshot', 2024)
# Returns: 'snap_2024_0831'
```

## Summary

**Use `get_snapshot_bands()`** when:
- Loading data for a specific training window
- You need only bands for one year
- You want validated, error-checked access

**It replaces manual filtering** and provides:
- ✅ Clear intent
- ✅ Validation
- ✅ Better error messages
- ✅ Less code