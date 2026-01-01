# DataReader - Zarr Data Loading System

Complete system for reading raw data from Zarr arrays based on spatial and temporal windows with support for temporal, static, irregular, and snapshot input groups.

## Quick API Reference

```python
from data.loaders.bindings.parser import BindingsParser
from data.loaders.data_reader import DataReader
from data.loaders.windows import SpatialWindow, TemporalWindow

# Initialize
parser = BindingsParser('bindings.yaml')
config = parser.parse()
reader = DataReader(config)

# Define windows
spatial = SpatialWindow.from_upper_left_and_hw((row, col), (height, width))
temporal = TemporalWindow(end_year=2020, window_length=10)

# Read individual groups
temporal_result = reader.read_temporal_group(name, spatial, temporal, return_full_temporal=False)
irregular_result = reader.read_irregular_group(name, spatial, temporal)
static_result = reader.read_static_group(name, spatial)
snapshot_result = reader.read_snapshot_group(name, spatial, year)

# Read all groups by category
all_temporal = reader.read_all_temporal_groups(spatial, temporal, return_full_temporal=False)
all_irregular = reader.read_all_irregular_groups(spatial, temporal)
all_static = reader.read_all_static_groups(spatial)
all_snapshot = reader.read_all_snapshot_groups(spatial, year)

# Multi-window snapshots (for training bundles)
windows = reader.read_snapshots_for_windows(spatial, anchor_end=2024, offsets=[0, -2, -4])
# Returns: {'t0': {...}, 't2': {...}, 't4': {...}}
```

**Return Type**: All read methods return `GroupReadResult`:
```python
result.data          # np.ndarray - [C, H, W] or [C, T, H, W]
result.band_names    # List[str] - Band names in order
result.group_name    # str - Input group name
result.category      # str - 'temporal', 'irregular', 'static', 'snapshot'
result.metadata      # Dict - Windows, zarr paths, years, etc.
```

**Shapes**:
- Temporal: `[C, T, H, W]` (full) or `[C, H, W]` (snapshot)
- Irregular: `[C, T_obs, H, W]` where T_obs = observations in window
- Static: `[C, H, W]`
- Snapshot: `[C, H, W]` for specific year

---

## Overview

The `DataReader` class provides a unified interface for loading data from hierarchical Zarr datasets. It handles:

- **Temporal data**: Time series with configurable window lengths
- **Static data**: Unchanging spatial data (topography, land cover)
- **Irregular data**: Sparse temporal observations (NAIP imagery)
- **Snapshot data**: Point-in-time metrics that vary by year (CCDC snapshots)
- **Spatial padding**: NaN-padding for out-of-bounds regions
- **Missing values**: Configurable fill value policies
- **Band metadata**: Tracking for downstream normalization

## Quick Start

```python
from data.loaders.bindings.parser import BindingsParser
from data.loaders.data_reader import DataReader
from data.loaders.windows import SpatialWindow, TemporalWindow

# 1. Parse configuration
parser = BindingsParser('bindings.yaml')
config = parser.parse()

# 2. Initialize reader
reader = DataReader(config)

# 3. Define windows
spatial_window = SpatialWindow.from_upper_left_and_hw(
    upper_left=(1000, 2000),
    hw=(256, 256)
)

temporal_window = TemporalWindow(
    end_year=2020,
    window_length=10
)

# 4. Read data
temporal_data = reader.read_temporal_group(
    'ls8day',
    spatial_window,
    temporal_window,
    return_full_temporal=False
)

static_data = reader.read_static_group(
    'topo',
    spatial_window
)

snapshot_data = reader.read_snapshot_group(
    'ccdc_snapshot',
    spatial_window,
    year=2024
)
```

## Input Categories

The DataReader supports four input categories:

| Category   | Time Behavior | Shape | Example Use Case |
|------------|---------------|-------|------------------|
| `temporal` | Time series | `[C, T, H, W]` or `[C, H, W]` | Landsat time series |
| `irregular` | Sparse observations | `[C, H, W]` | NAIP imagery |
| `static` | Unchanging | `[C, H, W]` | Topography, soils |
| `snapshot` | Point-in-time per year | `[C, H, W]` | CCDC metrics |

## API Reference

### Constructor

```python
reader = DataReader(
    config: Dict[str, Any],      # Parsed bindings configuration
    zarr_path: Optional[str] = None  # Override zarr path from config
)
```

**Args:**
- `config`: Parsed configuration dictionary from `BindingsParser`
- `zarr_path`: Optional path to Zarr dataset (uses `config['zarr']['path']` if not provided)

**Example:**
```python
parser = BindingsParser('bindings.yaml')
config = parser.parse()

# Use path from config
reader = DataReader(config)

# Override path
reader = DataReader(config, zarr_path='/path/to/different.zarr')
```

### Core Methods

#### read_temporal_group()

Read data from a temporal input group.

```python
result = reader.read_temporal_group(
    group_name: str,                      # e.g., 'ls8day', 'lcms_chg'
    spatial_window: SpatialWindow,        # Spatial region to read
    temporal_window: TemporalWindow,      # Temporal range to read
    return_full_temporal: bool = False    # Return full [C,T,H,W] or snapshot [C,H,W]
) -> GroupReadResult
```

**Returns:** `GroupReadResult` with:
- `data`: `[C, T, H, W]` if `return_full_temporal=True`, else `[C, H, W]`
- `band_names`: List of band names
- `group_name`: Name of the group
- `category`: `'temporal'`
- `metadata`: Dict with window info, zarr paths, etc.

**Example:**
```python
# Read full time series (10 years)
result = reader.read_temporal_group(
    'ls8day',
    spatial_window,
    temporal_window,
    return_full_temporal=True
)
print(result.data.shape)  # [7, 10, 256, 256] - 7 bands, 10 years

# Read only end year
result = reader.read_temporal_group(
    'ls8day',
    spatial_window,
    temporal_window,
    return_full_temporal=False
)
print(result.data.shape)  # [7, 256, 256] - 7 bands, end year only
```

#### read_static_group()

Read data from a static input group.

```python
result = reader.read_static_group(
    group_name: str,              # e.g., 'topo', 'evt'
    spatial_window: SpatialWindow # Spatial region to read
) -> GroupReadResult
```

**Returns:** `GroupReadResult` with:
- `data`: `[C, H, W]`
- `band_names`: List of band names
- `group_name`: Name of the group
- `category`: `'static'`
- `metadata`: Dict with spatial window, zarr path, etc.

**Example:**
```python
result = reader.read_static_group('topo', spatial_window)
print(result.data.shape)  # [8, 256, 256] - 8 topo bands
print(result.band_names)  # ['elevation', 'slope', 'aspect', ...]
```

#### read_snapshot_group()

Read data from a snapshot input group for a specific year.

```python
result = reader.read_snapshot_group(
    group_name: str,              # e.g., 'ccdc_snapshot'
    spatial_window: SpatialWindow,
    year: int                     # Snapshot year (e.g., 2024)
) -> GroupReadResult
```

**Returns:** `GroupReadResult` with:
- `data`: `[C, H, W]` - Only bands for the specified year
- `band_names`: List of band names (e.g., `['green_2024', 'red_2024', ...]`)
- `group_name`: Name of the group
- `category`: `'snapshot'`
- `metadata`: Dict with `year`, `zarr_prefix`, spatial window, etc.

**Example:**
```python
# Read snapshot for 2024
result = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2024)
print(result.data.shape)  # [20, 256, 256] - 20 bands for 2024
print(result.metadata['year'])  # 2024
print(result.metadata['zarr_prefix'])  # 'snap_2024_0831'

# Read snapshot for 2022
result = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2022)
print(result.data.shape)  # [20, 256, 256] - 20 bands for 2022
```

### Batch Reading Methods

#### read_all_temporal_groups()

Read all temporal groups at once.

```python
results = reader.read_all_temporal_groups(
    spatial_window: SpatialWindow,
    temporal_window: TemporalWindow,
    return_full_temporal: bool = False
) -> Dict[str, GroupReadResult]
```

**Returns:** Dictionary mapping `group_name` → `GroupReadResult`

**Example:**
```python
results = reader.read_all_temporal_groups(
    spatial_window,
    temporal_window,
    return_full_temporal=False
)

# Access individual groups
ls8day = results['ls8day']
lcms_chg = results['lcms_chg']
```

#### read_all_static_groups()

Read all static groups at once.

```python
results = reader.read_all_static_groups(
    spatial_window: SpatialWindow
) -> Dict[str, GroupReadResult]
```

**Returns:** Dictionary mapping `group_name` → `GroupReadResult`

**Example:**
```python
results = reader.read_all_static_groups(spatial_window)

topo = results['topo']
evt = results['evt']
```

#### read_all_snapshot_groups()

Read all snapshot groups for a specific year.

```python
results = reader.read_all_snapshot_groups(
    spatial_window: SpatialWindow,
    year: int
) -> Dict[str, GroupReadResult]
```

**Returns:** Dictionary mapping `group_name` → `GroupReadResult`

**Example:**
```python
results = reader.read_all_snapshot_groups(spatial_window, 2024)

ccdc_snapshot = results['ccdc_snapshot']
print(ccdc_snapshot.data.shape)  # [20, 256, 256]
```

#### read_snapshots_for_windows()

Read snapshot data for multiple training windows.

**This is the recommended method for training with multi-window bundles.**

```python
results = reader.read_snapshots_for_windows(
    spatial_window: SpatialWindow,
    anchor_end: int,              # e.g., 2024
    offsets: List[int]            # e.g., [0, -2, -4] for t0, t2, t4
) -> Dict[str, Dict[str, GroupReadResult]]
```

**Returns:** Nested dict: `{window_label: {group_name: GroupReadResult}}`

**Example:**
```python
# Read snapshots for training windows t0, t2, t4
results = reader.read_snapshots_for_windows(
    spatial_window,
    anchor_end=2024,
    offsets=[0, -2, -4]
)

# Access by window
t0_data = results['t0']['ccdc_snapshot']  # Year 2024
t2_data = results['t2']['ccdc_snapshot']  # Year 2022
t4_data = results['t4']['ccdc_snapshot']  # Year 2020

print(t0_data.data.shape)  # [20, 256, 256]
print(t0_data.metadata['year'])  # 2024

print(t2_data.data.shape)  # [20, 256, 256]
print(t2_data.metadata['year'])  # 2022
```

### Utility Methods

#### get_band_metadata()

Get band configuration for downstream processing (e.g., normalization).

```python
band_config = reader.get_band_metadata(
    category: str,      # 'temporal', 'static', 'irregular', 'snapshot'
    group_name: str,
    band_name: str
) -> BandConfig
```

**Returns:** `BandConfig` object with:
- `name`: Band name
- `array`: Zarr array name
- `norm`: Normalization preset name
- `mask`: List of mask references
- `quality_weight`: List of quality weight references
- `loss_weight`: Loss weight reference

**Example:**
```python
band_config = reader.get_band_metadata(
    'temporal',
    'ls8day',
    'NDVI_summer_p95'
)

print(band_config.norm)  # 'zscore'
print(band_config.mask)  # ['shared.masks.summer_obs_mask']
```

## Data Structures

### SpatialWindow

Defines a rectangular spatial region to read.

```python
# From upper-left corner and size
window = SpatialWindow.from_upper_left_and_hw(
    upper_left=(row, col),  # (1000, 2000)
    hw=(height, width)      # (256, 256)
)

# From center and size
window = SpatialWindow.from_center_and_hw(
    center=(row, col),
    hw=(height, width)
)

# Access properties
print(window.row_start)  # 1000
print(window.col_start)  # 2000
print(window.height)     # 256
print(window.width)      # 256
```

### TemporalWindow

Defines a temporal range to read.

```python
window = TemporalWindow(
    end_year=2020,      # Last year in window
    window_length=10    # Number of years
)

# This reads years 2011-2020 (10 years ending in 2020)
```

### GroupReadResult

Result object returned by all read methods.

```python
@dataclass
class GroupReadResult:
    data: np.ndarray              # Raw data array
    band_names: List[str]         # Band names in order
    group_name: str               # Input group name
    category: str                 # 'temporal', 'static', 'irregular', 'snapshot'
    metadata: Dict[str, Any]      # Additional metadata
```

**Example:**
```python
result = reader.read_temporal_group('ls8day', spatial_window, temporal_window)

# Access data
data = result.data  # [C, T, H, W] or [C, H, W]

# Access band names
for i, band_name in enumerate(result.band_names):
    print(f"Band {i}: {band_name}")

# Access metadata
print(result.metadata['zarr_group'])  # 'annual/ls8day/data'
print(result.metadata['spatial_window'])
print(result.metadata['temporal_window'])
```

## Usage Patterns

### Pattern 1: Basic Data Loading

```python
from data.loaders.bindings.parser import BindingsParser
from data.loaders.data_reader import DataReader
from data.loaders.windows import SpatialWindow, TemporalWindow

# Initialize
parser = BindingsParser('bindings.yaml')
config = parser.parse()
reader = DataReader(config)

# Define windows
spatial_window = SpatialWindow.from_upper_left_and_hw((1000, 2000), (256, 256))
temporal_window = TemporalWindow(end_year=2020, window_length=10)

# Read temporal data
ls8day = reader.read_temporal_group('ls8day', spatial_window, temporal_window)

# Read static data
topo = reader.read_static_group('topo', spatial_window)

# Read snapshot data
ccdc_2024 = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2024)
```

### Pattern 2: Load All Groups

```python
# Read all data for a spatial region
spatial_window = SpatialWindow.from_center_and_hw((5000, 5000), (256, 256))
temporal_window = TemporalWindow(end_year=2020, window_length=10)

# Load everything
temporal_data = reader.read_all_temporal_groups(
    spatial_window,
    temporal_window,
    return_full_temporal=False
)

static_data = reader.read_all_static_groups(spatial_window)

snapshot_data = reader.read_all_snapshot_groups(spatial_window, 2024)

# Access individual groups
ls8day = temporal_data['ls8day']
topo = static_data['topo']
ccdc = snapshot_data['ccdc_snapshot']
```

### Pattern 3: Training Window Bundles

**Recommended pattern for training with multi-window bundles.**

```python
# Configuration
anchor_end = 2024
offsets = [0, -2, -4]  # t0, t2, t4

# Read snapshots for all windows at once
snapshot_windows = reader.read_snapshots_for_windows(
    spatial_window,
    anchor_end,
    offsets
)

# Each window has the appropriate year's snapshot
data_t0 = snapshot_windows['t0']['ccdc_snapshot'].data  # [20, 256, 256] - 2024
data_t2 = snapshot_windows['t2']['ccdc_snapshot'].data  # [20, 256, 256] - 2022
data_t4 = snapshot_windows['t4']['ccdc_snapshot'].data  # [20, 256, 256] - 2020

# Combine for model input
bundle_snapshots = np.concatenate([data_t0, data_t2, data_t4], axis=0)
# [60, 256, 256] - all three windows
```

### Pattern 4: PyTorch Dataset Integration

```python
import torch
from torch.utils.data import Dataset

class ForestDataset(Dataset):
    def __init__(self, config_path, zarr_path, patch_size=256):
        # Parse config
        parser = BindingsParser(config_path)
        self.config = parser.parse()
        
        # Initialize reader
        self.reader = DataReader(self.config, zarr_path)
        
        self.patch_size = patch_size
        
        # Pre-compute sample locations
        self.samples = self._generate_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Define windows
        spatial_window = SpatialWindow.from_center_and_hw(
            sample['center'],
            (self.patch_size, self.patch_size)
        )
        
        temporal_window = TemporalWindow(
            end_year=sample['year'],
            window_length=10
        )
        
        # Read temporal data
        temporal_data = self.reader.read_all_temporal_groups(
            spatial_window,
            temporal_window,
            return_full_temporal=False
        )
        
        # Read static data
        static_data = self.reader.read_all_static_groups(spatial_window)
        
        # Read snapshot data for training windows
        snapshot_data = self.reader.read_snapshots_for_windows(
            spatial_window,
            anchor_end=sample['year'],
            offsets=[0, -2, -4]
        )
        
        # Convert to tensors
        return self._to_tensors(temporal_data, static_data, snapshot_data)
    
    def _to_tensors(self, temporal_data, static_data, snapshot_data):
        # Stack temporal groups
        temporal_tensor = torch.cat([
            torch.from_numpy(result.data)
            for result in temporal_data.values()
        ], dim=0)
        
        # Stack static groups
        static_tensor = torch.cat([
            torch.from_numpy(result.data)
            for result in static_data.values()
        ], dim=0)
        
        # Stack snapshot windows
        snapshot_tensors = {}
        for window_label, groups in snapshot_data.items():
            snapshot_tensors[window_label] = torch.cat([
                torch.from_numpy(result.data)
                for result in groups.values()
            ], dim=0)
        
        return {
            'temporal': temporal_tensor,
            'static': static_tensor,
            'snapshots': snapshot_tensors
        }
```

### Pattern 5: Full Training Pipeline

```python
from data.loaders.bindings.parser import BindingsParser
from data.loaders.data_reader import DataReader
from normalization import NormalizationManager
from zarr_stats_loader import ZarrStatsLoader

class TrainingDataPipeline:
    def __init__(self, config_path, zarr_path):
        # Parse configuration
        parser = BindingsParser(config_path)
        self.config = parser.parse()
        
        # Initialize components
        self.reader = DataReader(self.config, zarr_path)
        self.stats_loader = ZarrStatsLoader(zarr_path)
        self.norm_manager = NormalizationManager(self.config, self.stats_loader)
        
        # Pre-load all stats
        self.norm_manager.preload_stats()
    
    def load_and_normalize(self, spatial_window, anchor_end, offsets):
        """Load and normalize all data for a training sample."""
        
        temporal_window = TemporalWindow(anchor_end, 10)
        
        # Read raw data
        temporal_data = self.reader.read_all_temporal_groups(
            spatial_window,
            temporal_window,
            return_full_temporal=False
        )
        
        static_data = self.reader.read_all_static_groups(spatial_window)
        
        snapshot_data = self.reader.read_snapshots_for_windows(
            spatial_window,
            anchor_end,
            offsets
        )
        
        # Normalize temporal data
        normalized_temporal = self._normalize_group_results(
            temporal_data,
            'temporal'
        )
        
        # Normalize static data
        normalized_static = self._normalize_group_results(
            static_data,
            'static'
        )
        
        # Normalize snapshot data
        normalized_snapshots = {}
        for window_label, groups in snapshot_data.items():
            normalized_snapshots[window_label] = self._normalize_group_results(
                groups,
                'snapshot'
            )
        
        return {
            'temporal': normalized_temporal,
            'static': normalized_static,
            'snapshots': normalized_snapshots
        }
    
    def _normalize_group_results(self, group_results, category):
        """Normalize all bands in a set of group results."""
        normalized = {}
        
        for group_name, result in group_results.items():
            # Get group config
            group_config = self.config['inputs'][category][group_name]
            
            # Normalize each band
            normalized_bands = []
            for i, band_name in enumerate(result.band_names):
                # Get band config
                band_config = next(
                    b for b in group_config.bands if b.name == band_name
                )
                
                # Normalize
                norm_data = self.norm_manager.normalize_band(
                    data=result.data[i],
                    group_path=group_config.zarr.group,
                    band_config={
                        'array': band_config.array,
                        'norm': band_config.norm
                    }
                )
                
                normalized_bands.append(norm_data)
            
            normalized[group_name] = np.stack(normalized_bands, axis=0)
        
        return normalized
```

## Special Features

### Out-of-Bounds Padding

The DataReader automatically handles spatial regions that extend beyond the dataset boundaries.

```python
# Dataset is 10000 x 10000
# Request a window that goes off the edge
spatial_window = SpatialWindow.from_upper_left_and_hw(
    upper_left=(9500, 9500),  # Near edge
    hw=(1000, 1000)           # Extends beyond bounds
)

result = reader.read_static_group('topo', spatial_window)
# Returns [C, 1000, 1000] with NaN padding where out-of-bounds
```

**Behavior:**
- Out-of-bounds regions are filled with `np.nan`
- Valid data is preserved
- No errors or exceptions

### Missing Value Policies

Apply configured missing value policies:

```yaml
# In YAML config
static:
  topo:
    missing_policy:
      nan_from_fill: [-9999, -9998]  # Convert these to NaN
```

```python
# DataReader automatically applies this policy
result = reader.read_static_group('topo', spatial_window)
# Any pixels with values -9999 or -9998 are now NaN
```

### Temporal Padding

For temporal groups, missing timesteps are padded with NaN:

```python
# Dataset has years 2000-2020 (21 years)
# Request window ending in 2020 with 25-year length
temporal_window = TemporalWindow(end_year=2020, window_length=25)

result = reader.read_temporal_group(
    'ls8day',
    spatial_window,
    temporal_window,
    return_full_temporal=True
)

# Returns [C, 25, H, W]
# First 4 timesteps are NaN (years 1996-1999 don't exist)
# Last 21 timesteps have data (years 2000-2020)
```

## Performance Tips

### 1. Batch Reading

Use batch methods when loading multiple groups:

```python
# SLOW: Individual calls
ls8day = reader.read_temporal_group('ls8day', ...)
lcms_chg = reader.read_temporal_group('lcms_chg', ...)
lcms_lc = reader.read_temporal_group('lcms_lc_p', ...)

# FAST: Single batch call
all_temporal = reader.read_all_temporal_groups(...)
```

### 2. Window Reuse

Reuse window objects:

```python
# Create once
spatial_window = SpatialWindow.from_center_and_hw((5000, 5000), (256, 256))
temporal_window = TemporalWindow(2020, 10)

# Reuse many times
for group_name in ['ls8day', 'lcms_chg', 'lcms_lc_p']:
    result = reader.read_temporal_group(
        group_name,
        spatial_window,  # Same window
        temporal_window  # Same window
    )
```

### 3. Pre-compute Spatial Windows

For training datasets, pre-compute all sample locations:

```python
class ForestDataset(Dataset):
    def __init__(self, config_path, zarr_path):
        # ... initialization ...
        
        # Pre-compute ALL windows at init
        self.spatial_windows = self._generate_all_windows()
    
    def __getitem__(self, idx):
        # Just look up pre-computed window (fast!)
        spatial_window = self.spatial_windows[idx]
        
        # Read data
        result = self.reader.read_static_group('topo', spatial_window)
        return result
```

### 4. Snapshot Window Batching

Use `read_snapshots_for_windows()` instead of individual calls:

```python
# SLOW: Three separate calls
t0 = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2024)
t2 = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2022)
t4 = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2020)

# FAST: Single batched call
windows = reader.read_snapshots_for_windows(
    spatial_window,
    anchor_end=2024,
    offsets=[0, -2, -4]
)
```

## Error Handling

### Missing Groups

```python
try:
    result = reader.read_temporal_group('nonexistent', spatial_window, temporal_window)
except ValueError as e:
    print(f"Group not found: {e}")
    # "Temporal group 'nonexistent' not found"
```

### Missing Arrays

If an array doesn't exist in the Zarr dataset, it's logged and filled with NaN:

```python
result = reader.read_temporal_group('ls8day', spatial_window, temporal_window)
# If 'NDVI_summer_p95' array is missing:
# WARNING: Array 'NDVI_summer_p95' not found in annual/ls8day/data, using NaN
# Result still contains data, but that band is all NaN
```

### Invalid Snapshot Years

```python
try:
    result = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2026)
except ValueError as e:
    print(f"Invalid year: {e}")
    # "Year 2026 not available for snapshot 'ccdc_snapshot'. Available years: [2020, 2022, 2024]"
```

### Out of Bounds Windows

Out-of-bounds regions are handled gracefully with NaN padding (no errors):

```python
# Window completely out of bounds
spatial_window = SpatialWindow.from_upper_left_and_hw((50000, 50000), (256, 256))

result = reader.read_static_group('topo', spatial_window)
# Returns [C, 256, 256] filled with NaN (no error)
```

## Testing

### Basic Functionality Test

```python
# test_data_reader.py
from data.loaders.bindings.parser import BindingsParser
from data.loaders.data_reader import DataReader
from data.loaders.windows import SpatialWindow, TemporalWindow

def test_basic_reading():
    # Parse config
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    config = parser.parse()
    
    # Initialize reader
    reader = DataReader(config)
    
    # Define windows
    spatial_window = SpatialWindow.from_upper_left_and_hw((1000, 2000), (256, 256))
    temporal_window = TemporalWindow(end_year=2020, window_length=10)
    
    # Test temporal reading
    result = reader.read_temporal_group(
        'ls8day',
        spatial_window,
        temporal_window,
        return_full_temporal=False
    )
    
    assert result.data.shape[1:] == (256, 256), "Wrong spatial shape"
    assert result.category == 'temporal', "Wrong category"
    assert len(result.band_names) == result.data.shape[0], "Band count mismatch"
    
    # Test static reading
    static_result = reader.read_static_group('topo', spatial_window)
    assert static_result.data.shape[1:] == (256, 256), "Wrong spatial shape"
    assert static_result.category == 'static', "Wrong category"
    
    # Test snapshot reading
    snapshot_result = reader.read_snapshot_group('ccdc_snapshot', spatial_window, 2024)
    assert snapshot_result.data.shape[1:] == (256, 256), "Wrong spatial shape"
    assert snapshot_result.category == 'snapshot', "Wrong category"
    assert snapshot_result.metadata['year'] == 2024, "Wrong year"
    
    print("✅ All tests passed!")

if __name__ == '__main__':
    test_basic_reading()
```

## Troubleshooting

### "Zarr group not found"

**Cause:** The zarr path in the config doesn't match the actual Zarr structure.

**Solution:** Check that `config['inputs'][category][group_name].zarr.group` matches the actual Zarr hierarchy.

### "Array not found, using NaN"

**Cause:** An array specified in the config doesn't exist in Zarr.

**Solution:** This is a warning, not an error. Check:
1. Array name spelling in config
2. Array exists in Zarr at the expected path
3. Zarr dataset was built correctly

### Unexpected NaN values

**Cause:** Could be:
1. Out-of-bounds padding
2. Missing arrays
3. Missing value policy conversion

**Solution:** Check:
```python
result = reader.read_static_group('topo', spatial_window)

# Check for NaN
import numpy as np
nan_count = np.isnan(result.data).sum()
print(f"NaN pixels: {nan_count} / {result.data.size}")

# Check metadata
print(f"Spatial window: {result.metadata['spatial_window']}")
print(f"Zarr group: {result.metadata['zarr_group']}")
```

### Shape mismatches

**Cause:** Config expects different band count than returned.

**Solution:** Verify bands in config match Zarr:
```python
# Check config
group = config['inputs']['temporal']['ls8day']
config_bands = len(group.bands)
print(f"Config expects {config_bands} bands")

# Check result
result = reader.read_temporal_group('ls8day', spatial_window, temporal_window)
actual_bands = result.data.shape[0]
print(f"Actually got {actual_bands} bands")

# Check band names
print(f"Band names: {result.band_names}")
```

## Integration Examples

See the included example at the bottom of `data_reader.py`:

```bash
# Run example
python -m data.loaders.data_reader config/frl_bindings_v0.yaml
```

This demonstrates:
- Basic temporal, static, and snapshot reading
- Multi-window snapshot loading
- Error handling
- Result inspection

## Summary

The DataReader provides a unified, high-level interface for loading all types of input data from Zarr datasets. Key features:

- ✅ **Four input categories**: temporal, static, irregular, snapshot
- ✅ **Automatic padding**: NaN-padding for out-of-bounds regions
- ✅ **Missing value handling**: Configurable policies
- ✅ **Batch operations**: Read all groups at once
- ✅ **Training-friendly**: Multi-window snapshot loading
- ✅ **Type-safe**: Strong typing with dataclasses
- ✅ **Well-tested**: Comprehensive error handling

Use it as the foundation for your data loading pipeline, combining with the normalization system for complete preprocessing.