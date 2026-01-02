"""
Forest Patch Sampler - Spatial and Temporal Sampling for Training

Generates (spatial_window, anchor_year) pairs for training the forest
representation model. Uses a deterministic checkerboard split pattern for
train/val/test separation and weighted sampling for anchor year selection.

Key features:
- Deterministic spatial splits (train/val/test) using block-based checkerboard
- AOI-based filtering (only patches with sufficient valid coverage)
- Weighted anchor year sampling from endpoint years
- Flexible epoch modes: full (all patches) or sampled (subset per epoch)
- Debug window support for development
"""

import zarr
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass
from pathlib import Path
import logging

from data.loaders.windows import SpatialWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test"]


@dataclass
class SamplerConfig:
    """Configuration extracted from bindings and training configs."""
    # From bindings
    zarr_path: str
    aoi_zarr_group: str
    aoi_zarr_array: str
    
    # From training - spatial
    patch_size: int
    block_height: int
    block_width: int
    use_debug_window: bool
    window_origin: Optional[Tuple[int, int]]  # (row, col) in pixels
    window_size: Optional[Tuple[int, int]]    # (height, width) in pixels
    
    # From training - temporal
    endpoint_years: List[int]
    anchor_weights: Dict[int, float]
    
    # From training - epoch
    epoch_mode: str  # 'full', 'frac', 'number'
    sample_frac: Optional[float]
    sample_number: Optional[int]
    
    # Filtering
    min_aoi_fraction: float = 0.3


class ForestPatchSampler:
    """
    Sampler for (spatial_window, anchor_year) pairs.
    
    Uses a deterministic checkerboard pattern to assign patches to train/val/test
    splits based on global patch indices. Only patches with sufficient AOI
    coverage are kept.
    
    Anchor years are sampled according to weights specified in training config.
    
    Usage:
        sampler = ForestPatchSampler(bindings_config, training_config, split='train')
        
        # Get number of samples for this epoch
        n_samples = len(sampler)
        
        # Sample a batch
        for idx in range(n_samples):
            spatial_window, anchor_year = sampler[idx]
    """
    
    def __init__(
        self,
        bindings_config: Dict[str, Any],
        training_config: Dict[str, Any],
        split: Optional[SplitName] = None
    ):
        """
        Initialize ForestPatchSampler.
        
        Args:
            bindings_config: Parsed bindings configuration
            training_config: Parsed training configuration
            split: Optional split to filter ('train', 'val', 'test', or None for all)
        """
        self.split = split
        
        # Extract configuration
        self.config = self._extract_config(bindings_config, training_config)
        
        # Open Zarr dataset
        self.zarr_path = Path(self.config.zarr_path)
        logger.info(f"Opening Zarr dataset: {self.zarr_path}")
        self.zarr_root = zarr.open(str(self.zarr_path), mode='r')
        
        # Load and process AOI
        self._load_aoi()
        
        # Build valid patch grid and apply split
        self._build_patch_grid()
        
        # Setup anchor year sampling
        self._setup_anchor_sampling()
        
        # Setup epoch sampling strategy
        self._setup_epoch_sampling()
        
        logger.info(
            f"ForestPatchSampler initialized: split='{split}', "
            f"{len(self.valid_patches)} patches, "
            f"epoch_mode='{self.config.epoch_mode}'"
        )
    
    def _extract_config(
        self,
        bindings_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> SamplerConfig:
        """Extract all needed config parameters."""
        
        # From bindings - zarr path
        zarr_path = bindings_config['zarr']['path']
        
        # From bindings - AOI mask location
        aoi_mask_config = bindings_config['shared']['masks']['aoi']
        aoi_zarr_group = aoi_mask_config['zarr']['group']
        aoi_zarr_array = aoi_mask_config['zarr']['array']
        
        # From training - spatial domain
        spatial_config = training_config['spatial_domain']
        use_debug = spatial_config.get('debug_mode', False)
        
        if use_debug and 'debug_window' in spatial_config:
            domain_config = spatial_config['debug_window']
            window_origin = tuple(domain_config['origin'])  # [row, col]
            window_size = tuple(domain_config['size'])      # [H, W]
            block_grid = domain_config['block_grid']        # [block_h, block_w]
        else:
            domain_config = spatial_config['full_domain']
            window_origin = None
            window_size = None
            block_grid = domain_config['block_grid']
        
        block_height, block_width = block_grid
        
        # From training - sampling
        patch_size = training_config['sampling']['patch_size']
        
        # From training - temporal domain
        temporal_config = training_config['temporal_domain']
        endpoint_years = temporal_config['end_years']
        
        sampling_config = temporal_config['sampling']
        anchor_weights = sampling_config['weights']
        
        # From training - epoch config
        epoch_config = training_config['training']['epoch']
        epoch_mode = epoch_config['mode']
        sample_frac = epoch_config.get('sample_frac')
        sample_number = epoch_config.get('sample_number')
        
        return SamplerConfig(
            zarr_path=zarr_path,
            aoi_zarr_group=aoi_zarr_group,
            aoi_zarr_array=aoi_zarr_array,
            patch_size=patch_size,
            block_height=block_height,
            block_width=block_width,
            use_debug_window=use_debug,
            window_origin=window_origin,
            window_size=window_size,
            endpoint_years=endpoint_years,
            anchor_weights=anchor_weights,
            epoch_mode=epoch_mode,
            sample_frac=sample_frac,
            sample_number=sample_number
        )
    
    def _load_aoi(self):
        """Load and process AOI mask."""
        # Get AOI array from Zarr
        # Try to build path intelligently based on config
        # If group is empty or same as array, use array name directly
        if not self.config.aoi_zarr_group or self.config.aoi_zarr_group == self.config.aoi_zarr_array:
            aoi_path = self.config.aoi_zarr_array
        else:
            aoi_path = f"{self.config.aoi_zarr_group}/{self.config.aoi_zarr_array}"
        
        try:
            aoi_array = self.zarr_root[aoi_path]
        except KeyError as e:
            raise ValueError(
                f"AOI not found at '{aoi_path}': {e}\n"
                f"Available arrays at root: {list(self.zarr_root.keys())}"
            )
        
        # Get full AOI shape
        self.full_height, self.full_width = aoi_array.shape
        logger.info(f"Full AOI shape: {self.full_height} x {self.full_width}")
        
        # Determine window to load
        if self.config.use_debug_window:
            row0, col0 = self.config.window_origin
            h, w = self.config.window_size
            
            # Validate alignment
            ps = self.config.patch_size
            if row0 % ps != 0 or col0 % ps != 0:
                raise ValueError(
                    f"window_origin {self.config.window_origin} must be "
                    f"multiples of patch_size={ps}"
                )
            if h % ps != 0 or w % ps != 0:
                raise ValueError(
                    f"window_size {self.config.window_size} must be "
                    f"multiples of patch_size={ps}"
                )
            
            # Clip to valid bounds
            row1 = min(row0 + h, self.full_height)
            col1 = min(col0 + w, self.full_width)
            
            # Load windowed AOI
            aoi_data = aoi_array[row0:row1, col0:col1]
            
            self.window_origin = (row0, col0)
            self.patch_row_offset = row0 // ps
            self.patch_col_offset = col0 // ps
            
        else:
            # Load full domain, snapped to patch grid
            ps = self.config.patch_size
            max_row = (self.full_height // ps) * ps
            max_col = (self.full_width // ps) * ps
            
            aoi_data = aoi_array[0:max_row, 0:max_col]
            
            self.window_origin = (0, 0)
            self.patch_row_offset = 0
            self.patch_col_offset = 0
        
        # Convert to boolean
        self.aoi = (aoi_data != 0)
        logger.info(
            f"Loaded AOI window: {self.aoi.shape}, "
            f"offset=({self.patch_row_offset}, {self.patch_col_offset})"
        )
    
    def _build_patch_grid(self):
        """
        Build grid of valid patches and apply spatial split.
        
        Creates lists of:
        - patch_origins_raw: All patches passing AOI threshold (before split filter)
        - patch_split_codes_raw: Split codes for all patches (1=train, 2=val, 3=test)
        - valid_patches: Filtered list based on requested split
        """
        ps = self.config.patch_size
        
        # Compute AOI fraction for each patch
        # Each element is the fraction of valid pixels in that patch (0.0 to 1.0)
        h, w = self.aoi.shape
        n_patches_y = h // ps
        n_patches_x = w // ps
        
        logger.info(
            f"Computing patch AOI fractions: {n_patches_y}×{n_patches_x} = "
            f"{n_patches_y * n_patches_x} total patches"
        )
        
        patch_aoi_fractions = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)
        
        for py in range(n_patches_y):
            for px in range(n_patches_x):
                patch_aoi = self.aoi[py*ps:(py+1)*ps, px*ps:(px+1)*ps]
                patch_aoi_fractions[py, px] = patch_aoi.mean()
        
        logger.info(f"Computed patch AOI fractions: {patch_aoi_fractions.shape}")
        
        # Log AOI statistics
        logger.info(
            f"Patch AOI fraction stats: "
            f"min={patch_aoi_fractions.min():.3f}, "
            f"max={patch_aoi_fractions.max():.3f}, "
            f"mean={patch_aoi_fractions.mean():.3f}"
        )
        
        # Count patches above threshold
        n_above_threshold = (patch_aoi_fractions >= self.config.min_aoi_fraction).sum()
        logger.info(
            f"Patches with AOI >= {self.config.min_aoi_fraction}: "
            f"{n_above_threshold} / {patch_aoi_fractions.size} "
            f"({100 * n_above_threshold / patch_aoi_fractions.size:.1f}%)"
        )
        
        # Build patch lists
        self.patch_origins_raw: List[Tuple[int, int]] = []
        self.patch_split_codes_raw: List[int] = []
        
        for py_local in range(n_patches_y):
            for px_local in range(n_patches_x):
                frac_valid = float(patch_aoi_fractions[py_local, px_local])
                
                if frac_valid < self.config.min_aoi_fraction:
                    continue
                
                # Global patch indices (for deterministic split)
                py_global = self.patch_row_offset + py_local
                px_global = self.patch_col_offset + px_local
                
                # Compute split code
                split_code = self._compute_split_code(py_global, px_global)
                
                # Local (window-relative) origin in pixels
                row0_local = py_local * ps
                col0_local = px_local * ps
                
                self.patch_origins_raw.append((row0_local, col0_local))
                self.patch_split_codes_raw.append(split_code)
        
        # Count by split
        split_counts = {1: 0, 2: 0, 3: 0}
        for code in self.patch_split_codes_raw:
            split_counts[code] += 1
        
        logger.info(
            f"Patches after AOI filter: "
            f"train={split_counts[1]:,}, val={split_counts[2]:,}, "
            f"test={split_counts[3]:,}"
        )
        
        # Filter by requested split
        self.valid_patches: List[Tuple[int, int]] = []
        self.valid_split_codes: List[int] = []
        
        for (row0, col0), code in zip(
            self.patch_origins_raw, self.patch_split_codes_raw
        ):
            if self.split == 'train' and code != 1:
                continue
            if self.split == 'val' and code != 2:
                continue
            if self.split == 'test' and code != 3:
                continue
            
            self.valid_patches.append((row0, col0))
            self.valid_split_codes.append(code)
        
        logger.info(
            f"Loaded split='{self.split}' -> {len(self.valid_patches):,} patches"
        )
    
    def _compute_split_code(self, patch_row: int, patch_col: int) -> int:
        """
        Compute spatial split code using checkerboard pattern.
        
        Uses global patch indices to ensure deterministic splits across
        different window configurations.
        
        Args:
            patch_row: Global patch index along y-axis
            patch_col: Global patch index along x-axis
        
        Returns:
            Split code: 1 (train), 2 (val), or 3 (test)
        """
        block_row = patch_row // self.config.block_height
        block_col = patch_col // self.config.block_width
        
        A = (block_row // 2 + block_col // 2) % 2
        B = (block_row + block_col) % 4
        
        if A == 0 and B == 0:
            return 3  # test
        elif A == 0 and B == 2:
            return 2  # val
        else:
            return 1  # train
    
    def _setup_anchor_sampling(self):
        """Setup weighted sampling for anchor years."""
        # Normalize weights
        weights_dict = self.config.anchor_weights
        total_weight = sum(weights_dict.values())
        
        self.anchor_years = []
        self.anchor_probabilities = []
        
        for year in self.config.endpoint_years:
            weight = weights_dict.get(year, 0.0)
            self.anchor_years.append(year)
            self.anchor_probabilities.append(weight / total_weight)
        
        logger.info(
            f"Anchor year sampling: {dict(zip(self.anchor_years, self.anchor_probabilities))}"
        )
    
    def _setup_epoch_sampling(self):
        """Setup epoch sampling strategy based on config."""
        mode = self.config.epoch_mode
        n_total = len(self.valid_patches)
        
        if mode == 'full':
            self.samples_per_epoch = n_total
            self.sampling_mode = 'full'
            logger.info(f"Epoch mode: full ({self.samples_per_epoch:,} samples)")
            
        elif mode == 'frac':
            frac = self.config.sample_frac
            if frac is None:
                raise ValueError("epoch_mode='frac' requires sample_frac in config")
            self.samples_per_epoch = max(1, int(n_total * frac))
            self.sampling_mode = 'sampled'
            logger.info(
                f"Epoch mode: frac ({frac:.2%} of {n_total:,} = "
                f"{self.samples_per_epoch:,} samples)"
            )
            
        elif mode == 'number':
            num = self.config.sample_number
            if num is None:
                raise ValueError("epoch_mode='number' requires sample_number in config")
            self.samples_per_epoch = min(num, n_total)
            self.sampling_mode = 'sampled'
            logger.info(f"Epoch mode: number ({self.samples_per_epoch:,} samples)")
            
        else:
            raise ValueError(f"Unknown epoch_mode: {mode}")
        
        # Initialize epoch state
        self.current_epoch_indices = None
        self._regenerate_epoch()
    
    def _regenerate_epoch(self):
        """Generate sample indices for this epoch."""
        n_patches = len(self.valid_patches)
        
        if self.sampling_mode == 'full':
            # Use all patches, shuffled
            self.current_epoch_indices = np.random.permutation(n_patches)
        else:
            # Sample with replacement
            self.current_epoch_indices = np.random.choice(
                n_patches,
                size=self.samples_per_epoch,
                replace=True
            )
    
    def new_epoch(self):
        """Call at the start of each epoch to regenerate sample indices."""
        self._regenerate_epoch()
        logger.debug(f"New epoch: {len(self.current_epoch_indices)} samples")
    
    def __len__(self) -> int:
        """Return number of samples for this epoch."""
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[SpatialWindow, int]:
        """
        Sample a (spatial_window, anchor_year) pair.
        
        Args:
            idx: Index into current epoch's sample list
        
        Returns:
            Tuple of (spatial_window, anchor_year)
        
        Example:
            >>> sampler = ForestPatchSampler(bindings_config, training_config, 'train')
            >>> spatial_window, anchor_year = sampler[0]
            >>> print(spatial_window)  # SpatialWindow(row_start=..., ...)
            >>> print(anchor_year)     # 2024
        """
        if idx >= len(self.current_epoch_indices):
            raise IndexError(f"Index {idx} out of range for epoch size {len(self)}")
        
        # Get patch index for this sample
        patch_idx = self.current_epoch_indices[idx]
        
        # Get patch origin (local coordinates within window)
        row0_local, col0_local = self.valid_patches[patch_idx]
        
        # Convert to global coordinates
        row0_global = self.window_origin[0] + row0_local
        col0_global = self.window_origin[1] + col0_local
        
        # Create spatial window
        spatial_window = SpatialWindow.from_upper_left_and_hw(
            upper_left=(row0_global, col0_global),
            hw=(self.config.patch_size, self.config.patch_size)
        )
        
        # Sample anchor year
        anchor_year = np.random.choice(
            self.anchor_years,
            p=self.anchor_probabilities
        )
        
        return spatial_window, anchor_year
    
    def get_all_patches(self) -> List[Tuple[SpatialWindow, int]]:
        """
        Get all valid patches with a random anchor year for each.
        
        Useful for validation/testing with full coverage.
        
        Returns:
            List of (spatial_window, anchor_year) tuples
        """
        samples = []
        
        for row0_local, col0_local in self.valid_patches:
            row0_global = self.window_origin[0] + row0_local
            col0_global = self.window_origin[1] + col0_local
            
            spatial_window = SpatialWindow.from_upper_left_and_hw(
                upper_left=(row0_global, col0_global),
                hw=(self.config.patch_size, self.config.patch_size)
            )
            
            anchor_year = np.random.choice(
                self.anchor_years,
                p=self.anchor_probabilities
            )
            
            samples.append((spatial_window, anchor_year))
        
        return samples
