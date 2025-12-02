# src/data/patch_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Literal, Optional

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


class PatchDataset(Dataset):
    """
    Patch-based dataset over a Zarr cube.

    Patches are aligned on a regular grid of size (patch_size × patch_size)
    in (y, x) coordinates. The AOI is coarsened to this grid, and only patches
    with sufficient AOI coverage are kept.

    A spatial train/val/test split can be applied directly in patch space using
    a deterministic checkerboard + diagonal tiling pattern:

        Let (py, px) be the GLOBAL patch indices (units of patch_size pixels).
        Let block_height and block_width be block dimensions in patch units.

            block_row = py // block_height
            block_col = px // block_width

        A = (block_row // 2 + block_col // 2) % 2
        B = (block_row + block_col) % 4

        split = 3  (test) if A == 0 and B == 0
        split = 2  (val)  if A == 0 and B == 2
        split = 1  (train) otherwise

    This reproduces a spatially stratified “checkerboard” split, ensuring that
    train/val/test patches are separated beyond typical spatial autocorrelation
    distances while remaining broadly distributed across the domain.

    Optional debug window:
        A small rectangular region can be cropped from the full Zarr cube by
        specifying `window_origin=(y0, x0)` and `window_size=(h, w)`, both in
        pixel coordinates. These values must align to multiples of patch_size.
        Only patches inside this window are sampled. Split logic continues to
        use GLOBAL patch indices so that debug windows behave like cutouts of
        the full domain.

    Returned dictionary for each patch:
        {
            "x_cont": Tensor[T, C_cont, H, W] or None,   # continuous features
            "x_cat":  Tensor[T, C_cat,  H, W] or None,   # categorical features
            "x_mask": Tensor[T, C_mask, H, W] or None,   # optional mask layers
            "aoi":    Tensor[H, W] bool,                 # AOI mask per patch
        }

    Ideas for future:
        - Export the split-grid as a raster for inspection (e.g., GeoTIFF).
        - Add a built-in Matplotlib visualizer for A/B patterns or splits.
        - Enable alternative spatial CV strategies (e.g., radial, clustered,
          k-fold block CV, or inverted A/B rules).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        patch_size: int = 256,
        split: SplitName | None = None,
        block_width: int = 7,
        block_height: int = 7,
        min_aoi_fraction: float = 0.3,
        window_origin: Optional[Tuple[int, int]] = None,  # (y0, x0) in pixels
        window_size: Optional[Tuple[int, int]] = None,    # (height, width) in pixels
    ):
        """
        Initialize a PatchDataset over a Zarr cube.

        Args:
            zarr_path:
                Path to the Zarr store. Must be a directory containing .zgroup
                or .zarray metadata.
            patch_size:
                Spatial patch size in pixels (both y and x). Each patch is
                patch_size × patch_size.
            split:
                Optional spatial split to apply:
                  - None: keep all valid patches (no train/val/test filtering).
                  - "train": keep only patches assigned to the train split.
                  - "val":   keep only patches assigned to the validation split.
                  - "test":  keep only patches assigned to the test split.
                Split assignment is deterministic in patch space using the A/B
                pattern described in the class docstring.
            block_width:
                Block width in patch units for the spatial split. Controls how
                many patches along x form one "block" for A/B tiling.
            block_height:
                Block height in patch units for the spatial split. Controls how
                many patches along y form one "block" for A/B tiling.
            min_aoi_fraction:
                Minimum fraction of AOI coverage required for a patch to be
                kept. Computed on the coarsened AOI grid at patch resolution.
                Patches below this threshold are discarded before any split
                logic is applied.
            window_origin:
                Optional (y0, x0) in pixel coordinates specifying the upper-left
                corner of a debug window. Must be multiples of patch_size if
                provided. If None, the full spatial extent is used (snapped to
                full patches).
            window_size:
                Optional (height, width) in pixel coordinates specifying the
                size of the debug window. Must be multiples of patch_size if
                provided. Must be given together with window_origin. If None and
                window_origin is None, the full domain is used.
        """
        self.zarr_path = Path(zarr_path)
        self.patch_size = int(patch_size)
        self.split = split
        self.block_width = int(block_width)
        self.block_height = int(block_height)
        self.min_aoi_fraction = float(min_aoi_fraction)

        self.window_origin = window_origin
        self.window_size = window_size

        # ------------------------------------------------------------
        # Zarr sanity checks
        # ------------------------------------------------------------
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr path does not exist:\n  {self.zarr_path}")

        if not self.zarr_path.is_dir():
            raise ValueError(f"Expected directory for Zarr store:\n  {self.zarr_path}")

        has_zgroup = (self.zarr_path / ".zgroup").exists()
        has_zarray = (self.zarr_path / ".zarray").exists()
        if not (has_zgroup or has_zarray):
            raise ValueError(f"Directory does not look like a Zarr store:\n  {self.zarr_path}")

        # Open dataset
        self.ds = xr.open_zarr(str(self.zarr_path), consolidated=False)

        self.has_continuous = "continuous" in self.ds
        self.has_categorical = "categorical" in self.ds
        self.has_mask = "mask" in self.ds

        self.da_cont = self.ds["continuous"] if self.has_continuous else None
        self.da_cat = self.ds["categorical"] if self.has_categorical else None
        self.da_mask = self.ds["mask"] if self.has_mask else None

        if "aoi" not in self.ds:
            raise ValueError("Zarr store has no 'aoi' variable.")
        aoi_da = self.ds["aoi"]

        # Boolean AOI
        aoi_bool = (aoi_da != 0)

        # Spatial sizes
        if self.has_continuous:
            ny = self.da_cont.sizes["y"]
            nx = self.da_cont.sizes["x"]
        else:
            ny = aoi_bool.sizes["y"]
            nx = aoi_bool.sizes["x"]

        ps = self.patch_size

        # ------------------------------------------------------------
        # Determine window (global coords), then crop AOI to window
        # ------------------------------------------------------------
        if self.window_origin is not None or self.window_size is not None:
            if self.window_origin is None or self.window_size is None:
                raise ValueError("Both window_origin and window_size must be provided together.")
            y0_win, x0_win = self.window_origin
            h_win, w_win = self.window_size

            # Bounds checks
            if y0_win < 0 or x0_win < 0 or y0_win >= ny or x0_win >= nx:
                raise ValueError(
                    f"window_origin {self.window_origin} is out of bounds "
                    f"for (ny={ny}, nx={nx})"
                )

            # Require alignment to patch grid
            if (y0_win % ps != 0) or (x0_win % ps != 0):
                raise ValueError(
                    f"window_origin {self.window_origin} must be multiples of patch_size={ps}"
                )
            if (h_win % ps != 0) or (w_win % ps != 0):
                raise ValueError(
                    f"window_size {self.window_size} must be multiples of patch_size={ps}"
                )

            y1_win = min(y0_win + h_win, ny)
            x1_win = min(x0_win + w_win, nx)

            # Sanity: ensure window size is still patch aligned
            if ((y1_win - y0_win) % ps) != 0 or ((x1_win - x0_win) % ps) != 0:
                raise RuntimeError(
                    "Internal window alignment error; check window_size vs patch_size."
                )

            # Crop AOI to this window
            aoi_bool = aoi_bool.isel(
                y=slice(y0_win, y1_win),
                x=slice(x0_win, x1_win),
            )

            # Patch-grid offsets in GLOBAL patch indices
            patch_row_offset = y0_win // ps
            patch_col_offset = x0_win // ps

        else:
            # Full domain, snapped to full patches starting at (0,0)
            max_y = (ny // ps) * ps
            max_x = (nx // ps) * ps

            aoi_bool = aoi_bool.isel(
                y=slice(0, max_y),
                x=slice(0, max_x),
            )
            y0_win = 0
            x0_win = 0
            patch_row_offset = 0
            patch_col_offset = 0

        self.patch_row_offset = patch_row_offset
        self.patch_col_offset = patch_col_offset
        self.y0_win = y0_win
        self.x0_win = x0_win

        # ------------------------------------------------------------
        # Build coarse AOI grid at patch resolution (within window)
        # ------------------------------------------------------------
        aoi_coarse = (
            aoi_bool
            .coarsen(y=ps, x=ps, boundary="trim")
            .mean()
            .compute()
        )
        aoi_coarse_np = aoi_coarse.values

        # ------------------------------------------------------------
        # Build full list of patches BEFORE filtering by split
        # ------------------------------------------------------------
        self.patch_origins_raw: List[Tuple[int, int]] = []
        self.patch_split_codes_raw: List[int] = []

        n_blocks_y, n_blocks_x = aoi_coarse_np.shape

        for py_local in range(n_blocks_y):
            for px_local in range(n_blocks_x):
                frac_valid = float(aoi_coarse_np[py_local, px_local])
                if frac_valid <= self.min_aoi_fraction:
                    continue

                # Global patch indices in the full patch grid
                py_global = patch_row_offset + py_local
                px_global = patch_col_offset + px_local

                code = self._compute_split_code_for_patch(py_global, px_global)

                # Local (window-relative) origin in pixels
                y0_local = py_local * ps
                x0_local = px_local * ps

                self.patch_origins_raw.append((y0_local, x0_local))
                self.patch_split_codes_raw.append(code)

        # ------------------------------------------------------------
        # Print pre-split counts
        # ------------------------------------------------------------
        if self.patch_split_codes_raw:
            split_arr = np.array(self.patch_split_codes_raw)
            num_train = int((split_arr == 1).sum())
            num_val   = int((split_arr == 2).sum())
            num_test  = int((split_arr == 3).sum())
        else:
            num_train = num_val = num_test = 0

        print(
            f"[PatchDataset] Patches after AOI mask (within window): "
            f"train={num_train:,}  val={num_val:,}  test={num_test:,}"
        )

        # ------------------------------------------------------------
        # Apply filtering by desired split
        # ------------------------------------------------------------
        self.patch_origins: List[Tuple[int, int]] = []
        self.patch_split_codes: List[int] = []

        for (y0_local, x0_local), code in zip(
            self.patch_origins_raw, self.patch_split_codes_raw
        ):
            if self.split == "train" and code != 1:
                continue
            if self.split == "val" and code != 2:
                continue
            if self.split == "test" and code != 3:
                continue

            self.patch_origins.append((y0_local, x0_local))
            self.patch_split_codes.append(code)

        print(
            f"[PatchDataset] Loaded split='{self.split}' "
            f"-> {len(self.patch_origins):,} patches"
        )

        # Save AOI (already cropped to window) for masking
        self.aoi = aoi_bool

    # ----------------------------------------------------------------------
    # Spatial split logic (GLOBAL patch indices)
    # ----------------------------------------------------------------------
    def _compute_split_code_for_patch(self, patch_row: int, patch_col: int) -> int:
        """
        Compute spatial split code for a patch at GLOBAL patch index (patch_row, patch_col).

        Args:
            patch_row:
                Global patch index along the y-axis (0-based), measured in
                units of patch_size pixels.
            patch_col:
                Global patch index along the x-axis (0-based), measured in
                units of patch_size pixels.

        Returns:
            int:
                Split code for this patch:
                  - 1 = train
                  - 2 = validation
                  - 3 = test

        Notes:
            This uses the A/B pattern:

                block_row = patch_row // block_height
                block_col = patch_col // block_width

                A = (block_row // 2 + block_col // 2) % 2
                B = (block_row + block_col) % 4

            and applies:

                if A == 0 and B == 0: test (3)
                elif A == 0 and B == 2: val (2)
                else: train (1)
        """
        block_row = patch_row // self.block_height
        block_col = patch_col // self.block_width

        A = (block_row // 2 + block_col // 2) % 2
        B = (block_row + block_col) % 4

        if A == 0 and B == 0:
            return 3  # test
        elif A == 0 and B == 2:
            return 2  # val
        else:
            return 1  # train

    # ----------------------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        """
        Return the number of patches available in this dataset.

        Returns:
            int:
                Number of patch origins remaining after AOI filtering and
                optional split selection (train/val/test) within the chosen
                window (if any).
        """
        return len(self.patch_origins)

    def __getitem__(self, idx: int):
        """
        Load a single patch as a dictionary of tensors.

        Args:
            idx:
                Integer index into the list of available patch origins. Must be
                in the range [0, len(self) - 1].

        Returns:
            dict:
                A dictionary with the following keys:
                  - "x_cont": Tensor[T, C_cont, H, W] or None
                  - "x_cat": Tensor[T, C_cat, H, W] or None
                  - "x_mask": Tensor[T, C_mask, H, W] or None
                  - "aoi": Tensor[H, W] bool

                All patches are of spatial size H = W = patch_size.
        """
        y0_local, x0_local = self.patch_origins[idx]
        ps = self.patch_size

        # Convert local window coords back to global coords for slicing
        y0 = self.y0_win + y0_local
        x0 = self.x0_win + x0_local

        sl = dict(
            y=slice(y0, y0 + ps),
            x=slice(x0, x0 + ps),
        )

        # Continuous
        if self.has_continuous:
            da_cont_patch = self.da_cont.isel(**sl).transpose(
                "time", "feature_continuous", "y", "x"
            )
            x_cont = torch.from_numpy(da_cont_patch.values).float()
        else:
            x_cont = None

        # Categorical
        if self.has_categorical:
            da_cat_patch = self.da_cat.isel(**sl).transpose(
                "time", "feature_categorical", "y", "x"
            )
            x_cat = torch.from_numpy(da_cat_patch.values).long()
        else:
            x_cat = None

        # Mask
        if self.has_mask:
            da_mask_patch = self.da_mask.isel(**sl).transpose(
                "time", "feature_mask", "y", "x"
            )
            x_mask = torch.from_numpy(da_mask_patch.values.astype(bool))
        else:
            x_mask = None

        # AOI patch (AOI is already cropped to the window)
        aoi_patch = self.aoi.isel(
            y=slice(y0 - self.y0_win, y0 - self.y0_win + ps),
            x=slice(x0 - self.x0_win, x0 - self.x0_win + ps),
        )
        aoi = torch.from_numpy((aoi_patch.values != 0))

        return {
            "x_cont": x_cont,
            "x_cat": x_cat,
            "x_mask": x_mask,
            "aoi": aoi,
        }
