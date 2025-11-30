# data/patch_dataset.py
#
# Minimal dataset for VAE prototyping.
#
# Loads the "continuous" group from a spatiotemporal Zarr cube and
# slices it into non-overlapping 256×256 spatial patches. Each patch
# includes *all* time steps and *all* continuous features.
#
# Returned tensor shape: [T, C_cont, 256, 256]
#
# This v0 dataset intentionally ignores:
#   - categorical features
#   - masks
#   - normalization
#   - train/val/test partitioning
#   - random cropping
#
# Its sole job is to deliver real patches to the model so you can
# develop the earliest VAE without battling the full cube complexity.


import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PatchDataset(Dataset):
    def __init__(self, zarr_path: str | Path, patch_size: int = 256):
        # Normalize the path
        self.zarr_path = Path(zarr_path)

        # (1) Path exists?
        if not self.zarr_path.exists():
            raise FileNotFoundError(
                f"Zarr path does not exist:\n  {self.zarr_path}"
            )

        # (2) A Zarr root *must* be a directory
        if not self.zarr_path.is_dir():
            raise ValueError(
                f"Expected a directory for Zarr store, got a file:\n  {self.zarr_path}"
            )

        # (3) Check for Zarr structure markers
        has_zarray = any((self.zarr_path / ".zarray").exists()
                         for _ in [0])  # array-style root
        has_zgroup = (self.zarr_path / ".zgroup").exists()

        if not (has_zgroup or has_zarray):
            raise ValueError(
                f"The directory does not look like a Zarr store "
                f"(no .zgroup or .zarray):\n  {self.zarr_path}"
            )

        # -------------------------
        # Now open the dataset
        # -------------------------

        try:
            self.ds = xr.open_zarr(str(self.zarr_path), consolidated=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open Zarr store with xarray:\n  {self.zarr_path}\n"
                f"Original error:\n  {e}"
            )

        # (4) Must have 'continuous' group for this v0 dataset
        if "continuous" not in self.ds:
            raise ValueError(
                f"Zarr store does not contain a 'continuous' variable.\n"
                f"Variables found: {list(self.ds.data_vars.keys())}"
            )


        # The continuous DataArray: (time, feature_continuous, y, x)
        self.da = self.ds["continuous"]   # (time, feature_continuous, y, x)
        self.patch_size = patch_size

        ny = self.da.sizes["y"]
        nx = self.da.sizes["x"]
        
        # Restrict to full patches only (no ragged edges in v0)
        max_y = (ny // patch_size) * patch_size
        max_x = (nx // patch_size) * patch_size

        # Compute grid-aligned patch origins
        ys = range(0, max_y, patch_size)
        xs = range(0, max_x, patch_size)

        self.patch_origins = [(y0, x0) for y0 in ys for x0 in xs]

    def __len__(self):
        # Number of spatial patches available
        return len(self.patch_origins)

    def __getitem__(self, idx):
        y0, x0 = self.patch_origins[idx]
        ps = self.patch_size

        # Slice the cube: full time × full features × spatial window
        da_patch = self.da.isel(
            y=slice(y0, y0 + ps),
            x=slice(x0, x0 + ps),
        )
        
        da_patch = da_patch.transpose("time", "feature_continuous", "y", "x")
        # Convert xarray → NumPy → torch.FloatTensor
        arr = da_patch.values  # NumPy: [T, C, ps, ps]

        return torch.from_numpy(arr).float()
