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
        self.zarr_path = Path(zarr_path)

        # ---- open Zarr ----
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr path does not exist:\n  {self.zarr_path}")

        if not self.zarr_path.is_dir():
            raise ValueError(f"Expected directory for Zarr store:\n  {self.zarr_path}")

        has_zgroup = (self.zarr_path / ".zgroup").exists()
        has_zarray = (self.zarr_path / ".zarray").exists()
        if not (has_zgroup or has_zarray):
            raise ValueError(
                f"Directory does not look like a Zarr store:\n  {self.zarr_path}"
            )

        self.ds = xr.open_zarr(str(self.zarr_path), consolidated=False)
        self.da = self.ds["continuous"]
        self.patch_size = patch_size

        if "aoi" not in self.ds:
            raise ValueError("Zarr store has no 'aoi' variable.")
        aoi_da = self.ds["aoi"]  # (y, x)

        # Ensure boolean-ish
        aoi_bool = (aoi_da != 0)

        ny = self.da.sizes["y"]
        nx = self.da.sizes["x"]

        max_y = (ny // patch_size) * patch_size
        max_x = (nx // patch_size) * patch_size

        # Trim AOI to full tiles
        aoi_bool = aoi_bool.isel(
            y=slice(0, max_y),
            x=slice(0, max_x),
        )

        # Coarsen AOI to patch grid and compute fraction valid per patch
        # Result shape: (ny // patch_size, nx // patch_size)
        aoi_coarse = (
            aoi_bool
            .coarsen(y=patch_size, x=patch_size, boundary="trim")
            .mean()
            .compute()  # this touches AOI once, but returns small array
        )

        # Bring small coarse mask into NumPy
        aoi_coarse_np = aoi_coarse.values  # maybe 78×78, trivial

        # Build patch_origins using this coarse grid
        self.patch_origins = []
        n_blocks_y, n_blocks_x = aoi_coarse_np.shape
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                frac_valid = aoi_coarse_np[by, bx]
                if frac_valid > 0.1:  # keep if >10% AOI coverage
                    y0 = by * patch_size
                    x0 = bx * patch_size
                    self.patch_origins.append((y0, x0))

        # Optionally, keep the *full-res* AOI around for masking in __getitem__
        # but DO NOT call .values here.
        self.aoi = aoi_bool  # xarray/dask, not NumPy

    def __len__(self):
        # Number of spatial patches available
        return len(self.patch_origins)

    def __getitem__(self, idx):
        y0, x0 = self.patch_origins[idx]
        ps = self.patch_size

        da_patch = self.da.isel(
            y=slice(y0, y0 + ps),
            x=slice(x0, x0 + ps),
        ).transpose("time", "feature_continuous", "y", "x")
        x_arr = da_patch.values  # [T, C, ps, ps]

        aoi_patch = self.aoi.isel(
            y=slice(y0, y0 + ps),
            x=slice(x0, x0 + ps),
        )
        aoi_arr = (aoi_patch.values != 0)  # [ps, ps]

        x = torch.from_numpy(x_arr).float()
        aoi = torch.from_numpy(aoi_arr)

        return x, aoi
