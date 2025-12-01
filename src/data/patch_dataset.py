# src/data/patch_dataset.py

import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PatchDataset(Dataset):
    def __init__(
        self,
        zarr_path: str | Path,
        patch_size: int = 256,
    ):
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
        self.patch_size = patch_size

        self.has_continuous = "continuous" in self.ds
        self.has_categorical = "categorical" in self.ds
        self.has_mask = "mask" in self.ds

        # Keep direct handles for clarity
        self.da_cont = self.ds["continuous"] if self.has_continuous else None
        self.da_cat = self.ds["categorical"] if self.has_categorical else None
        self.da_mask = self.ds["mask"] if self.has_mask else None

        if "aoi" not in self.ds:
            raise ValueError("Zarr store has no 'aoi' variable.")
        aoi_da = self.ds["aoi"]  # (y, x) or (time,y,x) depending on how you stored it

        # Ensure boolean-ish
        aoi_bool = (aoi_da != 0)

        # Use spatial sizes from continuous group if present, otherwise AOI
        if self.has_continuous:
            ny = self.da_cont.sizes["y"]
            nx = self.da_cont.sizes["x"]
        else:
            ny = aoi_bool.sizes["y"]
            nx = aoi_bool.sizes["x"]

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
                if frac_valid > 0.3:  # keep if >30% AOI coverage
                    y0 = by * patch_size
                    x0 = bx * patch_size
                    self.patch_origins.append((y0, x0))

        # Keep the full-res AOI around for masking in __getitem__
        self.aoi = aoi_bool  # xarray/dask, not NumPy

    def __len__(self):
        # Number of spatial patches available
        return len(self.patch_origins)

    def __getitem__(self, idx):
        y0, x0 = self.patch_origins[idx]
        ps = self.patch_size

        # Slice dictionary for spatial window
        sl = dict(
            y=slice(y0, y0 + ps),
            x=slice(x0, x0 + ps),
        )

        # ---------------------------
        # Continuous: [T, C_cont, H, W]
        # ---------------------------
        if self.has_continuous:
            da_cont_patch = self.da_cont.isel(**sl).transpose(
                "time", "feature_continuous", "y", "x"
            )
            cont_arr = da_cont_patch.values  # [T, C_cont, ps, ps]
            x_cont = torch.from_numpy(cont_arr).float()
        else:
            x_cont = None

        # ---------------------------
        # Categorical: [T, C_cat, H, W]
        # ---------------------------
        if self.has_categorical:
            da_cat_patch = self.da_cat.isel(**sl).transpose(
                "time", "feature_categorical", "y", "x"
            )
            cat_arr = da_cat_patch.values  # [T, C_cat, ps, ps]
            x_cat = torch.from_numpy(cat_arr).long()
        else:
            x_cat = None

        # ---------------------------
        # Mask: [T, C_mask, H, W]
        # ---------------------------
        if self.has_mask:
            da_mask_patch = self.da_mask.isel(**sl).transpose(
                "time", "feature_mask", "y", "x"
            )
            mask_arr = da_mask_patch.values  # [T, C_mask, ps, ps]
            x_mask = torch.from_numpy(mask_arr.astype(bool))
        else:
            x_mask = None

        # ---------------------------
        # AOI patch: [H, W] bool
        # ---------------------------
        aoi_patch = self.aoi.isel(
            y=slice(y0, y0 + ps),
            x=slice(x0, x0 + ps),
        )
        aoi_arr = (aoi_patch.values != 0)  # [ps, ps]
        aoi = torch.from_numpy(aoi_arr)

        return {
            "x_cont": x_cont,   # Tensor[T, C_cont, H, W] or None
            "x_cat":  x_cat,    # Tensor[T, C_cat, H, W]  or None
            "x_mask": x_mask,   # Tensor[T, C_mask, H, W] or None
            "aoi":    aoi,      # Tensor[H, W] bool
        }
