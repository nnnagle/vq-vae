# utils/raster_ops.py
# ======================================================================
# Geospatial IO + alignment utilities
# - Mask metadata loading (template CRS/transform/bounds)
# - Spatial windowing (bounds intersection -> source/dest windows)
# - Reading rasters directly into the mask grid
# - NAIP (10 m) alignment to a 30 m mask via 3× scaling
#   (Dask-lazy, per-chunk reads; never materializes the full mosaic)
# ======================================================================

from typing import Tuple, List, Dict, Optional
import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
import rasterio
from rasterio.windows import from_bounds as win_from_bounds, Window
from rasterio.windows import bounds as win_bounds
from affine import Affine

from utils.log import log, warn, fail, ensure


# --------------------------- Mask meta --------------------------------
def load_mask_template(mask_path: str):
    """
    Load the 30 m mask as an xarray.DataArray and return template dict.
    Returns
    -------
    mask_da : xr.DataArray [y,x] uint8/bool
    template : dict(crs, transform, bounds, res)
    """
    with rasterio.open(mask_path) as ds:
        mask = ds.read(1).astype("uint8")
        H, W = ds.height, ds.width
        da_mask = xr.DataArray(mask, dims=("y", "x"), name="mask")
        template = {
            "crs": ds.crs,
            "transform": ds.transform,
            "bounds": ds.bounds,
            "res": ds.res,
        }
    return da_mask, template


# ---------------------- Bounds + window helpers -----------------------
def _round_window(win: Window) -> Window:
    return Window(
        col_off=int(round(win.col_off)),
        row_off=int(round(win.row_off)),
        width=int(round(win.width)),
        height=int(round(win.height)),
    )


def read_into_mask_grid(ds_path: str,
                        mask_shape: Tuple[int, int],
                        mask_bounds,
                        mask_transform: Affine,
                        dtype=np.float32) -> np.ndarray:
    """
    Read a single-band raster into the mask grid (y,x) using bounds/windowing.
    Returns a NumPy array (H,W). This is used behind Dask-delayed calls so
    it never constructs large arrays except per-chunk read in upper layers.
    """
    H, W = mask_shape
    with rasterio.open(ds_path) as src:
        # Intersect AOI bounds with source bounds
        L, B, R, T = mask_bounds.left, mask_bounds.bottom, mask_bounds.right, mask_bounds.top
        win = win_from_bounds(L, B, R, T, transform=src.transform)
        win = _round_window(win)

        # Read; crop/resample responsibility assumed handled upstream (same CRS/res)
        arr = src.read(1, window=win, out_dtype=dtype, boundless=True, fill_value=np.nan)

        # Ensure output shape matches mask grid (best-effort guard)
        if arr.shape != (H, W):
            h = min(H, arr.shape[0])
            w = min(W, arr.shape[1])
            canvas = np.full((H, W), np.nan, dtype=dtype)
            canvas[:h, :w] = arr[:h, :w]
            arr = canvas
        return arr


# ------------------------ NAIP alignment (Dask) ------------------------
def _mask_chunk_bounds(mask_transform: Affine,
                       y0: int, y1: int, x0: int, x1: int) -> Tuple[float, float, float, float]:
    """
    Compute geospatial bounds for a mask chunk window (y0:y1, x0:x1).
    """
    win = Window(x0, y0, (x1 - x0), (y1 - y0))
    return win_bounds(win, transform=mask_transform)  # (L,B,R,T)


def _read_naip_block(naip_path: str,
                     mask_transform: Affine,
                     y0: int, y1: int, x0: int, x1: int,
                     kshape: Tuple[int, int],
                     out_dtype: str) -> np.ndarray:
    """
    Read a NAIP block covering mask window [y0:y1, x0:x1] and reshape to patches:
      returns (yblk, xblk, krow, kcol, band)
    """
    krow, kcol = kshape
    yblk, xblk = (y1 - y0), (x1 - x0)
    expect_h = yblk * krow
    expect_w = xblk * kcol

    with rasterio.open(naip_path) as src:
        L, B, R, T = _mask_chunk_bounds(mask_transform, y0, y1, x0, x1)
        win = win_from_bounds(L, B, R, T, transform=src.transform)
        win = _round_window(win)

        arr = src.read(window=win, out_dtype=out_dtype, boundless=True, fill_value=np.nan)  # (B,H,W)
        BANDS, Hh, Ww = arr.shape

        # Best-effort enforce expected shape (assumes alignment/res match); crop/pad
        Hh2 = min(expect_h, Hh)
        Ww2 = min(expect_w, Ww)
        buf = np.full((BANDS, expect_h, expect_w), np.nan, dtype=arr.dtype)
        buf[:, :Hh2, :Ww2] = arr[:, :Hh2, :Ww2]
        arr = buf

        # Reshape to patches: (B, yblk, krow, xblk, kcol) -> (yblk,xblk,krow,kcol,B)
        arr = arr.reshape(BANDS, yblk, krow, xblk, kcol)
        arr = np.transpose(arr, (1, 3, 2, 4, 0))
        return arr  # (yblk, xblk, krow, kcol, band)


def align_naip_to_mask_as_patches(naip_path: str,
                                  mask_da: xr.DataArray,
                                  template: Dict,
                                  out_dtype: str = "float32",
                                  kshape: Tuple[int, int] = (3, 3),
                                  chunks: Dict[str, int] = None) -> xr.DataArray:
    """
    Build a Dask-backed NAIP array aligned to the mask grid, shaped:
      (y, x, krow, kcol, band)
    Each (y,x) cell is a krow×kcol patch at 3× resolution (assumes alignment).

    Implementation detail:
      - We probe band count once up front.
      - We create delayed blocks for each (y,x) chunk.
      - We CONCATENATE along X within each row, then CONCATENATE rows along Y.
        This handles ragged right/bottom edges (e.g., last X block 150 wide).
    """
    chunks = chunks or {}
    H = int(mask_da.sizes["y"])
    W = int(mask_da.sizes["x"])
    cy = max(1, int(chunks.get("y", 512)))
    cx = max(1, int(chunks.get("x", 512)))
    krow, kcol = kshape

    # Probe band count once
    with rasterio.open(naip_path) as src_probe:
        bands = int(src_probe.count)

    row_arrays: List[da.Array] = []
    for y0 in range(0, H, cy):
        y1 = min(H, y0 + cy)
        # Build blocks across X for this row
        x_blocks: List[da.Array] = []
        for x0 in range(0, W, cx):
            x1 = min(W, x0 + cx)
            d = delayed(_read_naip_block)(
                naip_path, template["transform"], y0, y1, x0, x1, kshape, out_dtype
            )
            arr = da.from_delayed(
                d,
                shape=(y1 - y0, x1 - x0, krow, kcol, bands),
                dtype=np.float32 if out_dtype == "float32" else np.uint16
            )
            x_blocks.append(arr)  # shapes may differ in axis=1 (x) at the rightmost edge
        # Concatenate this row along X (axis=1)
        row_arr = da.concatenate(x_blocks, axis=1)
        row_arrays.append(row_arr)  # shapes may differ in axis=0 (y) for bottom edge

    # Concatenate rows along Y (axis=0)
    data = da.concatenate(row_arrays, axis=0)  # (H, W, krow, kcol, bands)

    # Rechunk to requested sizes
    data = data.rechunk({
        0: cy,
        1: cx,
        2: chunks.get("krow", krow),
        3: chunks.get("kcol", kcol),
        4: chunks.get("band", bands),
    })

    da_xr = xr.DataArray(
        data,
        dims=("y", "x", "krow", "kcol", "band"),
        name="naip_patch",
    )
    return da_xr
