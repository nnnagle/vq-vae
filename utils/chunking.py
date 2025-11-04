"""
utils/chunking.py
------------
Helper functions for mapping raster pixel coordinates to Zarr chunk bins.

This module supports ragged chunk layouts, where the final chunks along each
axis may be smaller than the nominal chunk size.  The goal is to make it
possible to group random samples by their underlying Zarr chunk for efficient
I/O during model training.

Typical usage:
    from utils.chunking import compute_xy_chunks, pack_xy_by_chunk

    y_bins, x_bins = compute_xy_chunks(xy, y_chunk_sizes, x_chunk_sizes)
    xy_by_chunk = pack_xy_by_chunk(xy, y_bins, x_bins, len(y_chunk_sizes), len(x_chunk_sizes))

Used by:
    - VQVAEDataset (in loader.py) to pre-bucket valid sample coordinates.
    - ChunkBatchSampler (in samplers.py) to draw batches confined to a single chunk.

Design notes:
    * Works for any ragged chunk geometry; does not assume uniform chunk sizes.
    * Avoids loading the full raster—only uses precomputed coordinate arrays.
    * Keeps all operations NumPy-side for low overhead.

Author: (your name or lab)
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

def _bin_index(pos: int, chunk_sizes: Tuple[int, ...]) -> int:
    # Map absolute index -> which ragged chunk bin it falls into
    # e.g., chunk_sizes=(256,256,256,128) for 896-length axis
    cum = 0
    for i, sz in enumerate(chunk_sizes):
        nxt = cum + sz
        if pos < nxt:
            return i
        cum = nxt
    return len(chunk_sizes) - 1  # last bin as fallback

def compute_xy_chunks(xy: np.ndarray,
                      y_chunk_sizes: Tuple[int, ...],
                      x_chunk_sizes: Tuple[int, ...]) -> np.ndarray:
    # xy: shape [N,2] of (y, x)
    y_bins = np.fromiter((_bin_index(int(y), y_chunk_sizes) for y, _ in xy), dtype=np.int64, count=xy.shape[0])
    x_bins = np.fromiter((_bin_index(int(x), x_chunk_sizes) for _, x in xy), dtype=np.int64, count=xy.shape[0])
    return y_bins, x_bins

def pack_xy_by_chunk(xy: np.ndarray,
                     y_bins: np.ndarray,
                     x_bins: np.ndarray,
                     n_y: int, n_x: int) -> List[np.ndarray]:
    # returns list length n_y*n_x, each entry is an array of integer row indices into xy
    chunk_ids = y_bins * n_x + x_bins
    buckets = [list() for _ in range(n_y * n_x)]
    for i, cid in enumerate(chunk_ids.tolist()):
        buckets[cid].append(i)
    return [np.array(b, dtype=np.int64) if b else np.empty((0,), dtype=np.int64) for b in buckets]
