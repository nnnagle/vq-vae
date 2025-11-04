"""
utils/samplers.py
------------
Custom PyTorch Samplers for chunk-aware batch generation.

This module provides `ChunkBatchSampler`, which ensures that each batch of samples
comes from a single Zarr chunk of the raster dataset.  The sampler is designed to
minimize random I/O by preserving spatial locality during training.

Typical usage:
    from utils.samplers import ChunkBatchSampler

    batch_sampler = ChunkBatchSampler(
        dataset.xy_by_chunk,
        batch_size=64,
        drop_last=False,
        replacement_within_chunk=False,
        seed=42
    )

    loader = DataLoader(dataset, batch_sampler=batch_sampler, ...)

Used by:
    - train_vqvae.py to build DataLoaders with chunk-locked batching.
    - Datasets (e.g., VQVAEDataset) that expose `xy_by_chunk`.

Design notes:
    * Each batch is restricted to one (y,x) chunk → reduces Zarr read overhead.
    * Handles ragged chunks and small tail chunks via optional replacement.
    * Shuffles both chunk order and indices within each chunk every epoch.
    * Compatible with multi-worker DataLoader setups (persistent_workers=True recommended).

"""
from __future__ import annotations
import math
import random
from typing import Iterator, List, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler

class ChunkBatchSampler(Sampler[List[int]]):
    """
    Yields lists of dataset indices. Each batch comes from a single (y,x) chunk.

    Assumptions:
      - dataset.xy_by_chunk: List[np.ndarray] of integer indices into dataset.__getitem__ space
      - All arrays are on CPU; we only return python ints to DataLoader workers.
    """
    def __init__(self,
                 xy_by_chunk: Sequence[np.ndarray],
                 batch_size: int,
                 drop_last: bool = False,
                 replacement_within_chunk: bool = False,
                 seed: int | None = None) -> None:
        self.xy_by_chunk = [np.asarray(a, dtype=np.int64) for a in xy_by_chunk]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.replacement = bool(replacement_within_chunk)
        self.rng = random.Random(seed)

        # Precompute per-chunk sizes and non-empty chunk list
        self.chunk_sizes = [int(a.size) for a in self.xy_by_chunk]
        self.non_empty = [i for i, n in enumerate(self.chunk_sizes) if n > 0]
        self.total = sum(self.chunk_sizes)

    def __iter__(self) -> Iterator[List[int]]:
        # Fresh epoch shuffle
        chunk_order = self.non_empty[:]
        self.rng.shuffle(chunk_order)

        # Within-chunk permutations
        perms = {}
        for cid in chunk_order:
            idxs = self.xy_by_chunk[cid]
            if not self.replacement:
                perm = idxs.copy()
                np.random.shuffle(perm)
                perms[cid] = perm
            else:
                perms[cid] = idxs  # sampling with replacement

        # Emit batches
        for cid in chunk_order:
            idxs = perms[cid]
            if self.replacement:
                # number of batches ~ ceil(size/batch)
                n_batches = math.ceil(max(1, idxs.size) / self.batch_size)
                for _ in range(n_batches):
                    batch = self.rng.choices(idxs.tolist(), k=self.batch_size)
                    yield batch
            else:
                # chunk-local contiguous batches
                n_full = idxs.size // self.batch_size
                for b in range(n_full):
                    sl = idxs[b*self.batch_size:(b+1)*self.batch_size]
                    yield sl.tolist()
                # tail
                rem = idxs.size % self.batch_size
                if rem and not self.drop_last:
                    yield idxs[-rem:].tolist()

    def __len__(self) -> int:
        # Loose upper bound assuming no replacement
        if self.drop_last:
            return sum(n // self.batch_size for n in self.chunk_sizes)
        else:
            return sum((n + self.batch_size - 1) // self.batch_size for n in self.chunk_sizes)
