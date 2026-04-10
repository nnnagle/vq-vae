"""
Streaming sampling utilities.
"""

from __future__ import annotations

import numpy as np


class ReservoirSampler:
    """Fixed-capacity uniform random sample from a stream (Algorithm R, Vitter 1985).

    Maintains a uniform random sample of exactly ``capacity`` items from a
    stream of unknown length without storing the full stream.

    Args:
        capacity: Maximum number of items to keep.
        dim: Feature dimension of each item.
        seed: Random seed for reproducibility.
    """

    def __init__(self, capacity: int, dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty((capacity, dim), dtype=np.float32)
        self.n_seen = 0  # total items offered so far

    def add(self, vectors: np.ndarray) -> None:
        """Add a batch of vectors to the reservoir.

        Args:
            vectors: [N, dim] float32 array.
        """
        n = vectors.shape[0]
        for i in range(n):
            self.n_seen += 1
            if self.n_seen <= self.capacity:
                self.buffer[self.n_seen - 1] = vectors[i]
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.capacity:
                    self.buffer[j] = vectors[i]

    @property
    def filled(self) -> int:
        """Number of valid rows currently in the buffer."""
        return min(self.n_seen, self.capacity)

    def get(self) -> np.ndarray:
        """Return the sampled buffer as [filled, dim]."""
        return self.buffer[: self.filled]
