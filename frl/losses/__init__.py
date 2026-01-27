"""
Loss functions for representation learning.

This module provides various loss functions for training neural networks,
including contrastive, reconstruction, and quantization losses.

Available losses:
    - contrastive_loss: InfoNCE-style contrastive loss with weighted pairs
    - reconstruction_loss: Masked reconstruction loss for continuous targets (L1, L2, Huber)
    - categorical_loss: Cross-entropy loss for discrete/categorical targets
    - count_loss: Poisson/Negative Binomial loss for count data
"""

from __future__ import annotations

from losses.categorical import categorical_loss
from losses.contrastive import contrastive_loss
from losses.count import count_loss
from losses.reconstruction import reconstruction_loss

__all__ = [
    "categorical_loss",
    "contrastive_loss",
    "count_loss",
    "reconstruction_loss",
]
