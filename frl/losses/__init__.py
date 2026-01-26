"""
Loss functions for representation learning.

This module provides various loss functions for training neural networks,
including contrastive, reconstruction, and quantization losses.

Available losses:
    - contrastive_loss: InfoNCE-style contrastive loss with weighted pairs
    - reconstruction_loss: Masked reconstruction loss (L1, L2, Huber, etc.)
"""

from __future__ import annotations

from losses.contrastive import contrastive_loss
from losses.reconstruction import reconstruction_loss

__all__ = [
    "contrastive_loss",
    "reconstruction_loss",
]
