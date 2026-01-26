"""
Loss functions for representation learning.

This module provides various loss functions for training neural networks,
including contrastive, reconstruction, and quantization losses.

Available losses:
    - contrastive_loss: InfoNCE-style contrastive loss with weighted pairs
"""

from __future__ import annotations

from frl.losses.contrastive import contrastive_loss

__all__ = [
    "contrastive_loss",
]
