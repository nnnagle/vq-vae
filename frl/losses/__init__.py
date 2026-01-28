"""
Loss functions for representation learning.

This module provides various loss functions for training neural networks,
including contrastive, reconstruction, and quantization losses.

Available losses:
    - contrastive_loss: InfoNCE-style contrastive loss with weighted pairs
    - reconstruction_loss: Masked reconstruction loss for continuous targets (L1, L2, Huber)
    - categorical_loss: Cross-entropy loss for discrete/categorical targets
    - count_loss: Poisson/Negative Binomial loss for count data
    - variance_covariance_loss: VICReg-style loss to prevent embedding collapse
    - variance_loss: Variance component only
    - covariance_loss: Covariance component only

Pair generation for contrastive learning:
    - pairs_knn: k-nearest neighbor pairs
    - pairs_mutual_knn: mutual k-nearest neighbor pairs
    - pairs_quantile: pairs from quantile range of distances
    - pairs_radius: pairs from absolute distance range
"""

from __future__ import annotations

from losses.categorical import categorical_loss
from losses.contrastive import contrastive_loss
from losses.count import count_loss
from losses.pairs import (
    pairs_knn,
    pairs_mutual_knn,
    pairs_quantile,
    pairs_radius,
    pairs_with_spatial_constraint,
    apply_spatial_constraint,
)
from losses.reconstruction import reconstruction_loss
from losses.variance_covariance import (
    covariance_loss,
    variance_covariance_loss,
    variance_loss,
)

__all__ = [
    "apply_spatial_constraint",
    "categorical_loss",
    "contrastive_loss",
    "count_loss",
    "covariance_loss",
    "pairs_knn",
    "pairs_mutual_knn",
    "pairs_quantile",
    "pairs_radius",
    "pairs_with_spatial_constraint",
    "reconstruction_loss",
    "variance_covariance_loss",
    "variance_loss",
]
