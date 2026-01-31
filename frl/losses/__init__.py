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
from losses.phase_neighborhood import (
    average_features_by_ysfc,
    build_phase_neighborhood_batch,
    build_ysfc_overlap,
    phase_neighborhood_loss,
)
from losses.reconstruction import reconstruction_loss
from losses.soft_neighborhood import soft_neighborhood_matching_loss
from losses.triplet_phase import (
    build_triplet_constraints_batch,
    classify_triplet,
    phase_triplet_loss,
)
from losses.variance_covariance import (
    covariance_loss,
    variance_covariance_loss,
    variance_loss,
)

__all__ = [
    "apply_spatial_constraint",
    "build_phase_neighborhood_batch",
    "build_triplet_constraints_batch",
    "average_features_by_ysfc",
    "build_ysfc_overlap",
    "categorical_loss",
    "classify_triplet",
    "contrastive_loss",
    "count_loss",
    "covariance_loss",
    "pairs_knn",
    "pairs_mutual_knn",
    "pairs_quantile",
    "pairs_radius",
    "pairs_with_spatial_constraint",
    "phase_neighborhood_loss",
    "phase_triplet_loss",
    "reconstruction_loss",
    "soft_neighborhood_matching_loss",
    "variance_covariance_loss",
    "variance_loss",
]
