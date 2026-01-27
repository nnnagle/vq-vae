"""
VICReg-style variance-covariance loss for preventing embedding collapse.

This loss encourages:
1. Each embedding dimension to have sufficient variance (prevents collapse)
2. Embedding dimensions to be decorrelated (encourages full use of embedding space)
"""

from typing import Literal

import torch


def variance_covariance_loss(
    embeddings: torch.Tensor,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VICReg-style variance-covariance loss.

    This loss prevents embedding collapse by:
    1. Encouraging each dimension to have std >= variance_target
    2. Encouraging off-diagonal covariance elements to be zero

    Args:
        embeddings: [N, D] embedding matrix where N is number of samples
            and D is embedding dimension
        variance_weight: Weight for variance loss term (default: 1.0)
        covariance_weight: Weight for covariance loss term (default: 1.0)
        variance_target: Target standard deviation for each dimension (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-4)

    Returns:
        Tuple of (total_loss, variance_loss, covariance_loss)
        - total_loss: variance_weight * variance_loss + covariance_weight * covariance_loss
        - variance_loss: Mean hinge loss on standard deviations
        - covariance_loss: Mean squared off-diagonal covariance

    Example:
        >>> embeddings = torch.randn(1000, 64)  # 1000 samples, 64 dimensions
        >>> total, var_loss, cov_loss = variance_covariance_loss(embeddings)

    Notes:
        - For remote sensing with [B, D, H, W] latents, reshape to [B*H*W, D] first
        - Variance loss uses a hinge: max(0, target - std), so no penalty if std >= target
        - Covariance loss penalizes correlation between dimensions
        - Typical weights from VICReg paper: variance=25, covariance=25
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [N, D], got shape {embeddings.shape}")

    N, D = embeddings.shape

    if N < 2:
        # Need at least 2 samples for meaningful statistics
        zero = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        return zero, zero, zero

    # Center the embeddings (subtract mean)
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # === Variance loss ===
    # Compute standard deviation per dimension
    std = torch.sqrt(embeddings_centered.var(dim=0) + eps)

    # Hinge loss: penalize if std < variance_target
    variance_loss = torch.relu(variance_target - std).mean()

    # === Covariance loss ===
    # Compute covariance matrix [D, D]
    cov = (embeddings_centered.T @ embeddings_centered) / (N - 1)

    # Zero out diagonal (we only penalize off-diagonal elements)
    # Off-diagonal elements should be 0 for decorrelated dimensions
    cov_off_diag = cov.clone()
    cov_off_diag.fill_diagonal_(0.0)

    # Mean squared off-diagonal covariance
    # Divide by D to get per-dimension average
    covariance_loss = (cov_off_diag**2).sum() / D

    # Total loss
    total_loss = variance_weight * variance_loss + covariance_weight * covariance_loss

    return total_loss, variance_loss, covariance_loss


def variance_loss(
    embeddings: torch.Tensor,
    target: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute only the variance component of VICReg loss.

    Encourages each embedding dimension to have std >= target.

    Args:
        embeddings: [N, D] embedding matrix
        target: Target standard deviation (default: 1.0)
        eps: Small constant for numerical stability

    Returns:
        Variance loss (scalar)
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [N, D], got shape {embeddings.shape}")

    N, D = embeddings.shape

    if N < 2:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.relu(target - std).mean()


def covariance_loss(
    embeddings: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute only the covariance component of VICReg loss.

    Encourages embedding dimensions to be decorrelated.

    Args:
        embeddings: [N, D] embedding matrix
        eps: Small constant for numerical stability

    Returns:
        Covariance loss (scalar)
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [N, D], got shape {embeddings.shape}")

    N, D = embeddings.shape

    if N < 2:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    # Center embeddings
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Covariance matrix
    cov = (embeddings_centered.T @ embeddings_centered) / (N - 1)

    # Zero diagonal and compute mean squared off-diagonal
    cov_off_diag = cov.clone()
    cov_off_diag.fill_diagonal_(0.0)

    return (cov_off_diag**2).sum() / D
