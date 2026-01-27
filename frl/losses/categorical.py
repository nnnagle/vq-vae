"""
Categorical loss functions with masking support.

This module provides cross-entropy loss for discrete/categorical targets
with proper handling of masked (invalid) regions and class weighting.

Features:
    - Cross-entropy loss for multi-class classification
    - Proper masking with ignore_index support
    - Optional class weights for imbalanced data
    - Label smoothing support

Example:
    >>> from losses import categorical_loss
    >>> logits = torch.randn(4, 10, 32, 32)  # 10 classes
    >>> target = torch.randint(0, 10, (4, 32, 32))
    >>> mask = torch.ones(4, 32, 32, dtype=torch.bool)
    >>> loss = categorical_loss(logits, target, mask)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for categorical targets with optional masking.

    This is a wrapper around PyTorch's cross_entropy that adds proper mask support
    and flexible reduction modes.

    Args:
        logits: Predicted logits of shape [B, C, ...] where C is the number of
            classes. For spatial data, typically [B, C, H, W].
        target: Target class indices of shape [B, ...]. Values should be in
            range [0, C-1]. For spatial data, typically [B, H, W].
        mask: Optional boolean mask of shape [B, ...] where True = valid (compute
            loss), False = ignore. Must match target shape. If None, all elements
            are used.
        class_weights: Optional class weights of shape [C] for handling class
            imbalance. Higher weight = more importance.
        reduction: How to reduce the loss:
            - "mean": Average over valid elements (default)
            - "sum": Sum over valid elements
            - "none": Return per-element loss
        label_smoothing: Label smoothing factor in [0, 1]. Smooths target
            distribution: (1 - smoothing) * one_hot + smoothing / C. Default: 0.0.
        ignore_index: Target value to ignore (in addition to mask). Useful for
            padding. Default: -100.

    Returns:
        Loss tensor. Scalar if reduction is "mean" or "sum", shape [B, ...] if
        reduction is "none".

    Edge cases:
        - If mask is all False: returns 0.0 for "mean"/"sum"
        - Masked positions are set to ignore_index internally
        - NaN logits at masked locations: ignored

    Example:
        >>> # Basic usage - semantic segmentation
        >>> logits = torch.randn(4, 10, 32, 32)  # B=4, C=10, H=32, W=32
        >>> target = torch.randint(0, 10, (4, 32, 32))
        >>> loss = categorical_loss(logits, target)

        >>> # With mask (e.g., ignore background)
        >>> mask = target != 0  # Ignore class 0
        >>> loss = categorical_loss(logits, target, mask)

        >>> # With class weights for imbalanced data
        >>> weights = torch.tensor([0.1, 1.0, 1.0, 2.0, ...])  # 10 classes
        >>> loss = categorical_loss(logits, target, class_weights=weights)

        >>> # With label smoothing
        >>> loss = categorical_loss(logits, target, label_smoothing=0.1)
    """
    # Apply mask by setting masked positions to ignore_index
    if mask is not None:
        # Clone target to avoid modifying original
        target = target.clone()
        target[~mask] = ignore_index

    # Check if all elements are masked
    if mask is not None and not mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )

    # Apply reduction
    if reduction == "none":
        # Zero out masked positions for consistency
        if mask is not None:
            loss = torch.where(mask, loss, torch.zeros_like(loss))
        return loss
    elif reduction == "sum":
        if mask is not None:
            return loss[mask].sum()
        return loss.sum()
    else:  # mean
        if mask is not None:
            valid_loss = loss[mask]
            if valid_loss.numel() == 0:
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            return valid_loss.mean()
        return loss.mean()
