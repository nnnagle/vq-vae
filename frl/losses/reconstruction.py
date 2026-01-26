"""
Reconstruction loss functions with masking support.

This module provides flexible reconstruction losses that support various loss types
and proper handling of masked (invalid) regions.

Features:
    - Multiple loss types: L1, L2/MSE, Huber, Smooth L1
    - Proper masking to ignore invalid pixels/values
    - Flexible reduction modes
    - Works with any tensor shape

Example:
    >>> from losses import reconstruction_loss
    >>> input = torch.randn(4, 3, 32, 32)
    >>> target = torch.randn(4, 3, 32, 32)
    >>> mask = torch.ones(4, 32, 32, dtype=torch.bool)
    >>> loss = reconstruction_loss(input, target, mask)
"""

from __future__ import annotations

from typing import Literal

import torch


def reconstruction_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    loss_type: Literal["l1", "l2", "mse", "huber", "smooth_l1"] = "l2",
    reduction: Literal["mean", "sum", "none"] = "mean",
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Compute reconstruction loss with optional masking.

    Computes element-wise loss between input and target, optionally ignoring
    masked regions. Supports multiple loss functions and reduction modes.

    Loss types:
        - "l1": Mean Absolute Error. Robust to outliers.
            L = |input - target|
        - "l2" / "mse": Mean Squared Error. Standard choice, penalizes large errors.
            L = (input - target)^2
        - "huber": Huber loss (smooth L1). Quadratic for small errors, linear for
            large errors. Controlled by delta parameter.
            L = 0.5 * (input - target)^2  if |input - target| < delta
            L = delta * (|input - target| - 0.5 * delta)  otherwise
        - "smooth_l1": PyTorch's smooth L1 loss (Huber with delta=1 and different
            scaling). Common in object detection.

    Args:
        input: Predicted tensor of any shape.
        target: Target tensor, must be broadcastable with input.
        mask: Optional boolean mask where True = valid (compute loss), False = ignore.
            Must be broadcastable with input. If None, all elements are used.
        loss_type: Type of loss function. One of "l1", "l2", "mse", "huber",
            "smooth_l1". Note: "l2" and "mse" are equivalent. Default: "l2".
        reduction: How to reduce the loss:
            - "mean": Average over valid elements (default)
            - "sum": Sum over valid elements
            - "none": Return per-element loss (mask still applied as zeros)
        delta: Delta parameter for Huber loss. Determines the threshold between
            quadratic and linear regions. Default: 1.0.

    Returns:
        Loss tensor. Scalar if reduction is "mean" or "sum", same shape as input
        if reduction is "none".

    Edge cases:
        - If mask is all False (no valid elements): returns 0.0 for "mean"/"sum"
        - If mask is None: all elements are considered valid
        - NaN in input/target at masked locations: ignored (not propagated)

    Example:
        >>> # Basic usage
        >>> input = torch.randn(4, 3, 32, 32)
        >>> target = torch.randn(4, 3, 32, 32)
        >>> loss = reconstruction_loss(input, target)

        >>> # With mask (e.g., ignore cloudy pixels)
        >>> mask = torch.ones(4, 32, 32, dtype=torch.bool)
        >>> mask[:, :10, :10] = False  # Mask out top-left corner
        >>> loss = reconstruction_loss(input, target, mask, loss_type="l1")

        >>> # Huber loss with custom delta
        >>> loss = reconstruction_loss(input, target, loss_type="huber", delta=0.5)

        >>> # Get per-element loss
        >>> loss_map = reconstruction_loss(input, target, reduction="none")
    """
    # Compute element-wise difference
    diff = input - target

    # Compute per-element loss based on loss type
    if loss_type == "l1":
        loss = torch.abs(diff)
    elif loss_type in ("l2", "mse"):
        loss = diff * diff
    elif loss_type == "huber":
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * diff * diff
        linear = delta * (abs_diff - 0.5 * delta)
        loss = torch.where(abs_diff < delta, quadratic, linear)
    elif loss_type == "smooth_l1":
        loss = torch.nn.functional.smooth_l1_loss(input, target, reduction="none")
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Expected one of: l1, l2, mse, huber, smooth_l1"
        )

    # Apply mask if provided
    if mask is not None:
        # Expand mask to match loss shape if needed
        # e.g., mask [B, H, W] -> loss [B, C, H, W]
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(1)  # Add channel dimension

        # Expand to match (handles broadcasting)
        mask = mask.expand_as(loss)

        if reduction == "none":
            # Zero out masked elements
            loss = torch.where(mask, loss, torch.zeros_like(loss))
        else:
            # Select only valid elements
            valid_loss = loss[mask]

            if valid_loss.numel() == 0:
                # No valid elements
                return torch.tensor(0.0, device=input.device, dtype=input.dtype)

            if reduction == "mean":
                return valid_loss.mean()
            else:  # sum
                return valid_loss.sum()

    # No mask - use all elements
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # none
        return loss
