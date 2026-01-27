"""
Count loss functions with masking support.

This module provides loss functions for count/rate data including Poisson
and Negative Binomial distributions.

Features:
    - Poisson loss for count data
    - Negative Binomial loss for overdispersed counts
    - Proper masking support
    - Numerical stability for edge cases

Example:
    >>> from losses import count_loss
    >>> predicted_rate = torch.exp(model(x))  # Ensure positive
    >>> target_counts = torch.randint(0, 100, (4, 32, 32)).float()
    >>> loss = count_loss(predicted_rate, target_counts, loss_type="poisson")
"""

from __future__ import annotations

from typing import Literal

import torch


def count_loss(
    rate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    loss_type: Literal["poisson", "negative_binomial"] = "poisson",
    reduction: Literal["mean", "sum", "none"] = "mean",
    dispersion: torch.Tensor | float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute loss for count data with optional masking.

    Supports Poisson and Negative Binomial distributions for modeling count data.

    Loss types:
        - "poisson": Poisson negative log-likelihood. Assumes variance = mean.
            L = rate - target * log(rate)
            Use when counts follow Poisson distribution (variance ≈ mean).

        - "negative_binomial": Negative Binomial NLL. Handles overdispersion.
            Variance = mean + mean²/dispersion.
            Use when variance > mean (overdispersed counts).

    Args:
        rate: Predicted rate/mean parameter (λ). Must be positive. Shape [B, ...].
            Typically output of softplus or exp to ensure positivity.
        target: Target counts. Non-negative values. Shape must match rate.
        mask: Optional boolean mask where True = valid (compute loss), False = ignore.
            Must be broadcastable with rate. If None, all elements are used.
        loss_type: Type of count distribution. One of "poisson" or "negative_binomial".
            Default: "poisson".
        reduction: How to reduce the loss:
            - "mean": Average over valid elements (default)
            - "sum": Sum over valid elements
            - "none": Return per-element loss
        dispersion: Dispersion parameter for negative binomial (also called 'r' or
            'n' in some parameterizations). Can be a scalar or tensor matching rate
            shape. Higher values -> closer to Poisson. Default: 1.0.
        eps: Small constant for numerical stability. Default: 1e-8.

    Returns:
        Loss tensor. Scalar if reduction is "mean" or "sum", same shape as rate
        if reduction is "none".

    Edge cases:
        - If mask is all False: returns 0.0 for "mean"/"sum"
        - Zero rates: clamped to eps to avoid log(0)
        - Negative rates: will produce incorrect results (ensure positive input)

    Example:
        >>> # Poisson loss for count prediction
        >>> log_rate = model(features)  # Model outputs log-rate
        >>> rate = torch.exp(log_rate)  # Convert to rate (positive)
        >>> target = actual_counts.float()
        >>> loss = count_loss(rate, target, loss_type="poisson")

        >>> # With mask (e.g., ignore missing data)
        >>> mask = ~torch.isnan(target)
        >>> target = torch.nan_to_num(target, 0)
        >>> loss = count_loss(rate, target, mask=mask)

        >>> # Negative binomial for overdispersed data
        >>> loss = count_loss(rate, target, loss_type="negative_binomial", dispersion=10.0)

        >>> # Learned dispersion per location
        >>> dispersion = torch.exp(model.dispersion_head(features))
        >>> loss = count_loss(rate, target, loss_type="negative_binomial", dispersion=dispersion)
    """
    # Ensure numerical stability
    rate = rate.clamp(min=eps)

    if loss_type == "poisson":
        # Poisson NLL: rate - target * log(rate)
        # (Ignoring constant term log(target!) which doesn't affect gradients)
        loss = rate - target * torch.log(rate)

    elif loss_type == "negative_binomial":
        # Negative Binomial NLL
        # Parameterization: mean = rate, variance = rate + rate^2 / dispersion
        # NLL = -log(NB(target | rate, dispersion))
        #     = -log(Gamma(target + r) / (Gamma(target + 1) * Gamma(r)))
        #       - r * log(r / (r + rate)) - target * log(rate / (r + rate))
        # where r = dispersion

        r = dispersion
        if isinstance(r, (int, float)):
            r = torch.tensor(r, device=rate.device, dtype=rate.dtype)

        # Compute log-probabilities for numerical stability
        # p = r / (r + rate)  # probability parameter
        log_p = torch.log(r) - torch.log(r + rate)
        log_1mp = torch.log(rate) - torch.log(r + rate)

        # NLL = -lgamma(target + r) + lgamma(target + 1) + lgamma(r)
        #       - r * log(p) - target * log(1-p)
        loss = (
            -torch.lgamma(target + r)
            + torch.lgamma(target + 1)
            + torch.lgamma(r)
            - r * log_p
            - target * log_1mp
        )

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Expected one of: poisson, negative_binomial"
        )

    # Apply mask if provided
    if mask is not None:
        # Expand mask to match loss shape if needed
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(loss)

        if reduction == "none":
            loss = torch.where(mask, loss, torch.zeros_like(loss))
        else:
            valid_loss = loss[mask]

            if valid_loss.numel() == 0:
                return torch.tensor(0.0, device=rate.device, dtype=rate.dtype)

            if reduction == "mean":
                return valid_loss.mean()
            else:  # sum
                return valid_loss.sum()

    # No mask
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # none
        return loss
