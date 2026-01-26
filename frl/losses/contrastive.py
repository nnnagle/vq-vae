"""
Contrastive loss functions for representation learning.

This module provides vectorized contrastive loss implementations that support
weighted positive and negative pairs with grouping by anchor indices.

Features:
    - InfoNCE-style contrastive loss with temperature scaling
    - Support for weighted positive and negative pairs
    - Efficient grouping by anchors using scatter operations
    - Fully vectorized implementation (no Python loops)

Example:
    >>> from frl.losses import contrastive_loss
    >>> embeddings = torch.randn(100, 128)  # 100 samples, 128-dim embeddings
    >>> pos_pairs = torch.tensor([[0, 1], [0, 2], [1, 3]])  # anchor, positive
    >>> neg_pairs = torch.tensor([[0, 5], [0, 6], [1, 7]])  # anchor, negative
    >>> loss = contrastive_loss(embeddings, pos_pairs, neg_pairs)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def contrastive_loss(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
    temperature: float = 0.07,
    similarity: Literal["l2", "cosine", "dot"] = "l2",
) -> torch.Tensor:
    """
    Compute contrastive loss with weighted positive and negative pairs.

    This implements an InfoNCE-style contrastive loss that groups pairs by their
    anchor indices. For each anchor, the loss encourages high similarity with
    positive samples and low similarity with negative samples.

    The loss for each anchor is:
        L_a = -log(sum_p(w_p * exp(sim(a,p)/t)) /
                   (sum_p(w_p * exp(sim(a,p)/t)) + sum_n(w_n * exp(sim(a,n)/t))))

    Similarity functions:
        - "l2": sim(a, b) = -||a - b||^2 (negative squared L2 distance)
        - "cosine": sim(a, b) = (a · b) / (||a|| ||b||) (cosine similarity)
        - "dot": sim(a, b) = a · b (dot product)

    Note: For normalized embeddings, "cosine" and "dot" are equivalent, and "l2"
    is a linear transformation of both: -||a-b||^2 = -2 + 2(a·b) for unit vectors.

    Args:
        embeddings: Embedding vectors of shape [N, D] where N is the number of
            samples and D is the embedding dimension.
        pos_pairs: Positive pair indices of shape [P, 2] where P is the number of
            positive pairs. Each row is (anchor_idx, positive_idx).
        neg_pairs: Negative pair indices of shape [M, 2] where M is the number of
            negative pairs. Each row is (anchor_idx, negative_idx).
        pos_weights: Optional weights for positive pairs of shape [P]. If None,
            uniform weights of 1.0 are used.
        neg_weights: Optional weights for negative pairs of shape [M]. If None,
            uniform weights of 1.0 are used.
        temperature: Temperature scaling factor for the softmax. Lower values
            make the distribution sharper. Default: 0.07.
        similarity: Similarity function to use. One of "l2", "cosine", or "dot".
            Default: "l2".

    Returns:
        Scalar loss value averaged over all unique anchors.

    Example:
        >>> embeddings = torch.randn(100, 128)
        >>> pos_pairs = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 4]])
        >>> neg_pairs = torch.tensor([[0, 5], [0, 6], [1, 7], [2, 8]])
        >>> pos_weights = torch.tensor([1.0, 0.5, 1.0, 1.0])
        >>> neg_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        >>> loss = contrastive_loss(
        ...     embeddings, pos_pairs, neg_pairs,
        ...     pos_weights, neg_weights, temperature=0.1
        ... )
    """
    # Validate inputs
    if pos_pairs.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    # Extract anchor and target indices
    pos_anchors = pos_pairs[:, 0]  # [P]
    pos_targets = pos_pairs[:, 1]  # [P]

    neg_anchors = neg_pairs[:, 0]  # [M]
    neg_targets = neg_pairs[:, 1]  # [M]

    # Default weights if not provided
    if pos_weights is None:
        pos_weights = torch.ones(pos_pairs.shape[0], device=embeddings.device)
    if neg_weights is None:
        neg_weights = torch.ones(neg_pairs.shape[0], device=embeddings.device)

    # Compute similarities based on chosen similarity function
    pos_emb_a = embeddings[pos_anchors]  # [P, D]
    pos_emb_b = embeddings[pos_targets]  # [P, D]
    neg_emb_a = embeddings[neg_anchors]  # [M, D]
    neg_emb_b = embeddings[neg_targets]  # [M, D]

    if similarity == "l2":
        # sim(a, b) = -||a - b||^2
        pos_diff = pos_emb_a - pos_emb_b
        neg_diff = neg_emb_a - neg_emb_b
        pos_sims = -(pos_diff * pos_diff).sum(dim=1)  # [P]
        neg_sims = -(neg_diff * neg_diff).sum(dim=1)  # [M]
    elif similarity == "cosine":
        # sim(a, b) = (a · b) / (||a|| ||b||)
        pos_emb_a = F.normalize(pos_emb_a, p=2, dim=1)
        pos_emb_b = F.normalize(pos_emb_b, p=2, dim=1)
        neg_emb_a = F.normalize(neg_emb_a, p=2, dim=1)
        neg_emb_b = F.normalize(neg_emb_b, p=2, dim=1)
        pos_sims = (pos_emb_a * pos_emb_b).sum(dim=1)  # [P]
        neg_sims = (neg_emb_a * neg_emb_b).sum(dim=1)  # [M]
    elif similarity == "dot":
        # sim(a, b) = a · b
        pos_sims = (pos_emb_a * pos_emb_b).sum(dim=1)  # [P]
        neg_sims = (neg_emb_a * neg_emb_b).sum(dim=1)  # [M]
    else:
        raise ValueError(f"Unknown similarity function: {similarity}")

    # Apply temperature scaling
    pos_sims = pos_sims / temperature
    neg_sims = neg_sims / temperature

    # Compute weighted exponentials
    pos_exp = pos_weights * torch.exp(pos_sims)  # [P]
    neg_exp = neg_weights * torch.exp(neg_sims)  # [M]

    # Get unique anchors and create mapping for scatter
    # We need to sum exp values for each unique anchor
    all_anchors = torch.cat([pos_anchors, neg_anchors])
    unique_anchors = torch.unique(all_anchors)
    num_anchors = unique_anchors.shape[0]

    # Create index mapping: original anchor idx -> contiguous idx
    anchor_to_idx = torch.zeros(
        embeddings.shape[0], dtype=torch.long, device=embeddings.device
    )
    anchor_to_idx[unique_anchors] = torch.arange(
        num_anchors, device=embeddings.device
    )

    # Map pair anchors to contiguous indices
    pos_anchor_idx = anchor_to_idx[pos_anchors]  # [P]
    neg_anchor_idx = anchor_to_idx[neg_anchors]  # [M]

    # Sum positive and negative exponentials per anchor using scatter_add
    pos_sum = torch.zeros(num_anchors, device=embeddings.device, dtype=embeddings.dtype)
    neg_sum = torch.zeros(num_anchors, device=embeddings.device, dtype=embeddings.dtype)

    pos_sum.scatter_add_(0, pos_anchor_idx, pos_exp)
    neg_sum.scatter_add_(0, neg_anchor_idx, neg_exp)

    # Compute loss: -log(pos_sum / (pos_sum + neg_sum))
    # = -log(pos_sum) + log(pos_sum + neg_sum)
    denominator = pos_sum + neg_sum

    # Avoid log(0) by clamping
    eps = 1e-8
    loss_per_anchor = -torch.log(pos_sum.clamp(min=eps)) + torch.log(
        denominator.clamp(min=eps)
    )

    # Average over anchors
    return loss_per_anchor.mean()
