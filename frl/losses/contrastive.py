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
    >>> from losses import contrastive_loss
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
        - "l2": sim(a, b) = -||a - b||^2 / D (negative mean squared distance).
            Normalized by embedding dimension D for consistent temperature scaling.
            Works well with default temperature. Recommended for general use.
        - "cosine": sim(a, b) = (a · b) / (||a|| ||b||) (cosine similarity).
            Bounded in [-1, 1]. Works well with default temperature.
        - "dot": sim(a, b) = a · b (dot product).
            Unbounded - use only with normalized embeddings or higher temperature.
            For unnormalized embeddings, dot products scale with dimension and
            can cause the softmax to concentrate on a single pair.

    For normalized embeddings:
        - "cosine" and "dot" are equivalent
        - "l2" is a linear transformation: -||a-b||^2/D = (-2 + 2*dot)/D

    Edge cases:
        - Anchors with positives but no negatives: loss = 0 (nothing to contrast)
        - Anchors with negatives but no positives: ignored (filtered out)
        - Empty pos_pairs: returns 0.0
        - Very low temperature (e.g., 0.01): loss approaches 0 if the positive
          pair is closer than all negatives (perfect separation)

    Args:
        embeddings: Embedding vectors of shape [N, D] where N is the number of
            samples and D is the embedding dimension.
        pos_pairs: Positive pair indices of shape [P, 2] where P is the number of
            positive pairs. Each row is (anchor_idx, positive_idx).
        neg_pairs: Negative pair indices of shape [M, 2] where M is the number of
            negative pairs. Each row is (anchor_idx, negative_idx). Anchors that
            don't appear in pos_pairs are ignored.
        pos_weights: Optional weights for positive pairs of shape [P]. If None,
            uniform weights of 1.0 are used. Can be used for importance sampling
            or confidence weighting.
        neg_weights: Optional weights for negative pairs of shape [M]. If None,
            uniform weights of 1.0 are used. Can be used for hard negative mining.
        temperature: Temperature scaling factor for the softmax. Lower values
            make the distribution sharper (harder contrastive learning). Higher
            values make it softer (smoother gradients). Default: 0.07.
        similarity: Similarity function to use. One of "l2", "cosine", or "dot".
            Default: "l2".

    Returns:
        Scalar loss value averaged over all unique anchors that have positives.

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
        # sim(a, b) = -||a - b||^2 / D (normalized by dimension)
        pos_diff = pos_emb_a - pos_emb_b
        neg_diff = neg_emb_a - neg_emb_b
        dim = embeddings.shape[1]
        pos_sims = -(pos_diff * pos_diff).sum(dim=1) / dim  # [P]
        neg_sims = -(neg_diff * neg_diff).sum(dim=1) / dim  # [M]
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

    # Apply temperature scaling and convert to log-space for numerical stability
    # logits = log(weight) + sim/temperature
    pos_logits = torch.log(pos_weights) + pos_sims / temperature  # [P]
    neg_logits = torch.log(neg_weights) + neg_sims / temperature  # [M]

    # Get unique anchors from POSITIVE pairs only (anchors must have at least one positive)
    # Negatives for anchors without positives are ignored
    unique_anchors = torch.unique(pos_anchors)
    num_anchors = unique_anchors.shape[0]

    # Filter negative pairs to only include anchors that have positives
    valid_neg_mask = torch.isin(neg_anchors, unique_anchors)
    neg_anchors = neg_anchors[valid_neg_mask]
    neg_targets = neg_targets[valid_neg_mask]
    neg_logits = neg_logits[valid_neg_mask]

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

    # Compute log-sum-exp per anchor using the identity:
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # This avoids overflow/underflow from direct exp()

    # Concatenate all logits and anchor indices
    all_logits = torch.cat([pos_logits, neg_logits])  # [P + M]
    all_anchor_idx = torch.cat([pos_anchor_idx, neg_anchor_idx])  # [P + M]

    # Find max logit per anchor for numerical stability
    max_logits = torch.full(
        (num_anchors,), float("-inf"), device=embeddings.device, dtype=embeddings.dtype
    )
    max_logits.scatter_reduce_(0, all_anchor_idx, all_logits, reduce="amax")

    # Compute stable exp: exp(logit - max_per_anchor)
    all_exp = torch.exp(all_logits - max_logits[all_anchor_idx])
    pos_exp = torch.exp(pos_logits - max_logits[pos_anchor_idx])

    # Sum exponentials per anchor
    all_sum = torch.zeros(num_anchors, device=embeddings.device, dtype=embeddings.dtype)
    pos_sum = torch.zeros(num_anchors, device=embeddings.device, dtype=embeddings.dtype)

    all_sum.scatter_add_(0, all_anchor_idx, all_exp)
    pos_sum.scatter_add_(0, pos_anchor_idx, pos_exp)

    # Compute log sums (add back the max)
    log_pos_sum = torch.log(pos_sum) + max_logits  # log(sum(w_p * exp(s_p/t)))
    log_all_sum = torch.log(all_sum) + max_logits  # log(sum(w * exp(s/t)))

    # Loss = -log(pos_sum / all_sum) = -log_pos_sum + log_all_sum
    loss_per_anchor = -log_pos_sum + log_all_sum

    # Average over anchors
    return loss_per_anchor.mean()
