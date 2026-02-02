"""
Soft Neighborhood Matching Loss.

A generic loss that trains learned embeddings to reproduce the neighborhood
structure of a reference distance space.  Given two aligned distance matrices
— one from a reference signal and one from learned embeddings — the loss
computes per-row softmax distributions and minimises their KL divergence.

The loss is agnostic to the semantics of the reference and learned spaces.
Problem-specific logic (e.g. temporal alignment, pair selection, weighting)
lives in wrapper functions that prepare the inputs.

Loss formulation
----------------
For a single pair *b* in the batch, with aligned distance matrices
``d_ref[b]`` and ``d_learned[b]`` of shape ``[M, M]`` and a boolean mask:

.. math::

    p_{tt'} = \\frac{\\mathbb{1}[\\text{mask}(t,t')] \\cdot
              \\exp(-d_{\\text{ref}}(t,t') / \\tau_{\\text{ref}})}
              {\\sum_{t''} \\mathbb{1}[\\text{mask}(t,t'')] \\cdot
              \\exp(-d_{\\text{ref}}(t,t'') / \\tau_{\\text{ref}})}

    q_{tt'} = \\frac{\\mathbb{1}[\\text{mask}(t,t')] \\cdot
              \\exp(-d_{\\text{learned}}(t,t') / \\tau_{\\text{learned}})}
              {\\sum_{t''} \\mathbb{1}[\\text{mask}(t,t'')] \\cdot
              \\exp(-d_{\\text{learned}}(t,t'') / \\tau_{\\text{learned}})}

    \\mathcal{L}_b = \\sum_t \\mathbb{1}[\\text{row\\_valid}(t)] \\cdot
                     D_{\\text{KL}}(p_{t,\\cdot} \\| q_{t,\\cdot})

The final loss is a weighted mean across pairs:

.. math::

    \\mathcal{L} = \\frac{\\sum_b w_b \\, \\mathcal{L}_b}{\\sum_b w_b}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_neighborhood_matching_loss(
    d_reference: torch.Tensor,
    d_learned: torch.Tensor,
    mask: torch.Tensor,
    tau_ref: float = 1.0,
    tau_learned: float = 1.0,
    pair_weights: torch.Tensor | None = None,
    min_valid_per_row: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute soft neighborhood matching loss via KL divergence.

    Parameters
    ----------
    d_reference : Tensor ``[B, M, M]``
        Pairwise distances in the reference space (e.g. spectral).
        Must be non-negative.  Padded entries should be 0 and masked out.
    d_learned : Tensor ``[B, M, M]``
        Pairwise distances in the learned embedding space.
        Same shape and alignment as ``d_reference``.
    mask : BoolTensor ``[B, M, M]``
        ``True`` for valid entries.  The diagonal should normally be
        ``False`` (self-distances excluded).  Padded positions must be
        ``False``.
    tau_ref : float
        Temperature for the reference distribution.  Smaller values
        produce sharper distributions.
    tau_learned : float
        Temperature for the learned distribution.
    pair_weights : Tensor ``[B]``, optional
        Per-pair weight (e.g. from type similarity).  If ``None``,
        all pairs are weighted equally.
    min_valid_per_row : int
        Minimum number of valid (unmasked, non-self) entries in a row
        for that row to contribute to the loss.  Rows with fewer valid
        entries are skipped.  Must be >= 2 (need at least 2 neighbours
        to form a meaningful distribution).

    Returns
    -------
    loss : scalar Tensor
        Weighted mean KL divergence.  Returns 0 with ``requires_grad``
        if no valid rows exist.
    stats : dict
        Diagnostic statistics:

        - ``n_pairs``: number of pairs in the batch
        - ``n_pairs_active``: pairs with at least one valid row
        - ``n_rows_total``: total rows across all pairs
        - ``n_rows_valid``: rows that contributed to the loss
        - ``mean_kl``: mean KL divergence per valid row
        - ``mean_overlap``: mean number of valid entries per valid row
    """
    B, M, _ = d_reference.shape
    device = d_reference.device
    dtype = d_reference.dtype

    if min_valid_per_row < 2:
        raise ValueError(
            f"min_valid_per_row must be >= 2, got {min_valid_per_row}"
        )

    # --- Build masked logits -----------------------------------------------
    # Set masked positions to a large negative value so they get near-zero
    # probability after softmax.  Using -inf would produce NaN on rows that
    # are entirely masked (padding rows) because softmax(-inf, ..., -inf) is
    # 0/0.  A finite sentinel (-1e9) underflows to 0.0 cleanly in float32.
    large_neg = torch.tensor(-1e9, device=device, dtype=dtype)

    logits_ref = torch.where(mask, -d_reference / tau_ref, large_neg)      # [B, M, M]
    logits_learned = torch.where(mask, -d_learned / tau_learned, large_neg)  # [B, M, M]

    # --- Determine valid rows ----------------------------------------------
    # A row is valid if it has enough unmasked entries (excluding self).
    valid_per_row = mask.sum(dim=2)  # [B, M]
    row_valid = valid_per_row >= min_valid_per_row  # [B, M]

    n_rows_valid = row_valid.sum().item()

    if n_rows_valid == 0:
        return (
            torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
            {
                "n_pairs": B,
                "n_pairs_active": 0,
                "n_rows_total": B * M,
                "n_rows_valid": 0,
                "mean_kl": 0.0,
                "mean_overlap": 0.0,
            },
        )

    # --- Compute distributions ---------------------------------------------
    # Log-softmax along the last dimension (neighbours).
    log_p = logits_ref.log_softmax(dim=2)    # [B, M, M]
    log_q = logits_learned.log_softmax(dim=2)  # [B, M, M]

    # p in probability space (needed for KL = sum p * (log_p - log_q))
    p = logits_ref.softmax(dim=2)  # [B, M, M]

    # --- KL divergence per row ---------------------------------------------
    # KL(p_row || q_row) = sum_t' p(t,t') * (log p(t,t') - log q(t,t'))
    # Only sum over valid (masked) entries — others have p=0 and contribute 0.
    kl_per_row = (p * (log_p - log_q)).sum(dim=2)  # [B, M]

    # Zero out invalid rows.
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))

    # --- Aggregate per pair ------------------------------------------------
    # Sum KL across valid rows within each pair.
    rows_per_pair = row_valid.float().sum(dim=1)         # [B]
    kl_per_pair = kl_per_row.sum(dim=1)                  # [B]

    # Normalise per pair: mean KL per valid row.
    pair_active = rows_per_pair > 0
    kl_per_pair_normed = torch.where(
        pair_active,
        kl_per_pair / rows_per_pair.clamp(min=1),
        torch.zeros_like(kl_per_pair),
    )

    # --- Weighted mean across pairs ----------------------------------------
    if pair_weights is None:
        pair_weights = torch.ones(B, device=device, dtype=dtype)

    # Only count active pairs.
    weights = pair_weights * pair_active.float()
    total_weight = weights.sum()

    if total_weight > 0:
        loss = (weights * kl_per_pair_normed).sum() / total_weight
    else:
        loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    # --- Diagnostics -------------------------------------------------------
    with torch.no_grad():
        mean_overlap = (
            valid_per_row[row_valid].float().mean().item()
            if n_rows_valid > 0
            else 0.0
        )

    stats = {
        "n_pairs": B,
        "n_pairs_active": pair_active.sum().item(),
        "n_rows_total": B * M,
        "n_rows_valid": n_rows_valid,
        "mean_kl": loss.detach().item(),
        "mean_overlap": mean_overlap,
    }

    return loss, stats
