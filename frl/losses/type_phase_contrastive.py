"""
Type-local ranking InfoNCE on the phase embedding (Step 5).

Defines the metric: two pixel-times are close in ``z_phase`` iff they match on
**both** type and phase, so kNN in ``[z_type, z_phase]`` returns "same kind of
forest, same recovery stage". Closeness for the loss is computed from
**observables** (standardized ``z_type`` and the flow-state ``(a, Δa)``), never from
``z_phase`` itself (that would be circular / collapse-prone). See
``docs/phase_rethink_design.md`` Step 5.

Kernels (one drives everything):
    d_type²  = ‖ẑ_type_i − ẑ_type_j‖²          (standardized z_type)
    d_flow²  = ‖s_i − s_j‖²,   s = (a, Δa)      (observable flow-state)
    k_type   = exp(−d_type²/2σ_type²),  k_flow = exp(−d_flow²/2σ_flow²)
    p        = k_type · k_flow                   (positive: type ∧ flow)
    w_neg    = k_type · (1 − k_flow)             (negative: TYPE-LOCAL, diff-phase)

Similarity is **negative squared Euclidean** ``ℓ_ij = −‖z_phase_i−z_phase_j‖²/τ``
with a **fixed** τ (the absolute radial ruler). Positives are **cross-pixel** top-k
by ``p`` (same-pixel excluded — Step-4 OU owns within-pixel continuity, and this
prevents the "each pixel a private curve" collapse). Negatives are top-n by
``w_neg`` (same-type / different-phase; cross-type → ``w_neg≈0`` → not pushed apart,
i.e. incommensurate). Ranking, not distance-regression — z_phase stays free to
straighten/denoise.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def type_phase_contrastive_loss(
    z_phase: torch.Tensor,       # [M, D]
    z_type: torch.Tensor,        # [M, dt] (standardized)
    flow_state: torch.Tensor,    # [M, F] = (a, Δa) per pixel-time
    pixel_id: torch.Tensor,      # [M] long — which pixel each sample belongs to
    tau_phase: float = 1.0,
    sigma_type: float = 1.0,
    sigma_flow: float = 1.0,
    n_pos: int = 5,
    n_neg: int = 20,
    eps: float = 1e-8,
):
    """Type-local ranking InfoNCE. Returns (loss, diag).

    Args:
        z_phase: phase embeddings to shape.
        z_type: standardized type embeddings (detached upstream).
        flow_state: observable ``(a, Δa)`` per pixel-time (detached).
        pixel_id: pixel index per sample (same pixel ⇒ excluded from each other's
            positive selection).
        tau_phase: fixed temperature (the ruler) — never learned.
        sigma_type, sigma_flow: kernel bandwidths (peakedness knobs).
        n_pos, n_neg: cross-pixel positives / mined negatives per anchor.
    """
    M = z_phase.shape[0]
    device = z_phase.device
    if M < n_pos + 2:
        z = z_phase.sum() * 0.0
        return z, {"n_anchors": 0, "gap_over_tau": 0.0, "pos_d2": 0.0, "neg_d2": 0.0,
                   "n_pos_mean": 0.0, "n_pos_req": int(n_pos),
                   "k_type_pos": 0.0, "k_flow_pos": 0.0, "p_pos": 0.0,
                   "k_type_neg": 0.0, "k_flow_neg": 0.0, "p_neg": 0.0}

    z_type = z_type.detach()
    flow_state = flow_state.detach()

    # Observable kernels (no grad — they define the target neighborhood).
    with torch.no_grad():
        d_type2 = torch.cdist(z_type, z_type).pow(2)
        d_flow2 = torch.cdist(flow_state, flow_state).pow(2)
        k_type = torch.exp(-d_type2 / (2 * sigma_type ** 2))
        k_flow = torch.exp(-d_flow2 / (2 * sigma_flow ** 2))
        p = k_type * k_flow                                  # [M, M]
        w_neg = k_type * (1.0 - k_flow)                      # [M, M]

        self_mask = torch.eye(M, dtype=torch.bool, device=device)
        same_pixel = pixel_id.view(-1, 1) == pixel_id.view(1, -1)

        # Positives: cross-pixel only (exclude self + same-pixel), top-k by p.
        p_sel = p.masked_fill(same_pixel, 0.0)
        pos_val, pos_idx = p_sel.topk(min(n_pos, M - 1), dim=1)   # [M, n_pos]
        pos_valid = pos_val > eps                                 # ignore empty slots

        # Negatives: top-n by w_neg (self excluded). Weighted in the denom by w_neg.
        wn = w_neg.masked_fill(self_mask, 0.0)
        neg_wval, neg_idx = wn.topk(min(n_neg, M - 1), dim=1)     # [M, n_neg]

    # Similarity ℓ_ij = −‖z_phase_i − z_phase_j‖² / τ  (Euclidean, fixed τ).
    d_zp2 = torch.cdist(z_phase, z_phase).pow(2)                  # [M, M] (grad)
    logits = -d_zp2 / tau_phase

    rows = torch.arange(M, device=device).view(-1, 1)
    pos_logits = logits[rows, pos_idx]                           # [M, n_pos]
    neg_logits = logits[rows, neg_idx]                           # [M, n_neg]
    neg_w = neg_wval                                            # [M, n_neg]

    # Weighted-negative InfoNCE per (anchor, positive): the positive competes with
    # the anchor's mined negatives. denom = exp(ℓ_pos) + Σ_k w_neg_k·exp(ℓ_neg_k).
    neg_lse = torch.logsumexp(
        neg_logits + torch.log(neg_w + eps), dim=1, keepdim=True)  # [M, 1]
    # log(exp(pos) + exp(neg_lse)) per positive:
    denom = torch.logaddexp(pos_logits, neg_lse.expand_as(pos_logits))
    per_pos_nll = denom - pos_logits                            # −log softmax
    # Average over valid positives, then over anchors with ≥1 positive.
    pos_count = pos_valid.sum(dim=1).clamp(min=1)
    per_anchor = (per_pos_nll * pos_valid).sum(dim=1) / pos_count
    has_pos = pos_valid.any(dim=1)
    loss = per_anchor[has_pos].mean() if has_pos.any() else z_phase.sum() * 0.0

    with torch.no_grad():
        pos_d2 = (d_zp2[rows, pos_idx] * pos_valid).sum() / pos_valid.sum().clamp(min=1)
        neg_d2 = d_zp2[rows, neg_idx].mean()

        # Kernel diagnostics: the observable affinities at the *selected* pairs, so
        # we can see whether the σ bandwidths are sane. Positives should have high
        # k_type AND high k_flow (→ large p); negatives high k_type but low k_flow
        # (→ small p). A tiny mean p_pos / few valid positives ⇒ σ too peaked and
        # anchors are positive-starved (the pool clears the eps floor too rarely).
        pos_denom = pos_valid.sum().clamp(min=1)
        neg_valid = neg_wval > eps                                   # [M, n_neg]
        neg_denom = neg_valid.sum().clamp(min=1)
        kt_pos = k_type[rows, pos_idx]
        kf_pos = k_flow[rows, pos_idx]
        p_pos_m = p[rows, pos_idx]
        kt_neg = k_type[rows, neg_idx]
        kf_neg = k_flow[rows, neg_idx]
        p_neg_m = p[rows, neg_idx]

        diag = {
            "n_anchors": int(has_pos.sum().item()),
            "pos_d2": float(pos_d2),
            "neg_d2": float(neg_d2),
            # gap/τ: how many τ-units separate neg from pos distances (target ~2–3).
            "gap_over_tau": float((neg_d2 - pos_d2) / tau_phase),
            # mean valid positives per anchor (of n_pos requested) — starvation check.
            "n_pos_mean": float(pos_valid.sum(dim=1).float().mean()),
            "n_pos_req": int(n_pos),
            # observable affinities at selected positive pairs (masked to valid).
            "k_type_pos": float((kt_pos * pos_valid).sum() / pos_denom),
            "k_flow_pos": float((kf_pos * pos_valid).sum() / pos_denom),
            "p_pos": float((p_pos_m * pos_valid).sum() / pos_denom),
            # observable affinities at selected negative pairs (masked to valid).
            "k_type_neg": float((kt_neg * neg_valid).sum() / neg_denom),
            "k_flow_neg": float((kf_neg * neg_valid).sum() / neg_denom),
            "p_neg": float((p_neg_m * neg_valid).sum() / neg_denom),
        }
    return loss, diag
