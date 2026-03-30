"""
EVT-guided soft neighbourhood loss.

Encourages the z_type embedding space to mirror the network structure of
the LANDFIRE EVT confusion graph.  For each anchor pixel, the distribution
of distances to neighbouring pixels in the learned latent space should match
the distribution of diffusion distances in the EVT confusion graph.

Design
------
1.  **Similarity matrix** — built from the combined NE+SE EVT contingency
    table.  Codes absent from the regional histogram (below ``min_count``) are
    excluded so rare / out-of-region types don't dilute the signal.

2.  **Diffusion** — the raw confusion matrix is symmetrised and row-normalised
    to a stochastic transition matrix P, then raised to the k-th power (P^k).
    This captures transitive similarity: if A↔B and B↔C are both confused,
    the diffused matrix assigns nonzero similarity to A↔C even when their
    direct confusion is zero.  k=1 is direct confusion only; k=2 is the
    default.

3.  **Soft neighbourhood matching** — per-anchor KL divergence between
    softmax(−d_ref / τ_ref) and softmax(−d_learned / τ_learned), where
    d_ref comes from the diffused EVT graph and d_learned from pairwise
    L2 distances in the embedding.  This matches the full distributional
    structure, not just binary positive/negative pairs.

4.  **Inverse-frequency weighting** — each anchor's row KL is weighted by
    median_freq / freq(code), capped at max_weight.  This prevents the loss
    from being dominated by the few most common EVT codes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


_SUMMARY_COLS = ["Row Totals", "Percent Row Agreement"]
_SUMMARY_ROWS = ["Column Totals", "Percent Column Agreement"]


class EvtDiffusionMetric:
    """Diffusion-distance metric derived from the EVT confusion table.

    Parameters
    ----------
    confusion_csv:
        Path to the combined EVT contingency table CSV produced by
        ``data/combine_evt_contingency_tables.py``.  Row and column labels
        are integer LANDFIRE codes.
    code_counts:
        Mapping of integer LANDFIRE EVT code → regional pixel count, as
        found in the stats file at
        ``stats["evt_class"]["static_categorical.evt"]["counts"]``
        (keys may be strings or ints).  Codes below ``min_count`` are
        excluded from the metric.
    min_count:
        Minimum regional pixel count for a code to be included.
    diffusion_steps:
        Number of random-walk steps (k in P^k).  k=1 uses direct confusion
        only; k=2 (default) captures one level of transitivity.
    max_weight:
        Cap on inverse-frequency anchor weights to prevent extreme upweighting
        of very rare codes that do pass the min_count filter.
    """

    def __init__(
        self,
        confusion_csv: str | Path,
        code_counts: dict,
        min_count: int = 100,
        min_confusion_samples: int = 30,
        diffusion_steps: int = 2,
        laplace_smoothing: float = 0.0,
        binary_threshold: float = 0.0,
        max_weight: float = 10.0,
    ) -> None:
        self.max_weight = max_weight
        self._device = torch.device("cpu")

        # ---- Load and filter the confusion table ----------------------------
        conf = pd.read_csv(confusion_csv, index_col=0)
        # Drop summary rows / columns
        conf = conf.drop(
            columns=[c for c in _SUMMARY_COLS if c in conf.columns],
            errors="ignore",
        )
        conf = conf.drop(
            index=[r for r in _SUMMARY_ROWS if r in conf.index],
            errors="ignore",
        )
        conf = conf[conf.index.notna()]
        conf.index = conf.index.astype(int)
        conf.columns = conf.columns.astype(int)
        conf = conf.astype(float)

        # ---- Filter codes by regional pixel count ---------------------------
        # code_counts keys may be strings (from JSON) or ints
        int_counts: dict[int, float] = {
            int(k): float(v) for k, v in code_counts.items()
        }
        valid_codes = {code for code, cnt in int_counts.items() if cnt >= min_count}

        # Keep only codes present in both the confusion table AND the histogram
        keep = sorted(
            c for c in conf.index.tolist() if c in valid_codes
        )

        # ---- Filter codes with too few confusion table samples --------------
        # Row sum (including diagonal) = total field samples predicted as that
        # code.  Sparse rows produce unreliable transition probabilities.
        if min_confusion_samples > 0:
            row_sums_orig = conf.reindex(index=keep, columns=keep, fill_value=0.0).sum(axis=1)
            keep = sorted(c for c in keep if row_sums_orig.get(c, 0) >= min_confusion_samples)

        if len(keep) < 2:
            raise ValueError(
                f"Fewer than 2 EVT codes survive the filters "
                f"(min_count={min_count}, min_confusion_samples={min_confusion_samples}). "
                f"Lower the thresholds or check that the stats file covers your region."
            )

        conf = conf.reindex(index=keep, columns=keep, fill_value=0.0)

        # ---- Build symmetric confusion matrix and diffuse -------------------
        C = conf.values  # [K, K]
        C_sym = (C + C.T) / 2.0

        # Optional Laplace smoothing — add epsilon uniformly to every cell.
        # Regularises sparse rows toward uniform; min_confusion_samples already
        # removes the most extreme cases, so a small value (e.g. 0.1) is enough.
        if laplace_smoothing > 0.0:
            C_sym = C_sym + laplace_smoothing

        # Row-normalise → stochastic transition matrix P
        row_sums = C_sym.sum(axis=1, keepdims=True)
        # Rows with no confusion at all → uniform distribution
        uniform = np.full(C_sym.shape, 1.0 / C_sym.shape[0])
        P = np.where(row_sums > 0, C_sym / np.where(row_sums > 0, row_sums, 1.0), uniform)

        # Raise to the k-th power
        Pk = np.linalg.matrix_power(P, diffusion_steps)

        # Optional dichotomization: threshold P^k → binary, then re-normalize rows.
        # Eliminates weakly-confused pairs (P^k < threshold) that inflate eff_n_ref.
        # After thresholding, surviving confused pairs receive equal reference weight
        # (uniform distribution over the ~3-12 strongly-confused neighbors per code).
        if binary_threshold > 0.0:
            Pk_bin = (Pk > binary_threshold).astype(float)
            np.fill_diagonal(Pk_bin, 0.0)  # diagonal handled by same-code mask in loss
            row_sums_bin = Pk_bin.sum(axis=1, keepdims=True)
            uniform_bin = np.full(Pk_bin.shape, 1.0 / Pk_bin.shape[0])
            Pk = np.where(
                row_sums_bin > 0,
                Pk_bin / np.where(row_sums_bin > 0, row_sums_bin, 1.0),
                uniform_bin,
            )

        # S[i,j] ∈ [0, 1]; convert to distances: d = 1 - S
        self._S = torch.tensor(Pk, dtype=torch.float32)  # [K, K]
        self._code_to_idx: dict[int, int] = {code: i for i, code in enumerate(keep)}

        # ---- Inverse-frequency weights from pixel counts --------------------
        counts = np.array([int_counts.get(c, 0.0) for c in keep], dtype=np.float64)
        # Normalise to [0, 1] range so median is well-defined
        total = counts.sum()
        freqs = counts / total if total > 0 else np.ones_like(counts) / len(counts)
        median_freq = float(np.median(freqs[freqs > 0])) if (freqs > 0).any() else 1.0
        raw_weights = np.where(freqs > 0, median_freq / freqs, 0.0)
        raw_weights = np.clip(raw_weights, 0.0, max_weight)
        self._freq_weights = torch.tensor(raw_weights, dtype=torch.float32)  # [K]

    # ------------------------------------------------------------------
    def to(self, device: torch.device | str) -> "EvtDiffusionMetric":
        """Move internal tensors to *device*; returns self."""
        self._device = torch.device(device)
        self._S = self._S.to(self._device)
        self._freq_weights = self._freq_weights.to(self._device)
        return self

    # ------------------------------------------------------------------
    def reference_distances(
        self,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pairwise diffusion distances for a set of EVT codes.

        Parameters
        ----------
        codes:
            ``[N]`` int tensor of LANDFIRE EVT codes at anchor locations.

        Returns
        -------
        d_ref : ``[N, N]`` float tensor
            Pairwise distances (1 − diffused_similarity).  Unknown-code pairs
            get distance 1.0.
        valid : ``[N]`` bool tensor
            True for anchors whose EVT code is present in the metric.
        """
        N = codes.shape[0]
        valid = torch.tensor(
            [c.item() in self._code_to_idx for c in codes],
            dtype=torch.bool,
            device=self._device,
        )
        idx = torch.tensor(
            [self._code_to_idx.get(c.item(), 0) for c in codes],
            dtype=torch.long,
            device=self._device,
        )
        # Similarity matrix for these anchors
        sim = self._S[idx[:, None], idx[None, :]]  # [N, N]
        # Unknown-code entries: set similarity to 0 (distance = 1)
        inv_valid = ~valid
        sim[inv_valid, :] = 0.0
        sim[:, inv_valid] = 0.0
        # Self-similarity for unknowns stays 0 (they'll be masked out anyway)
        d_ref = 1.0 - sim
        return d_ref, valid

    # ------------------------------------------------------------------
    def anchor_weights(self, codes: torch.Tensor) -> torch.Tensor:
        """Return per-anchor inverse-frequency weights.

        Parameters
        ----------
        codes:
            ``[N]`` int tensor of LANDFIRE EVT codes.

        Returns
        -------
        weights : ``[N]`` float tensor
            Inverse-frequency weight for each anchor.  Anchors with unknown
            codes get weight 0.0 (they are excluded from the loss).
        """
        weights = torch.tensor(
            [
                self._freq_weights[self._code_to_idx[c.item()]].item()
                if c.item() in self._code_to_idx
                else 0.0
                for c in codes
            ],
            dtype=torch.float32,
            device=self._device,
        )
        return weights

    @property
    def n_codes(self) -> int:
        """Number of EVT codes in the metric."""
        return len(self._code_to_idx)

    @property
    def valid_codes(self) -> set:
        """Set of integer EVT codes accepted by this metric."""
        return set(self._code_to_idx.keys())


# ---------------------------------------------------------------------------

def evt_soft_neighborhood_loss(
    embeddings: torch.Tensor,
    evt_codes: torch.Tensor,
    metric: EvtDiffusionMetric,
    tau_ref: float = 0.5,
    tau_learned: float = 0.5,
    min_valid_anchors: int = 4,
) -> tuple[torch.Tensor, dict]:
    """EVT-guided soft neighbourhood loss.

    For each anchor with a known EVT code, minimises the KL divergence
    between the softmax distribution of diffusion distances in the EVT
    confusion graph and the softmax distribution of L2 distances in the
    learned embedding space.

    Anchors with EVT codes absent from the regional histogram are excluded.
    Remaining anchors are weighted by the inverse frequency of their code,
    so the loss is not dominated by the most common types.

    Parameters
    ----------
    embeddings : ``[N, D]``
        z_type embeddings at anchor locations.
    evt_codes : ``[N]``
        Integer LANDFIRE EVT code for each anchor.
    metric :
        Prebuilt :class:`EvtDiffusionMetric`.
    tau_ref :
        Temperature for the reference (EVT graph) softmax distribution.
        Smaller → sharper, focusing on nearest EVT neighbours.
    tau_learned :
        Temperature for the learned (embedding) softmax distribution.
    min_valid_anchors :
        Minimum number of anchors with known EVT codes required to compute
        the loss.  Returns 0 if not met.

    Returns
    -------
    loss : scalar tensor
    stats : dict
        Diagnostic keys: ``n_anchors_in``, ``n_anchors_valid``,
        ``n_rows_active``, ``mean_kl``, ``mean_entropy_ref``,
        ``mean_entropy_learned``.
    """
    device = embeddings.device
    zero = torch.tensor(0.0, device=device, dtype=embeddings.dtype,
                        requires_grad=True)
    empty_stats = dict(
        n_anchors_in=embeddings.shape[0],
        n_anchors_valid=0,
        n_rows_active=0,
        mean_kl=0.0,
        mean_entropy_ref=0.0,
        mean_entropy_learned=0.0,
    )

    # ---- Reference distances and per-anchor weights ----------------------
    d_ref, valid = metric.reference_distances(evt_codes)   # [N,N], [N]
    weights = metric.anchor_weights(evt_codes)             # [N]

    n_valid = int(valid.sum().item())
    if n_valid < min_valid_anchors:
        empty_stats["n_anchors_valid"] = n_valid
        return zero, empty_stats

    # ---- Filter to valid anchors only ------------------------------------
    emb_v = embeddings[valid]           # [M, D]
    d_ref_v = d_ref[valid][:, valid]    # [M, M]
    w_v = weights[valid]                # [M]  inverse-freq weights
    codes_v = evt_codes[valid]          # [M]  EVT code for each valid anchor
    M = emb_v.shape[0]

    # ---- Learned distances -----------------------------------------------
    d_learned_v = torch.cdist(emb_v, emb_v)  # [M, M]

    # ---- Mask: off-diagonal AND different-code pairs only ---------------
    # Same-code pairs are excluded: they have d_ref = 1 - P^k[c,c] ≈ 0.3,
    # which would dominate the softmax and reduce the loss to within-class
    # clustering rather than a cross-type topology constraint.
    self_mask = torch.eye(M, dtype=torch.bool, device=device)
    same_code = codes_v.unsqueeze(0) == codes_v.unsqueeze(1)  # [M, M]
    mask = ~self_mask & ~same_code  # [M, M]

    # ---- Softmax distributions -------------------------------------------
    large_neg = torch.tensor(-1e9, device=device, dtype=embeddings.dtype)
    logits_ref = torch.where(mask, -d_ref_v / tau_ref, large_neg)        # [M, M]
    logits_lrn = torch.where(mask, -d_learned_v / tau_learned, large_neg)  # [M, M]

    # Rows with ≥2 valid neighbours (all off-diagonal entries are valid here)
    valid_per_row = mask.sum(dim=1)   # [M]  — always M-1 for a full matrix
    row_active = valid_per_row >= 2   # [M]

    n_rows_active = int(row_active.sum().item())
    if n_rows_active == 0:
        empty_stats["n_anchors_valid"] = n_valid
        return zero, empty_stats

    # ---- Per-row KL divergence -------------------------------------------
    log_p = logits_ref.log_softmax(dim=1)    # [M, M]
    log_q = logits_lrn.log_softmax(dim=1)    # [M, M]
    p = logits_ref.softmax(dim=1)            # [M, M]

    kl_per_row = (p * (log_p - log_q)).sum(dim=1)   # [M]
    kl_per_row = torch.where(row_active, kl_per_row, torch.zeros_like(kl_per_row))

    # ---- Weighted mean (inverse-frequency weights) -----------------------
    row_weights = w_v * row_active.float()   # [M]
    total_weight = row_weights.sum()

    if total_weight > 0:
        loss = (row_weights * kl_per_row).sum() / total_weight
    else:
        empty_stats["n_anchors_valid"] = n_valid
        empty_stats["n_rows_active"] = n_rows_active
        return zero, empty_stats

    # ---- Diagnostics (no grad) -------------------------------------------
    with torch.no_grad():
        import math
        active = row_active
        mean_kl = loss.item()
        q_dist = logits_lrn.softmax(dim=1)
        entropy_ref = -(p * log_p).sum(dim=1)
        entropy_lrn = -(q_dist * log_q).sum(dim=1)
        mean_entropy_ref = entropy_ref[active].mean().item() if active.any() else 0.0
        mean_entropy_lrn = entropy_lrn[active].mean().item() if active.any() else 0.0
        off_diag_d = d_learned_v[mask]
        mean_d_learned = off_diag_d.median().item() if off_diag_d.numel() > 0 else 0.0

        # ---- Retrieval diagnostics ------------------------------------------
        # Confused pairs: P^k[i,j] > 0  ↔  d_ref < 1
        confused_mask = (d_ref_v < (1.0 - 1e-6)) & mask   # [M, M]
        noncf_mask    = (d_ref_v >= (1.0 - 1e-6)) & mask  # [M, M]

        confused_d = d_learned_v[confused_mask]
        noncf_d    = d_learned_v[noncf_mask]
        d_lrn_confused = confused_d.mean().item() if confused_d.numel() > 0 else 0.0
        d_lrn_noncf    = noncf_d.mean().item()    if noncf_d.numel()    > 0 else 0.0

        n_confused_per_row = confused_mask.sum(dim=1).float()  # [M]
        n_confused_pairs = (
            n_confused_per_row[active].mean().item() if active.any() else 0.0
        )

        # Normalized rank within the different-code pool only (mask already
        # excludes self and same-code pairs).  Setting excluded positions to
        # inf pushes them to the bottom of the argsort so they don't affect
        # the ranks of different-code pairs.  Result: 0=nearest, 0.5=random.
        d_for_rank = d_learned_v.clone()
        d_for_rank[~mask] = float('inf')
        raw_ranks = d_for_rank.argsort(dim=1).argsort(dim=1).float()  # [M, M]
        n_diff = mask.sum(dim=1).float()                               # [M]
        ranks_norm = raw_ranks / (n_diff.unsqueeze(1) - 1).clamp(min=1)  # [M, M]
        confused_ranks = ranks_norm[confused_mask]
        mean_rank_confused = (
            confused_ranks.mean().item() if confused_ranks.numel() > 0 else 0.5
        )

        eff_n_ref = math.exp(mean_entropy_ref)

    stats = dict(
        n_anchors_in=embeddings.shape[0],
        n_anchors_valid=n_valid,
        n_rows_active=n_rows_active,
        mean_kl=mean_kl,
        mean_entropy_ref=mean_entropy_ref,
        mean_entropy_learned=mean_entropy_lrn,
        median_d_learned=mean_d_learned,
        d_lrn_confused=d_lrn_confused,
        d_lrn_noncf=d_lrn_noncf,
        n_confused_pairs=n_confused_pairs,
        mean_rank_confused=mean_rank_confused,
        eff_n_ref=eff_n_ref,
    )
    return loss, stats
