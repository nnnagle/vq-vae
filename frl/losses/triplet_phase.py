"""
Triplet loss for the temporal phase encoder.

Given three sorted years t0 < t1 < t2 and a per-year ``ysfc`` indicator
(years-since-fast-change, where 0 = disturbance that year), this module:

1.  Determines disturbance status in each inter-year interval.
2.  Emits a set of (closer_pair, farther_pair, margin_class) ordering
    constraints appropriate for the ecological case.
3.  Computes a soft-margin triplet loss::

        loss = log(1 + exp(d_close - d_far + margin))

Cases
-----
* **d01 only** (disturbance between t0 and t1, not between t1 and t2):
    d(0,1)>d(1,2)+L,  d(0,2)>d(1,2)+L,  d(0,1)>d(0,2)+S
* **d12 only** (disturbance between t1 and t2, not between t0 and t1):
    d(1,2)>d(0,1)+L,  d(0,2)>d(0,1)+L,  d(1,2)>d(0,2)+S
* **no disturbance**:
    d(0,2)>d(0,1)+S,  d(0,2)>d(1,2)+S
* **disturbance in both intervals** or **disturbance at a sampled year**:
    skip (no loss contribution).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class MarginClass(Enum):
    """Semantic label for the margin size."""
    LARGE = auto()
    SMALL = auto()


@dataclass
class TripletConstraint:
    """A single ordering constraint: d(closer) should be < d(farther) - margin.

    ``closer`` and ``farther`` are tuples of year-slot indices (0, 1, or 2)
    identifying which pair of the three sampled years is involved.
    """
    closer: tuple[int, int]
    farther: tuple[int, int]
    margin_class: MarginClass


# ---------------------------------------------------------------------------
# Pair builder (pure logic, no tensors)
# ---------------------------------------------------------------------------

# Pre-built constraint lists for each ecological case.
_CONSTRAINTS_D01 = [
    # Disturbance between t0-t1 → d(0,1) and d(0,2) are large; d(1,2) is small
    TripletConstraint(closer=(1, 2), farther=(0, 1), margin_class=MarginClass.LARGE),
    TripletConstraint(closer=(1, 2), farther=(0, 2), margin_class=MarginClass.LARGE),
    TripletConstraint(closer=(0, 2), farther=(0, 1), margin_class=MarginClass.SMALL),
]

_CONSTRAINTS_D12 = [
    # Disturbance between t1-t2 → d(1,2) and d(0,2) are large; d(0,1) is small
    TripletConstraint(closer=(0, 1), farther=(1, 2), margin_class=MarginClass.LARGE),
    TripletConstraint(closer=(0, 1), farther=(0, 2), margin_class=MarginClass.LARGE),
    TripletConstraint(closer=(0, 2), farther=(1, 2), margin_class=MarginClass.SMALL),
]

_CONSTRAINTS_NONE = [
    # No disturbance → gradual drift, widest span is most different
    TripletConstraint(closer=(0, 1), farther=(0, 2), margin_class=MarginClass.SMALL),
    TripletConstraint(closer=(1, 2), farther=(0, 2), margin_class=MarginClass.SMALL),
]


def classify_triplet(
    ysfc: torch.Tensor,
    t0_idx: int,
    t1_idx: int,
    t2_idx: int,
) -> list[TripletConstraint] | None:
    """Determine ordering constraints for a pixel given its ysfc time series.

    Parameters
    ----------
    ysfc : Tensor of shape ``[T]``
        Years-since-fast-change for one pixel.  ``ysfc[t] == 0`` means a
        disturbance occurred in the year corresponding to temporal index *t*.
    t0_idx, t1_idx, t2_idx : int
        Temporal indices (into the T dimension) of the three sampled years,
        **already sorted** so that ``t0_idx < t1_idx < t2_idx``.

    Returns
    -------
    list[TripletConstraint] | None
        Constraint list for the applicable case, or *None* if the triplet
        should be skipped (disturbance at a sampled year, or disturbance in
        both intervals).
    """
    # Skip if any of the sampled years is itself a disturbance year.
    if ysfc[t0_idx] == 0 or ysfc[t1_idx] == 0 or ysfc[t2_idx] == 0:
        return None

    # Check for disturbance in each interval (years strictly between).
    d01 = (ysfc[t0_idx + 1 : t1_idx] == 0).any().item() if t1_idx > t0_idx + 1 else False
    d12 = (ysfc[t1_idx + 1 : t2_idx] == 0).any().item() if t2_idx > t1_idx + 1 else False

    if d01 and d12:
        return None
    if d01:
        return _CONSTRAINTS_D01
    if d12:
        return _CONSTRAINTS_D12
    return _CONSTRAINTS_NONE


# ---------------------------------------------------------------------------
# Vectorised pair builder (operates on a batch of pixels)
# ---------------------------------------------------------------------------

def build_triplet_constraints_batch(
    ysfc: torch.Tensor,
    t0_idx: int,
    t1_idx: int,
    t2_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ordering constraints for a batch of pixels.

    Parameters
    ----------
    ysfc : Tensor of shape ``[N, T]``
        Per-pixel ysfc time series for *N* pixels.
    t0_idx, t1_idx, t2_idx : int
        Temporal indices of the three sampled years (sorted).

    Returns
    -------
    closer_slots : LongTensor ``[K, 2]``
        Pairs of year-slot indices (values in {0, 1, 2}) for the pair that
        should be *closer* in embedding space.
    farther_slots : LongTensor ``[K, 2]``
        Corresponding pairs that should be *farther*.
    margin_is_large : BoolTensor ``[K]``
        True where the constraint uses the large margin.
    pixel_indices : LongTensor ``[K]``
        Which pixel (in 0..N-1) each constraint belongs to.

    Where *K* is the total number of constraints across all valid pixels.
    Pixels that should be skipped contribute zero constraints.
    """
    N = ysfc.shape[0]
    device = ysfc.device

    # --- Vectorised validity & interval checks -------------------------
    # Disturbance at sampled years → skip.
    at_t0 = ysfc[:, t0_idx] == 0  # [N]
    at_t1 = ysfc[:, t1_idx] == 0
    at_t2 = ysfc[:, t2_idx] == 0
    skip_sampled = at_t0 | at_t1 | at_t2

    # Disturbance in intervals.
    if t1_idx > t0_idx + 1:
        d01 = (ysfc[:, t0_idx + 1 : t1_idx] == 0).any(dim=1)  # [N]
    else:
        d01 = torch.zeros(N, dtype=torch.bool, device=device)

    if t2_idx > t1_idx + 1:
        d12 = (ysfc[:, t1_idx + 1 : t2_idx] == 0).any(dim=1)
    else:
        d12 = torch.zeros(N, dtype=torch.bool, device=device)

    skip_both = d01 & d12
    valid = ~(skip_sampled | skip_both)

    # --- Case masks (mutually exclusive within valid pixels) -----------
    case_d01 = valid & d01 & ~d12   # disturbance only in first interval
    case_d12 = valid & ~d01 & d12   # disturbance only in second interval
    case_none = valid & ~d01 & ~d12 # no disturbance

    # --- Materialise constraints per case ------------------------------
    # Each case has a fixed list of (closer, farther, is_large) tuples.
    # We replicate these for every pixel that matches the case.

    # Helper: expand a case's constraints for a boolean pixel mask.
    def _expand(mask: torch.Tensor, constraints: list[TripletConstraint]):
        pixel_ids = mask.nonzero(as_tuple=False).squeeze(1)  # [n]
        if pixel_ids.numel() == 0:
            return (
                torch.empty((0, 2), dtype=torch.long, device=device),
                torch.empty((0, 2), dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
        n = pixel_ids.shape[0]
        c = len(constraints)
        closer = torch.tensor(
            [con.closer for con in constraints], dtype=torch.long, device=device
        )  # [c, 2]
        farther = torch.tensor(
            [con.farther for con in constraints], dtype=torch.long, device=device
        )
        is_large = torch.tensor(
            [con.margin_class == MarginClass.LARGE for con in constraints],
            dtype=torch.bool, device=device,
        )  # [c]
        # Broadcast: [n * c, ...]
        closer_exp = closer.unsqueeze(0).expand(n, c, 2).reshape(-1, 2)
        farther_exp = farther.unsqueeze(0).expand(n, c, 2).reshape(-1, 2)
        is_large_exp = is_large.unsqueeze(0).expand(n, c).reshape(-1)
        pixel_exp = pixel_ids.unsqueeze(1).expand(n, c).reshape(-1)
        return closer_exp, farther_exp, is_large_exp, pixel_exp

    parts = [
        _expand(case_d01, _CONSTRAINTS_D01),
        _expand(case_d12, _CONSTRAINTS_D12),
        _expand(case_none, _CONSTRAINTS_NONE),
    ]

    closer_all = torch.cat([p[0] for p in parts], dim=0)
    farther_all = torch.cat([p[1] for p in parts], dim=0)
    is_large_all = torch.cat([p[2] for p in parts], dim=0)
    pixel_all = torch.cat([p[3] for p in parts], dim=0)

    return closer_all, farther_all, is_large_all, pixel_all


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def phase_triplet_loss(
    embeddings_t0: torch.Tensor,
    embeddings_t1: torch.Tensor,
    embeddings_t2: torch.Tensor,
    ysfc: torch.Tensor,
    t0_idx: int,
    t1_idx: int,
    t2_idx: int,
    large_margin: float = 1.0,
    small_margin: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute soft-margin triplet loss for the phase encoder.

    Parameters
    ----------
    embeddings_t0, embeddings_t1, embeddings_t2 : Tensor ``[N, D]``
        Embeddings for the three sampled years at *N* spatial locations.
    ysfc : Tensor ``[N, T]``
        Per-pixel ysfc time series.
    t0_idx, t1_idx, t2_idx : int
        Temporal indices of the three sampled years (sorted).
    large_margin, small_margin : float
        Margin values for constraints with/without disturbance.

    Returns
    -------
    loss : scalar Tensor
        Mean soft-margin triplet loss across all valid constraints.
        Returns 0 if no valid constraints exist.
    stats : dict
        Diagnostic statistics:
        - ``n_constraints``: total number of constraints
        - ``n_large``: number using the large margin
        - ``n_small``: number using the small margin
        - ``n_pixels_valid``: pixels contributing at least one constraint
        - ``n_pixels_skipped``: pixels contributing zero constraints
        - ``frac_satisfied``: fraction of constraints already satisfied
    """
    N = embeddings_t0.shape[0]
    device = embeddings_t0.device

    closer_slots, farther_slots, margin_is_large, pixel_indices = (
        build_triplet_constraints_batch(ysfc, t0_idx, t1_idx, t2_idx)
    )

    K = closer_slots.shape[0]

    if K == 0:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            {
                "n_constraints": 0,
                "n_large": 0,
                "n_small": 0,
                "n_pixels_valid": 0,
                "n_pixels_skipped": N,
                "frac_satisfied": 1.0,
            },
        )

    # Stack embeddings for easy slot indexing: [N, 3, D]
    emb_stack = torch.stack([embeddings_t0, embeddings_t1, embeddings_t2], dim=1)

    # Gather embeddings for each constraint.
    # closer_slots[:, 0], closer_slots[:, 1] are the two year-slot indices.
    pix = pixel_indices  # [K]

    closer_a = emb_stack[pix, closer_slots[:, 0]]   # [K, D]
    closer_b = emb_stack[pix, closer_slots[:, 1]]   # [K, D]
    farther_a = emb_stack[pix, farther_slots[:, 0]]  # [K, D]
    farther_b = emb_stack[pix, farther_slots[:, 1]]  # [K, D]

    # L2 distances.
    d_close = (closer_a - closer_b).pow(2).sum(dim=1)   # [K] (squared L2)
    d_far = (farther_a - farther_b).pow(2).sum(dim=1)

    # Margins.
    margin = torch.where(
        margin_is_large,
        torch.tensor(large_margin, device=device, dtype=d_close.dtype),
        torch.tensor(small_margin, device=device, dtype=d_close.dtype),
    )  # [K]

    # Soft-margin loss: log(1 + exp(d_close - d_far + margin))
    # We want d_far > d_close + margin, so violation = d_close - d_far + margin > 0.
    violation = d_close - d_far + margin
    loss_per_constraint = torch.nn.functional.softplus(violation)  # log(1+exp(x))

    loss = loss_per_constraint.mean()

    # --- Diagnostics ---------------------------------------------------
    n_large = margin_is_large.sum().item()
    n_small = K - n_large
    n_valid_pixels = pixel_indices.unique().numel()
    with torch.no_grad():
        frac_satisfied = (violation < 0).float().mean().item()

    stats = {
        "n_constraints": K,
        "n_large": n_large,
        "n_small": n_small,
        "n_pixels_valid": n_valid_pixels,
        "n_pixels_skipped": N - n_valid_pixels,
        "frac_satisfied": frac_satisfied,
    }

    return loss, stats
