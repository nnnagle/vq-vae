#!/usr/bin/env python3
"""Diagnostic C — disturbance produces ejection.

At ``ysfc = 0`` (a disturbance year), how often is there a big jump
``‖z_phase[t] − z_phase[t−1]‖``? Reports the jump-magnitude distribution at
ysfc=0 vs ysfc>0 and a separation metric — the ROC-AUC of "is this a disturbance
year" from jump magnitude alone. This validates the slow-feature + ejecta
geometry: stable years should drift slowly, disturbance years should jump.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from training.phase_eval.common import extract_pixel_series, iter_batches
from utils.sampling import ReservoirSampler

logger = logging.getLogger("phase_eval.ejection")

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def transition_jumps(
    z_phase: torch.Tensor, ysfc: torch.Tensor, valid: torch.Tensor,
):
    """Per-transition jump magnitudes and disturbance labels (pure, testable).

    A transition ``t`` (``t-1 → t``) is kept only when both timesteps are valid;
    it is a disturbance transition when ``ysfc[t] == 0`` and a stable transition
    when ``ysfc[t] > 0``.

    Args:
        z_phase: ``[N, T, zp]``   ysfc: ``[N, T]``   valid: ``[N, T]`` bool.

    Returns:
        ``(jumps, is_disturbance)`` numpy arrays over kept (disturbance ∪ stable)
        transitions: ``jumps`` float32, ``is_disturbance`` float32 in {0, 1}.
    """
    v_pair = valid[:, 1:] & valid[:, :-1]                          # [N, T-1]
    jumps = torch.linalg.vector_norm(z_phase[:, 1:] - z_phase[:, :-1], dim=2)
    ysfc_t = ysfc[:, 1:]
    vm = v_pair.reshape(-1)
    j = jumps.reshape(-1)[vm].numpy().astype(np.float32)
    y = ysfc_t.reshape(-1)[vm].numpy()
    keep = (y == 0) | (y > 0)
    return j[keep], (y[keep] == 0).astype(np.float32)


def run_ejection(
    ctx: dict,
    loader,
    halo: int = 16,
    max_pixels_per_sample: int = 2000,
    max_batches: int = 0,
    reservoir_cap: int = 4_000_000,
    seed: int = 0,
) -> dict:
    """Compute jump-magnitude distributions and disturbance-year ROC-AUC.

    A transition ``t`` (from ``t-1`` to ``t``) contributes only when both
    timesteps are valid. It is labelled a disturbance year when ``ysfc[t] == 0``.
    """
    # Paired (jump, label) reservoir for the AUC; label in column 1.
    res = ReservoirSampler(capacity=reservoir_cap, dim=2, seed=seed)
    # Separate magnitude reservoirs for clean per-class quantiles.
    res_dist = ReservoirSampler(capacity=reservoir_cap, dim=1, seed=seed + 1)
    res_stable = ReservoirSampler(capacity=reservoir_cap, dim=1, seed=seed + 2)
    rng = np.random.default_rng(seed)

    n_dist = 0
    n_stable = 0
    for batch in iter_batches(loader, max_batches):
        data = extract_pixel_series(
            batch, ctx, halo, max_pixels_per_sample=max_pixels_per_sample,
            need_pre_film=False, require_evt=False, rng=rng,
        )
        if data is None:
            continue
        if data["z_phase"].shape[1] < 2:
            continue
        j, lbl = transition_jumps(data["z_phase"], data["ysfc"], data["valid_tp"])
        if j.size == 0:
            continue

        is_dist = lbl == 1.0
        is_stable = ~is_dist
        n_dist += int(is_dist.sum())
        n_stable += int(is_stable.sum())

        if is_dist.any():
            res_dist.add(j[is_dist][:, None])
        if is_stable.any():
            res_stable.add(j[is_stable][:, None])
        res.add(np.stack([j, lbl], axis=1))

    def _quantiles(sampler: ReservoirSampler) -> Dict[str, float]:
        buf = sampler.get()[:, 0]
        if buf.size == 0:
            return {f"q{int(q * 100)}": 0.0 for q in _QUANTILES}
        return {f"q{int(q * 100)}": float(np.quantile(buf, q)) for q in _QUANTILES}

    dist_q = _quantiles(res_dist)
    stable_q = _quantiles(res_stable)

    # ROC-AUC of disturbance-year classification from jump magnitude alone.
    pairs = res.get()
    auc = None
    if pairs.shape[0] > 1 and len(np.unique(pairs[:, 1])) == 2:
        auc = float(roc_auc_score(pairs[:, 1], pairs[:, 0]))

    dist_median = dist_q["q50"]
    stable_median = stable_q["q50"]
    result = {
        "n_disturbance_transitions": n_dist,
        "n_stable_transitions": n_stable,
        "jump_quantiles_ysfc0": dist_q,
        "jump_quantiles_ysfc_gt0": stable_q,
        "median_jump_ratio": (dist_median / stable_median) if stable_median > 1e-12 else None,
        "roc_auc_disturbance_from_jump": auc,
        "auc_sample_size": int(pairs.shape[0]),
    }
    logger.info(
        f"[C] ejection: AUC={auc if auc is None else round(auc, 4)} | "
        f"median jump ysfc=0 {dist_median:.4f} vs ysfc>0 {stable_median:.4f} | "
        f"n_dist={n_dist:,} n_stable={n_stable:,}"
    )
    return result
