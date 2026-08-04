#!/usr/bin/env python3
"""Diagnostic B — type-conditional recovery curves (upgrade of phase_recovery_curves.py).

For selected EVT classes: (i) the **actual** mean signal vs ysfc and (ii) a light
probe ``z_phase → NBR`` whose **predicted mean vs ysfc** is plotted stratified by
EVT. Success = the embedding reproduces the *type-specific* shape, not one pooled
curve — quantified here by a per-EVT **shape-agreement** metric (correlation
between the predicted-vs-ysfc and observed-vs-ysfc bin curves).

Differences from the original ``phase_recovery_curves.py``:
* the probe is **phase-only** (``z_phase`` alone, not type+phase+interaction), so
  the curve isolates what z_phase carries;
* it emits a machine-readable shape-agreement metric to ``metrics.json``;
* it is driven by the shared harness extractor and reports on **test**.

The observed NBR stands in for "actual anomaly" until the Step-1 μ/σ readout lands
(the anomaly target is deferred, per docs/phase_rethink_design.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from training.phase_eval.common import (
    PHASE_TARGET_FEATURE,
    extract_pixel_series,
    iter_batches,
)
from training.phase_eval.reconstruction import _Standardizer, _solve_ridge
# ``training.phase_recovery_curves`` and ``fit_phase_linear_probe`` pull the heavy
# GDAL/matplotlib stack, so they are imported lazily inside the functions that use
# them (keeps this module importable for unit tests of the pure helpers).

logger = logging.getLogger("phase_eval.recovery_curves")

NBR_CHANNEL = "annual.nbr"
RIDGE_LAMBDA_GRID = (1e-3, 1e-2, 1e-1)

# ysfc bins for the shape-agreement metric. Mirrors
# ``training.phase_recovery_curves.YSFC_BINS`` (kept local so the pure metric does
# not depend on that module's matplotlib/GDAL import chain). Right endpoint
# exclusive.
YSFC_BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 20), (20, 31)]


# ---------------------------------------------------------------------------
# Phase-only ridge z_phase -> NBR
# ---------------------------------------------------------------------------

def _nbr_index(feature_builder) -> int:
    from training.fit_phase_linear_probe import _get_target_channels

    channels = _get_target_channels(feature_builder)
    if NBR_CHANNEL not in channels:
        raise RuntimeError(
            f"'{NBR_CHANNEL}' not in {PHASE_TARGET_FEATURE} channels: {channels}"
        )
    return channels.index(NBR_CHANNEL)


def _extract_for_curves(batch, ctx, halo, max_pixels, nbr_idx, rng):
    """Extract z_phase, ysfc, evt, and observed NBR at valid pixels."""
    data = extract_pixel_series(
        batch, ctx, halo, max_pixels_per_sample=max_pixels,
        need_pre_film=False, require_evt=True,
        extra_target_features=[PHASE_TARGET_FEATURE], rng=rng,
    )
    if data is None:
        return None
    nbr = data["targets"][PHASE_TARGET_FEATURE][:, nbr_idx, :]   # [N, T]
    return data, nbr


def _fit_phase_nbr_ridge(
    train_loader, val_loader, ctx, halo, max_pixels, max_batches, nbr_idx,
):
    """Fit a standardized phase-only ridge z_phase→NBR; select λ on val R²."""
    rng = np.random.default_rng(0)
    std: Optional[_Standardizer] = None
    for batch in iter_batches(train_loader, max_batches):
        got = _extract_for_curves(batch, ctx, halo, max_pixels, nbr_idx, rng)
        if got is None:
            continue
        data, _ = got
        valid = data["valid_tp"]
        fv = data["z_phase"][valid]
        if std is None:
            std = _Standardizer(fv.shape[1])
        std.update(fv)
    if std is None:
        raise RuntimeError("no valid pixels for the recovery-curve probe fit")
    std.finalize()

    rng = np.random.default_rng(0)
    A = B = None
    D = 0
    for batch in iter_batches(train_loader, max_batches):
        got = _extract_for_curves(batch, ctx, halo, max_pixels, nbr_idx, rng)
        if got is None:
            continue
        data, nbr = got
        valid = data["valid_tp"]
        X = std.apply(data["z_phase"])[valid].double()
        Y = nbr[valid].double().unsqueeze(1)
        ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
        Xa = torch.cat([X, ones], dim=1)
        if A is None:
            D = X.shape[1]
            A = torch.zeros(D + 1, D + 1, dtype=torch.float64)
            B = torch.zeros(D + 1, 1, dtype=torch.float64)
        A += Xa.T @ Xa
        B += Xa.T @ Y
    if A is None:
        raise RuntimeError("no valid pixels while accumulating the probe normal equations")

    # Select λ by validation R² on NBR.
    best = (-1e9, None, None, None)
    for lam in RIDGE_LAMBDA_GRID:
        W, b = _solve_ridge(A, B, D, lam)
        r2 = _val_r2(val_loader, ctx, halo, max_pixels, max_batches, nbr_idx, std, W, b)
        logger.info(f"  [B] λ={lam:g}: val NBR R²={r2:.4f}")
        if r2 > best[0]:
            best = (r2, lam, W, b)
    _, lam, W, b = best
    return std, W, b, lam


def _val_r2(val_loader, ctx, halo, max_pixels, max_batches, nbr_idx, std, W, b) -> float:
    rng = np.random.default_rng(1)
    sse = ssum = ssum2 = 0.0
    n = 0
    for batch in iter_batches(val_loader, max_batches):
        got = _extract_for_curves(batch, ctx, halo, max_pixels, nbr_idx, rng)
        if got is None:
            continue
        data, nbr = got
        valid = data["valid_tp"]
        X = std.apply(data["z_phase"])[valid]
        y = nbr[valid].double()
        pred = (X @ W + b).squeeze(1).double()
        sse += float(((pred - y) ** 2).sum())
        ssum += float(y.sum()); ssum2 += float((y * y).sum()); n += y.numel()
    if n == 0:
        return 0.0
    sst = ssum2 - ssum ** 2 / n
    return 1.0 - sse / sst if sst > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Shape-agreement metric
# ---------------------------------------------------------------------------

def _bin_medians(ysfc: np.ndarray, vals: np.ndarray, min_n: int) -> np.ndarray:
    """Per-ysfc-bin median (NaN where a bin has < ``min_n`` samples)."""
    out = np.full(len(YSFC_BINS), np.nan)
    for i, (lo, hi) in enumerate(YSFC_BINS):
        m = (ysfc >= lo) & (ysfc < hi)
        if int(m.sum()) >= min_n:
            out[i] = np.median(vals[m])
    return out


def _shape_agreement(reservoir: EvtReservoir, evt_codes: List[int], min_n: int) -> dict:
    """Per-EVT correlation between predicted-vs-ysfc and observed-vs-ysfc curves."""
    per_evt: Dict[str, float] = {}
    for code in evt_codes:
        d = reservoir.get(code)
        if d is None:
            continue
        ysfc, pred, obs = d[:, 0], d[:, 1], d[:, 2]
        pc = _bin_medians(ysfc, pred, min_n)
        oc = _bin_medians(ysfc, obs, min_n)
        both = np.isfinite(pc) & np.isfinite(oc)
        if int(both.sum()) >= 3 and np.std(pc[both]) > 1e-9 and np.std(oc[both]) > 1e-9:
            per_evt[str(code)] = float(np.corrcoef(pc[both], oc[both])[0, 1])
    mean = float(np.mean(list(per_evt.values()))) if per_evt else 0.0
    return {"per_evt": per_evt, "mean": mean, "n_evt_scored": len(per_evt)}


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_recovery_curves(
    ctx: dict,
    train_loader,
    val_loader,
    test_loader,
    evt_code_to_label: Dict[int, str] | None = None,
    top_k_evt: int = 20,
    halo: int = 16,
    max_pixels_per_sample: int = 2000,
    max_batches: int = 0,
    max_ysfc: float = 30.0,
    max_samples_per_evt: int = 10_000,
    min_bin_samples: int = 20,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """Fit a phase-only z_phase→NBR probe and produce type-conditional curves."""
    from training.phase_recovery_curves import (
        EvtReservoir, plot_recovery_curves, save_csv,
    )

    evt_code_to_label = evt_code_to_label or {}
    fb = ctx["feature_builder"]
    nbr_idx = _nbr_index(fb)

    logger.info("[B] fitting phase-only z_phase→NBR probe")
    std, W, b, lam = _fit_phase_nbr_ridge(
        train_loader, val_loader, ctx, halo, max_pixels_per_sample, max_batches, nbr_idx,
    )

    logger.info("[B] streaming test set into per-EVT reservoir")
    reservoir = EvtReservoir(max_per_evt=max_samples_per_evt, seed=seed)
    rng = np.random.default_rng(seed)
    for batch in iter_batches(test_loader, max_batches):
        got = _extract_for_curves(batch, ctx, halo, max_pixels_per_sample, nbr_idx, rng)
        if got is None:
            continue
        data, nbr = got
        valid = data["valid_tp"]
        ysfc = data["ysfc"]
        evt = data["evt"]                                # [N]
        N, T = ysfc.shape
        Xs = std.apply(data["z_phase"]).reshape(N * T, -1)
        pred = (Xs @ W + b).squeeze(1).reshape(N, T)
        # Keep valid (pixel, timestep) with ysfc in [0, max_ysfc].
        in_range = valid & (ysfc >= 0) & (ysfc <= max_ysfc)
        evt_bt = torch.from_numpy(evt).unsqueeze(1).expand(N, T)
        m = in_range.reshape(-1)
        if not m.any():
            continue
        reservoir.add_batch(
            evt_bt.reshape(-1)[m].numpy().astype(np.int32),
            ysfc.reshape(-1)[m].numpy().astype(np.float32),
            pred.reshape(-1)[m].numpy().astype(np.float32),
            nbr.reshape(-1)[m].numpy().astype(np.float32),
        )

    if reservoir.n_total() == 0:
        raise RuntimeError("no valid observations for recovery curves")

    sorted_counts = sorted(reservoir.pixel_counts().items(), key=lambda x: x[1], reverse=True)
    top_codes = [c for c, _ in sorted_counts[: min(top_k_evt, len(sorted_counts))]]

    shape = _shape_agreement(reservoir, top_codes, min_bin_samples)
    logger.info(
        f"[B] shape-agreement mean over {shape['n_evt_scored']} EVTs = {shape['mean']:.4f}"
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_csv(reservoir, top_codes, evt_code_to_label, out / "recovery_nbr_by_ysfc_by_evt.csv")
        plot_recovery_curves(
            reservoir, top_codes, evt_code_to_label,
            out / "recovery_curves.png", min_bin_samples=min_bin_samples,
        )

    return {
        "ridge_lambda": lam,
        "n_observations": reservoir.n_total(),
        "top_evt_codes": top_codes,
        "shape_agreement": shape,
    }
