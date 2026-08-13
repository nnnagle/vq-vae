#!/usr/bin/env python3
"""Diagnostic A — reconstruction probes: does z_phase retain the trajectory?

Fits light probes ``features → target`` and reports **total** variance explained
*and* — the diagnostic the spec calls the important one — **within-pixel**
variance explained (target decomposed into between-pixel and within-pixel parts;
R² on the within-pixel part is the phase signal). A model that scores high on
total but low on within-pixel has merely re-encoded type/level.

* **Feature sources** (the doc: probe post-FiLM ``z_phase``; contrast with pre-FiLM
  ``h`` and ``z_type``): selectable, defaults to all three so the contrast is in
  one report.
* **Target**: the raw phase-encoder input ``x`` (live now via
  :class:`~training.phase_eval.common.AnomalyTargetProvider`). The
  ``mature_baseline`` anomaly target is the Step-1 seam and is skipped until μ/σ
  exists.
* **Probes**: a closed-form streaming **ridge** (primary) and a small **MLP**
  (ceiling). Ridge λ is selected on val; everything is reported on test.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.phase_eval.common import (
    AnomalyTargetProvider,
    extract_pixel_series,
    iter_batches,
)

logger = logging.getLogger("phase_eval.reconstruction")

FEATURE_SOURCES = ("z_phase", "h", "z_type")
RIDGE_LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _warn_lambda_edge(best_lam: float, grid, label: str, scores=None) -> None:
    """Warn if the selected λ sits on the grid boundary.

    A boundary pick means the true optimum is probably *outside* the grid (the
    validation curve is still improving at the edge), so the reported fit is
    clamped and the grid should be widened. Interior picks are silent — the
    optimum is bracketed. Skips single-point grids (no interior to speak of).

    ``scores`` (the per-λ validation scores) lets the guard skip a **degenerate**
    sweep where λ had no effect (max−min ≈ 0): there the λ pick is arbitrary and a
    "boundary" warning would be a false alarm — e.g. z_type, whose within-pixel R²
    is identically 0 by construction, always defaulting the pick to the grid min.
    """
    lo, hi = min(grid), max(grid)
    if len(grid) < 2 or lo == hi:
        return
    if scores is not None and len(scores) and (max(scores) - min(scores)) < 1e-9:
        return
    if best_lam == lo:
        logger.warning(
            f"[{label}] selected λ={best_lam:g} is the grid MINIMUM ({lo:g}); the "
            f"optimum may be smaller — widen RIDGE_LAMBDA_GRID downward (less "
            f"regularization / possible overfitting at the reported fit)."
        )
    elif best_lam == hi:
        logger.warning(
            f"[{label}] selected λ={best_lam:g} is the grid MAXIMUM ({hi:g}); the "
            f"optimum may be larger — widen RIDGE_LAMBDA_GRID upward (more "
            f"regularization wanted than the grid allows)."
        )


# ---------------------------------------------------------------------------
# Feature source + target
# ---------------------------------------------------------------------------

def _feature_series(data: dict, source: str) -> torch.Tensor:
    """Return per-pixel feature time series ``[N, T, D]`` for the chosen source."""
    if source == "z_phase":
        return data["z_phase"]
    if source == "h":
        if "h" not in data:
            raise KeyError("pre-FiLM 'h' not extracted; pass need_pre_film=True")
        return data["h"]
    if source == "z_type":
        T = data["z_phase"].shape[1]
        return data["z_type"].unsqueeze(1).expand(-1, T, -1).contiguous()
    raise ValueError(f"unknown feature source: {source!r}")


# Target channels to drop from the reconstruction. ``temporal_position`` is the
# calendar/year index — a smooth per-pixel ramp that is trivially predictable and
# semantically empty for forest dynamics. Post-FiLM z_phase reconstructs it at
# ~0.92 within-R² and it was inflating the weighted aggregate (~22% of z_phase's
# score) without reflecting any real temporal signal, so it is excluded from the
# fit, the evaluation, and the aggregate.
EXCLUDE_TARGET_CHANNELS = ("temporal_position",)


def _full_target_channel_names(ctx: dict) -> Optional[List[str]]:
    """Phase-input target channel names from config, or None if unresolvable."""
    try:
        from training.phase_eval.common import PHASE_INPUT_FEATURE
        names = list(ctx["feature_builder"].config.get_feature(PHASE_INPUT_FEATURE).channels.keys())
        return [n.split(".", 1)[-1] for n in names]   # strip "annual." group prefix
    except Exception:
        return None


def _target_channels_kept(ctx: dict) -> Tuple[Optional[List[str]], Optional[torch.Tensor]]:
    """Kept target channel names + their column indices after applying the
    ``EXCLUDE_TARGET_CHANNELS`` filter. Returns ``(None, None)`` when channel names
    can't be resolved (→ keep all channels; the caller falls back to x0..xN names).
    """
    names = _full_target_channel_names(ctx)
    if names is None:
        return None, None
    keep = [i for i, n in enumerate(names) if n not in EXCLUDE_TARGET_CHANNELS]
    dropped = [n for n in names if n in EXCLUDE_TARGET_CHANNELS]
    if not dropped:
        logger.warning(
            f"[A] EXCLUDE_TARGET_CHANNELS={EXCLUDE_TARGET_CHANNELS} matched none of "
            f"the target channels {names}; nothing excluded."
        )
    else:
        logger.info(f"[A] excluding target channels {dropped} from reconstruction")
    return [names[i] for i in keep], torch.tensor(keep, dtype=torch.long)


def _target_series(
    data: dict, provider: AnomalyTargetProvider, keep_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return per-pixel target time series ``[N, T, C]`` (kept channels only)."""
    tgt = provider(data["x"], data["z_type"])       # [N, Cx, T]
    tgt = tgt.permute(0, 2, 1).contiguous()          # [N, T, Cx]
    if keep_idx is not None:
        tgt = tgt.index_select(2, keep_idx.to(tgt.device))
    return tgt


def _target_channel_names(ctx: dict, kind: str, C: int) -> List[str]:
    """Human-readable target channel names (falls back to x0..xN-1)."""
    names = _full_target_channel_names(ctx)
    if names is not None and len(names) == C:
        return names
    return [f"x{c}" for c in range(C)]


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------

class _Standardizer:
    """Column mean/std of features, fit on valid (pixel, timestep) rows."""

    def __init__(self, dim: int):
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sumsq = torch.zeros(dim, dtype=torch.float64)
        self.n = 0
        self.mean = torch.zeros(dim)
        self.std = torch.ones(dim)

    def update(self, feat_valid: torch.Tensor) -> None:
        f = feat_valid.double()
        self.sum += f.sum(dim=0)
        self.sumsq += (f * f).sum(dim=0)
        self.n += f.shape[0]

    def finalize(self) -> None:
        if self.n == 0:
            return
        mean = self.sum / self.n
        var = (self.sumsq / self.n - mean * mean).clamp(min=1e-12)
        self.mean = mean.float()
        self.std = var.sqrt().float()

    def apply(self, feat: torch.Tensor) -> torch.Tensor:
        return (feat - self.mean) / self.std


# ---------------------------------------------------------------------------
# Metric accumulator (total + within-pixel R²)
# ---------------------------------------------------------------------------

class _R2Accumulator:
    """Streams total and within-pixel R² accumulators per target channel."""

    def __init__(self, C: int):
        self.C = C
        self.sse = torch.zeros(C, dtype=torch.float64)
        self.sum_y = torch.zeros(C, dtype=torch.float64)
        self.sum_y2 = torch.zeros(C, dtype=torch.float64)
        self.n = 0
        self.sse_w = torch.zeros(C, dtype=torch.float64)
        self.sst_w = torch.zeros(C, dtype=torch.float64)
        self.n_pix = 0
        self.m_w = 0                                   # within-pixel valid obs count
        # For the within-pixel variance *fraction* of the target.
        self.sst_total = torch.zeros(C, dtype=torch.float64)

    def update(self, pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> None:
        """pred/y ``[N, T, C]``; valid ``[N, T]`` bool."""
        vm = valid.unsqueeze(-1)                       # [N, T, 1]
        p = pred.double()
        t = y.double()
        # --- total (flattened over valid observations) ---
        pf = p[valid]                                  # [M, C]
        tf = t[valid]
        self.sse += ((pf - tf) ** 2).sum(dim=0)
        self.sum_y += tf.sum(dim=0)
        self.sum_y2 += (tf * tf).sum(dim=0)
        self.n += pf.shape[0]
        # --- within-pixel (per pixel with >= 2 valid timesteps) ---
        cnt = valid.sum(dim=1)                         # [N]
        has = cnt >= 2
        if has.any():
            vv = vm[has].double()
            cc = valid[has].sum(dim=1, keepdim=True).clamp(min=1).double()  # [Nh, 1]
            ph = p[has] * vv
            th = t[has] * vv
            p_mean = ph.sum(dim=1, keepdim=True) / cc.unsqueeze(-1)
            t_mean = th.sum(dim=1, keepdim=True) / cc.unsqueeze(-1)
            p_anom = (p[has] - p_mean) * vv
            t_anom = (t[has] - t_mean) * vv
            self.sse_w += ((p_anom - t_anom) ** 2).sum(dim=(0, 1))
            self.sst_w += (t_anom ** 2).sum(dim=(0, 1))
            self.n_pix += int(has.sum())
            self.m_w += int(valid[has].sum())          # within-pixel valid obs

    def result(self, channels: List[str]) -> dict:
        def _r2(sse, sst):
            return {
                ch: (1.0 - float(sse[c]) / float(sst[c])) if sst[c] > 1e-12 else 0.0
                for c, ch in enumerate(channels)
            }
        sst_total = (self.sum_y2 - self.sum_y ** 2 / max(self.n, 1)).clamp(min=0.0)
        r2_total = _r2(self.sse, sst_total)
        r2_within = _r2(self.sse_w, self.sst_w)
        # within-pixel fraction of total target variance (per channel)
        var_frac = {
            ch: (float(self.sst_w[c]) / float(sst_total[c])) if sst_total[c] > 1e-12 else 0.0
            for c, ch in enumerate(channels)
        }
        # Per-channel absolute variance (the weight) and MSE (the annotation).
        # Targets are z-scored per channel, so these are in comparable units and a
        # near-zero within_variance flags a channel where a wild negative R² is
        # numerically meaningless (tiny denominator), not a real reconstruction
        # failure. within_mse == within_variance would mean R²=0 (the mean baseline).
        within_var = {
            ch: (float(self.sst_w[c]) / self.m_w) if self.m_w > 0 else 0.0
            for c, ch in enumerate(channels)
        }
        total_var = {
            ch: float(sst_total[c]) / max(self.n, 1) for c, ch in enumerate(channels)
        }
        within_mse = {
            ch: (float(self.sse_w[c]) / self.m_w) if self.m_w > 0 else 0.0
            for c, ch in enumerate(channels)
        }
        total_mse = {
            ch: float(self.sse[c]) / max(self.n, 1) for c, ch in enumerate(channels)
        }
        # Variance-weighted aggregate R²: pool residual/total SS across channels
        # *before* the ratio. Identical to weighting each channel's R² by its
        # within-pixel variance, so low-variance channels with wild negative R²
        # can no longer dominate the unweighted per-channel mean. This is the
        # honest headline for "does the embedding retain temporal signal?".
        sse_w_tot, sst_w_tot = float(self.sse_w.sum()), float(self.sst_w.sum())
        sse_t_tot, sst_t_tot = float(self.sse.sum()), float(sst_total.sum())
        r2_within_weighted = (1.0 - sse_w_tot / sst_w_tot) if sst_w_tot > 1e-12 else 0.0
        r2_total_weighted = (1.0 - sse_t_tot / sst_t_tot) if sst_t_tot > 1e-12 else 0.0
        return {
            "n_observations": self.n,
            "n_pixels": self.n_pix,
            "n_within_observations": self.m_w,
            "r2_total_per_channel": r2_total,
            "r2_total_mean": float(np.mean(list(r2_total.values()))) if r2_total else 0.0,
            "r2_total_weighted": r2_total_weighted,
            "r2_within_per_channel": r2_within,
            "r2_within_mean": float(np.mean(list(r2_within.values()))) if r2_within else 0.0,
            "r2_within_weighted": r2_within_weighted,
            "within_variance_fraction_per_channel": var_frac,
            "within_variance_fraction_mean": float(np.mean(list(var_frac.values()))) if var_frac else 0.0,
            "within_variance_per_channel": within_var,
            "total_variance_per_channel": total_var,
            "within_mse_per_channel": within_mse,
            "total_mse_per_channel": total_mse,
        }


# ---------------------------------------------------------------------------
# Ridge: fit normal equations once, solve per lambda
# ---------------------------------------------------------------------------

def _fit_standardizer_and_normal_eq(
    loader, ctx, source, provider, halo, max_pixels, max_batches, keep_idx=None,
) -> Tuple[_Standardizer, torch.Tensor, torch.Tensor, int, int]:
    """Two extraction passes over train: (1) feature stats, (2) normal equations.

    Returns ``(standardizer, A, B, D, C)`` where ``A = [X̃|1]^T[X̃|1]`` and
    ``B = [X̃|1]^T Y`` accumulate the ridge normal equations on standardized
    features with a bias column, **averaged over the M observations** (divided by
    M). Averaging implements the per-sample ridge objective
    ``(1/M)‖X̃w − y‖² + λ‖w‖²`` so the λ grid is meaningful and independent of
    dataset size: with standardized features ``A`` then has a unit diagonal, and
    λ∈[1e-4, 1] spans negligible→strong. Without the 1/M, ``A``'s diagonal is ~M
    (10⁷–10⁸) and every λ≤1 is a ~10⁻⁷ perturbation — no regularization at all.
    """
    need_h = source == "h"
    rng = np.random.default_rng(0)

    # Pass 1 — feature mean/std.
    std = None
    for batch in iter_batches(loader, max_batches):
        data = extract_pixel_series(
            batch, ctx, halo, max_pixels_per_sample=max_pixels,
            need_pre_film=need_h, require_evt=False, rng=rng,
        )
        if data is None:
            continue
        feat = _feature_series(data, source)          # [N, T, D]
        valid = data["valid_tp"]
        fv = feat[valid]
        if std is None:
            std = _Standardizer(fv.shape[1])
        std.update(fv)
    if std is None:
        raise RuntimeError("no valid pixels found while fitting the standardizer")
    std.finalize()

    # Pass 2 — normal equations on standardized features.
    rng = np.random.default_rng(0)   # identical pixel subsampling as pass 1
    A = B = None
    D = C = 0
    M = 0
    for batch in iter_batches(loader, max_batches):
        data = extract_pixel_series(
            batch, ctx, halo, max_pixels_per_sample=max_pixels,
            need_pre_film=need_h, require_evt=False, rng=rng,
        )
        if data is None:
            continue
        feat = _feature_series(data, source)
        tgt = _target_series(data, provider, keep_idx)
        valid = data["valid_tp"]
        X = std.apply(feat)[valid].double()           # [M, D]
        Y = tgt[valid].double()                       # [M, C]
        ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
        Xa = torch.cat([X, ones], dim=1)
        if A is None:
            D, C = X.shape[1], Y.shape[1]
            A = torch.zeros(D + 1, D + 1, dtype=torch.float64)
            B = torch.zeros(D + 1, C, dtype=torch.float64)
        A += Xa.T @ Xa
        B += Xa.T @ Y
        M += X.shape[0]
    if A is None or M == 0:
        raise RuntimeError("no valid pixels found while accumulating normal equations")
    # Average the normal equations so λ is on a dataset-size-independent scale
    # (see docstring): A/M has a unit diagonal for standardized features.
    A /= M
    B /= M
    return std, A, B, D, C


def _solve_ridge(A: torch.Tensor, B: torch.Tensor, D: int, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    reg = torch.eye(D + 1, dtype=torch.float64) * lam
    reg[-1, -1] = 0.0                                  # do not penalize the bias
    Wb = torch.linalg.solve(A + reg, B)
    return Wb[:-1].float(), Wb[-1].float()             # W [D, C], b [C]


def _evaluate(
    loader, ctx, source, provider, std, predict_fn, channels, halo, max_pixels, max_batches,
    keep_idx=None,
) -> dict:
    """Stream a split and compute total + within-pixel R² via ``predict_fn``."""
    need_h = source == "h"
    acc = _R2Accumulator(len(channels))
    rng = np.random.default_rng(1)
    for batch in iter_batches(loader, max_batches):
        data = extract_pixel_series(
            batch, ctx, halo, max_pixels_per_sample=max_pixels,
            need_pre_film=need_h, require_evt=False, rng=rng,
        )
        if data is None:
            continue
        feat = _feature_series(data, source)          # [N, T, D]
        tgt = _target_series(data, provider, keep_idx)  # [N, T, C]
        valid = data["valid_tp"]
        N, T, _ = feat.shape
        Xs = std.apply(feat).reshape(N * T, -1)
        pred = predict_fn(Xs).reshape(N, T, len(channels))
        acc.update(pred, tgt, valid)
    return acc.result(channels)


# ---------------------------------------------------------------------------
# MLP ceiling
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _fit_mlp(
    loader, ctx, source, provider, std, halo, max_pixels, max_batches,
    device, cache_cap=2_000_000, epochs=20, lr=1e-3, seed=0, keep_idx=None,
) -> _MLP:
    """Fit the MLP ceiling on a bounded in-memory cache of standardized rows."""
    need_h = source == "h"
    rng = np.random.default_rng(2)
    Xs: List[torch.Tensor] = []
    Ys: List[torch.Tensor] = []
    kept = 0
    for batch in iter_batches(loader, max_batches):
        data = extract_pixel_series(
            batch, ctx, halo, max_pixels_per_sample=max_pixels,
            need_pre_film=need_h, require_evt=False, rng=rng,
        )
        if data is None:
            continue
        feat = _feature_series(data, source)
        tgt = _target_series(data, provider, keep_idx)
        valid = data["valid_tp"]
        X = std.apply(feat)[valid]
        Y = tgt[valid]
        if kept < cache_cap:
            take = min(X.shape[0], cache_cap - kept)
            Xs.append(X[:take]); Ys.append(Y[:take]); kept += take
        if kept >= cache_cap:
            break
    Xc = torch.cat(Xs).to(device)
    Yc = torch.cat(Ys).to(device)
    torch.manual_seed(seed)
    mlp = _MLP(Xc.shape[1], Yc.shape[1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    lossf = nn.MSELoss()
    bs = 65536
    n = Xc.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(mlp(Xc[idx]), Yc[idx])
            loss.backward(); opt.step()
            tot += float(loss) * idx.numel()
        logger.info(f"  MLP[{source}] epoch {ep + 1}/{epochs} train MSE={tot / n:.5f}")
    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_reconstruction(
    ctx: dict,
    train_loader,
    val_loader,
    test_loader,
    sources=FEATURE_SOURCES,
    anomaly_kinds=("raw",),
    halo: int = 16,
    max_pixels_per_sample: int = 2000,
    max_batches: int = 0,
    fit_mlp: bool = True,
) -> dict:
    """Run diagnostic A across feature sources and anomaly targets.

    Returns a nested metrics dict keyed by ``"{source}__{anomaly_kind}"``, each
    holding ridge test metrics (with the selected λ) and, if enabled, MLP-ceiling
    test metrics.
    """
    results: Dict[str, dict] = {}
    for kind in anomaly_kinds:
        provider = AnomalyTargetProvider(kind)
        # A non-'raw' target may still be a Step-1 seam that isn't wired up; probe
        # it once and record the deferral instead of crashing the whole run.
        try:
            provider(torch.zeros(1, 1, 1), torch.zeros(1, 1))
        except NotImplementedError as e:
            logger.warning(f"anomaly kind '{kind}' is deferred: {e}")
            results[f"__deferred__{kind}"] = {"deferred": str(e)}
            continue
        # Target channels are the same across sources; resolve the exclusion once.
        kept_names, keep_idx = _target_channels_kept(ctx)
        for source in sources:
            key = f"{source}__{kind}"
            logger.info(f"[A] fitting ridge: source={source} target={kind}")
            std, A, B, D, C = _fit_standardizer_and_normal_eq(
                train_loader, ctx, source, provider, halo,
                max_pixels_per_sample, max_batches, keep_idx=keep_idx,
            )
            channels = kept_names if kept_names is not None else _target_channel_names(ctx, kind, C)

            # Select ridge λ on val by variance-weighted within-pixel R² (the
            # phase signal, weighted so near-dead channels don't drive the pick).
            best_lam, best_val, best_W, best_b = None, -1e9, None, None
            val_scores = []
            for lam in RIDGE_LAMBDA_GRID:
                W, b = _solve_ridge(A, B, D, lam)
                predict = lambda Xs, W=W, b=b: Xs @ W + b
                vm = _evaluate(
                    val_loader, ctx, source, provider, std, predict, channels,
                    halo, max_pixels_per_sample, max_batches, keep_idx=keep_idx,
                )
                score = vm["r2_within_weighted"]
                val_scores.append(score)
                logger.info(
                    f"  λ={lam:g}: val within-R² weighted={score:.4f} "
                    f"mean={vm['r2_within_mean']:.4f} total={vm['r2_total_weighted']:.4f}"
                )
                if score > best_val:
                    best_lam, best_val, best_W, best_b = lam, score, W, b
            _warn_lambda_edge(best_lam, RIDGE_LAMBDA_GRID, f"A:{key}", val_scores)

            predict = lambda Xs, W=best_W, b=best_b: Xs @ W + b
            test_ridge = _evaluate(
                test_loader, ctx, source, provider, std, predict, channels,
                halo, max_pixels_per_sample, max_batches, keep_idx=keep_idx,
            )
            entry = {"ridge_lambda": best_lam, "ridge_test": test_ridge}
            logger.info(
                f"  [A] {key} RIDGE test: within-R² weighted={test_ridge['r2_within_weighted']:.4f} "
                f"mean={test_ridge['r2_within_mean']:.4f} | total weighted={test_ridge['r2_total_weighted']:.4f}"
            )

            if fit_mlp:
                mlp = _fit_mlp(
                    train_loader, ctx, source, provider, std, halo,
                    max_pixels_per_sample, max_batches, ctx["device"], keep_idx=keep_idx,
                )
                predict_mlp = lambda Xs, m=mlp: m(Xs.float().to(ctx["device"])).detach().cpu()
                test_mlp = _evaluate(
                    test_loader, ctx, source, provider, std, predict_mlp, channels,
                    halo, max_pixels_per_sample, max_batches, keep_idx=keep_idx,
                )
                entry["mlp_test"] = test_mlp
                logger.info(
                    f"  [A] {key} MLP  test: within-R² weighted={test_mlp['r2_within_weighted']:.4f} "
                    f"mean={test_mlp['r2_within_mean']:.4f} | total weighted={test_mlp['r2_total_weighted']:.4f}"
                )
            results[key] = entry

        # Cross-source summary — the h → z_phase within-pixel R² gap. A positive
        # gap means pre-FiLM h retains temporal signal that post-FiLM z_phase
        # loses across the FiLM step (the phase-rethink thesis, quantified). Report
        # it on both the weighted headline and the unweighted mean, on ridge (and
        # MLP when fit), so the number is tracked automatically each run.
        h_key, z_key = f"h__{kind}", f"z_phase__{kind}"
        if h_key in results and z_key in results:
            summary = {}
            for probe in ("ridge_test", "mlp_test"):
                if probe in results[h_key] and probe in results[z_key]:
                    h_t, z_t = results[h_key][probe], results[z_key][probe]
                    tag = probe.split("_")[0]              # "ridge" / "mlp"
                    summary[f"{tag}_within_r2_gap_weighted"] = (
                        h_t["r2_within_weighted"] - z_t["r2_within_weighted"])
                    summary[f"{tag}_within_r2_gap_mean"] = (
                        h_t["r2_within_mean"] - z_t["r2_within_mean"])
                    summary[f"{tag}_h_within_r2_weighted"] = h_t["r2_within_weighted"]
                    summary[f"{tag}_z_phase_within_r2_weighted"] = z_t["r2_within_weighted"]
            if summary:
                results[f"__summary__{kind}"] = summary
                logger.info(
                    f"  [A] h→z_phase within-R² gap ({kind}): ridge weighted="
                    f"{summary.get('ridge_within_r2_gap_weighted', float('nan')):.4f} "
                    f"(h={summary.get('ridge_h_within_r2_weighted', float('nan')):.4f}, "
                    f"z_phase={summary.get('ridge_z_phase_within_r2_weighted', float('nan')):.4f})"
                )
    return results
