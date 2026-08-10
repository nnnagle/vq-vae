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
from typing import Callable, Dict, List, Tuple

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


def _target_series(data: dict, provider: AnomalyTargetProvider) -> torch.Tensor:
    """Return per-pixel target time series ``[N, T, C]``."""
    tgt = provider(data["x"], data["z_type"])   # [N, Cx, T]
    return tgt.permute(0, 2, 1).contiguous()


def _target_channel_names(ctx: dict, kind: str, C: int) -> List[str]:
    """Human-readable target channel names (falls back to x0..xN-1).

    For the ``raw`` target these are the phase-input feature's channels
    (e.g. temporal_position, red, nir, ...); the anomaly target keeps the same
    per-channel layout.
    """
    try:
        from training.phase_eval.common import PHASE_INPUT_FEATURE
        names = list(ctx["feature_builder"].config.get_feature(PHASE_INPUT_FEATURE).channels.keys())
        # strip the "annual." group prefix for readability
        names = [n.split(".", 1)[-1] for n in names]
        if len(names) == C:
            return names
    except Exception:
        pass
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
        return {
            "n_observations": self.n,
            "n_pixels": self.n_pix,
            "r2_total_per_channel": r2_total,
            "r2_total_mean": float(np.mean(list(r2_total.values()))) if r2_total else 0.0,
            "r2_within_per_channel": r2_within,
            "r2_within_mean": float(np.mean(list(r2_within.values()))) if r2_within else 0.0,
            "within_variance_fraction_per_channel": var_frac,
            "within_variance_fraction_mean": float(np.mean(list(var_frac.values()))) if var_frac else 0.0,
        }


# ---------------------------------------------------------------------------
# Ridge: fit normal equations once, solve per lambda
# ---------------------------------------------------------------------------

def _fit_standardizer_and_normal_eq(
    loader, ctx, source, provider, halo, max_pixels, max_batches,
) -> Tuple[_Standardizer, torch.Tensor, torch.Tensor, int, int]:
    """Two extraction passes over train: (1) feature stats, (2) normal equations.

    Returns ``(standardizer, A, B, D, C)`` where ``A = [X̃|1]^T[X̃|1]`` and
    ``B = [X̃|1]^T Y`` accumulate the ridge normal equations on standardized
    features with a bias column.
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
    for batch in iter_batches(loader, max_batches):
        data = extract_pixel_series(
            batch, ctx, halo, max_pixels_per_sample=max_pixels,
            need_pre_film=need_h, require_evt=False, rng=rng,
        )
        if data is None:
            continue
        feat = _feature_series(data, source)
        tgt = _target_series(data, provider)
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
    if A is None:
        raise RuntimeError("no valid pixels found while accumulating normal equations")
    return std, A, B, D, C


def _solve_ridge(A: torch.Tensor, B: torch.Tensor, D: int, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    reg = torch.eye(D + 1, dtype=torch.float64) * lam
    reg[-1, -1] = 0.0                                  # do not penalize the bias
    Wb = torch.linalg.solve(A + reg, B)
    return Wb[:-1].float(), Wb[-1].float()             # W [D, C], b [C]


def _evaluate(
    loader, ctx, source, provider, std, predict_fn, channels, halo, max_pixels, max_batches,
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
        tgt = _target_series(data, provider)          # [N, T, C]
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
    device, cache_cap=2_000_000, epochs=20, lr=1e-3, seed=0,
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
        tgt = _target_series(data, provider)
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
        for source in sources:
            key = f"{source}__{kind}"
            logger.info(f"[A] fitting ridge: source={source} target={kind}")
            std, A, B, D, C = _fit_standardizer_and_normal_eq(
                train_loader, ctx, source, provider, halo,
                max_pixels_per_sample, max_batches,
            )
            channels = _target_channel_names(ctx, kind, C)

            # Select ridge λ on val by within-pixel R² (the phase signal).
            best_lam, best_val, best_W, best_b = None, -1e9, None, None
            for lam in RIDGE_LAMBDA_GRID:
                W, b = _solve_ridge(A, B, D, lam)
                predict = lambda Xs, W=W, b=b: Xs @ W + b
                vm = _evaluate(
                    val_loader, ctx, source, provider, std, predict, channels,
                    halo, max_pixels_per_sample, max_batches,
                )
                score = vm["r2_within_mean"]
                logger.info(f"  λ={lam:g}: val within-R²={score:.4f} total-R²={vm['r2_total_mean']:.4f}")
                if score > best_val:
                    best_lam, best_val, best_W, best_b = lam, score, W, b

            predict = lambda Xs, W=best_W, b=best_b: Xs @ W + b
            test_ridge = _evaluate(
                test_loader, ctx, source, provider, std, predict, channels,
                halo, max_pixels_per_sample, max_batches,
            )
            entry = {"ridge_lambda": best_lam, "ridge_test": test_ridge}
            logger.info(
                f"  [A] {key} RIDGE test: within-R²={test_ridge['r2_within_mean']:.4f} "
                f"total-R²={test_ridge['r2_total_mean']:.4f}"
            )

            if fit_mlp:
                mlp = _fit_mlp(
                    train_loader, ctx, source, provider, std, halo,
                    max_pixels_per_sample, max_batches, ctx["device"],
                )
                predict_mlp = lambda Xs, m=mlp: m(Xs.float().to(ctx["device"])).detach().cpu()
                test_mlp = _evaluate(
                    test_loader, ctx, source, provider, std, predict_mlp, channels,
                    halo, max_pixels_per_sample, max_batches,
                )
                entry["mlp_test"] = test_mlp
                logger.info(
                    f"  [A] {key} MLP  test: within-R²={test_mlp['r2_within_mean']:.4f} "
                    f"total-R²={test_mlp['r2_total_mean']:.4f}"
                )
            results[key] = entry
    return results
