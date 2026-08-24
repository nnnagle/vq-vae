"""Phase-0 classical AR(1)+noise recovery estimator (data prior for ρ(z_type)).

Model-free, offline grounding for the differentiable Kalman filter
(``losses.kalman_filter``): fit a scalar AR(1)-plus-measurement-noise model to
observed recovery series, per EVT, to answer three questions *before* any
training:

1. **ρ̂(type)** — a data prior to seed/anchor the type-conditional transition
   head. Estimated from **lags ≥ 1** (``γ(2)/γ(1)``), which are free of
   measurement noise, so ρ̂ is **de-attenuated** — unlike the naive lag-1
   autocorrelation ``γ(1)/γ(0)`` = ρ · reliability, biased toward 0.
2. **AR-order adequacy** — for a true AR(1) the noise-free autocovariance decays
   geometrically, so ``γ(k+1)/γ(k)`` is constant for k ≥ 1. Divergent successive
   ratios ⇒ AR(1) is too rigid (longer memory / non-monotone recovery ⇒ AR(2) /
   complex modes).
3. **reliability** ``γ_x/(γ_x+R) = ρ_naive/ρ_ratio`` — how noisy the series is,
   i.e. how much attenuation the plug-in OU penalty would have suffered.

Autocovariances are pooled over many short **within-segment** recovery runs:
a lag-k pair ``(t-k, t)`` contributes only if both timesteps are valid and no
disturbance reset falls in ``(t-k, t]`` (segments don't cross an outward jump) —
the same segmentation the filter uses. Pure NumPy; unit-tested on synthetic
AR(1)+noise. The ISAAC CLI (``run_ar1_recovery.py``) feeds it real series.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def pooled_autocov(
    x: np.ndarray,               # [P, T] series (rows = pixels)
    lag: int,
    valid: Optional[np.ndarray] = None,   # [P, T] bool
    reset: Optional[np.ndarray] = None,   # [P, T] bool (True = segment start)
    demean: bool = True,
) -> tuple[float, int]:
    """Pooled within-segment lag-``lag`` autocovariance and the pair count.

    A pair ``(t-lag, t)`` is used iff both ends are valid and no reset occurs in
    ``(t-lag, t]`` (i.e. the two ends share a recovery segment).
    """
    P, T = x.shape
    if valid is None:
        valid = np.ones((P, T), dtype=bool)
    if reset is None:
        reset = np.zeros((P, T), dtype=bool)
    if lag >= T:
        return 0.0, 0

    mu = x[valid].mean() if (demean and valid.any()) else 0.0
    xc = x - mu

    # A reset anywhere in (t-lag, t] breaks the pair. Cumulative reset count lets
    # us test "no reset in the window" as an equality of prefix sums.
    rcum = np.cumsum(reset.astype(np.int64), axis=1)          # [P, T]
    t = np.arange(lag, T)
    left, right = t - lag, t
    # resets strictly after left, up to and including right:
    breaks = rcum[:, right] - rcum[:, left]                   # [P, T-lag]
    ok = valid[:, left] & valid[:, right] & (breaks == 0)     # [P, T-lag]
    prod = xc[:, left] * xc[:, right]
    n = int(ok.sum())
    if n == 0:
        return 0.0, 0
    return float(prod[ok].sum() / n), n


def summarize_ar1(
    x: np.ndarray,
    valid: Optional[np.ndarray] = None,
    reset: Optional[np.ndarray] = None,
    maxlag: int = 3,
) -> dict:
    """Fit the scalar AR(1)+noise summary. Returns a dict of estimates.

    Keys: ``rho_ratio`` (γ2/γ1, de-attenuated), ``rho_naive`` (γ1/γ0, attenuated),
    ``reliability`` (γ_x/(γ_x+R) ≈ rho_naive/rho_ratio), ``lag_ratios``
    (γ(k+1)/γ(k) for k≥1 — constant ⇒ AR(1) adequate), ``gammas`` (γ0..γmaxlag),
    ``n_pairs_lag1``.
    """
    gammas, ns = [], []
    for k in range(maxlag + 1):
        g, n = pooled_autocov(x, k, valid, reset)
        gammas.append(g)
        ns.append(n)
    g0, g1 = gammas[0], gammas[1]
    rho_ratio = gammas[2] / g1 if abs(g1) > 1e-12 else float("nan")
    rho_naive = g1 / g0 if abs(g0) > 1e-12 else float("nan")
    reliability = (rho_naive / rho_ratio
                   if (rho_ratio == rho_ratio and abs(rho_ratio) > 1e-12)
                   else float("nan"))
    lag_ratios = [gammas[k + 1] / gammas[k] if abs(gammas[k]) > 1e-12 else float("nan")
                  for k in range(1, maxlag)]
    return {
        "rho_ratio": rho_ratio,
        "rho_naive": rho_naive,
        "reliability": reliability,
        "lag_ratios": lag_ratios,
        "gammas": gammas,
        "n_pairs_lag1": ns[1],
    }


def ar1_adequacy(summary: dict, tol: float = 0.1) -> bool:
    """AR(1) adequate iff the successive noise-free ratios γ(k+1)/γ(k) (k≥1) are
    all within ``tol`` of the lag2/lag1 ratio (geometric decay)."""
    ratios = [r for r in summary["lag_ratios"] if r == r]       # drop NaNs
    if len(ratios) < 2:
        return True
    return max(abs(r - ratios[0]) for r in ratios[1:]) < tol
