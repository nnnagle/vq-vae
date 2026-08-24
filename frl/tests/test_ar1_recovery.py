"""Tests for analysis.ar1_recovery — the Phase-0 AR(1)+noise recovery estimator.

Confirms: (1) γ2/γ1 recovers ρ de-attenuated while γ1/γ0 is attenuated by
measurement noise; (2) the reliability ratio is recovered; (3) segment resets
keep pairs from crossing outward jumps; (4) the AR-order adequacy check fires on
AR(2). Prints the recovered numbers so ``pytest -s`` is the Phase-0 eval log.
"""

from __future__ import annotations

import math

import numpy as np

from analysis.ar1_recovery import summarize_ar1, ar1_adequacy, pooled_autocov


def _ar1_noise(P, T, rho, q, r, seed):
    g = np.random.default_rng(seed)
    x = np.zeros((P, T))
    x[:, 0] = g.normal(0, math.sqrt(q / (1 - rho * rho)), P)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + g.normal(0, math.sqrt(q), P)
    return x + g.normal(0, math.sqrt(r), (P, T))


def test_ratio_deattenuates_naive_attenuates(capsys):
    rho, q, r = 0.85, 0.3, 0.9
    a = _ar1_noise(400, 60, rho, q, r, seed=0)
    s = summarize_ar1(a)
    print(f"\n[ar1] true ρ={rho:.3f}  γ2/γ1={s['rho_ratio']:.3f}  "
          f"naive γ1/γ0={s['rho_naive']:.3f}  reliability={s['reliability']:.3f}")
    assert abs(s["rho_ratio"] - rho) < 0.05        # de-attenuated
    assert s["rho_naive"] < rho - 0.05             # attenuated
    # reliability = γ_x/(γ_x+R); with these q,r ≈ 0.65
    assert 0.4 < s["reliability"] < 0.9


def test_reset_recovers_rho_across_outward_jumps():
    # Piecewise recovery: each segment restarts high (disturbed) and decays.
    # The cross-segment transition is an outward jump; without a reset it
    # contaminates ρ̂, with the reset it is excluded and ρ̂ is recovered.
    rho, q, r, x0, seg = 0.8, 0.2, 0.3, 15.0, 20
    P, T = 200, 80
    g = np.random.default_rng(1)
    x = np.zeros((P, T))
    reset = np.zeros((P, T), dtype=bool)
    for start in range(0, T, seg):
        reset[:, start] = True
        x[:, start] = x0 + g.normal(0, 1, P)
        for t in range(start + 1, min(start + seg, T)):
            x[:, t] = rho * x[:, t - 1] + g.normal(0, math.sqrt(q), P)
    a = x + g.normal(0, math.sqrt(r), (P, T))

    _, n_no = pooled_autocov(a, 1, reset=None)
    _, n_yes = pooled_autocov(a, 1, reset=reset)
    assert n_no - n_yes == P * (T // seg - 1)       # boundary pairs removed

    rho_reset = summarize_ar1(a, reset=reset)["rho_ratio"]
    rho_none = summarize_ar1(a, reset=None)["rho_ratio"]
    assert abs(rho_reset - rho) < 0.05              # gating recovers ρ
    assert abs(rho_reset - rho) < abs(rho_none - rho)


def test_ar_order_flags_ar2():
    # AR(2) with a complex/decaying-oscillatory character → non-geometric
    # autocovariance → adequacy check should fail.
    P, T = 300, 80
    g = np.random.default_rng(2)
    phi1, phi2 = 0.6, -0.5
    x = np.zeros((P, T))
    for t in range(2, T):
        x[:, t] = phi1 * x[:, t - 1] + phi2 * x[:, t - 2] + g.normal(0, 0.3, P)
    ar1 = _ar1_noise(P, T, 0.85, 0.2, 0.1, seed=3)
    assert ar1_adequacy(summarize_ar1(ar1)) is True
    assert ar1_adequacy(summarize_ar1(x)) is False


def test_empty_and_short_are_safe():
    a = np.zeros((3, 1))
    g, n = pooled_autocov(a, 1)
    assert g == 0.0 and n == 0
