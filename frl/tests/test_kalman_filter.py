"""Tests for losses.kalman_filter — the differentiable within-pixel filter + NLL.

Covers: (1) correctness vs an independent scalar AR(1)+noise filter, (2) autograd
vs finite-difference gradient wrt ρ, (3) the payoff — the filter NLL recovers ρ
while the naive plug-in / lag-1 estimate attenuates it, (4) outward-jump gating,
(5) masks / resets / batch independence. Tests 3–4 print the numbers they assert
so ``pytest -s`` doubles as the Phase-1 evaluation log.
"""

from __future__ import annotations

import math

import torch

from losses.kalman_filter import kalman_filter_nll


def _scalar_ref_nll(a, rho, q, r, m0, p0):
    """Independent textbook scalar AR(1)+noise filter NLL (t=0 = unscored init)."""
    T = a.shape[0]
    x, P = m0, p0
    nll = 0.0
    for t in range(T):
        if t == 0:
            xp, Pp = m0, p0
        else:
            xp, Pp = rho * x, rho * rho * P + q
        S = Pp + r
        y = a[t] - xp
        K = Pp / S
        x = xp + K * y
        P = (1 - K) * Pp
        if t > 0:
            nll += 0.5 * (y * y / S + math.log(S) + math.log(2 * math.pi))
    return nll / (T - 1)


def _ar1_noise_series(T, rho, q, r, seed, n=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T)
    x[:, 0] = torch.randn(n, generator=g) * math.sqrt(q / (1 - rho * rho))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + torch.randn(n, generator=g) * math.sqrt(q)
    a = x + torch.randn(n, T, generator=g) * math.sqrt(r)
    return a  # [n, T]


def _run(a, rho, q, r, m0=0.0, p0=1.0, valid=None, reset=None):
    """Scalar (d=Cobs=1) filter wrapper: a is [N, T]."""
    N, T = a.shape
    return kalman_filter_nll(
        a.unsqueeze(-1),
        A_diag=torch.full((N, 1), float(rho)),
        Q_diag=torch.full((N, 1), float(q)),
        C=torch.ones(1, 1),
        R_diag=torch.full((1,), float(r)),
        m0=torch.full((1,), float(m0)),
        P0_diag=torch.full((1,), float(p0)),
        valid=valid, reset=reset,
    )


class TestCorrectness:
    def test_matches_scalar_reference(self):
        a = _ar1_noise_series(40, 0.8, 0.3, 0.5, seed=0)   # [1, 40]
        nll, x_filt, diag = _run(a, 0.8, 0.3, 0.5)
        ref = _scalar_ref_nll(a[0], 0.8, 0.3, 0.5, 0.0, 1.0)
        assert abs(float(nll) - ref) < 1e-4
        assert x_filt.shape == (1, 40, 1)
        assert diag["n_scored"] == 39          # t=0 unscored

    def test_batch_independence(self):
        a = _ar1_noise_series(30, 0.7, 0.2, 0.4, seed=1, n=5)
        nll_batch, _, _ = _run(a, 0.7, 0.2, 0.4)
        per = [float(_run(a[i:i + 1], 0.7, 0.2, 0.4)[0]) for i in range(5)]
        assert abs(float(nll_batch) - sum(per) / len(per)) < 1e-5

    def test_multi_dim_runs_and_is_finite(self):
        N, T, d, Cobs = 4, 15, 3, 6
        a = torch.randn(N, T, Cobs)
        nll, x_filt, diag = kalman_filter_nll(
            a, A_diag=torch.full((N, d), 0.9), Q_diag=torch.full((N, d), 0.2),
            C=torch.randn(Cobs, d), R_diag=torch.full((Cobs,), 0.5),
            m0=torch.zeros(d), P0_diag=torch.ones(d),
        )
        assert torch.isfinite(nll) and x_filt.shape == (N, T, d)
        assert math.isfinite(diag["nis_mean"])


class TestGradient:
    def test_autograd_matches_finite_difference_in_rho(self):
        a = _ar1_noise_series(50, 0.75, 0.3, 0.4, seed=2)
        rho = torch.tensor(0.6, requires_grad=True)

        def loss_of(rho_t):
            return kalman_filter_nll(
                a.unsqueeze(-1), A_diag=rho_t.view(1, 1),
                Q_diag=torch.full((1, 1), 0.3), C=torch.ones(1, 1),
                R_diag=torch.full((1,), 0.4), m0=torch.zeros(1),
                P0_diag=torch.ones(1))[0]

        loss_of(rho).backward()
        g_auto = float(rho.grad)
        eps = 1e-4
        with torch.no_grad():
            g_fd = (float(loss_of(torch.tensor(0.6 + eps)))
                    - float(loss_of(torch.tensor(0.6 - eps)))) / (2 * eps)
        assert abs(g_auto - g_fd) < 1e-2


class TestDeAttenuation:
    """The payoff: filter NLL recovers ρ; naive lag-1 of the noisy obs attenuates."""

    def test_filter_recovers_rho_naive_attenuates(self, capsys):
        true_rho, q, r = 0.85, 0.25, 0.8          # sizeable measurement noise
        # Batch of pixels (short T, many N) — same data volume, fast T-loop.
        a = _ar1_noise_series(80, true_rho, q, r, seed=7, n=300)   # [300, 80]
        N = a.shape[0]

        # Naive pooled lag-1 autocorrelation of the noisy observation (attenuated
        # by the reliability ratio γ_x/(γ_x+r)).
        s = a - a.mean(dim=1, keepdim=True)
        naive = float((s[:, 1:] * s[:, :-1]).sum() / (s * s).sum())

        # Fit a single shared ρ by minimising the filter NLL (Q, R known).
        rho = torch.tensor(_inv_sigmoid(0.5), requires_grad=True)
        opt = torch.optim.Adam([rho], lr=0.05)
        for _ in range(250):
            opt.zero_grad()
            nll = kalman_filter_nll(
                a.unsqueeze(-1), A_diag=torch.sigmoid(rho).expand(N, 1),
                Q_diag=torch.full((N, 1), q), C=torch.ones(1, 1),
                R_diag=torch.full((1,), r), m0=torch.zeros(1),
                P0_diag=torch.full((1,), q / (1 - 0.85 ** 2)))[0]
            nll.backward()
            opt.step()
        fit = float(torch.sigmoid(rho.detach()))

        print(f"\n[de-attenuation] true ρ={true_rho:.3f}  "
              f"filter ρ̂={fit:.3f}  naive lag-1={naive:.3f}  "
              f"(attenuation gap={fit - naive:.3f})")
        assert abs(fit - true_rho) < 0.05          # filter is ~unbiased
        assert naive < true_rho - 0.05             # naive is attenuated
        assert fit - naive > 0.05                  # filter beats naive


class TestOutwardJumpGating:
    def test_reset_excludes_the_jump_from_the_nll(self, capsys):
        # A recovery segment, then a disturbance (outward jump), then recovery.
        true_rho, q, r = 0.8, 0.2, 0.3
        a = _ar1_noise_series(60, true_rho, q, r, seed=3)      # [1, 60]
        a[0, 30] += 15.0                                       # outward jump
        a[0, 31] += 12.0
        reset = torch.zeros(1, 60, dtype=torch.bool)
        reset[0, 30] = True                                    # ysfc==0 marks it

        nll_gated, _, dg = _run(a, true_rho, q, r, reset=reset)
        nll_ungated, _, du = _run(a, true_rho, q, r)

        print(f"\n[gating] NLL ungated={float(nll_ungated):.3f}  "
              f"gated={float(nll_gated):.3f}  scored steps "
              f"{du['n_scored']:.0f}→{dg['n_scored']:.0f}")
        # Gating drops the contaminated step from the likelihood → lower NLL and
        # one fewer scored step; the huge jump no longer dominates.
        assert dg["n_scored"] == du["n_scored"] - 1
        assert float(nll_gated) < float(nll_ungated)

    def test_gated_fit_beats_ungated_fit_under_jumps(self):
        """Piecewise recovery: each disturbance restarts a decaying AR(1) segment
        from a large disturbed value. Gating (reset at each segment start) scores
        only within-segment recovery → recovers ρ; not gating also scores the
        outward jump *into* each segment → biases ρ."""
        true_rho, q, r, x0, seg = 0.85, 0.2, 0.3, 15.0, 16
        N, T = 200, 80
        g = torch.Generator().manual_seed(5)
        x = torch.zeros(N, T)
        reset = torch.zeros(N, T, dtype=torch.bool)
        for start in range(0, T, seg):
            reset[:, start] = True
            x[:, start] = x0 + torch.randn(N, generator=g)       # disturbed init
            for t in range(start + 1, min(start + seg, T)):
                x[:, t] = true_rho * x[:, t - 1] \
                    + torch.randn(N, generator=g) * math.sqrt(q)
        a = x + torch.randn(N, T, generator=g) * math.sqrt(r)

        def fit(use_reset):
            rho = torch.tensor(_inv_sigmoid(0.5), requires_grad=True)
            opt = torch.optim.Adam([rho], lr=0.05)
            rm = reset if use_reset else None
            for _ in range(200):
                opt.zero_grad()
                nll = kalman_filter_nll(
                    a.unsqueeze(-1), A_diag=torch.sigmoid(rho).expand(N, 1),
                    Q_diag=torch.full((N, 1), q), C=torch.ones(1, 1),
                    R_diag=torch.full((1,), r), m0=torch.zeros(1),
                    P0_diag=torch.full((1,), 100.0), reset=rm)[0]   # diffuse reset prior
                nll.backward()
                opt.step()
            return float(torch.sigmoid(rho.detach()))

        gated, ungated = fit(True), fit(False)
        assert abs(gated - true_rho) < 0.05                        # gating recovers ρ
        assert abs(gated - true_rho) < abs(ungated - true_rho)     # and beats ungated


class TestMasks:
    def test_invalid_step_coasts_and_is_unscored(self):
        a = _ar1_noise_series(20, 0.8, 0.2, 0.3, seed=4)
        valid = torch.ones(1, 20, dtype=torch.bool)
        valid[0, 10] = False
        nll, x_filt, diag = _run(a, 0.8, 0.2, 0.3, valid=valid)
        assert diag["n_scored"] == 18          # t=0 and the invalid step unscored
        assert torch.isfinite(x_filt).all()


def _inv_sigmoid(p):
    return math.log(p / (1 - p))
