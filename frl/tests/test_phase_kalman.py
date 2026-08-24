"""Tests for models.phase_kalman.PhaseKalman and the RepresentationModel wiring.

Covers: ρ-head seed (uniform rho_init at construction), forward shapes + finite
NLL + diagnostics, gradient flow to all state-space params, the ysfc reset mask,
and the model-level ``phase_kalman_forward`` path (anomaly a-block → filter).
"""

from __future__ import annotations

import math

import torch

from models.phase_kalman import PhaseKalman
from models.representation import RepresentationModel


def _kalman(dt=8, C=5, d=6, **kw):
    torch.manual_seed(0)
    return PhaseKalman(z_type_dim=dt, n_obs=C, state_dim=d, **kw)


class TestSeedAndParams:
    def test_rho_seed_is_uniform_rho_init(self):
        pk = _kalman(rho_init=0.861)
        rho = pk.rho(torch.randn(20, 8))               # [20, 6]
        # weight=0 at init → every pixel/mode == rho_init.
        assert torch.allclose(rho, torch.full_like(rho, 0.861), atol=1e-3)

    def test_default_rho_init_matches_20yr_5pct(self):
        # 0.05 ** (1/20) ≈ 0.861 — recorded design seed.
        assert abs(0.05 ** (1 / 20) - 0.861) < 1e-3
        pk = _kalman()
        assert abs(float(pk.rho(torch.zeros(1, 8)).mean().detach()) - 0.861) < 1e-3

    def test_positive_variances(self):
        pk = _kalman()
        assert (pk.q(torch.randn(4, 8)) > 0).all()
        assert (pk.r > 0).all() and (pk.p0 > 0).all()


class TestForward:
    def _inputs(self, N=16, C=5, T=12):
        a = torch.randn(N, C, T)
        z_type = torch.randn(N, 8)
        ysfc = torch.arange(T).float().expand(N, T).clone()   # rising = recovering
        valid = torch.ones(N, T, dtype=torch.bool)
        return a, z_type, ysfc, valid

    def test_shapes_and_diag(self):
        pk = _kalman()
        a, z_type, ysfc, valid = self._inputs()
        z_phase, nll, diag = pk(a, z_type, ysfc, valid)
        assert z_phase.shape == (16, 12, 6)
        assert torch.isfinite(nll)
        assert abs(diag["rho_mean"] - 0.861) < 1e-3
        assert math.isfinite(diag["nis_mean"])
        assert 0.0 < diag["scored_frac"] <= 1.0
        assert diag["nis_target"] == 5.0

    def test_gradients_flow_to_all_params(self):
        pk = _kalman()
        a, z_type, ysfc, valid = self._inputs()
        _, nll, _ = pk(a, z_type, ysfc, valid)
        nll.backward()
        for name, p in pk.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert p.grad.abs().sum() > 0, f"zero grad for {name}"

    def test_reset_from_ysfc_marks_disturbances(self):
        valid = torch.ones(2, 6, dtype=torch.bool)
        ysfc = torch.tensor([[3., 4., 0., 1., 2., 3.],     # disturbance (ysfc==0) at t=2
                             [1., 2., 3., 4., 5., 6.]])    # clean recovery, no reset
        reset = PhaseKalman.reset_from_ysfc(ysfc, valid)
        assert reset[0, 2].item() is True                  # ysfc drop 4→0
        assert not reset[0, 0].item()                      # t=0 handled by the filter
        assert not reset[1].any().item()                   # monotone rise = no reset


class TestModelWiring:
    def _model(self, enabled):
        torch.manual_seed(0)
        return RepresentationModel(
            type_in_channels=5, phase_in_channels=4,
            z_type_dim=8, z_phase_dim=6,
            type_encoder_channels=[16, 8],
            phase_tcn_channels=[8, 8], phase_tcn_dilations=[1, 2],
            phase_tcn_norm="none", rff_features=64,
            phase_kalman_enabled=enabled,
        )

    def test_disabled_by_default(self):
        assert self._model(enabled=False).phase_kalman is None

    def test_phase_kalman_forward_from_anomaly(self):
        m = self._model(enabled=True)
        assert m.phase_kalman is not None
        m.mature_baseline.update(torch.randn(200, 8), torch.randn(200, 4))
        N, C, T = 10, 4, 12
        # Build the anomaly the way step.py does, then run the filter path.
        x = torch.randn(N, C, T)
        z_type = torch.randn(N, 8)
        mu, sigma = m.mature_baseline.predict(z_type)
        feats, valid = m.anomaly_transform(x, mu, sigma)     # [N, 2C, T]
        ysfc = torch.arange(T).float().expand(N, T).clone()
        z_phase, nll, diag = m.phase_kalman_forward(feats, z_type, ysfc, valid)
        assert z_phase.shape == (N, T, 6)
        assert torch.isfinite(nll)

    def test_forward_raises_when_disabled(self):
        m = self._model(enabled=False)
        try:
            m.phase_kalman_forward(torch.randn(2, 8, 5), torch.randn(2, 8),
                                   torch.zeros(2, 5), torch.ones(2, 5, dtype=torch.bool))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
