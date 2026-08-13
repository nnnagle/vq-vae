"""Tests for the FiLM-free phase pathway (Step 3b model surgery).

Covers RepresentationModel.forward_phase_at_locations with the μ/σ readout +
anomaly transform wired in, FiLM removed, and a magnitude-preserving TCN norm.
"""

from __future__ import annotations

import pytest
import torch

from models.representation import RepresentationModel


def _model(**kw):
    torch.manual_seed(0)
    return RepresentationModel(
        type_in_channels=5,
        phase_in_channels=4,
        z_type_dim=8,
        z_phase_dim=6,
        type_encoder_channels=[16, 8],
        phase_tcn_channels=[8, 8],
        phase_tcn_dilations=[1, 2],
        phase_tcn_norm="none",
        rff_features=64,
        **kw,
    )


class TestPhaseForward:
    def test_output_shape(self):
        m = _model()
        m.mature_baseline.update(torch.randn(200, 8), torch.randn(200, 4))
        z = m.forward_phase_at_locations(torch.randn(12, 4, 10), torch.randn(12, 8))
        assert z.shape == (12, 10, 6)          # [N, T, z_phase_dim]

    def test_return_input_is_2C(self):
        m = _model()
        m.mature_baseline.update(torch.randn(50, 8), torch.randn(50, 4))
        z, feats = m.forward_phase_at_locations(
            torch.randn(5, 4, 10), torch.randn(5, 8), return_input=True
        )
        assert feats.shape == (5, 8, 10)       # [N, 2C, T] = [a ; Δa]

    def test_film_removed_readout_transform_are_submodules(self):
        m = _model()
        assert not hasattr(m, "phase_film")
        mods = dict(m.named_modules())
        assert "mature_baseline" in mods and "anomaly_transform" in mods
        # TCN input is 2*C (anomaly a + Δa)
        assert m.phase_tcn.in_channels == 8

    def test_grad_flows_to_tcn_not_ztype(self):
        """z_phase trains the TCN; the anomaly input is a grad-free constant, so
        no gradient reaches z_type (stop-grad by construction)."""
        m = _model()
        m.mature_baseline.update(torch.randn(100, 8), torch.randn(100, 4))
        z_type = torch.randn(6, 8, requires_grad=True)
        z = m.forward_phase_at_locations(torch.randn(6, 4, 10), z_type)
        z.pow(2).sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.phase_tcn.parameters()
        )
        assert z_type.grad is None             # anomaly is detached → no leak

    def test_dense_forward_phase_raises(self):
        m = _model()
        with pytest.raises(NotImplementedError):
            m.forward_phase(torch.randn(1, 4, 10, 3, 3), torch.randn(1, 8, 3, 3))

    def test_magnitude_survives_depth(self):
        """Deeper input excursion → larger ‖z_phase‖ (no per-sample norm dividing
        depth out). Rough monotonicity check on the FiLM-free pathway."""
        m = _model()
        m.mature_baseline.update(torch.randn(400, 8), 0.1 * torch.randn(400, 4))
        z_type = torch.randn(16, 8)
        shallow = 0.3 * torch.ones(16, 4, 10)
        deep = 3.0 * torch.ones(16, 4, 10)
        n_shallow = m.forward_phase_at_locations(shallow, z_type).norm(dim=-1).mean()
        n_deep = m.forward_phase_at_locations(deep, z_type).norm(dim=-1).mean()
        assert n_deep > n_shallow
