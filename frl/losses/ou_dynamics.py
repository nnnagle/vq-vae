"""
Within-pixel Gaussian Ornstein–Uhlenbeck dynamics loss (Step 4).

Models recovery as a discrete OU process mean-reverting to the single shared
origin: ``z_phase[t] | z_phase[t-1] ~ N(ρ·z_phase[t-1], s²(‖z_{t-1}‖)·I)`` with a
**scalar global** ``ρ ∈ (0,1)``. The loss is the transition NLL on **non-disturbance**
steps only (disturbance kicks are exogenous → gated out by a Δ-based gate). One term
subsumes smoothness (small step), radius contraction (ρ<1 → recovery ordering) and
direction preservation (target points along ``z_{t-1}`` → straight rays). See
``docs/phase_rethink_design.md`` Step 4.

``s`` (noise scale) is isotropic, global, radius-shrinking ``s²(r)=s0²+s1²·r²``. The
default is **fixed** ``s`` with ``s1=0``, so the NLL reduces to the weighted squared
residual (the ``log s`` term is constant and drops) — this avoids the ``z→0`` /
``s→0`` collapse magnet a learned ``s`` introduces. Enable ``learn_s`` / ``s1`` for a
tighter radius-shrinking basin once anti-collapse (Step 5 contrastive) is in place.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class OUDynamics(nn.Module):
    """Learnable ρ (and optionally s) + the OU transition-NLL loss.

    Args:
        rho_init: initial contraction ρ ∈ (0,1) (a single global scalar).
        learn_rho: if True, ρ is a learned sigmoid-constrained scalar.
        s0: base noise scale (also the maturity/basin scale). Fixed unless learn_s.
        s1: radius-growth coefficient (0 = homoscedastic). Fixed unless learn_s.
        learn_s: if True, s0/s1 become learned (softplus) params and the ``log s``
            NLL term is included. Default False → weighted squared residual only.
        tau: disturbance-gate scale on ``‖Δa‖`` (in the locked Δ-scale units;
            stable ‖Δa‖≈1, disturbance ≫1). Gate ``exp(-(‖Δa‖/τ)²)``.
    """

    def __init__(
        self,
        rho_init: float = 0.9,
        learn_rho: bool = True,
        s0: float = 0.5,
        s1: float = 0.0,
        learn_s: bool = False,
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        self.learn_s = bool(learn_s)
        self.register_buffer("tau", torch.tensor(float(tau)))

        if learn_rho:
            self._rho_logit = nn.Parameter(torch.tensor(_inv_sigmoid(rho_init)))
        else:
            self.register_buffer("_rho_fixed", torch.tensor(float(rho_init)))
        self._learn_rho = bool(learn_rho)

        if learn_s:
            # softplus params so s0,s1 ≥ 0; init to the requested values.
            self._s0_raw = nn.Parameter(torch.tensor(_inv_softplus(s0)))
            self._s1_raw = nn.Parameter(torch.tensor(_inv_softplus(max(s1, 1e-4))))
        else:
            self.register_buffer("_s0_fixed", torch.tensor(float(s0)))
            self.register_buffer("_s1_fixed", torch.tensor(float(s1)))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self._rho_logit) if self._learn_rho else self._rho_fixed

    @property
    def s0(self) -> torch.Tensor:
        return F.softplus(self._s0_raw) + 1e-3 if self.learn_s else self._s0_fixed

    @property
    def s1(self) -> torch.Tensor:
        return F.softplus(self._s1_raw) if self.learn_s else self._s1_fixed

    def forward(
        self,
        z_phase: torch.Tensor,
        delta_a_norm: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ):
        """OU transition NLL over non-disturbance, valid transitions.

        Args:
            z_phase: ``[N, T, D]`` per-timestep phase embeddings.
            delta_a_norm: ``[N, T]`` ``‖Δa[t]‖`` (disturbance signal; the anomaly
                transform's Δ block norm).
            valid: ``[N, T]`` bool timestep validity (or None → all valid).
        Returns:
            (loss scalar, diag dict).
        """
        N, T, D = z_phase.shape
        if T < 2:
            z = z_phase.sum() * 0.0
            return z, {"gate_mean": 0.0, "resid_rms": 0.0, "rho": float(self.rho),
                       "n_eff": 0.0}

        rho = self.rho
        z_prev = z_phase[:, :-1, :]                     # [N, T-1, D]
        z_next = z_phase[:, 1:, :]                      # [N, T-1, D]
        r2 = (z_next - rho * z_prev).pow(2).sum(-1)     # [N, T-1]
        r_prev2 = z_prev.pow(2).sum(-1)                 # [N, T-1] = ‖z_{t-1}‖²
        s2 = self.s0 ** 2 + self.s1 ** 2 * r_prev2      # [N, T-1]

        if self.learn_s:
            nll = r2 / (2.0 * s2) + 0.5 * D * torch.log(2 * math.pi * s2)
        else:
            # s constant → 1/(2s²) is a global scale folded into the loss weight,
            # log-s term constant → dropped. Just the weighted squared residual.
            nll = r2

        # Disturbance gate on the transition target step (t = the "next" index).
        da = delta_a_norm[:, 1:]                        # [N, T-1]
        w = torch.exp(-(da / self.tau) ** 2)            # [N, T-1]
        if valid is not None:
            w = w * (valid[:, 1:] & valid[:, :-1]).to(w.dtype)

        denom = w.sum().clamp(min=1.0)
        loss = (w * nll).sum() / denom
        with torch.no_grad():
            diag = {
                "gate_mean": (w.sum() / (w.numel() or 1)).item(),
                "resid_rms": ((w * r2).sum() / denom).sqrt().item(),
                "rho": float(rho),
                "s0": float(self.s0),
                "n_eff": float(denom),
            }
        return loss, diag


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-4)
    # inverse of softplus: x = log(exp(y) - 1)
    return math.log(math.expm1(y)) if y < 20 else y
