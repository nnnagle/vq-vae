"""
Type-conditional differentiable Kalman filter for the phase pathway (Step 6).

Wraps ``losses.kalman_filter.kalman_filter_nll`` with the learnable, **type-
conditional** state-space parameters, so the phase pathway becomes a reduced-rank
linear-Gaussian filter run per pixel on the anomaly:

    x_t = diag(ρ(z_type)) x_{t-1} + w_t,  w_t ~ N(0, Q(z_type))
    a_t = C x_t + v_t,                    v_t ~ N(0, R)     (shared C, R)

``z_phase = filtered state x_{t|t}``; the loss is the marginal NLL (de-attenuated
ρ). See CLAUDE.md → "Phase pathway: differentiable Kalman filter".

Decisions in force: shared emission ``C`` and measurement noise ``R`` (type-
conditional C = later revisit); ρ and Q are type-conditional heads off
**detached** z_type (stop-grad, as everywhere in the phase pathway); prior mean
``m0`` fixed at 0 (mature ⇒ anomaly 0 ⇒ state 0); prior variance ``P0`` learned
and diffuse (disturbed states are assimilated from data at a reset). The ρ head
is initialised so every pixel starts at ``rho_init`` (bias = logit(rho_init),
weight = 0) and learns type-variation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.kalman_filter import kalman_filter_nll


def _inv_sigmoid(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-4)
    return math.log(math.expm1(y)) if y < 20 else y


class PhaseKalman(nn.Module):
    """Learnable type-conditional AR(1)+noise filter over the anomaly.

    Args:
        z_type_dim: dim of z_type (the conditioning input).
        n_obs: number of anomaly channels ``C`` (the observation dim).
        state_dim: latent state dim (= z_phase_dim).
        rho_init: initial recovery persistence ρ (uniform across pixels at init).
            Default 0.861 = 0.05**(1/20) (within 5% of mature in 20 annual steps).
        q_init, r_init, p0_init: initial process / measurement / prior variances.
        floor: positivity floor added after softplus on Q, R, P0.
    """

    def __init__(
        self,
        z_type_dim: int,
        n_obs: int,
        state_dim: int,
        rho_init: float = 0.861,
        q_init: float = 0.1,
        r_init: float = 0.25,
        p0_init: float = 25.0,
        floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_obs = n_obs
        self.state_dim = state_dim
        self.floor = float(floor)

        # Type-conditional heads (off detached z_type). Init: bias = target,
        # weight = 0 → uniform at init, learns type-variation.
        self.rho_head = nn.Linear(z_type_dim, state_dim)
        nn.init.zeros_(self.rho_head.weight)
        nn.init.constant_(self.rho_head.bias, _inv_sigmoid(rho_init))

        self.q_head = nn.Linear(z_type_dim, state_dim)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, _inv_softplus(q_init))

        # Shared emission C [n_obs, state_dim] and measurement noise R [n_obs].
        self.C = nn.Parameter(torch.randn(n_obs, state_dim) / math.sqrt(state_dim))
        self.r_raw = nn.Parameter(torch.full((n_obs,), _inv_softplus(r_init)))

        # Prior: mean fixed at 0 (mature); variance learned + diffuse.
        self.register_buffer("m0", torch.zeros(state_dim))
        self.p0_raw = nn.Parameter(torch.full((state_dim,), _inv_softplus(p0_init)))

    # -- parameter views ----------------------------------------------------

    def rho(self, z_type: torch.Tensor) -> torch.Tensor:
        # clamp < 1 for a strictly stable, mean-reverting recovery.
        return torch.sigmoid(self.rho_head(z_type.detach())).clamp(max=0.999)

    def q(self, z_type: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.q_head(z_type.detach())) + self.floor

    @property
    def r(self) -> torch.Tensor:
        return F.softplus(self.r_raw) + self.floor

    @property
    def p0(self) -> torch.Tensor:
        return F.softplus(self.p0_raw) + self.floor

    # -- reset mask from ysfc ----------------------------------------------

    @staticmethod
    def reset_from_ysfc(ysfc: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Segment-start (reset) mask [N,T]: disturbance year (ysfc==0) or a ysfc
        decrease (a new recovery sequence begins). t=0 is handled by the filter."""
        reset = torch.zeros_like(valid, dtype=torch.bool)
        if ysfc.shape[1] > 1:
            y = torch.nan_to_num(ysfc, nan=1e9)
            reset[:, 1:] = (y[:, 1:] == 0) | (y[:, 1:] < y[:, :-1])
        return reset

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        a: torch.Tensor,        # [N, n_obs, T] anomaly (the `a` block, not Δa)
        z_type: torch.Tensor,   # [N, z_type_dim]
        ysfc: torch.Tensor,     # [N, T] raw years since change
        valid: torch.Tensor,    # [N, T] bool timestep validity
    ):
        """Returns ``(z_phase [N, T, state_dim], nll scalar, diag dict)``."""
        rho = self.rho(z_type)                        # [N, state]
        Q = self.q(z_type)                            # [N, state]
        reset = self.reset_from_ysfc(ysfc, valid)     # [N, T]
        a_ntc = a.permute(0, 2, 1).contiguous()       # [N, T, n_obs]

        nll, x_filt, diag = kalman_filter_nll(
            a_ntc, A_diag=rho, Q_diag=Q, C=self.C, R_diag=self.r,
            m0=self.m0, P0_diag=self.p0, valid=valid, reset=reset,
        )
        total = float(valid.sum().clamp(min=1))
        diag["scored_frac"] = diag["n_scored"] / total
        diag["nis_target"] = float(self.n_obs)        # NIS should track n_obs
        return x_filt, nll, diag
