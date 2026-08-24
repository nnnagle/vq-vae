"""
Differentiable within-pixel Kalman filter + marginal NLL (phase-pathway Step-6).

A reduced-rank linear-Gaussian state-space model of forest recovery, run **per
pixel** on the type-conditional anomaly ``a`` (self-pairs only — there is no
cross-pixel coupling here; cross-pixel structure lives in the contrastive):

    state:  x_t = A x_{t-1} + w_t,   A = diag(ρ),  w_t ~ N(0, Q)   (diag)
    obs:    a_t = C x_t + v_t,                     v_t ~ N(0, R)   (diag)

with latent state ``x_t ∈ R^d`` (= z_phase; d small, e.g. 8), observation
``a_t ∈ R^{C_obs}`` (the anomaly channels), and a **shared** learned emission
``C`` (C_obs × d). ρ and Q are **type-conditional** (passed in per pixel as
``[N, d]``); a low-dim state is still expressive because its dynamics vary with
z_type. R, C, and the prior (m0, P0) are shared.

Why a filter and not the plain OU residual penalty (``losses.ou_dynamics``): the
filter integrates the latent state out, so the loss is the **marginal**
likelihood (prediction-error decomposition). That de-attenuates ρ — the plug-in
penalty ‖z_t − ρ z_{t-1}‖² is the joint/complete-data MAP and biases ρ toward 0
by the reliability ratio γ_x/(γ_x+R). The filter's gain is the optimal
data-driven drive, so there is no free "f" to trade off against ρ: the ρ↔f
identifiability degeneracy is removed by construction.

**Outward-jump gating.** Recovery is the *inward* relaxation of the anomaly
toward 0 (the AR(1) mean-reversion that identifies ρ); a disturbance is a large
*outward* jump that is not AR(1)-modelable. Disturbance years (``reset``, from
ysfc==0) restart the segment prior and are **assimilated but not scored**, so
the jump never enters the likelihood and can't inflate Q or attenuate ρ. ρ is
thus estimated from recovery transitions only — the statistically correct
"estimate the AR coefficient on the relaxation regime".

The Kalman recursion is the exact marginaliser (not an optimiser) — Adam steps
on the NLL it returns; autodiff flows through the whole filter.

Design decisions currently in force (see CLAUDE.md → phase Kalman filter):
  * phase VCR disabled (the NLL + emission prevent state collapse);
  * **filtered** state x_{t|t} returned as z_phase (RTS smoother = later revisit);
  * **ysfc** reset gate (unlabelled/innovation-threshold gate = later revisit);
  * **shared** emission C (type-conditional C = later revisit).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


_LOG_2PI = math.log(2.0 * math.pi)


def _b(x: torch.Tensor, N: int) -> torch.Tensor:
    """Broadcast a shared [d]/[C]/[d,d] param to a leading batch dim N."""
    return x.unsqueeze(0).expand(N, *x.shape)


def kalman_filter_nll(
    a: torch.Tensor,               # [N, T, Cobs] anomaly observations
    A_diag: torch.Tensor,          # [N, d] transition ρ per mode, in (0,1)
    Q_diag: torch.Tensor,          # [N, d] process-noise variances (>0)
    C: torch.Tensor,               # [Cobs, d] shared emission
    R_diag: torch.Tensor,          # [Cobs] or [N, Cobs] meas-noise variances (>0)
    m0: torch.Tensor,              # [d] or [N, d] prior mean
    P0_diag: torch.Tensor,         # [d] or [N, d] prior variances (>0)
    valid: Optional[torch.Tensor] = None,   # [N, T] bool obs-present
    reset: Optional[torch.Tensor] = None,   # [N, T] bool segment start (ysfc==0)
    jitter: float = 1e-5,
):
    """Run the per-pixel Kalman filter and return the marginal NLL.

    Returns
    -------
    nll_mean : scalar Tensor
        Mean scored-step NLL (total scored NLL / number of scored steps).
    x_filt : Tensor ``[N, T, d]``
        Filtered state estimates x_{t|t} (→ z_phase).
    diag : dict
        ``nll_total``, ``n_scored``, ``nis_mean`` (mean normalised innovation
        squared eᵀS⁻¹e — should track Cobs when Q/R are calibrated; the free
        filter-consistency / identifiability check), ``rho_mean``.

    Notes
    -----
    * t=0 is treated as a reset for every pixel (initialise from the prior,
      assimilate a_0, do not score) so the first step never scores a
      prediction from an arbitrary pre-sample.
    * Reset steps restart the prior (drop the previous segment), assimilate the
      observation (so z_phase at a disturbance year reflects the disturbance),
      and are excluded from the NLL.
    * Invalid steps skip the measurement update (state coasts on the prediction)
      and are not scored.
    """
    N, T, Cobs = a.shape
    d = A_diag.shape[1]
    device, dtype = a.device, a.dtype

    if R_diag.dim() == 1:
        R_diag = _b(R_diag, N)                        # [N, Cobs]
    if m0.dim() == 1:
        m0 = _b(m0, N)                                # [N, d]
    if P0_diag.dim() == 1:
        P0_diag = _b(P0_diag, N)                      # [N, d]

    if valid is None:
        valid = torch.ones(N, T, dtype=torch.bool, device=device)
    if reset is None:
        reset = torch.zeros(N, T, dtype=torch.bool, device=device)

    eye_d = torch.eye(d, device=device, dtype=dtype)
    eye_o = torch.eye(Cobs, device=device, dtype=dtype)
    R_mat = R_diag.unsqueeze(-1) * eye_o              # [N, Cobs, Cobs] diag

    # Prior as the "posterior at t=-1".
    x_prev = m0                                       # [N, d]
    P_prev = P0_diag.unsqueeze(-1) * eye_d            # [N, d, d]
    P0_mat = P_prev

    x_filt = torch.empty(N, T, d, device=device, dtype=dtype)
    nll_total = a.new_zeros(())
    nis_total = a.new_zeros(())
    n_scored = a.new_zeros(())

    for t in range(T):
        is_reset = reset[:, t]
        if t == 0:
            is_reset = torch.ones_like(is_reset)      # t=0 initialises from prior
        is_valid = valid[:, t]
        do_score = is_valid & ~is_reset               # [N]

        # --- predict -----------------------------------------------------
        x_pred = A_diag * x_prev                                    # [N, d]
        P_pred = A_diag.unsqueeze(-1) * P_prev * A_diag.unsqueeze(-2) \
            + Q_diag.unsqueeze(-1) * eye_d                          # [N, d, d]
        # Reset (and t=0): drop the propagated state, restart from the prior.
        rmask = is_reset.view(N, 1)
        x_pred = torch.where(rmask, m0, x_pred)
        P_pred = torch.where(rmask.view(N, 1, 1), P0_mat, P_pred)

        # --- innovation --------------------------------------------------
        Cx = torch.einsum("od,nd->no", C, x_pred)                  # [N, Cobs]
        y = a[:, t, :] - Cx                                        # [N, Cobs]
        CP = torch.einsum("od,nde->noe", C, P_pred)                # [N, Cobs, d]
        S = torch.einsum("noe,fe->nof", CP, C) + R_mat             # [N, Cobs, Cobs]
        S = S + jitter * eye_o

        L = torch.linalg.cholesky(S)                               # [N, Cobs, Cobs]
        # Kalman gain K = P_pred Cᵀ S⁻¹  via cholesky_solve on Sᵀ=S.
        M = torch.einsum("nde,oe->ndo", P_pred, C)                 # P_pred Cᵀ [N,d,Cobs]
        Kt = torch.cholesky_solve(M.transpose(-1, -2), L)          # S⁻¹ Mᵀ [N,Cobs,d]
        K = Kt.transpose(-1, -2)                                   # [N, d, Cobs]

        # --- measurement update (Joseph form for PSD) --------------------
        x_upd = x_pred + torch.einsum("ndo,no->nd", K, y)          # [N, d]
        KC = torch.einsum("ndo,oe->nde", K, C)                     # [N, d, d]
        IKC = eye_d - KC
        P_upd = torch.einsum("nde,nef,ngf->ndg", IKC, P_pred, IKC) \
            + torch.einsum("ndo,no,nfo->ndf", K, R_diag, K)        # [N, d, d]

        # Invalid obs → coast on the prediction (skip the update).
        vmask = is_valid.view(N, 1)
        x_new = torch.where(vmask, x_upd, x_pred)
        P_new = torch.where(vmask.view(N, 1, 1), P_upd, P_pred)

        # --- score (marginal NLL of the one-step prediction error) -------
        u = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)   # S⁻¹ y
        maha = (y * u).sum(-1)                                     # [N]
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        nll_t = 0.5 * (maha + logdet + Cobs * _LOG_2PI)            # [N]

        sc = do_score.to(dtype)
        nll_total = nll_total + (sc * nll_t).sum()
        nis_total = nis_total + (sc * maha).sum()
        n_scored = n_scored + sc.sum()

        x_filt[:, t, :] = x_new
        x_prev, P_prev = x_new, P_new

    denom = n_scored.clamp(min=1.0)
    nll_mean = nll_total / denom
    diag = {
        "nll_total": float(nll_total.detach()),
        "n_scored": float(n_scored.detach()),
        "nis_mean": float((nis_total / denom).detach()),
        "rho_mean": float(A_diag.detach().mean()),
    }
    return nll_mean, x_filt, diag
