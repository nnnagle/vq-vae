"""
Anomaly input transform for the rethought phase pathway (Step 2).

Turns the bindings-defined phase feature ``x_it`` into the phase-encoder input: a
type-conditional **anomaly** ``a = (x − μ(z_type)) / σ(z_type)`` (mature ⇒ ``a ≈ 0``,
disturbance ⇒ large excursion) plus its **temporal difference** ``Δa`` (abruptness:
fast harvest/fire → large ``|Δa|``; slow insect → small ``|Δa|`` but sustained
``a``). μ/σ come from the Step-1 :class:`~models.mature_baseline.RFFMatureBaseline`
readout; this module is fed the readout's ``(μ, σ)`` so it stays decoupled and
unit-testable (see ``docs/phase_rethink_design.md``, Step 2).

Channel layout of the output: ``[a (C) ; Δa/scale (C)]`` → ``2C`` channels.

**Δ has its own locked robust scale.** Because the phase encoder drops per-sample
normalization (Step 3b, magnitude-preserving), ``Δa`` needs an explicit, *fixed*
scale so its dynamic range matches ``a`` for the TCN — a drifting normalizer would
destroy the depth→radius mapping. The scale is a single global robust statistic
(EMA of the per-batch median of ``‖Δa‖``) **observed during the readout-settling
warmup and locked when the anomaly input goes live** (the phase losses activate).
Before the lock the scale is identity (1.0); during warmup the output is only used
to *observe* the scale, not yet fed to the phase losses. The training loop calls
:meth:`lock_delta_scale` at the warmup boundary; it is frozen thereafter (mild
post-warmup staleness from slow z_type drift is the deliberate fixed-scale
tradeoff — re-estimate once at a checkpoint if it ever drifts badly).

Everything here is grad-free: μ/σ are stop-grad and ``x`` is data, so the anomaly
is a constant *input* to the encoder (the TCN's parameters receive gradient, the
input does not).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class AnomalyTransform(nn.Module):
    """Build ``[a ; Δa/scale]`` at anchor pixels from ``x`` and the readout's μ/σ.

    Args:
        n_channels: number of phase feature channels ``C``.
        delta_only: if True (default) emit ``[a, Δa]``; ``Δ²a`` is a later option.
        scale_decay: EMA decay for the observed Δ scale during warmup.
        eps: floor on the locked Δ scale (avoid divide-by-zero).
    """

    def __init__(
        self,
        n_channels: int,
        delta_only: bool = True,
        scale_decay: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not delta_only:
            raise NotImplementedError(
                "Δ²a channels are a deferred Step-2 option (add if the fast/slow "
                "separation in diagnostic C/D is weak); only Δ-only is implemented."
            )
        self.n_channels = n_channels
        self.delta_only = delta_only
        self.scale_decay = float(scale_decay)
        self.eps = float(eps)

        # Locked robust Δ scale (identity until locked at the warmup boundary).
        self.register_buffer("delta_scale", torch.ones(()))
        self.register_buffer("_scale_ema", torch.zeros(()))
        self.register_buffer("_scale_obs", torch.zeros((), dtype=torch.long))
        self.register_buffer("scale_locked", torch.zeros((), dtype=torch.long))

    @property
    def out_channels(self) -> int:
        return 2 * self.n_channels

    # -- Δ-scale lifecycle ---------------------------------------------------

    @torch.no_grad()
    def _observe_delta_scale(self, delta_a: torch.Tensor, valid_delta: torch.Tensor) -> None:
        """EMA of the per-batch median of ``‖Δa‖`` over valid Δ positions."""
        norms = torch.linalg.vector_norm(delta_a, dim=1)[valid_delta]   # [K]
        if norms.numel() == 0:
            return
        med = norms.median()
        d = 0.0 if int(self._scale_obs.item()) == 0 else self.scale_decay
        self._scale_ema.mul_(d).add_((1 - d) * med)
        self._scale_obs += 1

    @torch.no_grad()
    def lock_delta_scale(self) -> float:
        """Freeze the Δ scale to the observed EMA (call at the warmup boundary)."""
        if int(self._scale_obs.item()) > 0:
            self.delta_scale.copy_(torch.clamp(self._scale_ema, min=self.eps))
        self.scale_locked.fill_(1)
        return float(self.delta_scale.item())

    # -- transform -----------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: ``[N, C, T]`` phase feature at anchors.
            mu, sigma: ``[N, C]`` mature baseline from the readout (broadcast over T).
            valid: ``[N, T]`` bool timestep-validity, or None (all valid). Non-finite
                ``x`` timesteps are treated invalid regardless.
        Returns:
            features ``[N, 2C, T]`` = ``[a ; Δa/scale]`` (zeros at invalid positions),
            and ``valid_out`` ``[N, T]`` (the anomaly validity).
        """
        finite = torch.isfinite(x).all(dim=1)                       # [N, T]
        valid_out = finite if valid is None else (valid & finite)   # [N, T]

        a = (torch.nan_to_num(x) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)  # [N,C,T]
        a = torch.where(valid_out.unsqueeze(1), a, torch.zeros_like(a))

        # Δa[t] = a[t] − a[t−1]; t=0 has no predecessor; a Δ position is valid only
        # if both t and t−1 are valid.
        da = torch.zeros_like(a)
        da[:, :, 1:] = a[:, :, 1:] - a[:, :, :-1]
        valid_delta = torch.zeros_like(valid_out)
        valid_delta[:, 1:] = valid_out[:, 1:] & valid_out[:, :-1]
        da = torch.where(valid_delta.unsqueeze(1), da, torch.zeros_like(da))

        if self.training and int(self.scale_locked.item()) == 0:
            self._observe_delta_scale(da, valid_delta)

        da = da / self.delta_scale
        return torch.cat([a, da], dim=1), valid_out
