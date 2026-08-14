"""
Mature-baseline estimator for the rethought phase pathway (Step 1).

Estimates, for any pixel's type embedding ``z_type``, the per-channel mean
``μ(z_type)`` and scale ``σ(z_type)`` of **mature** forest of that type. These
define the type-conditional anomaly input ``a = (x − μ(z_type)) / σ(z_type)`` that
the redesigned phase encoder consumes (see ``docs/phase_rethink_design.md``, Step 1).

Form (locked): a **Random Fourier Features (RFF)** readout of standardized,
detached ``z_type`` — a fixed, centerless random basis
``φ(z) = √(2/D)·cos(Ω z + b)`` with ``Ω ~ N(0, h⁻²)`` drawn once. Chosen over RBF
(data-centers go stale under a drifting z_type) and a Lipschitz-MLP (unbounded
extrapolation): the basis is smooth global sinusoids, and the ridge readout reverts
toward a flat prior where mature data is sparse.

Fit: a pure **online conditional-moment estimator** — no gradient parameters. It
keeps EMA sufficient statistics over mature anchors and solves a centered kernel
ridge for the conditional mean ``μ = E[x|z]``; the scale comes from a **second
ridge on the squared residuals** ``E[(x − μ)²|z]`` (floored). Both reuse the same
regularized feature-covariance ``(Ac + λI)``.

Two properties worth knowing (both found empirically while implementing; see the
Step-1 tests):

* **``D`` (n_features) controls revert-to-prior sharpness.** RFF features are
  periodic, so with ``D`` too small a query far from the mature data is *not* far
  in feature space and only partially reverts to the prior. Larger ``D`` (→ a
  closer Gaussian-kernel approximation) sharpens both the revert-to-prior of μ/σ
  and the leverage calibration below. Default ``D`` is generous for this reason.
* **Scale via a residual-variance ridge, not ``E[x²]−μ²``.** Estimating the
  variance as a difference of two separately-fit moments collapses (σ→floor) when
  μ is under-regularized; fitting a ridge directly to the squared residuals of the
  mean fit is numerically stable and recovers the true per-channel noise.

Deviation from the locked spec (σ): Step 1 locked "closed-form μ + **gradient** σ".
While implementing, the conditional variance turned out to be closed-form too (a
ridge on squared residuals — same machinery, no gradient head, no extra loss term,
and it reverts to the prior variance far from data). This module implements the
closed-form σ; a gradient σ head is a small addition if preferred. Flagged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RFFMatureBaseline(nn.Module):
    """Online RFF conditional-moment estimator of mature μ(z_type), σ(z_type).

    All state is held in buffers (fixed random basis + EMA sufficient statistics),
    so the module carries **no gradient parameters** and is inherently stop-grad
    w.r.t. any downstream loss: it is updated only by :meth:`update` folding mature
    samples into the running statistics.

    Args:
        in_dim: dimension ``d`` of ``z_type`` (e.g. ``z_type_dim``).
        n_channels: number of phase feature channels ``C`` to normalize.
        n_features: RFF dimension ``D``. Larger = closer to the exact Gaussian
            kernel = sharper revert-to-prior / leverage (see module docstring).
        bandwidth: RFF length-scale ``h`` in **standardized z_type** units
            (``Ω ~ N(0, h⁻²)``); larger ``h`` = smoother μ/σ. The Step-1 knob.
        ridge_lambda: ridge ``λ`` on the centered feature covariance. Scale is
            relative to the feature covariance (trace ≈ 1 for normalized ``φ``);
            tune together with ``bandwidth`` by held-out (val-split) NLL.
        sigma_floor: lower bound on ``σ`` (and thus on the prior scale). Must be
            > 0 so the anomaly divide is safe.
        ema_decay: EMA decay for the sufficient statistics and the internal
            z_type standardization (closer to 1 = slower / smoother tracking).
        standardize: if True, maintain running mean/var of ``z_type`` and
            standardize before the RFF map (recommended — z_type is unconstrained
            in magnitude per CLAUDE.md).
        seed: RNG seed for the fixed ``Ω, b`` (reproducible basis).
    """

    def __init__(
        self,
        in_dim: int,
        n_channels: int,
        n_features: int = 1024,
        bandwidth: float = 1.0,
        ridge_lambda: float = 1e-3,
        sigma_floor: float = 1e-2,
        ema_decay: float = 0.99,
        standardize: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if sigma_floor <= 0:
            raise ValueError("sigma_floor must be > 0 (anomaly divides by σ)")
        if not (0.0 < ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")

        self.in_dim = in_dim
        self.n_channels = n_channels
        self.n_features = n_features
        self.bandwidth = float(bandwidth)
        self.ridge_lambda = float(ridge_lambda)
        self.sigma_floor = float(sigma_floor)
        self.ema_decay = float(ema_decay)
        self.standardize = bool(standardize)

        # Fixed random basis: Ω ~ N(0, h⁻²), b ~ U[0, 2π).  (buffers, not params)
        g = torch.Generator().manual_seed(seed)
        omega = torch.randn(in_dim, n_features, generator=g)   # UNIT scale; h applied in _features
        bias = torch.rand(n_features, generator=g) * (2.0 * math.pi)
        self.register_buffer("omega", omega)
        self.register_buffer("rff_bias", bias)

        # Active bandwidth is a BUFFER so it can be locked to the observed median
        # ‖Δz‖ (median heuristic) at the curriculum boundary — mirrors the Δ-scale
        # lock. Starts at the configured heuristic value; the readout fits with the
        # heuristic through warmup, then h snaps to the data scale and re-settles.
        self.register_buffer("active_bandwidth", torch.tensor(self.bandwidth))
        self.register_buffer("_dz_median_ema", torch.zeros(()))   # observed median ‖Δz‖
        self.register_buffer("_dz_obs", torch.zeros((), dtype=torch.long))
        self.register_buffer("bandwidth_locked", torch.zeros((), dtype=torch.long))

        # EMA sufficient statistics over mature samples (all buffers).
        D, C = n_features, n_channels
        self.register_buffer("phi_mean", torch.zeros(D))    # E[φ]
        self.register_buffer("x_mean", torch.zeros(C))      # E[x]        = μ_prior
        self.register_buffer("r2_mean", torch.zeros(C))     # E[(x−μ)²]   = var prior
        self.register_buffer("A", torch.zeros(D, D))        # E[φφᵀ]
        self.register_buffer("Cx", torch.zeros(D, C))       # E[φ xᵀ]
        self.register_buffer("Cr2", torch.zeros(D, C))      # E[φ (x−μ)²ᵀ]

        # Internal z_type standardization (EMA mean/var).
        self.register_buffer("z_mean", torch.zeros(in_dim))
        self.register_buffer("z_var", torch.ones(in_dim))

        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    # -- feature map ---------------------------------------------------------

    def _standardize_z(self, z: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return z
        return (z - self.z_mean) / torch.sqrt(self.z_var + 1e-6)

    def _features(self, z: torch.Tensor) -> torch.Tensor:
        """z ``[N, d]`` → φ ``[N, D]`` = √(2/D)·cos((z_std / h)·Ω + b)."""
        proj = (self._standardize_z(z) / self.active_bandwidth) @ self.omega + self.rff_bias
        return math.sqrt(2.0 / self.n_features) * torch.cos(proj)

    # -- median-heuristic bandwidth lifecycle --------------------------------

    @property
    def median_dz(self) -> float:
        """Observed EMA median pairwise ‖Δz‖ of standardized z_type (the natural
        bandwidth scale). 0 until the first update."""
        return float(self._dz_median_ema.item())

    @torch.no_grad()
    def lock_bandwidth(self) -> float:
        """Freeze the bandwidth to the observed median ‖Δz‖ (median heuristic).
        Call at the curriculum boundary, like the Δ-scale lock."""
        if int(self._dz_obs.item()) > 0 and self._dz_median_ema > 0:
            self.active_bandwidth.copy_(self._dz_median_ema)
        self.bandwidth_locked.fill_(1)
        return float(self.active_bandwidth.item())

    @torch.no_grad()
    def heldout_r2(self, z_type: torch.Tensor, x: torch.Tensor):
        """Held-out μ fit quality on mature (z, x): returns (ss_res, ss_tot) pooled
        over channels, where ss_tot is vs the prior mean E[x] (readout's x_mean).
        R² = 1 − Σss_res/Σss_tot > 0 iff the readout beats the flat prior — the
        direct 'is the bandwidth right' signal (R²<0 ⇒ h mis-calibrated)."""
        mu, _ = self.predict(z_type)
        ss_res = (x - mu).pow(2).sum()
        ss_tot = (x - self.x_mean).pow(2).sum()
        return ss_res, ss_tot

    # -- centered ridge system ----------------------------------------------

    def _system(self):
        """Regularized centered feature covariance ``K = Ac + λI`` and the
        centered cross-covariances for the mean and residual-variance fits."""
        Ac = self.A - torch.outer(self.phi_mean, self.phi_mean)
        cc = self.Cx - torch.outer(self.phi_mean, self.x_mean)
        cc_r = self.Cr2 - torch.outer(self.phi_mean, self.r2_mean)
        eye = torch.eye(self.n_features, device=Ac.device, dtype=Ac.dtype)
        K = Ac + self.ridge_lambda * eye
        return K, cc, cc_r

    def _mu_from(self, f: torch.Tensor, K: torch.Tensor, cc: torch.Tensor):
        """μ at centered features ``f`` given the solved system."""
        return self.x_mean + f @ torch.linalg.solve(K, cc)

    # -- online update -------------------------------------------------------

    @torch.no_grad()
    def update(self, z_type: torch.Tensor, x: torch.Tensor) -> None:
        """Fold a batch of **mature** samples into the running statistics.

        Mean statistics are folded first, then μ is evaluated on the batch and its
        squared residuals fold the variance statistics — so even a single update is
        self-consistent (variance = residual variance of that update's μ).

        Args:
            z_type: ``[M, d]`` type embeddings of the mature samples (detached
                internally; a pixel's z_type is repeated across its mature
                timesteps by the caller).
            x: ``[M, C]`` phase-feature values at those mature samples.
        """
        z_type = z_type.detach()
        x = x.detach()
        M = z_type.shape[0]
        if M == 0:
            return

        # First update overwrites (decay→0) so the EMA isn't biased toward zeros.
        d = 0.0 if int(self.num_updates.item()) == 0 else self.ema_decay

        if self.standardize:
            self.z_mean.mul_(d).add_((1 - d) * z_type.mean(0))
            self.z_var.mul_(d).add_((1 - d) * z_type.var(0, unbiased=False))

        # Observe the median pairwise ‖Δz‖ of standardized z (the median-heuristic
        # bandwidth scale) until it's locked at the curriculum boundary.
        if int(self.bandwidth_locked.item()) == 0 and M >= 2:
            z_std = self._standardize_z(z_type)
            dz = torch.cdist(z_std, z_std)
            med = dz[torch.triu(torch.ones_like(dz, dtype=torch.bool), diagonal=1)].median()
            dm = 0.0 if int(self._dz_obs.item()) == 0 else self.ema_decay
            self._dz_median_ema.mul_(dm).add_((1 - dm) * med)
            self._dz_obs += 1

        phi = self._features(z_type)                         # [M, D]

        # 1) mean statistics
        self.phi_mean.mul_(d).add_((1 - d) * phi.mean(0))
        self.x_mean.mul_(d).add_((1 - d) * x.mean(0))
        self.A.mul_(d).add_((1 - d) * (phi.t() @ phi) / M)
        self.Cx.mul_(d).add_((1 - d) * (phi.t() @ x) / M)

        # 2) residuals from the (now-updated) mean fit, then variance statistics
        K, cc, _ = self._system()
        mu = self._mu_from(phi - self.phi_mean, K, cc)       # [M, C]
        r2 = (x - mu) ** 2
        self.r2_mean.mul_(d).add_((1 - d) * r2.mean(0))
        self.Cr2.mul_(d).add_((1 - d) * (phi.t() @ r2) / M)

        self.num_updates += 1

    # -- prediction ----------------------------------------------------------

    @torch.no_grad()
    def predict(self, z_type: torch.Tensor):
        """μ, σ for query ``z_type`` ``[N, d]`` → each ``[N, C]``.

        μ reverts to the prior mean ``E[x]`` and σ to the prior scale where mature
        data is sparse (sharper with larger ``D``); both are labeled by
        :meth:`leverage`.
        """
        f = self._features(z_type.detach()) - self.phi_mean          # [N, D]
        K, cc, cc_r = self._system()
        W = torch.linalg.solve(K, torch.cat([cc, cc_r], dim=1))      # [D, 2C]
        Wm, Wr = W[:, : self.n_channels], W[:, self.n_channels :]
        mu = self.x_mean + f @ Wm
        var = torch.clamp(self.r2_mean + f @ Wr, min=self.sigma_floor ** 2)
        return mu, torch.sqrt(var)

    @torch.no_grad()
    def leverage(self, z_type: torch.Tensor) -> torch.Tensor:
        """Per-sample prior strength ``v(z) ∈ [0, 1]`` — a coverage score.

        ``v(z) = λ·fᵀK⁻¹f / (fᵀf)`` with ``f = φ(z) − E[φ]``, ``K = Ac + λI``.
        Monotone in local mature-data support: **higher = leaning more on the
        prior** (μ/σ reverting), lower = data-pinned. The absolute scale sharpens
        toward a clean [0,1] prior-fraction as ``D`` grows (periodic-feature
        caveat, module docstring). Aggregate per EVT for the coverage diagnostic.
        """
        f = self._features(z_type.detach()) - self.phi_mean
        K, _, _ = self._system()
        Kf = torch.linalg.solve(K, f.t()).t()
        num = self.ridge_lambda * (f * Kf).sum(dim=1)
        den = (f * f).sum(dim=1) + 1e-12
        return torch.clamp(num / den, min=0.0, max=1.0)

    @torch.no_grad()
    def anomaly(self, x: torch.Tensor, z_type: torch.Tensor) -> torch.Tensor:
        """Type-conditional anomaly ``(x − μ(z_type)) / σ(z_type)``.

        Args:
            x: ``[N, C]`` phase-feature values.
            z_type: ``[N, d]`` type embeddings (one per row of ``x``).
        Returns:
            ``[N, C]`` anomaly.
        """
        mu, sigma = self.predict(z_type)
        return (x - mu) / sigma

    @torch.no_grad()
    def nll(self, z_type: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-sample heteroscedastic Gaussian NLL (the h-selection criterion).

        Score this on **out-of-fit** (val-split) mature pixels to pick ``bandwidth``;
        on the fit data it falls monotonically with smaller ``h`` and gives no
        signal. Returns ``[N]`` (mean over channels).
        """
        mu, sigma = self.predict(z_type)
        var = sigma * sigma
        nll = 0.5 * (((x - mu) ** 2) / var + torch.log(2 * math.pi * var))
        return nll.mean(dim=1)
