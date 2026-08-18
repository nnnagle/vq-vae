#!/usr/bin/env python3
"""Type-conditional readouts for Diagnostic B (pure math; unit-tested).

The additive concat-ridge `ridge([z_type, z_phase]) → target` gives `z_phase` a
single global coefficient, but `z_phase` is a **type-collapsed shadow** — a given
`z_phase` displacement should mean a large signal swing for one forest type and a
small one for another. So the additive probe under-reads `z_phase` (it cannot give
it a type-specific gain). This module provides two richer, still-parsimonious
readouts that let `z_type` modulate how `z_phase` is read:

1. **Low-rank BILINEAR ridge.**
       pred = b + w_typeᵀ·z_type + w_phaseᵀ·z_phase + (Pᵀ z_type)ᵀ · V · z_phase
   `P` is a **fixed** whitened top-`r` PCA projection of standardized `z_type`, so
   the interaction is rank `r` (a handful of type-directions each setting a gain on
   `z_phase`). Equivalent to ridge on `[z_type, z_phase, (Pᵀz_type)⊗z_phase]`. The
   main-effect block (`w_type`, `w_phase`) and the interaction block (`V`) get
   **separate** ridge penalties (`λ_main`, `λ_bilinear`): the interaction block is
   higher-variance and generally wants heavier shrinkage, so tying them to one λ
   either under-regularizes the mains or over-regularizes the interaction.

2. **Type-local kNN (Nadaraya–Watson) with a PRODUCT kernel.**
       w(q,j) = exp(−‖zt_q−zt_j‖²/2σ_type²) · exp(−‖zp_q−zp_j‖²/2σ_phase²)
       pred(q) = Σ_j w_j y_j / Σ_j w_j
   Type-local by construction (the type kernel restricts the reference set to
   type-neighbors) and reads `z_phase` locally within that neighborhood. `σ_type`
   and `σ_phase` are tuned **independently** — the whole point is that the
   "same type" radius and the "same recovery state" radius are different scales.
   This mirrors the downstream use case (kNN post-stratification on `[z_type,
   z_phase]`) more faithfully than any single-metric readout.

All functions are pure (tensors in / tensors out) so they unit-test without the
loader/zarr stack; the streaming plumbing lives in ``recovery_curves.py``.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Bilinear readout
# ---------------------------------------------------------------------------

def whitened_pca(cov_zt: torch.Tensor, r: int) -> torch.Tensor:
    """Top-``r`` **whitened** PCA projection ``P`` ``[dt, r]`` of standardized z_type.

    ``cov_zt`` is the covariance of *standardized* ``z_type`` (unit-diagonal). The
    columns are scaled by ``1/sqrt(eigenvalue)`` so the projected coordinates
    ``z_type_std @ P`` have ~unit variance — that keeps the bilinear interaction
    block on the same unit-diagonal ridge scale as the standardized main-effect
    columns, without needing a separate standardizer for the interaction.
    """
    cov = 0.5 * (cov_zt + cov_zt.transpose(-1, -2))            # symmetrize
    evals, evecs = torch.linalg.eigh(cov.double())            # ascending eigenvalues
    r = int(min(r, evecs.shape[1]))
    top_vecs = evecs[:, -r:]                                   # [dt, r]
    top_vals = evals[-r:].clamp(min=1e-8)
    P = top_vecs / top_vals.sqrt()                            # whiten → unit-var proj
    return P.to(cov_zt.dtype)


def bilinear_interaction(zt_s: torch.Tensor, zp_s: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Rank-``r`` interaction features ``(Pᵀ z_type) ⊗ z_phase`` → ``[N, r*zp]``.

    Row order of the flattened outer product is ``(type_component, phase_dim)`` with
    ``phase_dim`` fastest — i.e. block ``k`` (of ``r``) is that type-component's gain
    on the full ``z_phase`` vector.
    """
    proj = zt_s @ P                                           # [N, r]
    inter = proj.unsqueeze(2) * zp_s.unsqueeze(1)             # [N, r, zp]
    return inter.reshape(zt_s.shape[0], -1)                   # [N, r*zp]


def bilinear_features(zt_s: torch.Tensor, zp_s: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Full bilinear design matrix ``[z_type, z_phase, (Pᵀz_type)⊗z_phase]``."""
    return torch.cat([zt_s, zp_s, bilinear_interaction(zt_s, zp_s, P)], dim=1)


def block_penalty(
    dt: int, zp: int, r: int, lam_main: float, lam_bilinear: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Per-column ridge penalty for ``[type, phase | interaction | bias]``.

    ``λ_main`` on the ``dt+zp`` main-effect columns, ``λ_bilinear`` on the ``r*zp``
    interaction columns, ``0`` on the trailing bias column (never penalized).
    """
    n_main = dt + zp
    n_int = r * zp
    pen = torch.empty(n_main + n_int + 1, dtype=dtype)
    pen[:n_main] = lam_main
    pen[n_main:n_main + n_int] = lam_bilinear
    pen[-1] = 0.0
    return pen


def solve_block_ridge(
    A: torch.Tensor, B: torch.Tensor, penalty: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``(A + diag(penalty)) β = B`` for a block-diagonal ridge penalty.

    ``A`` ``[D+1, D+1]`` and ``B`` ``[D+1, C]`` are the (M-averaged) augmented
    normal equations with the bias as the last row/col. ``penalty`` ``[D+1]`` is
    the per-column λ (0 at the bias). Returns ``(W [D, C], b [C])``.
    """
    reg = torch.diag(penalty.to(A.dtype))
    Wb = torch.linalg.solve(A + reg, B)
    return Wb[:-1], Wb[-1]


# ---------------------------------------------------------------------------
# Type-local kNN (product-kernel Nadaraya–Watson)
# ---------------------------------------------------------------------------

def product_kernel_predict(
    zt_q: torch.Tensor,
    zp_q: torch.Tensor,
    zt_ref: torch.Tensor,
    zp_ref: torch.Tensor,
    y_ref: torch.Tensor,
    sigma_type: float,
    sigma_phase: float,
    chunk: int = 4096,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Nadaraya–Watson regression with a **product** kernel and **independent**
    type / phase bandwidths. Returns ``pred [Nq]`` (single target ``y_ref [Nref]``).

    Queries are chunked to bound the ``[chunk, Nref]`` kernel matrix. A query with
    no reference mass within either bandwidth falls back toward the reference mean
    of whatever little weight survives (den is floored at ``eps``); with sane
    bandwidths that is a negligible tail.
    """
    nq = zt_q.shape[0]
    out = torch.empty(nq, dtype=torch.float64)
    inv_t = 1.0 / (2.0 * float(sigma_type) ** 2)
    inv_p = 1.0 / (2.0 * float(sigma_phase) ** 2)
    ztr = zt_ref.double()
    zpr = zp_ref.double()
    yr = y_ref.double()
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        dt2 = torch.cdist(zt_q[s:e].double(), ztr).pow(2)     # [c, Nref]
        dp2 = torch.cdist(zp_q[s:e].double(), zpr).pow(2)
        w = torch.exp(-(dt2 * inv_t) - (dp2 * inv_p))         # product kernel
        num = w @ yr                                          # [c]
        den = w.sum(dim=1).clamp(min=eps)
        out[s:e] = num / den
    return out
