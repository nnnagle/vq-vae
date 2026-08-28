"""
Ray / runs-kernel phase objective (the ysfc-free successor to the soft-neighborhood).

Three ingredients, all read from the data (a, Δa) and z_type — no ysfc:

1. **Pairwise jump gate** ``G[n,i,j] = exp(-(max_{i<s≤j} ‖Δa_s‖ / τ)²)`` — a soft
   gate that closes for any pair of timesteps with a disturbance *between* them
   (not just adjacent). It marks ejections and defines "jump-free run" softly,
   with no explicit segmentation.

2. **Ray contraction + mature anchor** (``ray_contraction_anchor_loss``): pin the
   mature part of the trajectory (small ‖a‖) to the origin, and require every
   jump-free lag to contract at a fixed nominal ρ: ``‖z_{t'} − ρ^{t'-t} z_t‖²``,
   gated by ``G``. This builds the cone geometry (origin = steady state; recovery
   = radial contraction along a ray).

3. **Runs-kernel metric matching** (``runs_kernel_matching_loss``): a jump-gated,
   window-normalized, evidence-weighted, type-gated similarity between (pixel,
   time) points — the continuous-signal analogue of a substring/local-alignment
   kernel, whose "words" are jump-free recovery runs. z_phase similarities are
   pulled to match it, so pixels on the same recovery run land together and the
   window self-aligns different calendar offsets (A's yr5-10 ↔ B's yr10-15).

Recovery *rate* is deliberately fixed (nominal ρ): rate variation is a *type*
difference (site index) and lives in z_type, so z_phase is the rate-normalized
shape. Anti-collapse is provided by (3)'s spread + phase VICReg + the anchor, not
by this module alone.

All functions are pure and unit-tested (``tests/test_ray_runs.py``); wiring lives
in ``training/representation/step.py``.
"""

from __future__ import annotations

import torch


def pairwise_jump_gate(delta_a_norm: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft gate over *intervening* jumps.

    Args:
        delta_a_norm: ``[N, T]`` per-timestep ``‖Δa_t‖``.
        tau: gate scale (in the locked Δ-scale units; stable ‖Δa‖≈1, jump ≫1).
    Returns:
        ``G`` ``[N, T, T]``, ``G[n,i,j] = exp(-(max_{min<s≤max} ‖Δa_s‖ / τ)²)``
        (symmetric; diagonal = 1). Close to 0 when a disturbance lies between i and j.
    """
    N, T = delta_a_norm.shape
    between = delta_a_norm.new_zeros(N, T, T)
    for i in range(T - 1):
        # max over s in (i, j] for each j > i  → cumulative max forward.
        between[:, i, i + 1:] = torch.cummax(delta_a_norm[:, i + 1:], dim=1).values
    between = between + between.transpose(1, 2)   # symmetric; diagonal stays 0
    return torch.exp(-(between / tau) ** 2)


def ray_contraction_anchor_loss(
    zp: torch.Tensor,             # [N, T, d] phase embeddings
    a_norm: torch.Tensor,         # [N, T] ‖a_t‖ (maturity signal)
    delta_a_norm: torch.Tensor,   # [N, T] ‖Δa_t‖ (jump signal)
    valid: torch.Tensor,          # [N, T] bool
    rho: float,
    tau_jump: float,
    sigma_mature: float,
    mature_quantile: float = 0.0,
):
    """Mature-anchor + fixed-ρ contraction over jump-free lags.

    The mature weight is ``exp(-(‖a‖/σ)²)``. With ``mature_quantile>0``, σ is set
    to that quantile of the batch's own valid ‖a‖ (self-calibrating), so a
    controlled fraction of the least-anomalous timesteps get pinned to the origin
    regardless of how well the readout centers ‖a‖ — a fixed absolute
    ``sigma_mature`` mis-fires (mature_frac→0) whenever the readout residual floor
    sits above it. ``mature_quantile=0`` uses the absolute ``sigma_mature``.

    Returns ``(anchor_loss, contraction_loss, diag)``. The caller weights each
    (they are reported separately so they can be tuned independently).
    """
    N, T, d = zp.shape
    dev = zp.device
    vf = valid.to(zp.dtype)

    # --- Anchor: where ‖a‖≈0 (mature), pin ‖z‖→0 (label-free origin). ----------
    if mature_quantile > 0.0 and bool(valid.any()):
        sig = torch.quantile(a_norm[valid].detach(), mature_quantile).clamp(min=1e-6)
    else:
        sig = a_norm.new_tensor(float(sigma_mature))
    w_mat = torch.exp(-(a_norm / sig) ** 2) * vf                   # [N, T]
    zr2 = zp.pow(2).sum(-1)                                        # [N, T]
    anchor = (w_mat * zr2).sum() / w_mat.sum().clamp(min=1.0)

    # --- Contraction: every jump-free lag i<j must be a ρ^(j-i) decay. --------
    G = pairwise_jump_gate(delta_a_norm, tau_jump)                # [N, T, T]
    ii = torch.arange(T, device=dev)
    lag = (ii[None, :] - ii[:, None]).clamp(min=0).to(zp.dtype)   # [T, T] = j-i (j≥i)
    rho_pow = torch.as_tensor(rho, device=dev, dtype=zp.dtype) ** lag  # [T, T]
    pred = rho_pow[None, :, :, None] * zp[:, :, None, :]          # [N, i, j, d] = ρ^(j-i) z_i
    resid2 = (zp[:, None, :, :] - pred).pow(2).sum(-1)            # [N, i, j] = ‖z_j - ρ^(j-i)z_i‖²
    triu = torch.triu(torch.ones(T, T, device=dev, dtype=zp.dtype), diagonal=1)
    w = G * triu[None] * vf[:, :, None] * vf[:, None, :]          # [N, i, j]
    wsum = w.sum().clamp(min=1.0)
    contraction = (w * resid2).sum() / wsum

    with torch.no_grad():
        offdiag = triu[None] * vf[:, :, None] * vf[:, None, :]
        gate_mean = (G * offdiag).sum() / offdiag.sum().clamp(min=1.0)
        diag = {
            "rho": float(rho),
            "anchor": float(anchor.detach()),
            "contraction": float(contraction.detach()),
            "gate_mean": float(gate_mean),
            "mature_frac": float((w_mat.sum() / vf.sum().clamp(min=1.0)).detach()),
            "sigma_mature_eff": float(sig.detach()),
            "a_norm_med": float(a_norm[valid].median().detach()) if bool(valid.any()) else 0.0,
            "resid_rms": float(((w * resid2).sum() / wsum).sqrt().detach()),
            "z_rms": float((zr2 * vf).sum().div(vf.sum().clamp(min=1.0)).sqrt().detach()),
        }
    return anchor, contraction, diag


def _type_grouped_pool(valid, z_type, n_seeds, group_size, max_points, generator):
    """Pool (pixel, time) points in ``z_type``-kNN pixel groups (the exp035 neighbor
    selection, ysfc-free): pick ``n_seeds`` seed pixels, take each seed's
    ``group_size`` nearest neighbors in standardized ``z_type`` (its forest-type
    twins), and return **all valid timesteps** of the selected pixels. This
    concentrates the O(M²) budget on same-type pairs — every point then has
    ~(group_size-1)·T same-type partners across *other* pixels (the cross-pixel
    recovery comparisons the kernel rewards), instead of the ~1-2 that random
    pooling yields. Returns ``(n_idx, t_idx)`` over the selected points."""
    dev = valid.device
    pix = valid.any(dim=1).nonzero(as_tuple=False).squeeze(1)     # candidate pixels
    C = int(pix.numel())
    if C == 0:
        return pix.new_zeros(0), pix.new_zeros(0)
    g = min(group_size, C)
    zt = z_type[pix]
    zt = (zt - zt.mean(0, keepdim=True)) / zt.std(0, keepdim=True).clamp(min=1e-6)
    D = torch.cdist(zt, zt)                                       # [C, C] type distances
    n_s = min(n_seeds, C)
    seed_local = torch.randperm(C, device=dev, generator=generator)[:n_s]
    knn = D[seed_local].topk(g, dim=1, largest=False).indices     # [n_s, g] local idx (incl self)
    sel_local = torch.unique(knn.reshape(-1))
    # Budget by whole pixels (preserve group structure): keep pixels until the
    # pooled point count would exceed max_points.
    counts = valid[pix[sel_local]].sum(dim=1)                     # valid points per selected pixel
    order = torch.randperm(int(sel_local.numel()), device=dev, generator=generator)
    sel_local, counts = sel_local[order], counts[order]
    keep = torch.cumsum(counts, 0) <= max_points
    if not bool(keep.any()):
        keep[0] = True
    sel_pix = pix[sel_local[keep]]
    vmask = valid[sel_pix]                                        # [S, T]
    nz = vmask.nonzero(as_tuple=False)                            # [M, 2] (local pixel, t)
    return sel_pix[nz[:, 0]], nz[:, 1]


def _extract_windows(flow, G, valid, n_idx, t_idx, half_window, window_sigma):
    """Per pooled point (n_idx[m], t_idx[m]): the ±half_window flow window and its
    weight = window-decay × jump-survival × validity. Returns win [M, 2W+1, F],
    gate [M, 2W+1]."""
    N, T, Fd = flow.shape
    M = n_idx.shape[0]
    W = half_window
    offs = torch.arange(-W, W + 1, device=flow.device)            # [2W+1]
    tt = t_idx[:, None] + offs[None, :]                           # [M, 2W+1]
    in_range = (tt >= 0) & (tt < T)
    tt_c = tt.clamp(0, T - 1)
    n_rep = n_idx[:, None].expand(M, 2 * W + 1)
    win = flow[n_rep, tt_c]                                       # [M, 2W+1, F]
    v = valid[n_rep, tt_c] & in_range                            # [M, 2W+1]
    surv = G[n_rep, t_idx[:, None].expand(M, 2 * W + 1), tt_c]    # [M, 2W+1] jump survival t→t+off
    wdecay = torch.exp(-(offs.to(flow.dtype) ** 2) / (2 * window_sigma ** 2))[None, :]
    gate = wdecay * surv * v.to(flow.dtype)                       # [M, 2W+1]
    win = win * v.to(flow.dtype)[..., None]
    return win, gate


def runs_kernel_matching_loss(
    zp: torch.Tensor,             # [N, T, d]
    flow: torch.Tensor,           # [N, T, F]  = (a, Δa) per timestep
    delta_a_norm: torch.Tensor,   # [N, T]
    z_type: torch.Tensor,         # [N, dt]
    valid: torch.Tensor,          # [N, T] bool
    *,
    tau_jump: float,
    half_window: int,
    window_sigma: float,
    sigma_flow: float,
    sigma_type: float,
    tau_metric: float,
    max_points: int,
    min_points: int,
    type_keep_threshold: float = 0.0,
    n_seeds: int = 0,
    group_size: int = 8,
    generator: torch.Generator | None = None,
):
    """Match z_phase similarities to the jump-gated runs-kernel similarity.

    Pools valid (pixel, time) points (``n_seeds>0`` ⇒ z_type-kNN pixel groups,
    the exp035 neighbor structure — see ``_type_grouped_pool``; else a legacy random
    subsample to ``max_points``), forms the windowed runs-kernel similarity ``S``
    (normalized per overlap), the evidence ``E`` (window overlap mass), and the type
    gate ``k_type``; then pulls the
    z_phase similarity ``L = exp(-‖Δz‖²/τ)`` toward ``S`` with a weighted MSE,
    weight = evidence × type-gate (self excluded).

    **Type-threshold pair selection.** z_phase is a type-collapsed *shadow* — type
    separation is z_type's job, so this loss only ever pulls *same-type* pairs
    together; it never pushes different types apart. Pairs whose type gate
    ``k_type < type_keep_threshold`` are dropped from the objective entirely (hard
    keep-mask), so the O(M²) budget is spent only on close-type neighbors. With
    ``type_keep_threshold = 0`` the mask is inert (all pairs kept, soft-weighted by
    ``k_type`` as before). The ``keep_*`` diagnostics report the realized
    neighborhood *bandwidth* so ``sigma_type`` / the threshold can be tuned.

    Returns ``(loss, diag)``.
    """
    N, T, d = zp.shape
    dev = zp.device
    # --- Pool (pixel, time) points --------------------------------------------
    # n_seeds>0: z_type-kNN pixel groups (all timesteps) — spends the O(M²) budget
    # on same-type pairs (the exp035 neighbor structure, ysfc-free). Else: legacy
    # random subsample (each point finds ~1-2 same-type partners; budget wasted).
    if n_seeds > 0:
        n_idx, t_idx = _type_grouped_pool(
            valid, z_type, n_seeds, group_size, max_points, generator)
        M = int(n_idx.shape[0])
        if M < min_points:
            return zp.sum() * 0.0, {"n_points": M, "active": 0.0}
    else:
        nz = valid.nonzero(as_tuple=False)                       # [P, 2] (n, t)
        P = nz.shape[0]
        if P < min_points:
            return zp.sum() * 0.0, {"n_points": P, "active": 0.0}
        if P > max_points:
            perm = torch.randperm(P, device=dev, generator=generator)[:max_points]
            nz = nz[perm]
        n_idx, t_idx = nz[:, 0], nz[:, 1]
        M = n_idx.shape[0]
    n_pixels = int(torch.unique(n_idx).numel())

    G = pairwise_jump_gate(delta_a_norm, tau_jump)               # [N, T, T]
    win, gate = _extract_windows(flow, G, valid, n_idx, t_idx, half_window, window_sigma)

    # --- Windowed runs-kernel similarity S and evidence E ----------------------
    num = zp.new_zeros(M, M)
    den = zp.new_zeros(M, M)
    two_sf2 = 2.0 * sigma_flow ** 2
    for w in range(2 * half_window + 1):
        gw = gate[:, w]                                          # [M]
        fw = win[:, w, :]                                        # [M, F]
        d2 = torch.cdist(fw, fw).pow(2)                         # [M, M]
        kw = torch.exp(-d2 / two_sf2)
        gg = gw[:, None] * gw[None, :]                           # [M, M]
        num = num + gg * kw
        den = den + gg
    S = num / den.clamp(min=1e-8)                                # [M, M] normalized similarity
    E = den                                                      # [M, M] evidence (overlap mass)

    # --- Type gate (standardized z_type) --------------------------------------
    zt = z_type[n_idx]                                           # [M, dt]
    zt = (zt - zt.mean(0, keepdim=True)) / zt.std(0, keepdim=True).clamp(min=1e-6)
    dt2 = torch.cdist(zt, zt).pow(2)                            # [M, M]
    k_type = torch.exp(-dt2 / (2.0 * sigma_type ** 2))

    # --- Learned z_phase similarity and the weighted match --------------------
    zpp = zp[n_idx, t_idx]                                       # [M, d]
    dl2 = torch.cdist(zpp, zpp).pow(2)                          # [M, M]
    L = torch.exp(-dl2 / tau_metric)                            # [M, M] in (0,1]

    eye = torch.eye(M, device=dev, dtype=zp.dtype)
    offdiag = 1.0 - eye
    # Hard keep-mask: only pull same-type pairs together (shadow embedding — never
    # push types apart). k_type ≥ threshold ⇒ standardized type-distance small.
    keep = offdiag
    if type_keep_threshold > 0.0:
        keep = keep * (k_type >= type_keep_threshold).to(zp.dtype)
    weight = E * k_type * keep                                  # evidence × type, kept pairs only
    wsum = weight.sum().clamp(min=1e-6)
    loss = (weight * (L - S) ** 2).sum() / wsum

    with torch.no_grad():
        # Split diagnostics by type-similar vs type-dissimilar for tuning.
        same = (k_type > 0.5) * offdiag
        diff = (k_type <= 0.5) * offdiag
        def _wmean(x, m):
            return float((x * m).sum() / m.sum().clamp(min=1.0))
        # --- Bandwidth monitors: realized neighborhood among KEPT pairs ---------
        n_off = offdiag.sum().clamp(min=1.0)
        n_keep = keep.sum()
        keep_frac = float(n_keep / n_off)                       # fraction of pairs surviving threshold
        # effective same-type neighbors per point (kept pairs / points)
        nbr_per_pt = float(n_keep / max(M, 1))
        diag = {
            "n_points": M,
            "n_pixels": n_pixels,                # distinct anchor pixels in the pool
            "active": 1.0,
            "S_same": _wmean(S, same),           # runs-sim, same type (want high for neighbors)
            "S_diff": _wmean(S, diff),           # runs-sim, diff type
            "L_same": _wmean(L, same),           # z_phase sim, same type
            "L_diff": _wmean(L, diff),           # z_phase sim, diff type
            "evidence_mean": _wmean(E, offdiag),
            "k_type_mean": _wmean(k_type, offdiag),
            # bandwidth of the retained neighborhood (tune sigma_type / threshold):
            "keep_frac": keep_frac,              # want a small-but-nonzero fraction
            "nbr_per_pt": nbr_per_pt,            # effective same-type neighbors per point
            "k_type_kept": _wmean(k_type, keep), # mean type-gate among kept pairs (→1 = very tight)
            "dt_kept": _wmean(dt2.sqrt(), keep), # mean standardized type-distance among kept pairs
            # calibration: how well L tracks S on kept pairs (evidence-weighted)
            "match_rmse": float((weight * (L - S) ** 2).sum().div(wsum).sqrt()),
        }
    return loss, diag
