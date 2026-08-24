"""Per-epoch diagnostic logging for representation training.

Extracted verbatim from the epoch loop of ``train_representation.py``. ``log_epoch``
emits the human-readable train/val loss summary and all the diagnostic blocks
(EVT, distribution stats, tau sweeps, phase pairs/loss, FiLM, type leakage).

The caller's ``logger`` is passed in so log records keep their original
``%(name)s`` (``training.train_representation``) — the output is byte-identical
to the former inline block.
"""

from __future__ import annotations

import logging


def fmt_stats(s: dict) -> str:
    return (
        f"mean={s['mean']:.3f}, std={s['std']:.3f}, "
        f"[q25={s['q25']:.3f}, q50={s['q50']:.3f}, q75={s['q75']:.3f}]"
    )


def log_epoch(
    logger: logging.Logger,
    epoch: int,
    num_epochs: int,
    train_stats: dict,
    val_stats: dict,
    loss_config: dict,
    phase_config: dict | None,
    evt_metric,
) -> None:
    """Log the full per-epoch summary and diagnostics for one completed epoch."""
    logger.info(f"Epoch {epoch+1}/{num_epochs} complete")
    logger.info(
        f"  Train: {train_stats['loss']:.4f} "
        f"spec={train_stats['spectral_loss']:.4f} spat={train_stats['spatial_loss']:.4f} "
        f"phase={train_stats['phase_loss']:.4f} spr={train_stats.get('phase_spread_loss', 0.0):.4f} "
        f"rdisc={train_stats.get('phase_recovery_disc_loss', 0.0):.4f} "
        f"leak={train_stats.get('phase_leakage_loss', 0.0):.2e} "
        f"vcr={train_stats['vcr_loss']:.4f} "
        f"pvcr={train_stats['phase_vcr_loss']:.4f} "
        f"anchor={train_stats.get('phase_anchor_loss', 0.0):.4f} "
        f"ou={train_stats.get('phase_ou_loss', 0.0):.4f} "
        f"evt={train_stats['evt_loss']:.4f}"
    )
    # RFF readout fit: held-out μ-R² is the "is the bandwidth right" signal (R²>0 ⇔
    # readout beats the flat prior; R²<0 ⇔ h mis-calibrated). h is the median ‖Δz‖.
    logger.info(
        f"  Readout fit: μ-R²(heldout) train={train_stats.get('readout_r2', 0.0):+.3f} "
        f"val={val_stats.get('readout_r2', 0.0):+.3f} | "
        f"h={train_stats.get('readout_bandwidth', 0.0):.2f} "
        f"median‖Δz‖={train_stats.get('readout_median_dz', 0.0):.2f} | "
        f"leverage={train_stats.get('readout_leverage', 0.0):.3f}"
    )
    # Within-pixel dynamics diagnostics. The Kalman filter (Step 6) reports
    # mean ρ, NIS (normalized innovation squared — should track n_obs when Q/R
    # are calibrated; the filter-consistency / identifiability check), and the
    # scored fraction. The plug-in OU (Step 4) reports ρ, gate mean, resid RMS.
    od = train_stats.get('ou_diag')
    if od and od.get('nis_mean') is not None:
        logger.info(
            f"  Phase Kalman: rho={od.get('rho_mean', 0.0):.3f} "
            f"nis={od.get('nis_mean', 0.0):.2f} (target {od.get('nis_target', 0.0):.0f}) "
            f"scored_frac={od.get('scored_frac', 0.0):.3f} "
            f"n_scored={od.get('n_scored', 0.0):.0f}"
        )
    elif od:
        logger.info(
            f"  OU dynamics: rho={od.get('rho', 0.0):.3f} "
            f"gate_mean={od.get('gate_mean', 0.0):.3f} "
            f"resid_rms={od.get('resid_rms', 0.0):.3f} "
            f"s0={od.get('s0', 0.0):.3f} n_eff={od.get('n_eff', 0.0):.0f}"
        )
    # Phase radius: RMS ‖z_phase‖ split by recovery state (hub-and-rim check).
    # mature_rms should sit near the origin (anchor loss pins it); disturbed_rms is
    # the ejection radius. neg_d2 ≈ disturbed_rms² + mature_rms² for cross-pixel
    # negatives, so a floating mature_rms inflates the gap without clean ejection.
    m_rms = train_stats.get('mature_radius_rms', 0.0)
    d_rms = train_stats.get('disturbed_radius_rms', 0.0)
    if m_rms > 0.0 or d_rms > 0.0:
        ratio = f" (ratio m/d={m_rms / d_rms:.2f})" if d_rms > 0 else ""
        logger.info(
            f"  Phase radius: mature_rms={m_rms:.3f} disturbed_rms={d_rms:.3f}{ratio}"
        )
    # Step-5 contrastive: gap/τ is the fixed-ruler calibration (target ~2–3).
    cd = train_stats.get('phase_contrastive_diag')
    if cd:
        logger.info(
            f"  Phase contrastive: gap/T={cd.get('gap_over_tau', 0.0):.2f} "
            f"(pos_d2={cd.get('pos_d2', 0.0):.3f} neg_d2={cd.get('neg_d2', 0.0):.3f}) "
            f"anchors={cd.get('n_anchors', 0)} n_disturbed={cd.get('n_disturbed', 0)}"
        )
        # Kernel health: mean valid positives per anchor (of n_pos) + the observable
        # affinities at selected pairs. Low n_pos/anchor or tiny p_pos ⇒ σ too peaked
        # (anchors positive-starved). Negatives want high k_type, low k_flow.
        logger.info(
            f"  Phase kernels: n_pos/anchor={cd.get('n_pos_mean', 0.0):.2f}/{cd.get('n_pos_req', 5)} | "
            f"pos: k_type={cd.get('k_type_pos', 0.0):.3f} k_flow={cd.get('k_flow_pos', 0.0):.3f} "
            f"p={cd.get('p_pos', 0.0):.2e} | "
            f"neg: k_type={cd.get('k_type_neg', 0.0):.3f} k_flow={cd.get('k_flow_neg', 0.0):.3f} "
            f"p={cd.get('p_neg', 0.0):.2e}"
        )
    logger.info(
        f"  Val:   {val_stats['loss']:.4f} "
        f"spec={val_stats['spectral_loss']:.4f} spat={val_stats['spatial_loss']:.4f} "
        f"phase={val_stats['phase_loss']:.4f} spr={val_stats.get('phase_spread_loss', 0.0):.4f} "
        f"rdisc={val_stats.get('phase_recovery_disc_loss', 0.0):.4f} "
        f"leak={val_stats.get('phase_leakage_loss', 0.0):.2e} "
        f"vcr={val_stats['vcr_loss']:.4f} "
        f"pvcr={val_stats['phase_vcr_loss']:.4f} evt={val_stats['evt_loss']:.4f}"
    )
    # EVT diagnostics — logged only when the EVT loss is active
    if evt_metric is not None:
        td = train_stats.get('evt_diag', {})
        vd = val_stats.get('evt_diag', {})
        logger.info(
            f"  EVT train | "
            f"kl={td.get('mean_kl', 0.0):.3f} "
            f"H_ref={td.get('mean_entropy_ref', 0.0):.3f} "
            f"H_lrn={td.get('mean_entropy_learned', 0.0):.3f} "
            f"med_d_lrn={td.get('median_d_learned', 0.0):.3f} "
            f"n_valid={td.get('n_anchors_valid', 0):.0f}"
        )
        logger.info(
            f"  EVT train | "
            f"rank_cf={td.get('mean_rank_confused', 0.5):.3f} "
            f"d_cf={td.get('d_lrn_confused', 0.0):.3f} "
            f"d_ncf={td.get('d_lrn_noncf', 0.0):.3f} "
            f"n_cf={td.get('n_confused_pairs', 0.0):.1f} "
            f"eff_n={td.get('eff_n_ref', 1.0):.1f}"
        )
        logger.info(
            f"  EVT val   | "
            f"kl={vd.get('mean_kl', 0.0):.3f} "
            f"H_ref={vd.get('mean_entropy_ref', 0.0):.3f} "
            f"H_lrn={vd.get('mean_entropy_learned', 0.0):.3f} "
            f"med_d_lrn={vd.get('median_d_learned', 0.0):.3f} "
            f"n_valid={vd.get('n_anchors_valid', 0):.0f}"
        )
        logger.info(
            f"  EVT val   | "
            f"rank_cf={vd.get('mean_rank_confused', 0.5):.3f} "
            f"d_cf={vd.get('d_lrn_confused', 0.0):.3f} "
            f"d_ncf={vd.get('d_lrn_noncf', 0.0):.3f} "
            f"n_cf={vd.get('n_confused_pairs', 0.0):.1f} "
            f"eff_n={vd.get('eff_n_ref', 1.0):.1f}"
        )

    # Log distribution statistics
    logger.info(
        f"  Gate values: {fmt_stats(train_stats['gate_stats'])}"
    )
    logger.info(
        f"  Spatial pos weights: {fmt_stats(train_stats['pos_weight_stats'])}"
    )
    logger.info(
        f"  Spatial neg weights: {fmt_stats(train_stats['neg_weight_stats'])}"
    )
    psd = train_stats.get('pos_spec_dist_stats', {})
    nsd = train_stats.get('neg_spec_dist_stats', {})
    if psd.get('mean', 0.0) != 0.0 or nsd.get('mean', 0.0) != 0.0:
        logger.info(
            f"  Spatial spec dists: pos={fmt_stats(psd)} | neg={fmt_stats(nsd)}"
        )
    if epoch == 0:
        tau_sweep = train_stats.get('tau_sweep', {})
        if tau_sweep:
            active_tau = loss_config.get('spatial_spectral_tau', 1.0)
            logger.info(f"  Spatial spectral weight τ sweep (epoch 0, active τ={active_tau}):")
            logger.info(f"    {'tau':>6}  {'pos_mean':>8}  {'pos_q25':>8}  {'pos_q50':>8}  {'neg_mean':>8}")
            for t, v in sorted(tau_sweep.items()):
                marker = " <-- active" if t == active_tau else ""
                logger.info(
                    f"    {t:>6.1f}  {v['pos_mean']:>8.3f}  {v['pos_q25']:>8.3f}  {v['pos_q50']:>8.3f}  {v['neg_mean']:>8.3f}{marker}"
                )
        spec_neg_sweep = train_stats.get('spectral_neg_tau_sweep', {})
        if spec_neg_sweep:
            active_tau_neg = loss_config.get('spectral_neg_tau', 1.0)
            logger.info(f"  Spectral neg weight τ sweep (epoch 0, active τ={active_tau_neg}):")
            logger.info(f"    {'tau':>6}  {'neg_mean':>8}  {'neg_q25':>8}  {'neg_q50':>8}")
            for t, v in sorted(spec_neg_sweep.items()):
                marker = " <-- active" if t == active_tau_neg else ""
                logger.info(
                    f"    {t:>6.1f}  {v['neg_mean']:>8.3f}  {v['neg_q25']:>8.3f}  {v['neg_q50']:>8.3f}{marker}"
                )
    ps = train_stats.get('pos_sim_stats', {})
    ns = train_stats.get('neg_sim_stats', {})
    if ps.get('mean', 0.0) != 0.0 or ns.get('mean', 0.0) != 0.0:
        gap = ps.get('mean', 0.0) - ns.get('mean', 0.0)
        _sw = loss_config.get('spatial_loss_weight', 1.0)
        _raw_spat = (train_stats.get('spatial_loss', 0.0) / _sw) if _sw > 0 else 0.0
        eff_confusers = f"{2.718 ** _raw_spat:.1f}"
        logger.info(
            f"  Spatial sims: pos={fmt_stats(ps)} | "
            f"neg mean={ns.get('mean', 0.0):.4f} | "
            f"gap={gap:.4f} | eff_confusers={eff_confusers}/{train_stats.get('spatial_neg_pairs', '?')}"
        )
    sss = train_stats.get('spectral_sim_stats')
    if sss is not None:
        _T = sss.get('temperature', 0.07) or 0.07
        _gap = sss['pos_mean'] - sss['neg_mean']
        _spw = loss_config.get('spectral_loss_weight', 1.0)
        _raw_spec = (train_stats.get('spectral_loss', 0.0) / _spw) if _spw > 0 else 0.0
        logger.info(
            f"  Spectral sims: pos_mean={sss['pos_mean']:.4f}±{sss['pos_std']:.4f} | "
            f"neg mean={sss['neg_mean']:.4f} | gap={_gap:.4f} | "
            f"gap/T={_gap / _T:.1f} (T={_T:g}) | "
            f"eff_confusers={2.718 ** _raw_spec:.1f}/{train_stats.get('spectral_neg_pairs', '?')}"
        )
    logger.info(
        f"  Pairs/batch: "
        f"spec(batch total) pos={train_stats.get('spectral_pos_pairs', 0)} neg={train_stats.get('spectral_neg_pairs', 0)} | "
        f"spat(per sample) pos={train_stats.get('spatial_pos_pairs', 0)} neg={train_stats.get('spatial_neg_pairs', 0)}"
    )

    # Log phase pair construction stats
    ps = train_stats.get('phase_pair_stats')
    if ps and ps['n_anchors'] > 0:
        logger.info(
            f"  Phase pairs: {ps['n_total_pairs']:.0f} total "
            f"({ps['n_self_pairs']:.0f} self + {ps['n_total_pairs'] - ps['n_self_pairs']:.0f} cross) | "
            f"Anchors: {ps['n_anchors_surviving']:.0f}/{ps['n_anchors']:.0f} surviving | "
            f"kNN candidates: {ps['n_candidates']:.0f} -> overlap filter: {ps['n_after_overlap']:.0f} | "
            f"Overlap: mean={ps['overlap_mean']:.1f}, min={ps['overlap_min']}"
        )
        logger.info(
            f"  Phase spec dist: mean={ps['dist_mean']:.2f}±{ps['dist_std']:.2f}, "
            f"[q25={ps['dist_q25']:.2f}, q50={ps['dist_q50']:.2f}, q75={ps['dist_q75']:.2f}], "
            f"range=[{ps['dist_min']:.2f}, {ps['dist_max']:.2f}] | "
            f"Weights(sigma={phase_config['sigma']}): {ps['weight_mean']:.3f}±{ps['weight_std']:.3f}"
        )

    # NOTE: the retired soft-neighborhood "Phase loss" block and the removed-FiLM
    # diagnostics used to log here. Both reflected retired features — the
    # soft-neighborhood loss (its phase_loss_stats.curriculum_w is a dead counter,
    # unrelated to the live anchor/OU/contrastive curriculum) and the FiLM head —
    # so they printed misleading "inactive" / "no data" lines even when the phase
    # pathway was fully active. Removed. Live phase state is the "Phase radius",
    # "OU dynamics", and "Phase contrastive" lines above.

    # Log pre-FiLM type-leakage diagnostics
    tls = train_stats.get('type_leakage_stats')
    if tls is not None:
        logger.info(
            f"  Pre-FiLM type leakage: cross_cov_frob={tls['cross_cov_frob']:.4f} | "
            f"z_type R² from h: mean={tls['r2_mean']:.4f}, max={tls['r2_max']:.4f}"
        )
