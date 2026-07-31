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
import math


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
        f"pvcr={train_stats['phase_vcr_loss']:.4f} evt={train_stats['evt_loss']:.4f}"
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

    # Log phase loss stats
    pls = train_stats.get('phase_loss_stats')
    if pls and pls.get('curriculum_w', 0) > 0:
        logger.info(
            f"  Phase loss: self={pls['loss_self']:.4f}, cross={pls['loss_cross']:.4f} | "
            f"Pairs: {pls['n_pairs_input']:.0f} input, "
            f"{pls['n_pairs_sufficient_overlap']:.0f} with overlap | "
            f"Curriculum weight: {pls['curriculum_w']:.2f}"
        )
        # Reference distance distributions (what tau_ref operates on)
        logger.info(
            f"  Phase d_ref_self:  mean={pls['d_ref_self_mean']:.3f}±{pls['d_ref_self_std']:.3f}, "
            f"[q25={pls['d_ref_self_q25']:.3f}, q50={pls['d_ref_self_q50']:.3f}, q75={pls['d_ref_self_q75']:.3f}]"
        )
        logger.info(
            f"  Phase d_ref_cross: mean={pls['d_ref_cross_mean']:.3f}±{pls['d_ref_cross_std']:.3f}, "
            f"[q25={pls['d_ref_cross_q25']:.3f}, q50={pls['d_ref_cross_q50']:.3f}, q75={pls['d_ref_cross_q75']:.3f}]"
        )
        # Entropy of softmax distributions (0=one-hot, log(M)=uniform)
        # With mean_overlap~11.5, log(M) ~ log(10) ~ 2.30 nats
        logger.info(
            f"  Phase entropy (nats): "
            f"self p={pls['self_mean_entropy_p']:.3f}, q={pls['self_mean_entropy_q']:.3f} | "
            f"cross p={pls['cross_mean_entropy_p']:.3f}, q={pls['cross_mean_entropy_q']:.3f} "
            f"[max~{pls['self_mean_overlap']:.1f} neighbors -> log(M)~{math.log(max(pls['self_mean_overlap'], 1)):.2f}]"
        )
    elif pls and phase_config is not None:
        # Phase loss configured but not yet active (curriculum still ramping in).
        # When phase_config is None the loss is fully disabled — there is no start
        # epoch to report, so skip the line rather than indexing a None config.
        logger.info(
            f"  Phase loss: inactive (curriculum_w={pls['curriculum_w']:.2f}, "
            f"starts epoch {phase_config['curriculum_start_epoch']+1})"
        )

    # Log FiLM diagnostics (data-dependent: actual gamma/beta across pixels)
    fs = train_stats.get('film_stats')
    if fs is not None:
        logger.info(
            f"  FiLM gamma (data): mean={fs['gamma_mean']:.4f}, "
            f"std={fs['gamma_std']:.4f}, "
            f"per_dim_std={fs['gamma_per_dim_std']:.4f}"
        )
        logger.info(
            f"  FiLM beta  (data): mean={fs['beta_mean']:.4f}, "
            f"std={fs['beta_std']:.4f}, "
            f"per_dim_std={fs['beta_per_dim_std']:.4f}"
        )
    else:
        logger.info("  FiLM: no data (phase pathway not active yet)")

    # Log pre-FiLM type-leakage diagnostics
    tls = train_stats.get('type_leakage_stats')
    if tls is not None:
        logger.info(
            f"  Pre-FiLM type leakage: cross_cov_frob={tls['cross_cov_frob']:.4f} | "
            f"z_type R² from h: mean={tls['r2_mean']:.4f}, max={tls['r2_max']:.4f}"
        )
