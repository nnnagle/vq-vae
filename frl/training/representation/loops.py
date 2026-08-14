#!/usr/bin/env python3
"""Epoch-level train / validate loops for representation learning.

``train_epoch`` and ``validate_epoch`` iterate the dataloader, call
``process_batch`` per batch, accumulate losses and diagnostic stats, and (when
profiling is enabled) log per-epoch dataloader wait/step and component timing.
Extracted verbatim from ``train_representation.py``.
"""

from __future__ import annotations

import logging
import time

import torch
from torch.utils.data import DataLoader

from data.loaders.builders.feature_builder import FeatureBuilder
from data.sampling.anchor_sampling import AnchorSampler
from models import RepresentationModel
from losses.evt_soft_neighborhood import EvtDiffusionMetric
from training.representation.profiling import is_profiling
from training.representation.step import process_batch
from training.representation.curriculum import ramp_weight

# Keep the original module's logger name so slurm-log records are byte-identical
# to the pre-refactor inline code (the messages moved, the %(name)s should not).
logger = logging.getLogger("training.train_representation")


def train_epoch(
    train_dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: dict,
    epoch: int,
    num_epochs: int,
    log_interval: int = 10,
    phase_sampler: AnchorSampler | None = None,
    phase_config: dict | None = None,
    spread_config: dict | None = None,
    recovery_disc_config: dict | None = None,
    evt_metric: EvtDiffusionMetric | None = None,
    evt_sampler: AnchorSampler | None = None,
    max_batches: int | None = None,
) -> dict:
    """Run training on entire training set for one epoch."""
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_phase_loss = 0.0
    total_phase_spread_loss = 0.0
    total_phase_recovery_disc_loss = 0.0
    total_phase_leakage_loss = 0.0
    total_vcr_loss = 0.0
    total_phase_vcr_loss = 0.0
    total_phase_anchor_loss = 0.0
    total_phase_ou_loss = 0.0
    total_readout_leverage = 0.0
    last_ou_diag = None
    last_contrastive_diag = None
    total_evt_loss = 0.0
    all_epoch_evt_diag: list[dict] = []
    total_spectral_pos_pairs = 0
    total_spectral_neg_pairs = 0
    total_spatial_pos_pairs = 0
    total_spatial_neg_pairs = 0
    total_batches = 0

    # Lock the anomaly-transform Δ scale once, at the phase-curriculum boundary
    # (it was observed through the type-only warmup; frozen thereafter).
    if phase_config is not None and not bool(model.anomaly_transform.scale_locked):
        if ramp_weight(epoch, phase_config.get('curriculum_start_epoch', 10),
                       phase_config.get('curriculum_ramp_epochs', 10)) > 0.0:
            locked = model.anomaly_transform.lock_delta_scale()
            logger.info(
                f"[epoch {epoch}] Locked anomaly Δ scale = {locked:.4f} "
                f"(phase curriculum active; readout μ/σ keep learning)"
            )

    # Keep last batch stats for epoch-level distribution logging
    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    last_gate_stats = empty_stats
    last_pos_weight_stats = empty_stats
    last_neg_weight_stats = empty_stats
    last_pos_sim_stats = empty_stats
    last_neg_sim_stats = empty_stats
    last_pos_spec_dist_stats = empty_stats
    last_neg_spec_dist_stats = empty_stats
    last_tau_sweep: dict = {}
    last_spectral_sim_stats = None
    last_phase_pair_stats = None
    last_phase_loss_stats = None
    last_film_stats = None

    # Split time spent blocked on the dataloader (next(...) starvation) from
    # time spent in the training step. The per-batch timers inside
    # process_batch only cover step compute, so this is the only place the
    # dataloader wait is visible.
    t_wait_total = 0.0
    t_step_total = 0.0
    t_step_steady = 0.0  # step time over steady-state batches only (batch_idx >= 1)
    n_batches_seen = 0
    _t_fetch = time.perf_counter()
    # Per-component step-timing accumulated over steady-state batches (batch 0
    # is skipped: it carries one-time warmup like the cuSOLVER SVD workspace).
    tm_accum: dict[str, float] = {}
    tm_n = 0

    for batch_idx, batch in enumerate(train_dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        t_wait_total += time.perf_counter() - _t_fetch
        _t_step = time.perf_counter()
        stats = process_batch(
            batch, feature_builder, model, device, config,
            training=True, optimizer=optimizer,
            phase_sampler=phase_sampler, phase_config=phase_config,
            spread_config=spread_config, recovery_disc_config=recovery_disc_config,
            epoch=epoch, evt_metric=evt_metric, evt_sampler=evt_sampler,
        )

        scheduler.step()

        if is_profiling() and batch_idx >= 1 and stats.get('timing'):
            for _k, _v in stats['timing'].items():
                if _k == 'cross_phase_detail':
                    continue
                tm_accum[_k] = tm_accum.get(_k, 0.0) + _v
            tm_n += 1

        if stats['n_valid'] > 0:
            total_loss += stats['loss']
            total_spectral_loss += stats['spectral_loss']
            total_spatial_loss += stats['spatial_loss']
            total_phase_loss += stats['phase_loss']
            total_phase_spread_loss += stats.get('phase_spread_loss', 0.0)
            total_phase_recovery_disc_loss += stats.get('phase_recovery_disc_loss', 0.0)
            total_phase_leakage_loss += stats.get('phase_leakage_loss', 0.0)
            total_vcr_loss += stats['vcr_loss']
            total_phase_vcr_loss += stats['phase_vcr_loss']
            total_phase_anchor_loss += stats.get('phase_anchor_loss', 0.0)
            total_phase_ou_loss += stats.get('phase_ou_loss', 0.0)
            if stats.get('ou_diag'):
                last_ou_diag = stats['ou_diag']
            if stats.get('phase_contrastive_diag'):
                last_contrastive_diag = stats['phase_contrastive_diag']
            total_readout_leverage += stats.get('readout_leverage', 0.0)
            total_evt_loss += stats.get('evt_loss', 0.0)
            if stats.get('evt_diag'):
                all_epoch_evt_diag.append(stats['evt_diag'])
            total_spectral_pos_pairs += stats['spectral_pos_pairs']
            total_spectral_neg_pairs += stats['spectral_neg_pairs']
            total_spatial_pos_pairs += stats['spatial_pos_pairs']
            total_spatial_neg_pairs += stats['spatial_neg_pairs']
            total_batches += 1

            # Update distribution stats from last valid batch
            last_gate_stats = stats['gate_stats']
            last_pos_weight_stats = stats['pos_weight_stats']
            last_neg_weight_stats = stats['neg_weight_stats']
            last_pos_sim_stats = stats.get('pos_sim_stats', empty_stats)
            last_neg_sim_stats = stats.get('neg_sim_stats', empty_stats)
            last_pos_spec_dist_stats = stats.get('pos_spec_dist_stats', empty_stats)
            last_neg_spec_dist_stats = stats.get('neg_spec_dist_stats', empty_stats)
            last_tau_sweep = stats.get('tau_sweep', {})
            last_spectral_sim_stats = stats.get('spectral_sim_stats')
            last_phase_pair_stats = stats.get('phase_pair_stats')
            last_phase_loss_stats = stats.get('phase_loss_stats')
            if stats.get('film_stats') is not None:
                last_film_stats = stats['film_stats']

            if is_profiling() and batch_idx == 0 and stats.get('timing'):
                tm = stats['timing']
                top_keys = [k for k in tm if k != 'cross_phase_detail']
                total_t = sum(tm[k] for k in top_keys)
                logger.info(
                    f"Batch timing (s): "
                    f"feat={tm['feature_build']:.2f} "
                    f"anchors={tm['anchor_sample']:.2f} "
                    f"spat_pairs={tm['spatial_pairs']:.2f} "
                    f"spec_wts={tm['spectral_weights']:.2f} "
                    f"gpu_fwd={tm['gpu_forward']:.2f} "
                    f"phase_pairs={tm['phase_pairs']:.2f} "
                    f"phase_fwd={tm['phase_forward']:.2f} "
                    f"loss={tm['loss_compute']:.2f} "
                    f"cross_spec={tm['cross_spectral']:.2f} "
                    f"cross_phase={tm['cross_phase']:.2f} "
                    f"| total={total_t:.2f}"
                )
                cp = tm.get('cross_phase_detail', {})
                if cp:
                    logger.info(
                        f"  cross_phase breakdown (s): "
                        f"cat={cp.get('cat', 0):.2f} "
                        f"svd={cp.get('svd', 0):.2f} "
                        f"knn={cp.get('knn', 0):.2f} "
                        f"build_batch={cp.get('build_batch', 0):.2f} "
                        f"neighborhood={cp.get('neighborhood_loss', 0):.2f} "
                        f"spread={cp.get('spread_loss', 0):.2f} "
                        f"rd={cp.get('rd_loss', 0):.2f} "
                        f"leakage={cp.get('leakage', 0):.2f}"
                    )

            if batch_idx % log_interval == 0:
                ps = stats.get('phase_pair_stats')
                pls = stats.get('phase_loss_stats')
                cw = pls.get('curriculum_w', 0) if pls else 0
                cw_str = f" cw={cw:.2f}" if 0 < cw < 1.0 else ""
                n_batches = len(train_dataloader)
                batch_width = len(str(n_batches))
                logger.info(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx+1:{batch_width}d}/{n_batches} | "
                    f"loss={stats['loss']:.4f} "
                    f"spec={stats['spectral_loss']:.4f} "
                    f"spat={stats['spatial_loss']:.4f} "
                    f"phase={stats['phase_loss']:.4f} "
                    f"spr={stats.get('phase_spread_loss', 0.0):.4f} "
                    f"vcr={stats['vcr_loss']:.4f} "
                    f"pvcr={stats['phase_vcr_loss']:.4f} "
                    f"evt={stats.get('evt_loss', 0.0):.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.2e}"
                )
                if ps and ps['n_anchors'] > 0 and pls and cw > 0:
                    logger.info(
                        f"  phase: {ps['n_total_pairs']:.0f} pairs "
                        f"({ps['n_anchors_surviving']:.0f}/{ps['n_anchors']:.0f} anchors, "
                        f"overlap={ps['overlap_mean']:.1f}) "
                        f"self={pls['loss_self']:.4f} cross={pls['loss_cross']:.4f}"
                        f"{cw_str}"
                    )

        _step_dt = time.perf_counter() - _t_step
        t_step_total += _step_dt
        if batch_idx >= 1:
            t_step_steady += _step_dt
        n_batches_seen += 1
        _t_fetch = time.perf_counter()

    if is_profiling() and n_batches_seen > 0:
        _wpb = t_wait_total / n_batches_seen
        _spb = t_step_total / n_batches_seen
        _tot = t_wait_total + t_step_total
        _frac = 100.0 * t_wait_total / max(_tot, 1e-9)
        logger.info(
            f"Epoch {epoch+1} dataloader: "
            f"wait={t_wait_total:.1f}s ({_wpb:.2f}/batch, {_frac:.0f}% of loop) | "
            f"step={t_step_total:.1f}s ({_spb:.2f}/batch) over {n_batches_seen} batches"
        )
        if tm_n > 0:
            _parts = " ".join(
                f"{_k}={tm_accum[_k] / tm_n:.2f}"
                for _k in sorted(tm_accum, key=lambda k: -tm_accum[k])
            )
            # Disjoint top-level buckets that should tile the whole step. The
            # unaccounted residual = measured step minus their sum; a large
            # residual means real work is happening outside every timer.
            _disjoint = ('pass1_total', 'gpu_forward', 'pass2_total',
                         'cross_spectral', 'cross_phase', 'backward')
            _accounted = sum(tm_accum.get(k, 0.0) for k in _disjoint)
            _resid = (t_step_steady - _accounted) / tm_n
            logger.info(
                f"Epoch {epoch+1} step breakdown (s/batch, avg over {tm_n} steady-state batches): "
                f"{_parts} | step={t_step_steady / tm_n:.2f} unaccounted={_resid:.2f}"
            )

    if total_batches == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0,
            'phase_loss': 0.0, 'phase_spread_loss': 0.0, 'phase_recovery_disc_loss': 0.0,
            'vcr_loss': 0.0, 'phase_vcr_loss': 0.0,
            'phase_anchor_loss': 0.0, 'phase_ou_loss': 0.0, 'ou_diag': None,
            'readout_leverage': 0.0,
            'evt_loss': 0.0,
            'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'pos_sim_stats': empty_stats, 'neg_sim_stats': empty_stats,
            'spectral_sim_stats': None,
            'pos_spec_dist_stats': empty_stats, 'neg_spec_dist_stats': empty_stats,
            'phase_pair_stats': None, 'phase_loss_stats': None,
            'film_stats': None,
        }

    _empty_evt_diag = dict(
        mean_entropy_ref=0.0, mean_entropy_learned=0.0,
        median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
        d_lrn_confused=0.0, d_lrn_noncf=0.0,
        n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
    )
    epoch_evt_diag = (
        {k: sum(d.get(k, _empty_evt_diag[k]) for d in all_epoch_evt_diag) / len(all_epoch_evt_diag)
         for k in _empty_evt_diag}
        if all_epoch_evt_diag else _empty_evt_diag
    )
    return {
        'loss': total_loss / total_batches,
        'spectral_loss': total_spectral_loss / total_batches,
        'spatial_loss': total_spatial_loss / total_batches,
        'phase_loss': total_phase_loss / total_batches,
        'phase_spread_loss': total_phase_spread_loss / total_batches,
        'phase_recovery_disc_loss': total_phase_recovery_disc_loss / total_batches,
        'phase_leakage_loss': total_phase_leakage_loss / total_batches,
        'vcr_loss': total_vcr_loss / total_batches,
        'phase_vcr_loss': total_phase_vcr_loss / total_batches,
        'phase_anchor_loss': total_phase_anchor_loss / total_batches,
        'phase_ou_loss': total_phase_ou_loss / total_batches,
        'ou_diag': last_ou_diag,
        'phase_contrastive_diag': last_contrastive_diag,
        'readout_leverage': total_readout_leverage / total_batches,
        'evt_loss': total_evt_loss / total_batches,
        'evt_diag': epoch_evt_diag,
        'spectral_pos_pairs': total_spectral_pos_pairs // total_batches,
        'spectral_neg_pairs': total_spectral_neg_pairs // total_batches,
        'spatial_pos_pairs': total_spatial_pos_pairs // total_batches,
        'spatial_neg_pairs': total_spatial_neg_pairs // total_batches,
        'batches': total_batches,
        'gate_stats': last_gate_stats,
        'pos_weight_stats': last_pos_weight_stats,
        'neg_weight_stats': last_neg_weight_stats,
        'pos_sim_stats': last_pos_sim_stats,
        'neg_sim_stats': last_neg_sim_stats,
        'pos_spec_dist_stats': last_pos_spec_dist_stats,
        'neg_spec_dist_stats': last_neg_spec_dist_stats,
        'tau_sweep': last_tau_sweep,
        'spectral_neg_tau_sweep': locals().get('spectral_neg_tau_sweep', {}),
        'spectral_sim_stats': last_spectral_sim_stats,
        'phase_pair_stats': last_phase_pair_stats,
        'phase_loss_stats': last_phase_loss_stats,
        'film_stats': last_film_stats,
    }

def validate_epoch(
    val_dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    config: dict,
    phase_sampler: AnchorSampler | None = None,
    phase_config: dict | None = None,
    spread_config: dict | None = None,
    recovery_disc_config: dict | None = None,
    epoch: int = 0,
    evt_metric: EvtDiffusionMetric | None = None,
    evt_sampler: AnchorSampler | None = None,
    max_batches: int | None = None,
) -> dict:
    """Run validation on entire validation set."""
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_phase_loss = 0.0
    total_phase_spread_loss = 0.0
    total_phase_recovery_disc_loss = 0.0
    total_phase_leakage_loss = 0.0
    total_vcr_loss = 0.0
    total_phase_vcr_loss = 0.0
    total_phase_anchor_loss = 0.0
    total_phase_ou_loss = 0.0
    total_readout_leverage = 0.0
    last_ou_diag = None
    last_contrastive_diag = None
    total_evt_loss = 0.0
    all_epoch_evt_diag: list[dict] = []
    total_batches = 0

    # Keep last batch stats for epoch-level distribution logging
    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    last_gate_stats = empty_stats
    last_pos_weight_stats = empty_stats
    last_neg_weight_stats = empty_stats
    last_pos_sim_stats = empty_stats
    last_neg_sim_stats = empty_stats
    last_pos_spec_dist_stats = empty_stats
    last_neg_spec_dist_stats = empty_stats
    last_spectral_sim_stats = None
    last_phase_pair_stats = None
    last_phase_loss_stats = None
    last_film_stats = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            stats = process_batch(
                batch, feature_builder, model, device, config,
                training=False,
                phase_sampler=phase_sampler, phase_config=phase_config,
                spread_config=spread_config, recovery_disc_config=recovery_disc_config,
                epoch=epoch, evt_metric=evt_metric, evt_sampler=evt_sampler,
            )
            if stats['n_valid'] > 0:
                total_loss += stats['loss']
                total_spectral_loss += stats['spectral_loss']
                total_spatial_loss += stats['spatial_loss']
                total_phase_loss += stats['phase_loss']
                total_phase_spread_loss += stats.get('phase_spread_loss', 0.0)
                total_phase_recovery_disc_loss += stats.get('phase_recovery_disc_loss', 0.0)
                total_phase_leakage_loss += stats.get('phase_leakage_loss', 0.0)
                total_vcr_loss += stats['vcr_loss']
                total_phase_vcr_loss += stats['phase_vcr_loss']
                total_phase_anchor_loss += stats.get('phase_anchor_loss', 0.0)
                total_phase_ou_loss += stats.get('phase_ou_loss', 0.0)
                if stats.get('ou_diag'):
                    last_ou_diag = stats['ou_diag']
                if stats.get('phase_contrastive_diag'):
                    last_contrastive_diag = stats['phase_contrastive_diag']
                total_readout_leverage += stats.get('readout_leverage', 0.0)
                total_evt_loss += stats.get('evt_loss', 0.0)
                if stats.get('evt_diag'):
                    all_epoch_evt_diag.append(stats['evt_diag'])
                total_batches += 1

                # Update distribution stats from last valid batch
                last_gate_stats = stats['gate_stats']
                last_pos_weight_stats = stats['pos_weight_stats']
                last_neg_weight_stats = stats['neg_weight_stats']
                last_pos_sim_stats = stats.get('pos_sim_stats', empty_stats)
                last_neg_sim_stats = stats.get('neg_sim_stats', empty_stats)
                last_spectral_sim_stats = stats.get('spectral_sim_stats')
                last_phase_pair_stats = stats.get('phase_pair_stats')
                last_phase_loss_stats = stats.get('phase_loss_stats')
                if stats.get('film_stats') is not None:
                    last_film_stats = stats['film_stats']

    if total_batches == 0:
        _empty_evt_diag = dict(
            mean_entropy_ref=0.0, mean_entropy_learned=0.0,
            median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
            d_lrn_confused=0.0, d_lrn_noncf=0.0,
            n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
        )
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0,
            'phase_loss': 0.0, 'phase_spread_loss': 0.0, 'phase_recovery_disc_loss': 0.0,
            'vcr_loss': 0.0, 'phase_vcr_loss': 0.0,
            'phase_anchor_loss': 0.0, 'phase_ou_loss': 0.0, 'ou_diag': None,
            'readout_leverage': 0.0,
            'evt_loss': 0.0, 'evt_diag': _empty_evt_diag,
            'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'spectral_sim_stats': None,
            'phase_pair_stats': None, 'phase_loss_stats': None,
            'film_stats': None,
        }

    _empty_evt_diag = dict(
        mean_entropy_ref=0.0, mean_entropy_learned=0.0,
        median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
        d_lrn_confused=0.0, d_lrn_noncf=0.0,
        n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
    )
    epoch_evt_diag = (
        {k: sum(d.get(k, _empty_evt_diag[k]) for d in all_epoch_evt_diag) / len(all_epoch_evt_diag)
         for k in _empty_evt_diag}
        if all_epoch_evt_diag else _empty_evt_diag
    )
    return {
        'loss': total_loss / total_batches,
        'spectral_loss': total_spectral_loss / total_batches,
        'spatial_loss': total_spatial_loss / total_batches,
        'phase_loss': total_phase_loss / total_batches,
        'phase_spread_loss': total_phase_spread_loss / total_batches,
        'phase_recovery_disc_loss': total_phase_recovery_disc_loss / total_batches,
        'phase_leakage_loss': total_phase_leakage_loss / total_batches,
        'vcr_loss': total_vcr_loss / total_batches,
        'phase_vcr_loss': total_phase_vcr_loss / total_batches,
        'phase_anchor_loss': total_phase_anchor_loss / total_batches,
        'phase_ou_loss': total_phase_ou_loss / total_batches,
        'ou_diag': last_ou_diag,
        'phase_contrastive_diag': last_contrastive_diag,
        'readout_leverage': total_readout_leverage / total_batches,
        'evt_loss': total_evt_loss / total_batches,
        'evt_diag': epoch_evt_diag,
        'batches': total_batches,
        'gate_stats': last_gate_stats,
        'pos_weight_stats': last_pos_weight_stats,
        'neg_weight_stats': last_neg_weight_stats,
        'pos_sim_stats': last_pos_sim_stats,
        'neg_sim_stats': last_neg_sim_stats,
        'pos_spec_dist_stats': last_pos_spec_dist_stats,
        'neg_spec_dist_stats': last_neg_spec_dist_stats,
        'spectral_sim_stats': last_spectral_sim_stats,
        'phase_pair_stats': last_phase_pair_stats,
        'phase_loss_stats': last_phase_loss_stats,
        'film_stats': last_film_stats,
    }

