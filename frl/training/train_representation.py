#!/usr/bin/env python3
"""
Minimal training script for representation learning.

This script trains a simple encoder using InfoNCE contrastive loss:
- Input: features.type_encoder_input
- Encoder: Conv2D kernel=1, two layers [128, 64]
- Output: z_type (64-dim embedding per pixel)
- Loss: InfoNCE with auxiliary distance-based pair selection

Usage:
    python scripts/train_representation.py

    # Or with custom config paths:
    python scripts/train_representation.py \
        --bindings config/frl_binding_v1.yaml \
        --training config/frl_training_v1.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from data.sampling import sample_anchors_grid_plus_supplement
from data.sampling.anchor_sampling import AnchorSampler, build_anchor_sampler
from models import RepresentationModel
from losses import contrastive_loss, pairs_mutual_knn, pairs_mutual_knn_chunked, pairs_quantile, pairs_with_spatial_constraint
from losses.phase_pairs import build_phase_pairs
from losses.phase_neighborhood import (
    build_phase_neighborhood_batch,
    compute_phase_spread_ranking,
    phase_neighborhood_loss,
)
from losses.triplet_phase import phase_recovery_discrimination_loss
from losses.variance_covariance import variance_covariance_loss
from losses.evt_soft_neighborhood import EvtDiffusionMetric, evt_soft_neighborhood_loss
from utils import (
    compute_spatial_distances,
    extract_at_locations,
    extract_temporal_at_locations,
    spatial_knn_pairs,
    spatial_negative_pairs,
)

logger = logging.getLogger(__name__)


def compute_input_dropout_rate(
    schedule_cfg: float | dict,
    epoch: int,
    total_epochs: int,
) -> float:
    """Return the input dropout rate for the current epoch.

    Args:
        schedule_cfg: Either a scalar float (constant rate) or a dict with keys:
            - schedule: 'constant' | 'linear' | 'cosine'
            - For 'constant': rate (float)
            - For 'linear' / 'cosine': start, end, epochs (ramp length)
        epoch: Current epoch index (0-based).
        total_epochs: Total training epochs (used as ramp length fallback).

    Returns:
        Dropout probability for this epoch.
    """
    if isinstance(schedule_cfg, (int, float)):
        return float(schedule_cfg)

    schedule = schedule_cfg.get("schedule", "constant")

    if schedule == "constant":
        return float(schedule_cfg.get("rate", 0.0))

    start = float(schedule_cfg.get("start", 0.0))
    end = float(schedule_cfg.get("end", 0.1))
    ramp_epochs = int(schedule_cfg.get("epochs", total_epochs))
    t = min(epoch / max(ramp_epochs, 1), 1.0)

    if schedule == "linear":
        return start + t * (end - start)
    elif schedule == "cosine":
        return start + (end - start) * (1 - math.cos(math.pi * t)) / 2
    else:
        raise ValueError(f"Unknown input_dropout schedule: {schedule!r}")


def pair_l2(a: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    # a: [N, C], pairs: [P, 2] -> returns [P]
    v1 = a[pairs[:, 0]]
    v2 = a[pairs[:, 1]]
    return torch.norm(v1 - v2, dim=1)


def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    config: dict,
    training: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    phase_sampler: AnchorSampler | None = None,
    phase_config: dict | None = None,
    spread_config: dict | None = None,
    recovery_disc_config: dict | None = None,
    epoch: int = 0,
    evt_metric: EvtDiffusionMetric | None = None,
    evt_sampler: AnchorSampler | None = None,
) -> dict:
    """
    Process a single batch for training or validation.

    Args:
        batch: Batch from dataloader
        feature_builder: FeatureBuilder instance
        model: RepresentationModel (encoder + spatial conv)
        device: Device to use
        config: Loss config dict
        training: If True, run in training mode (gradients, optimizer step).
                  If False, run in eval mode (no gradients, no jitter).
        optimizer: Optimizer (required if training=True)
        phase_sampler: Optional anchor sampler for phase loss pair construction
        phase_config: Optional phase loss config dict (k, min_overlap, etc.)
        epoch: Current epoch (0-indexed), used for curriculum weighting
        evt_sampler: Optional EVT-stratified anchor sampler; oversamples rare EVT codes

    Returns:
        Dict with loss values and stats
    """
    if training:
        if optimizer is None:
            raise ValueError("optimizer is required when training=True")
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # Use jitter only during training
    jitter_radius = config.get('jitter_radius', 4) if training else 0

    # Process each sample in batch (pair generation is per-patch)
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_phase_loss = 0.0
    total_phase_spread_loss = 0.0
    total_phase_recovery_disc_loss = 0.0
    total_vcr_loss = 0.0
    total_phase_vcr_loss = 0.0
    total_evt_loss = 0.0
    all_evt_diag: list[dict] = []  # accumulate per-sample EVT diagnostics
    n_valid = 0
    total_spectral_pos_pairs = 0
    total_spectral_neg_pairs = 0
    total_spatial_pos_pairs = 0
    total_spatial_neg_pairs = 0

    # Collectors for distribution logging (accumulated across samples)
    all_gate_values = []
    all_pos_weights = []
    all_neg_weights = []
    all_pos_sims = []
    all_neg_sims = []
    all_pos_spec_dists = []
    all_neg_spec_dists = []
    _TAU_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    tau_sweep_pos: dict[float, list] = {t: [] for t in _TAU_SWEEP}
    tau_sweep_neg: dict[float, list] = {t: [] for t in _TAU_SWEEP}

    # Phase pair stats accumulators
    all_phase_pair_stats = []
    all_phase_loss_stats = []

    # FiLM data-dependent stats accumulators
    all_film_gamma = []
    all_film_beta = []
    # Pre-FiLM type-leakage accumulators: h mean-pooled over T, and z_type
    all_pre_film_h_mean = []   # [N, zp] per batch
    all_z_type_at_phase = []   # [N, z_type_dim] per batch

    # Collectors for global spectral loss (pairs built cross-batch after loop)
    cross_patch_z_anchors: list[torch.Tensor] = []
    cross_patch_spec_features: list[torch.Tensor] = []
    cross_patch_anchor_coords: list[torch.Tensor] = []

    # Collectors for cross-batch phase loss (assembled after loop like spectral loss)
    cross_phase_z_type: list[torch.Tensor] = []
    cross_phase_spec: list[torch.Tensor] = []
    cross_phase_embeddings: list[torch.Tensor] = []
    cross_phase_ysfc: list[torch.Tensor] = []
    cross_phase_pairs: list[torch.Tensor] = []
    cross_phase_weights: list[torch.Tensor] = []
    cross_phase_dynamism: list[torch.Tensor] = []
    cross_phase_n_offset: int = 0  # running pixel count for pair index remapping

    # Hoist epoch-dependent curriculum weights (same for every patch in the batch)
    if phase_config is not None:
        _start = phase_config.get('curriculum_start_epoch', 10)
        _ramp  = phase_config.get('curriculum_ramp_epochs', 10)
        if epoch < _start:
            curriculum_w = 0.0
        elif epoch >= _start + _ramp:
            curriculum_w = 1.0
        else:
            curriculum_w = (epoch - _start) / _ramp
    else:
        curriculum_w = 0.0

    if spread_config is not None:
        _s_start = spread_config['curriculum_start_epoch']
        _s_ramp  = spread_config['curriculum_ramp_epochs']
        if epoch < _s_start:
            spread_w = 0.0
        elif epoch >= _s_start + _s_ramp:
            spread_w = 1.0
        else:
            spread_w = (epoch - _s_start) / _s_ramp
    else:
        spread_w = 0.0

    if recovery_disc_config is not None:
        _rd_start = recovery_disc_config['curriculum_start_epoch']
        _rd_ramp  = recovery_disc_config['curriculum_ramp_epochs']
        if epoch < _rd_start:
            rd_w = 0.0
        elif epoch >= _rd_start + _rd_ramp:
            rd_w = 1.0
        else:
            rd_w = (epoch - _rd_start) / _rd_ramp
    else:
        rd_w = 0.0

    batch_size = len(batch['metadata'])

    for i in range(batch_size):
        # Extract single sample from batch
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != 'metadata'
        }
        sample['metadata'] = batch['metadata'][i]

        # Build features
        encoder_feature = feature_builder.build_feature(config['type_encoder_feature'], sample)
        spec_dist_feature = feature_builder.build_feature('infonce_type_spectral', sample)

        # Convert to tensors
        encoder_data = torch.from_numpy(encoder_feature.data).float().to(device)
        spec_dist_data = torch.from_numpy(spec_dist_feature.data).float().to(device)
        mask = torch.from_numpy(encoder_feature.mask).to(device)

        # Also apply distance feature mask
        spec_dist_mask = torch.from_numpy(spec_dist_feature.mask).to(device)
        combined_mask = mask & spec_dist_mask

        # Sample anchor locations — use EVT-stratified sampler when available so
        # rare forest types are represented for cross-batch kNN pair construction.
        if evt_sampler is not None:
            anchors = evt_sampler(combined_mask, training=training, sample=sample)
        else:
            anchors = sample_anchors_grid_plus_supplement(
                combined_mask,
                stride=config.get('stride', 16),
                border=config.get('border', 16),
                jitter_radius=jitter_radius,
                supplement_n=config.get('supplement_n', 104),
            )

        if anchors.shape[0] < 10:
            continue

        # Extract features at anchor locations
        # spec_dist_at_anchors collected here; pairs built cross-batch after loop.
        encoder_at_anchors = extract_at_locations(encoder_data, anchors)
        spec_dist_at_anchors = extract_at_locations(spec_dist_data, anchors)

        # --- Spatial InfoNCE Loss ---
        # Use efficient offset-based pair generation (no large distance matrix)

        # Get spatial positive pairs (k nearest neighbors within max_radius)
        pos_anchor_idx, pos_neighbor_coords = spatial_knn_pairs(
            anchors,
            combined_mask,
            k=config.get('spatial_positive_k', 4),
            max_radius=int(config.get('spatial_positive_max_dist', 8)),
        )

        # Get spatial negative pairs (random sample from distance range)
        neg_anchor_idx, neg_neighbor_coords = spatial_negative_pairs(
            anchors,
            combined_mask,
            min_distance=config.get('spatial_negative_min_dist', 16.0),
            max_distance=config.get('spatial_negative_max_dist', None),
            n_per_anchor=config.get('spatial_negatives_per_anchor', 4),
        )

        # Build coordinate-to-index mapping for spatial loss
        # Collect all unique coordinates: anchors + positive neighbors + negative neighbors
        all_spatial_coords = [anchors]
        if pos_neighbor_coords.numel() > 0:
            all_spatial_coords.append(pos_neighbor_coords)
        if neg_neighbor_coords.numel() > 0:
            all_spatial_coords.append(neg_neighbor_coords)

        all_spatial_coords = torch.cat(all_spatial_coords, dim=0)  # [N+M+K, 2]

        # Get unique coordinates
        unique_coords, inverse_indices = torch.unique(
            all_spatial_coords, dim=0, return_inverse=True
        )

        # Map anchor indices (first N in all_spatial_coords)
        n_anchors_spatial = anchors.shape[0]
        anchor_to_unique = inverse_indices[:n_anchors_spatial]

        # Convert pairs to index into unique_coords
        n_pos = pos_neighbor_coords.shape[0] if pos_neighbor_coords.numel() > 0 else 0
        n_neg = neg_neighbor_coords.shape[0] if neg_neighbor_coords.numel() > 0 else 0

        spatial_pos_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
        spatial_neg_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)

        if n_pos > 0:
            # pos_neighbor indices in all_spatial_coords: [n_anchors_spatial : n_anchors_spatial + n_pos]
            pos_neighbor_unique = inverse_indices[n_anchors_spatial : n_anchors_spatial + n_pos]
            pos_anchor_unique = anchor_to_unique[pos_anchor_idx]
            spatial_pos_pairs = torch.stack([pos_anchor_unique, pos_neighbor_unique], dim=1)

        if n_neg > 0:
            # neg_neighbor indices in all_spatial_coords: [n_anchors_spatial + n_pos : ]
            neg_neighbor_unique = inverse_indices[n_anchors_spatial + n_pos :]
            neg_anchor_unique = anchor_to_unique[neg_anchor_idx]
            spatial_neg_pairs = torch.stack([neg_anchor_unique, neg_neighbor_unique], dim=1)
            
        # --- Spectral weighting for spatial pairs ---
        # Use spec_dist_data (Mahalanobis space) to measure spectral similarity at spatial coordinates
        spec_dist_unique = extract_at_locations(spec_dist_data, unique_coords)  # [Nuniq, Cdist]

        tau = config.get("spatial_spectral_tau", 1.0)  # tune this
        min_w = config.get("spatial_min_w", 0.05)
        
        pos_weights = None
        neg_weights = None
        
        if spatial_pos_pairs.numel() > 0:
            dpos = pair_l2(spec_dist_unique, spatial_pos_pairs)
            pos_weights = torch.exp(-dpos / tau).clamp(min=min_w, max=1.0)
            all_pos_weights.append(pos_weights.detach().cpu())
            all_pos_spec_dists.append(dpos.detach().cpu())
            dpos_cpu = dpos.detach().cpu()
            for t in _TAU_SWEEP:
                tau_sweep_pos[t].append(torch.exp(-dpos_cpu / t).clamp(min=min_w, max=1.0))

        if spatial_neg_pairs.numel() > 0:
            dneg = pair_l2(spec_dist_unique, spatial_neg_pairs)
            neg_weights = (1.0 - torch.exp(-dneg / tau)).clamp(min=min_w, max=1.0)
            all_neg_weights.append(neg_weights.detach().cpu())
            all_neg_spec_dists.append(dneg.detach().cpu())
            dneg_cpu = dneg.detach().cpu()
            for t in _TAU_SWEEP:
                tau_sweep_neg[t].append((1.0 - torch.exp(-dneg_cpu / t)).clamp(min=min_w, max=1.0))

        # Check if we have valid pairs for losses
        # Spectral: pairs are built cross-batch after the loop; just need valid anchors.
        has_spectral = spec_dist_at_anchors.shape[0] > 0
        has_spatial = spatial_pos_pairs.shape[0] > 0 and spatial_neg_pairs.shape[0] > 0

        if not has_spectral and not has_spatial:
            continue

        # Encode the full patch for efficient embedding extraction
        # encoder_data: [C, H, W] -> [1, C, H, W] -> model -> [1, D, H, W]
        encoder_input_full = encoder_data.unsqueeze(0)
        z_full, gate = model(encoder_input_full, return_gate=True)  # [1, D, H, W] each
        z_full = z_full.squeeze(0)  # [D, H, W]
        gate = gate.squeeze(0)  # [D, H, W]

        # Collect gate values on CPU — kept for the full epoch, avoid GPU fragmentation
        all_gate_values.append(gate.detach().flatten().cpu())

        # Extract embeddings at anchor locations for spectral loss
        z_anchors = extract_at_locations(z_full, anchors)  # [num_anchors, D]

        # EVT soft neighbourhood loss
        evt_loss_val = torch.tensor(0.0, device=device)
        if evt_metric is not None:
            evt_feature = feature_builder.build_feature('evt_class', sample)
            evt_data = torch.from_numpy(evt_feature.data).long().to(device)  # [1, H, W]
            if evt_sampler is not None:
                # Draw EVT-stratified anchors (oversamples rare EVT codes)
                evt_anchors = evt_sampler(combined_mask, training=training, sample=sample)
                z_evt = extract_at_locations(z_full, evt_anchors)   # [M, D]
                evt_at_anchors = extract_at_locations(evt_data, evt_anchors).squeeze(1)  # [M]
            else:
                z_evt = z_anchors
                evt_at_anchors = extract_at_locations(evt_data, anchors).squeeze(1)  # [N]
            evt_raw, evt_diag = evt_soft_neighborhood_loss(
                z_evt,
                evt_at_anchors,
                evt_metric,
                tau_ref=config.get('evt_tau_ref', 0.5),
                tau_learned=config.get('evt_tau_learned', 0.5),
            )
            all_evt_diag.append(evt_diag)
            evt_loss_val = config.get('evt_weight', 0.0) * evt_raw

        # Compute variance-covariance regularization on type embeddings
        vcr_loss_val = torch.tensor(0.0, device=device)
        if config.get('vcr_enabled', False) and z_anchors.shape[0] >= 2:
            vcr_total, _, _ = variance_covariance_loss(
                z_anchors,
                variance_weight=config.get('vcr_variance_weight', 1.0),
                covariance_weight=config.get('vcr_covariance_weight', 1.0),
                variance_target=config.get('vcr_variance_target', 1.0),
            )
            vcr_loss_val = config.get('vcr_weight', 0.1) * vcr_total

        # cross_patch_z_anchors is populated after the NaN check below so that
        # samples with non-finite loss don't corrupt the cross-batch spectral computation.

        # Compute spatial loss
        spatial_loss_val = torch.tensor(0.0, device=device)
        if has_spatial:
            # Extract embeddings at unique coordinate locations
            z_spatial = extract_at_locations(z_full, unique_coords)  # [num_unique, D]

            spatial_loss_val = contrastive_loss(
                z_spatial,
                spatial_pos_pairs,
                spatial_neg_pairs,
                pos_weights=pos_weights,
                neg_weights=neg_weights,
                temperature=config.get('spatial_temperature', 0.07),
                similarity='cosine',
            )

            # Collect cosine similarities for diagnostics (z_spatial is unit-norm)
            with torch.no_grad():
                p_a, p_b = z_spatial[spatial_pos_pairs[:, 0]], z_spatial[spatial_pos_pairs[:, 1]]
                n_a, n_b = z_spatial[spatial_neg_pairs[:, 0]], z_spatial[spatial_neg_pairs[:, 1]]
                all_pos_sims.append((p_a * p_b).sum(1).cpu())
                all_neg_sims.append((n_a * n_b).sum(1).cpu())

        # --- Phase pair construction + loss ---
        # Pair construction (kNN + overlap) runs on CPU.
        # Loss computation runs on GPU (requires gradients through phase encoder).
        phase_loss_val = torch.tensor(0.0, device=device)
        phase_spread_loss_val = torch.tensor(0.0, device=device)
        phase_recovery_disc_loss_val = torch.tensor(0.0, device=device)
        phase_vcr_loss_val = torch.tensor(0.0, device=device)
        if phase_sampler is not None and phase_config is not None:
            # Build ysfc feature via FeatureBuilder (returns numpy)
            ysfc_feature = feature_builder.build_feature('ysfc', sample)
            ysfc_data = torch.from_numpy(ysfc_feature.data).float()
            # ysfc_data: [1, T, H, W] (single channel, CPU)

            # Combined mask for phase anchors: encoder mask AND ysfc validity
            ysfc_mask = torch.from_numpy(ysfc_feature.mask)
            # ysfc_mask is [T, H, W] for temporal; collapse to [H, W]
            if ysfc_mask.ndim == 3:
                ysfc_spatial_mask = ysfc_mask.all(dim=0)  # valid across all timesteps
            else:
                ysfc_spatial_mask = ysfc_mask
            phase_mask = combined_mask.cpu() & ysfc_spatial_mask

            # Sample separate anchors for phase loss
            phase_anchors = phase_sampler(
                phase_mask, training=training, sample=sample
            )

            if phase_anchors.shape[0] >= 10:
                # Extract spectral features at phase anchors (CPU for kNN + weights)
                phase_spec_at_anchors = extract_at_locations(
                    spec_dist_data.cpu(), phase_anchors
                )  # [N_phase, C]

                # Extract ysfc time series at phase anchors
                # ysfc_data is [1, T, H, W]; squeeze channel dim for extraction
                ysfc_at_anchors = extract_temporal_at_locations(
                    ysfc_data, phase_anchors
                )  # [N_phase, T, 1]
                ysfc_at_anchors = ysfc_at_anchors.squeeze(-1)  # [N_phase, T]

                # Build pairs (all CPU)
                phase_pairs, phase_weights, phase_stats = build_phase_pairs(
                    spec_features=phase_spec_at_anchors,
                    ysfc=ysfc_at_anchors,
                    k=phase_config.get('k', 16),
                    min_overlap=phase_config.get('min_overlap', 3),
                    min_pairs=phase_config.get('min_pairs', 5),
                    include_self=phase_config.get('include_self', True),
                    sigma=phase_config.get('sigma', 5.0),
                    self_pair_weight=phase_config.get('self_pair_weight', 1.0),
                )

                all_phase_pair_stats.append(phase_stats)

                # --- Accumulate phase data for cross-batch loss (computed after loop) ---
                if phase_pairs.shape[0] > 0 and curriculum_w > 0.0:
                        # Build phase_ccdc temporal feature
                        phase_ccdc_feature = feature_builder.build_feature(config['phase_encoder_feature'], sample)
                        phase_ccdc_data = torch.from_numpy(
                            phase_ccdc_feature.data
                        ).float().to(device)

                        # Extract only anchor pixel time-series (avoid dense TCN)
                        phase_anchors_dev = phase_anchors.to(device)
                        phase_ccdc_at_anchors = extract_temporal_at_locations(
                            phase_ccdc_data, phase_anchors_dev
                        )  # [N_phase, T, C]
                        spec_at_phase_anchors = phase_ccdc_at_anchors  # [N_phase, T, C]
                        phase_ccdc_at_anchors = phase_ccdc_at_anchors.permute(0, 2, 1)  # [N_phase, C, T]

                        # Extract z_type at anchor locations (stop-grad)
                        z_type_at_anchors = extract_at_locations(
                            z_full.detach(), phase_anchors_dev
                        )  # [N_phase, z_type_dim]

                        # Run phase encoder on anchor pixels only
                        z_phase_at_anchors, film_gamma, film_beta, pre_film_h = model.forward_phase_at_locations(
                            phase_ccdc_at_anchors, z_type_at_anchors,
                            return_film=True,
                            return_pre_film=True,
                        )  # [N_phase, T, 12], [N_phase, 12], [N_phase, 12], [N_phase, 12, T]
                        all_film_gamma.append(film_gamma.detach())
                        all_film_beta.append(film_beta.detach())
                        all_pre_film_h_mean.append(pre_film_h.mean(dim=2).detach())
                        all_z_type_at_phase.append(z_type_at_anchors.detach())

                        # Phase VCR: prevent dimensional collapse in z_phase (per-patch)
                        phase_vcr_cfg = config.get('phase_vcr_config')
                        if phase_vcr_cfg is not None:
                            z_phase_flat = z_phase_at_anchors.reshape(-1, 12)
                            pvcr_total, _, _ = variance_covariance_loss(
                                z_phase_flat,
                                variance_weight=phase_vcr_cfg.get('variance_weight', 1.0),
                                covariance_weight=phase_vcr_cfg.get('covariance_weight', 1.0),
                                variance_target=phase_vcr_cfg.get('variance_target', 1.0),
                            )
                            phase_vcr_loss_val = phase_vcr_cfg.get('weight', 0.1) * curriculum_w * pvcr_total
                        else:
                            phase_vcr_loss_val = torch.tensor(0.0, device=device)

                        # Accumulate for cross-batch phase loss (offset pair indices)
                        ysfc_at_anchors_gpu = ysfc_at_anchors.to(device)
                        cross_phase_z_type.append(z_type_at_anchors)
                        cross_phase_spec.append(spec_at_phase_anchors)
                        cross_phase_embeddings.append(z_phase_at_anchors)
                        cross_phase_ysfc.append(ysfc_at_anchors_gpu)
                        cross_phase_pairs.append(phase_pairs.to(device) + cross_phase_n_offset)
                        cross_phase_weights.append(phase_weights.to(device))
                        cross_phase_n_offset += z_type_at_anchors.shape[0]

                        # Accumulate dynamism for spread loss
                        if spread_config is not None:
                            dynamism_data = torch.from_numpy(
                                feature_builder.build_feature(
                                    'phase_dynamism_supervision', sample
                                ).data
                            ).float()
                            cross_phase_dynamism.append(
                                extract_at_locations(dynamism_data, phase_anchors)
                            )

        # Combine per-patch losses (phase losses computed cross-batch after loop)
        spatial_weight = config.get('spatial_loss_weight', 1.0)
        loss = (spatial_weight * spatial_loss_val
                + vcr_loss_val
                + phase_vcr_loss_val
                + evt_loss_val)

        # Skip if loss is NaN or Inf (numerical instability)
        if not torch.isfinite(loss):
            md = sample['metadata']
            sw = md.get('spatial_window')
            origin = (sw.row_start, sw.col_start) if sw is not None else '?'
            patch_idx = md.get('patch_idx', '?')
            def _fmt(t):
                v = t.item() if hasattr(t, 'item') else float(t)
                return f"{v:.4f}"
            logger.warning(
                f"Skipping sample with non-finite loss: {loss.item()} | "
                f"patch_idx={patch_idx} origin={origin} | "
                f"spat={_fmt(spatial_loss_val)} "
                f"phase={_fmt(phase_loss_val)} "
                f"spr={_fmt(phase_spread_loss_val)} "
                f"vcr={_fmt(vcr_loss_val)} "
                f"pvcr={_fmt(phase_vcr_loss_val)}"
            )
            continue

        # Accumulate cross-batch spectral inputs only for finite-loss samples.
        if has_spectral:
            cross_patch_z_anchors.append(model.project_type(z_anchors))
            cross_patch_spec_features.append(spec_dist_at_anchors)
            cross_patch_anchor_coords.append(anchors)

        # Accumulate: keep as tensor for training (backward), use .item() for validation
        if training:
            total_loss += loss
            total_spatial_loss += spatial_loss_val
            total_vcr_loss += vcr_loss_val
            total_phase_vcr_loss += phase_vcr_loss_val
            total_evt_loss += evt_loss_val
        else:
            total_loss += loss.item()
            total_spatial_loss += spatial_loss_val.item()
            total_vcr_loss += vcr_loss_val.item()
            total_phase_vcr_loss += phase_vcr_loss_val.item()
            total_evt_loss += evt_loss_val.item()
        n_valid += 1
        total_spatial_pos_pairs += spatial_pos_pairs.shape[0]
        total_spatial_neg_pairs += spatial_neg_pairs.shape[0]

    # --- Global Spectral InfoNCE with Cross-Batch kNN ---
    # All anchors from all samples in the batch are pooled into a single
    # N_total × N_total spectral distance matrix. Positive pairs are mutual
    # kNN across the full pool — spectrally similar pixels from *different*
    # patches can now be positives, teaching location-invariant forest type.
    # Negatives are sampled from the [q_low, q_high) quantile range of spectral
    # distances and weighted by 1 - exp(-d/tau) to suppress false negatives
    # (cross-patch pairs that happen to be spectrally similar).
    spectral_weight = config.get('spectral_loss_weight', 1.0)
    global_spectral_loss_val = torch.tensor(0.0, device=device)
    spectral_neg_tau_sweep: dict = {}
    if cross_patch_z_anchors:
        n_patches = len(cross_patch_z_anchors)
        z_all = torch.cat(cross_patch_z_anchors, dim=0)        # [N_total, D]
        spec_all = torch.cat(cross_patch_spec_features, dim=0) # [N_total, C]

        offsets: list[int] = [0]
        for z in cross_patch_z_anchors:
            offsets.append(offsets[-1] + z.shape[0])

        # --- Positive pairs: chunked mutual kNN ---
        # Processes chunk_size queries at a time — peak memory O(chunk × N_total).
        # Keeps all anchors; no subsampling needed.
        global_pos = pairs_mutual_knn_chunked(
            spec_all,
            cross_patch_anchor_coords,
            offsets,
            k=config.get('positive_k', 16),
            pos_min_spatial=config.get('positive_min_spatial', 4.0),
            chunk_size=config.get('spectral_knn_chunk_size', 128),
        )

        # --- Negative pairs: random cross-patch sampling ---
        # Cross-patch pairs automatically satisfy the spatial distance constraint.
        # We sample randomly rather than computing the full distance matrix;
        # spectral weights handle false negatives (low weight ≈ spectrally similar).
        tau_neg = config.get('spectral_neg_tau', 1.0)
        min_w = config.get('spectral_neg_min_weight', 0.05)
        # Scale neg count by N_total so each anchor gets ~neg_per_anchor negatives on average.
        # (Analogous to spatial InfoNCE which uses spatial_negatives_per_anchor * N.)
        N_total = z_all.shape[0]
        neg_per_anchor = config.get('spectral_neg_per_anchor', 20)
        n_neg = neg_per_anchor * N_total
        n_patch_pairs = n_patches * (n_patches - 1)
        n_per = max(1, n_neg // n_patch_pairs) if n_patch_pairs > 0 else 0

        neg_i_parts, neg_j_parts = [], []
        for pi in range(n_patches):
            for pj in range(n_patches):
                if pi == pj:
                    continue
                is_s, is_e = offsets[pi], offsets[pi + 1]
                js_s, js_e = offsets[pj], offsets[pj + 1]
                neg_i_parts.append(torch.randint(is_s, is_e, (n_per,), device=device))
                neg_j_parts.append(torch.randint(js_s, js_e, (n_per,), device=device))

        global_neg_i = torch.cat(neg_i_parts)
        global_neg_j = torch.cat(neg_j_parts)
        global_neg = torch.stack([global_neg_i, global_neg_j], dim=1)

        # Distances computed only for sampled pairs — O(n_neg × C), not O(N²)
        neg_spec_dist = torch.norm(spec_all[global_neg_i] - spec_all[global_neg_j], dim=1)
        neg_weights = (1.0 - torch.exp(-neg_spec_dist / tau_neg)).clamp(min=min_w, max=1.0)
        if epoch == 0:
            _nsd_cpu = neg_spec_dist.detach().cpu()
            spectral_neg_tau_sweep = {
                t: {
                    'neg_mean': (1.0 - torch.exp(-_nsd_cpu / t)).clamp(min=min_w, max=1.0).mean().item(),
                    'neg_q25':  torch.quantile((1.0 - torch.exp(-_nsd_cpu / t)).clamp(min=min_w, max=1.0), 0.25).item(),
                    'neg_q50':  torch.quantile((1.0 - torch.exp(-_nsd_cpu / t)).clamp(min=min_w, max=1.0), 0.50).item(),
                }
                for t in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
            }
        else:
            spectral_neg_tau_sweep = {}

        global_spectral_loss_val = contrastive_loss(
            z_all, global_pos, global_neg,
            temperature=config.get('temperature', 0.07),
            similarity='l2',
            neg_weights=neg_weights,
        )
        total_spectral_pos_pairs = global_pos.shape[0]
        total_spectral_neg_pairs = global_neg.shape[0]

    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    empty_phase_stats = {
        'n_anchors': 0, 'n_anchors_surviving': 0,
        'n_candidates': 0, 'n_after_overlap': 0,
        'n_self_pairs': 0, 'n_total_pairs': 0,
        'overlap_mean': 0.0, 'overlap_min': 0,
        'weight_mean': 0.0, 'weight_std': 0.0,
        'dist_mean': 0.0, 'dist_std': 0.0,
        'dist_q25': 0.0, 'dist_q50': 0.0, 'dist_q75': 0.0,
        'dist_min': 0.0, 'dist_max': 0.0,
    }
    empty_phase_loss_stats = {
        'n_pairs_input': 0, 'n_pairs_sufficient_overlap': 0,
        'loss_self': 0.0, 'loss_cross': 0.0,
        'curriculum_w': 0.0,
        # Entropy of reference (p) and learned (q) distributions
        'self_mean_entropy_p': 0.0, 'self_mean_entropy_q': 0.0,
        'self_mean_overlap': 0.0,
        'cross_mean_entropy_p': 0.0, 'cross_mean_entropy_q': 0.0,
        'cross_mean_overlap': 0.0,
        # Distance distributions that tau operates on
        'd_ref_self_mean': 0.0, 'd_ref_self_std': 0.0,
        'd_ref_self_q25': 0.0, 'd_ref_self_q50': 0.0, 'd_ref_self_q75': 0.0,
        'd_ref_cross_mean': 0.0, 'd_ref_cross_std': 0.0,
        'd_ref_cross_q25': 0.0, 'd_ref_cross_q50': 0.0, 'd_ref_cross_q75': 0.0,
    }

    # Aggregate phase pair stats across samples
    def aggregate_phase_stats(stats_list: list[dict]) -> dict:
        if not stats_list:
            return empty_phase_stats
        agg = {}
        for key in empty_phase_stats:
            vals = [s[key] for s in stats_list]
            if key.startswith('n_'):
                agg[key] = sum(vals) / len(vals)  # mean per sample
            else:
                agg[key] = sum(vals) / len(vals)  # mean
        return agg

    def aggregate_phase_loss_stats(stats_list: list[dict]) -> dict:
        if not stats_list:
            return empty_phase_loss_stats
        agg = {}
        for key in empty_phase_loss_stats:
            vals = [s.get(key, 0.0) for s in stats_list]
            agg[key] = sum(vals) / len(vals)
        return agg

    _empty_evt_diag = dict(
        mean_entropy_ref=0.0, mean_entropy_learned=0.0,
        median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
        d_lrn_confused=0.0, d_lrn_noncf=0.0,
        n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
    )
    if n_valid == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0,
            'phase_loss': 0.0, 'phase_spread_loss': 0.0, 'phase_recovery_disc_loss': 0.0,
            'vcr_loss': 0.0, 'phase_vcr_loss': 0.0,
            'evt_loss': 0.0, 'evt_diag': _empty_evt_diag,
            'n_valid': 0,
            'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
            'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'phase_pair_stats': empty_phase_stats,
            'phase_loss_stats': empty_phase_loss_stats,
        }

    # --- Cross-batch phase losses ---
    # Phase neighborhood, spread, and recovery discrimination losses are computed
    # once over all patches in the batch. Spectral reference features are demeaned
    # by a type-local baseline computed via SVD rank reduction + kNN in type space.
    cross_phase_loss_val = torch.tensor(0.0, device=device)
    cross_phase_spread_val = torch.tensor(0.0, device=device)
    cross_phase_rd_val = torch.tensor(0.0, device=device)

    if cross_phase_embeddings:
        Z = torch.cat(cross_phase_z_type, dim=0)           # [N_total, z_type_dim]
        spec_all = torch.cat(cross_phase_spec, dim=0)      # [N_total, T, C]
        z_phase_all = torch.cat(cross_phase_embeddings, dim=0)  # [N_total, T, 12]
        ysfc_all = torch.cat(cross_phase_ysfc, dim=0)      # [N_total, T]
        pairs_all = torch.cat(cross_phase_pairs, dim=0)    # [B_total, 2]
        weights_all = torch.cat(cross_phase_weights, dim=0)  # [B_total]

        # SVD rank reduction on z_type for stable kNN type baseline
        K = phase_config.get('phase_type_proj_rank', 8)
        k_nbrs = phase_config.get('phase_type_proj_neighbors', 20)
        Z_c = Z - Z.mean(0, keepdim=True)
        U, _, _ = torch.linalg.svd(Z_c, full_matrices=False)
        Z_k = U[:, :K]  # [N_total, K]

        # kNN in reduced type space → type-local spectral baseline per pixel
        sim = Z_k @ Z_k.T  # [N_total, N_total]
        sim.fill_diagonal_(float('-inf'))
        k_nbrs = min(k_nbrs, sim.shape[0] - 1)
        topk_idx = sim.topk(k_nbrs, dim=1).indices  # [N_total, k_nbrs]
        S_mean = spec_all.mean(dim=1)                # [N_total, C] — pixel mean over T
        S_hat = S_mean[topk_idx].mean(dim=1)         # [N_total, C] — type-local baseline
        spec_demeaned = spec_all - S_hat.unsqueeze(1)  # [N_total, T, C]

        # Build aligned distance batch once; reuse for neighborhood + spread losses
        phase_batch = build_phase_neighborhood_batch(
            spectral_features=spec_demeaned,
            phase_embeddings=z_phase_all,
            ysfc=ysfc_all,
            pair_indices=pairs_all,
            min_overlap=phase_config.get('min_overlap', 3),
        )

        # Phase neighborhood loss
        p_loss, p_loss_stats = phase_neighborhood_loss(
            spectral_features=spec_demeaned,
            phase_embeddings=z_phase_all,
            ysfc=ysfc_all,
            pair_indices=pairs_all,
            pair_weights=weights_all,
            tau_ref=phase_config.get('tau_ref', 0.1),
            tau_learned=phase_config.get('tau_learned', 0.1),
            min_overlap=phase_config.get('min_overlap', 3),
            min_valid_per_row=phase_config.get('min_valid_per_row', 2),
            self_similarity_weight=phase_config.get('self_similarity_weight', 1.0),
            cross_pixel_weight=phase_config.get('cross_pixel_weight', 1.0),
            _batch=phase_batch,
        )
        cross_phase_loss_val = phase_config.get('weight', 1.0) * curriculum_w * p_loss
        p_loss_stats['curriculum_w'] = curriculum_w
        all_phase_loss_stats.append(p_loss_stats)

        # Phase spread ranking loss
        if spread_config is not None and spread_w > 0.0 and cross_phase_dynamism:
            dynamism_all = torch.cat(cross_phase_dynamism, dim=0)  # [N_total, C]
            dynamism_ref = dynamism_all.mean(dim=1).to(device)     # [N_total]
            valid_mask = phase_batch['valid_pair_mask']
            if valid_mask.any():
                idx_i_v = pairs_all[valid_mask, 0]
                idx_j_v = pairs_all[valid_mask, 1]
                spread_loss, _ = compute_phase_spread_ranking(
                    batch_result=phase_batch,
                    idx_i_valid=idx_i_v,
                    idx_j_valid=idx_j_v,
                    dynamism_ref=dynamism_ref,
                    margin=spread_config['margin'],
                    delta=spread_config['delta'],
                )
                cross_phase_spread_val = spread_config['weight'] * spread_w * spread_loss

        # Phase recovery discrimination loss
        if recovery_disc_config is not None and rd_w > 0.0:
            rd_loss, _ = phase_recovery_discrimination_loss(
                z_phase=z_phase_all,
                ysfc=ysfc_all,
                margin=recovery_disc_config['margin'],
                low_ysfc_max=recovery_disc_config['low_ysfc_max'],
                high_ysfc_min=recovery_disc_config['high_ysfc_min'],
            )
            cross_phase_rd_val = recovery_disc_config['weight'] * rd_w * rd_loss

    # Average losses over valid samples in batch.
    # Spectral and phase losses are computed globally (cross-patch) and added on top.
    _scalar = lambda t: t.item() if hasattr(t, 'item') else float(t)
    if training:
        mean_loss = (total_loss / n_valid
                     + spectral_weight * global_spectral_loss_val
                     + cross_phase_loss_val
                     + cross_phase_spread_val
                     + cross_phase_rd_val)
        mean_spectral_loss = global_spectral_loss_val
    else:
        mean_loss = (total_loss / n_valid
                     + spectral_weight * _scalar(global_spectral_loss_val)
                     + _scalar(cross_phase_loss_val)
                     + _scalar(cross_phase_spread_val)
                     + _scalar(cross_phase_rd_val))
        mean_spectral_loss = _scalar(global_spectral_loss_val)
    mean_spatial_loss = total_spatial_loss / n_valid
    mean_phase_loss = _scalar(cross_phase_loss_val)
    mean_phase_spread_loss = _scalar(cross_phase_spread_val)
    mean_phase_recovery_disc_loss = _scalar(cross_phase_rd_val)
    mean_vcr_loss = total_vcr_loss / n_valid
    mean_phase_vcr_loss = total_phase_vcr_loss / n_valid
    mean_evt_loss = total_evt_loss / n_valid

    if training:
        # Final NaN check before backward
        if not torch.isfinite(mean_loss):
            logger.warning(f"Skipping batch with non-finite mean loss: {mean_loss.item()}")
            return {
                'loss': float('nan'), 'spectral_loss': float('nan'),
                'spatial_loss': float('nan'), 'phase_loss': float('nan'),
                'phase_spread_loss': float('nan'), 'phase_recovery_disc_loss': float('nan'),
                'vcr_loss': float('nan'), 'phase_vcr_loss': float('nan'),
                'evt_loss': float('nan'), 'evt_diag': _empty_evt_diag,
                'n_valid': 0,
                'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
                'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
                'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
                'neg_weight_stats': empty_stats,
                'phase_pair_stats': empty_phase_stats,
                'phase_loss_stats': empty_phase_loss_stats,
            }

        # Backward
        mean_loss.backward()

        # Gradient clipping
        if config.get('gradient_clip_enabled', True):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.get('gradient_clip_max_norm', 1.0)
            )

        optimizer.step()
        mean_loss = mean_loss.item()
        mean_spectral_loss = mean_spectral_loss.item()
        mean_spatial_loss = mean_spatial_loss.item()
        mean_phase_loss = mean_phase_loss.item()
        mean_phase_spread_loss = mean_phase_spread_loss.item()
        mean_phase_recovery_disc_loss = mean_phase_recovery_disc_loss.item()
        mean_vcr_loss = mean_vcr_loss.item()
        mean_phase_vcr_loss = mean_phase_vcr_loss.item()
        mean_evt_loss = mean_evt_loss.item() if hasattr(mean_evt_loss, 'item') else float(mean_evt_loss)

    # Compute distribution statistics for gate values and weights
    def compute_stats(tensors: list[torch.Tensor]) -> dict:
        """Compute summary stats from list of tensors."""
        if not tensors:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
        combined = torch.cat(tensors)
        # torch.quantile requires < 2^24 elements; subsample if needed
        _MAX_QUANTILE_ELEMS = 2 ** 23
        q_input = (
            combined[torch.randperm(len(combined), device=combined.device)[:_MAX_QUANTILE_ELEMS]]
            if len(combined) > _MAX_QUANTILE_ELEMS
            else combined
        )
        return {
            'mean': combined.mean().item(),
            'std':  combined.std().item(),
            'min':  combined.min().item(),
            'max':  combined.max().item(),
            'q25':  torch.quantile(q_input, 0.25).item(),
            'q50':  torch.quantile(q_input, 0.50).item(),
            'q75':  torch.quantile(q_input, 0.75).item(),
        }

    # Compute data-dependent FiLM stats
    film_stats = None
    if all_film_gamma:
        gamma_cat = torch.cat(all_film_gamma, dim=0)  # [total_pixels, 12]
        beta_cat = torch.cat(all_film_beta, dim=0)    # [total_pixels, 12]
        film_stats = {
            'gamma_mean': gamma_cat.mean().item(),
            'gamma_std': gamma_cat.std().item(),
            'gamma_per_dim_std': gamma_cat.std(dim=0).mean().item(),
            'beta_mean': beta_cat.mean().item(),
            'beta_std': beta_cat.std().item(),
            'beta_per_dim_std': beta_cat.std(dim=0).mean().item(),
        }

    # Type-leakage diagnostics: how much type information is in pre-FiLM h?
    type_leakage_stats = None
    if all_pre_film_h_mean and all_z_type_at_phase:
        h_cat = torch.cat(all_pre_film_h_mean, dim=0).float()   # [N, zp]
        zt_cat = torch.cat(all_z_type_at_phase, dim=0).float()  # [N, z_type_dim]
        N = h_cat.shape[0]

        # Option 1: Cross-covariance Frobenius norm
        # Demean both to get unbiased cross-covariance
        h_c = h_cat - h_cat.mean(dim=0, keepdim=True)
        zt_c = zt_cat - zt_cat.mean(dim=0, keepdim=True)
        cross_cov = (h_c.T @ zt_c) / (N - 1)  # [zp, z_type_dim]
        cross_cov_frob = cross_cov.pow(2).sum().sqrt().item()

        # Option 2: Ridge regression R² of z_type predicted from h
        # Fit closed-form ridge: W = (h^T h + λI)^{-1} h^T zt
        lam = 1e-3
        A = h_c.T @ h_c + lam * torch.eye(h_c.shape[1], device=h_c.device)
        B = h_c.T @ zt_c
        W = torch.linalg.solve(A, B)  # [zp, z_type_dim]
        pred = h_c @ W               # [N, z_type_dim]
        ss_res = (zt_c - pred).pow(2).sum(dim=0)   # [z_type_dim]
        ss_tot = zt_c.pow(2).sum(dim=0).clamp(min=1e-8)
        r2_per_dim = (1.0 - ss_res / ss_tot)       # [z_type_dim]
        r2_mean = r2_per_dim.mean().item()
        r2_max = r2_per_dim.max().item()

        type_leakage_stats = {
            'cross_cov_frob': cross_cov_frob,
            'r2_mean': r2_mean,
            'r2_max': r2_max,
        }

    # Aggregate EVT diagnostics across samples
    empty_evt_diag = dict(
        mean_entropy_ref=0.0, mean_entropy_learned=0.0,
        median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
        d_lrn_confused=0.0, d_lrn_noncf=0.0,
        n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
    )
    if all_evt_diag:
        evt_diag_agg = {
            k: sum(d.get(k, empty_evt_diag[k]) for d in all_evt_diag) / len(all_evt_diag)
            for k in empty_evt_diag
        }
    else:
        evt_diag_agg = empty_evt_diag

    return {
        'loss': mean_loss,
        'spectral_loss': mean_spectral_loss,
        'spatial_loss': mean_spatial_loss,
        'phase_loss': mean_phase_loss,
        'phase_spread_loss': mean_phase_spread_loss,
        'phase_recovery_disc_loss': mean_phase_recovery_disc_loss if not hasattr(mean_phase_recovery_disc_loss, 'item') else mean_phase_recovery_disc_loss.item(),
        'vcr_loss': mean_vcr_loss,
        'phase_vcr_loss': mean_phase_vcr_loss,
        'evt_loss': mean_evt_loss if not hasattr(mean_evt_loss, 'item') else mean_evt_loss.item(),
        'evt_diag': evt_diag_agg,
        'n_valid': n_valid,
        # Spectral pairs are cross-batch totals (not per-sample); report raw counts.
        'spectral_pos_pairs': total_spectral_pos_pairs,
        'spectral_neg_pairs': total_spectral_neg_pairs,
        'spatial_pos_pairs': total_spatial_pos_pairs // n_valid if n_valid > 0 else 0,
        'spatial_neg_pairs': total_spatial_neg_pairs // n_valid if n_valid > 0 else 0,
        'gate_stats': compute_stats(all_gate_values),
        'pos_weight_stats': compute_stats(all_pos_weights),
        'neg_weight_stats': compute_stats(all_neg_weights),
        'pos_sim_stats': compute_stats(all_pos_sims),
        'neg_sim_stats': compute_stats(all_neg_sims),
        'pos_spec_dist_stats': compute_stats(all_pos_spec_dists),
        'neg_spec_dist_stats': compute_stats(all_neg_spec_dists),
        'tau_sweep': {
            t: {
                'pos_mean': torch.cat(tau_sweep_pos[t]).mean().item() if tau_sweep_pos[t] else 0.0,
                'pos_q25':  torch.quantile(torch.cat(tau_sweep_pos[t]), 0.25).item() if tau_sweep_pos[t] else 0.0,
                'pos_q50':  torch.quantile(torch.cat(tau_sweep_pos[t]), 0.50).item() if tau_sweep_pos[t] else 0.0,
                'neg_mean': torch.cat(tau_sweep_neg[t]).mean().item() if tau_sweep_neg[t] else 0.0,
            }
            for t in _TAU_SWEEP
        },
        'phase_pair_stats': aggregate_phase_stats(all_phase_pair_stats),
        'phase_loss_stats': aggregate_phase_loss_stats(all_phase_loss_stats),
        'film_stats': film_stats,
        'type_leakage_stats': type_leakage_stats,
    }


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
) -> dict:
    """Run training on entire training set for one epoch."""
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_phase_loss = 0.0
    total_phase_spread_loss = 0.0
    total_phase_recovery_disc_loss = 0.0
    total_vcr_loss = 0.0
    total_phase_vcr_loss = 0.0
    total_evt_loss = 0.0
    all_epoch_evt_diag: list[dict] = []
    total_spectral_pos_pairs = 0
    total_spectral_neg_pairs = 0
    total_spatial_pos_pairs = 0
    total_spatial_neg_pairs = 0
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
    last_tau_sweep: dict = {}
    last_phase_pair_stats = None
    last_phase_loss_stats = None
    last_film_stats = None

    for batch_idx, batch in enumerate(train_dataloader):
        stats = process_batch(
            batch, feature_builder, model, device, config,
            training=True, optimizer=optimizer,
            phase_sampler=phase_sampler, phase_config=phase_config,
            spread_config=spread_config, recovery_disc_config=recovery_disc_config,
            epoch=epoch, evt_metric=evt_metric, evt_sampler=evt_sampler,
        )

        scheduler.step()

        if stats['n_valid'] > 0:
            total_loss += stats['loss']
            total_spectral_loss += stats['spectral_loss']
            total_spatial_loss += stats['spatial_loss']
            total_phase_loss += stats['phase_loss']
            total_phase_spread_loss += stats.get('phase_spread_loss', 0.0)
            total_phase_recovery_disc_loss += stats.get('phase_recovery_disc_loss', 0.0)
            total_vcr_loss += stats['vcr_loss']
            total_phase_vcr_loss += stats['phase_vcr_loss']
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
            last_phase_pair_stats = stats.get('phase_pair_stats')
            last_phase_loss_stats = stats.get('phase_loss_stats')
            if stats.get('film_stats') is not None:
                last_film_stats = stats['film_stats']

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

    if total_batches == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0,
            'phase_loss': 0.0, 'phase_spread_loss': 0.0, 'phase_recovery_disc_loss': 0.0,
            'vcr_loss': 0.0, 'phase_vcr_loss': 0.0,
            'evt_loss': 0.0,
            'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'pos_sim_stats': empty_stats, 'neg_sim_stats': empty_stats,
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
        'vcr_loss': total_vcr_loss / total_batches,
        'phase_vcr_loss': total_phase_vcr_loss / total_batches,
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
) -> dict:
    """Run validation on entire validation set."""
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_phase_loss = 0.0
    total_phase_spread_loss = 0.0
    total_phase_recovery_disc_loss = 0.0
    total_vcr_loss = 0.0
    total_phase_vcr_loss = 0.0
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
    last_phase_pair_stats = None
    last_phase_loss_stats = None
    last_film_stats = None

    with torch.no_grad():
        for batch in val_dataloader:
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
                total_vcr_loss += stats['vcr_loss']
                total_phase_vcr_loss += stats['phase_vcr_loss']
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
            'evt_loss': 0.0, 'evt_diag': _empty_evt_diag,
            'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
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
        'vcr_loss': total_vcr_loss / total_batches,
        'phase_vcr_loss': total_phase_vcr_loss / total_batches,
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
        'phase_pair_stats': last_phase_pair_stats,
        'phase_loss_stats': last_phase_loss_stats,
        'film_stats': last_film_stats,
    }

def main():
    parser = argparse.ArgumentParser(description='Train representation encoder')
    parser.add_argument(
        '--bindings',
        type=str,
        default=None,
        help='Path to bindings config (overrides training config bindings_path)'
    )
    parser.add_argument(
        '--training',
        type=str,
        default='config/frl_training_v1.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to model architecture config (overrides training config model_path)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints (overrides config)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing experiment directory if it exists'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='CKPT',
        help='Path to a checkpoint to resume training from. Loads model weights and '
             'optimizer state; the scheduler is reinitialized as a fresh cosine decay '
             'from the checkpoint LR over the remaining epochs (num_epochs - start_epoch).'
    )
    args = parser.parse_args()

    # Parse configs first to get defaults
    logger.info(f"Loading training config from {args.training}")
    _training_parser = TrainingConfigParser(args.training)
    training_config = _training_parser.parse()
    raw_training_config: dict = _training_parser.raw_config or {}

    bindings_path = args.bindings or training_config.config_paths.get('bindings_path')
    if bindings_path is None:
        raise ValueError("No bindings path found. Set config.bindings_path in the training YAML or pass --bindings.")
    logger.info(f"Loading bindings config from {bindings_path}")
    bindings_config = DatasetBindingsParser(bindings_path).parse()

    model_config_path = args.model_config or training_config.config_paths.get('model_path')
    if model_config_path is None:
        raise ValueError("No model config path found. Set config.model_path in the training YAML or pass --model-config.")
    logger.info(f"Loading model config from {model_config_path}")
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Apply config values with command-line overrides
    num_epochs = args.epochs or training_config.training.num_epochs
    batch_size = args.batch_size or training_config.training.batch_size
    lr = args.lr or training_config.optimizer.lr
    weight_decay = training_config.optimizer.weight_decay
    device_str = args.device or training_config.hardware.device

    # Build run directory structure: {run_root}/{experiment_name}/{ckpt_dir|log_dir}
    run_root = Path(training_config.run.run_root)
    experiment_name = training_config.run.experiment_name
    experiment_dir = run_root / experiment_name

    # Check if experiment directory already exists
    if experiment_dir.exists():
        if args.overwrite:
            logger.info(f"Overwriting existing experiment directory: {experiment_dir}")
            shutil.rmtree(experiment_dir)
        else:
            logger.error(
                f"Experiment directory already exists: {experiment_dir}. "
                f"Use --overwrite to replace it."
            )
            raise SystemExit(1)

    checkpoint_dir = args.checkpoint_dir or str(experiment_dir / training_config.run.ckpt_dir)
    log_dir = experiment_dir / training_config.run.log_dir

    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Create train dataset
    logger.info("Creating train dataset...")
    patch_size = training_config.sampling.patch_size
    spatial = training_config.spatial_domain
    debug_window = None
    if spatial.debug_mode and spatial.debug_window is not None:
        w = spatial.debug_window
        debug_window = ((w.origin[0], w.origin[1]), (w.size[0], w.size[1]))
        logger.info(f"Debug mode: spatial window origin={w.origin} size={w.size}")

    epoch_cfg = training_config.training.epoch
    train_dataset = ForestDatasetV2(
        bindings_config,
        split='train',
        patch_size=patch_size,
        min_aoi_fraction=0.3,
        debug_window=debug_window,
        epoch_mode=epoch_cfg.mode,
        sample_frac=epoch_cfg.sample_frac,
        sample_number=epoch_cfg.sample_number,
    )
    logger.info(
        f"Train dataset has {len(train_dataset.patches)} total patches "
        f"(epoch_mode={epoch_cfg.mode}, "
        f"patches/epoch={len(train_dataset)})"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dataset = ForestDatasetV2(
        bindings_config,
        split='val',
        patch_size=patch_size,
        min_aoi_fraction=0.3,
        debug_window=debug_window,
    )
    logger.info(f"Validation dataset has {len(val_dataset)} patches")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    # Create feature builder
    logger.info("Creating feature builder...")
    feature_builder = FeatureBuilder(bindings_config)

    # Read feature dimensions from bindings config
    type_enc_feature = training_config.model_input.type_encoder_feature
    phase_enc_feature = training_config.model_input.phase_encoder_feature
    type_in_channels = len(bindings_config.get_feature(type_enc_feature).channels)
    phase_in_channels = len(bindings_config.get_feature(phase_enc_feature).channels)
    logger.info(
        f"Feature dimensions from config: "
        f"{type_enc_feature}={type_in_channels}, {phase_enc_feature}={phase_in_channels}"
    )

    # Create model
    logger.info("Creating representation model...")
    model = RepresentationModel.from_config(
        model_config,
        type_in_channels=type_in_channels,
        phase_in_channels=phase_in_channels,
    ).to(device)
    logger.info(f"Model (v{RepresentationModel.VERSION}): {model}")

    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    spatial_params = sum(p.numel() for p in model.spatial_conv.parameters() if p.requires_grad)
    phase_tcn_params = sum(p.numel() for p in model.phase_tcn.parameters() if p.requires_grad)
    phase_film_params = sum(p.numel() for p in model.phase_film.parameters() if p.requires_grad)
    phase_head_params = sum(p.numel() for p in model.phase_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Trainable parameters: type(encoder={encoder_params:,}, spatial={spatial_params:,}), "
        f"phase(tcn={phase_tcn_params:,}, film={phase_film_params:,}, head={phase_head_params:,}), "
        f"total={total_params:,}"
    )

    # Create optimizer
    logger.info(f"Creating optimizer: lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Load checkpoint for resume (model weights + optimizer state).
    # The scheduler is NOT restored — it is rebuilt as a fresh cosine from the
    # checkpoint LR over the remaining epochs so the full LR range is used.
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']  # saved as epoch+1 (the next epoch to run)
        resume_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Resumed: start_epoch={start_epoch}, resume_lr={resume_lr:.3e}")

    # Scheduler is created after phase_config below, so it can condition on whether
    # phase loss is active (needed for the two-phase LR schedule).

    # Loss and training config — sourced from parsed bindings YAML.
    spectral_loss_cfg = bindings_config.get_loss('infonce_type_spectral')
    spatial_loss_cfg = bindings_config.get_loss('infonce_type_spatial')
    sampling_cfg = bindings_config.get_sampling_strategy(
        spectral_loss_cfg.anchor_population if spectral_loss_cfg else 'grid-plus-supplement'
    )

    # Grid/sampling params from sampling strategy
    grid = sampling_cfg.grid if sampling_cfg and sampling_cfg.grid else None
    supplement = sampling_cfg.supplement if sampling_cfg else None

    loss_config = {
        # Sampling (from sampling-strategy config)
        'stride': grid.stride if grid else 16,
        'border': grid.exclude_border if grid else 16,
        'jitter_radius': grid.jitter.radius if grid and grid.jitter else 4,
        'supplement_n': supplement.n if supplement else 104,

        # Spectral InfoNCE loss (from losses.infonce_type_spectral config)
        'positive_k': (
            spectral_loss_cfg.positive_strategy.selection.k
            if spectral_loss_cfg and spectral_loss_cfg.positive_strategy
            and spectral_loss_cfg.positive_strategy.selection
            else 16
        ),
        'positive_min_spatial': (
            spectral_loss_cfg.positive_strategy.selection.min_distance
            if spectral_loss_cfg and spectral_loss_cfg.positive_strategy
            and spectral_loss_cfg.positive_strategy.selection
            else 4.0
        ),
        'negative_quantile_low': (
            spectral_loss_cfg.negative_strategy.selection.range[0]
            if spectral_loss_cfg and spectral_loss_cfg.negative_strategy
            and spectral_loss_cfg.negative_strategy.selection
            and spectral_loss_cfg.negative_strategy.selection.range
            else 0.5
        ),
        'negative_quantile_high': (
            spectral_loss_cfg.negative_strategy.selection.range[1]
            if spectral_loss_cfg and spectral_loss_cfg.negative_strategy
            and spectral_loss_cfg.negative_strategy.selection
            and spectral_loss_cfg.negative_strategy.selection.range
            else 0.75
        ),
        'negative_min_spatial': (
            spectral_loss_cfg.negative_strategy.selection.min_distance
            if spectral_loss_cfg and spectral_loss_cfg.negative_strategy
            and spectral_loss_cfg.negative_strategy.selection
            else 8.0
        ),
        'temperature': (
            spectral_loss_cfg.temperature
            if spectral_loss_cfg and spectral_loss_cfg.temperature is not None
            else 0.07
        ),

        # Spatial InfoNCE loss (from losses.infonce_type_spatial config)
        'spatial_positive_k': (
            spatial_loss_cfg.positive_strategy.selection.k
            if spatial_loss_cfg and spatial_loss_cfg.positive_strategy
            and spatial_loss_cfg.positive_strategy.selection
            else 4
        ),
        'spatial_positive_max_dist': (
            spatial_loss_cfg.positive_strategy.selection.max_distance
            if spatial_loss_cfg and spatial_loss_cfg.positive_strategy
            and spatial_loss_cfg.positive_strategy.selection
            else 8
        ),
        'spatial_negative_min_dist': (
            spatial_loss_cfg.negative_strategy.selection.min_distance
            if spatial_loss_cfg and spatial_loss_cfg.negative_strategy
            and spatial_loss_cfg.negative_strategy.selection
            and spatial_loss_cfg.negative_strategy.selection.min_distance is not None
            else 96.0
        ),
        'spatial_negative_max_dist': (
            spatial_loss_cfg.negative_strategy.selection.max_distance
            if spatial_loss_cfg and spatial_loss_cfg.negative_strategy
            and spatial_loss_cfg.negative_strategy.selection
            and spatial_loss_cfg.negative_strategy.selection.max_distance is not None
            else 192.0
        ),
        'spatial_negatives_per_anchor': (
            spatial_loss_cfg.negative_strategy.selection.n_per_anchor
            if spatial_loss_cfg and spatial_loss_cfg.negative_strategy
            and spatial_loss_cfg.negative_strategy.selection
            and spatial_loss_cfg.negative_strategy.selection.n_per_anchor is not None
            else 16
        ),
        'spatial_spectral_tau': (
            spatial_loss_cfg.spectral_weighting.tau
            if spatial_loss_cfg and spatial_loss_cfg.spectral_weighting
            else 200
        ),
        'spatial_min_w': (
            spatial_loss_cfg.spectral_weighting.min_weight
            if spatial_loss_cfg and spatial_loss_cfg.spectral_weighting
            else 0.03
        ),
        'spatial_temperature': (
            spatial_loss_cfg.temperature
            if spatial_loss_cfg and spatial_loss_cfg.temperature is not None
            else 0.07
        ),

        # Loss weights (from losses config)
        'spectral_loss_weight': spectral_loss_cfg.weight if spectral_loss_cfg else 1.0,
        'spatial_loss_weight': spatial_loss_cfg.weight if spatial_loss_cfg else 1.0,

        # Cross-batch spectral neg sampling: target this many negatives per anchor.
        # With N_total anchors, n_neg = spectral_neg_per_anchor * N_total pairs are sampled.
        'spectral_neg_per_anchor': 20,

        # Training (from training config)
        'gradient_clip_enabled': training_config.training.gradient_clip.enabled,
        'gradient_clip_max_norm': training_config.training.gradient_clip.max_norm,

        # Encoder input feature names (from model section of training config)
        'type_encoder_feature': training_config.model_input.type_encoder_feature,
        'phase_encoder_feature': training_config.model_input.phase_encoder_feature,
    }

    # Variance-covariance regularization (optional)
    vcr_cfg = bindings_config.get_loss('variance_covariance_type')
    if vcr_cfg is not None:
        loss_config['vcr_enabled'] = True
        loss_config['vcr_weight'] = vcr_cfg.weight if vcr_cfg.weight is not None else 0.1
        loss_config['vcr_variance_weight'] = vcr_cfg.variance_weight if vcr_cfg.variance_weight is not None else 1.0
        loss_config['vcr_covariance_weight'] = vcr_cfg.covariance_weight if vcr_cfg.covariance_weight is not None else 1.0
        loss_config['vcr_variance_target'] = vcr_cfg.variance_target if vcr_cfg.variance_target is not None else 1.0
        logger.info(
            f"Variance-covariance loss enabled: weight={loss_config['vcr_weight']}, "
            f"var_w={loss_config['vcr_variance_weight']}, "
            f"cov_w={loss_config['vcr_covariance_weight']}, "
            f"var_target={loss_config['vcr_variance_target']}"
        )
    else:
        logger.info("Variance-covariance loss (type) disabled (not in config)")

    # Phase VCR (variance-covariance on z_phase)
    phase_vcr_cfg = bindings_config.get_loss('variance_covariance_phase')
    if phase_vcr_cfg is not None:
        loss_config['phase_vcr_config'] = {
            'weight': phase_vcr_cfg.weight if phase_vcr_cfg.weight is not None else 0.1,
            'variance_weight': phase_vcr_cfg.variance_weight if phase_vcr_cfg.variance_weight is not None else 1.0,
            'covariance_weight': phase_vcr_cfg.covariance_weight if phase_vcr_cfg.covariance_weight is not None else 1.0,
            'variance_target': phase_vcr_cfg.variance_target if phase_vcr_cfg.variance_target is not None else 1.0,
        }
        logger.info(
            f"Variance-covariance loss (phase) enabled: "
            f"weight={loss_config['phase_vcr_config']['weight']}, "
            f"var_w={loss_config['phase_vcr_config']['variance_weight']}, "
            f"cov_w={loss_config['phase_vcr_config']['covariance_weight']}, "
            f"var_target={loss_config['phase_vcr_config']['variance_target']}"
        )
    else:
        logger.info("Variance-covariance loss (phase) disabled (not in config)")

    logger.info(
        f"Loss config from bindings: "
        f"stride={loss_config['stride']}, border={loss_config['border']}, "
        f"supplement_n={loss_config['supplement_n']}, "
        f"spectral(k={loss_config['positive_k']}, min_spatial={loss_config['positive_min_spatial']}, "
        f"neg_q=[{loss_config['negative_quantile_low']}, {loss_config['negative_quantile_high']}], "
        f"neg_min_spatial={loss_config['negative_min_spatial']}, temp={loss_config['temperature']}), "
        f"spatial(neg_dist=[{loss_config['spatial_negative_min_dist']}, {loss_config['spatial_negative_max_dist']}], "
        f"neg_per_anchor={loss_config['spatial_negatives_per_anchor']}, "
        f"spec_tau={loss_config['spatial_spectral_tau']}, min_w={loss_config['spatial_min_w']}, "
        f"temp={loss_config['spatial_temperature']}), "
        f"weights(spectral={loss_config['spectral_loss_weight']}, spatial={loss_config['spatial_loss_weight']}), "
        f"cross_patch_neg_k={loss_config.get('cross_patch_negatives_per_anchor', 8)}"
    )

    # --- Phase loss pair construction setup ---
    phase_loss_cfg = bindings_config.get_loss('soft_neighborhood_phase')
    phase_sampler = None
    phase_config = None

    if phase_loss_cfg is not None:
        # Build the ysfc-weighted anchor sampler
        phase_anchor_pop = (
            phase_loss_cfg.anchor_population
            if phase_loss_cfg.anchor_population
            else 'grid-plus-supplement-ysfc'
        )
        phase_sampler = build_anchor_sampler(bindings_config, phase_anchor_pop)

        # Extract pair construction + loss params from parsed config
        ps = phase_loss_cfg.pair_strategy
        pw = phase_loss_cfg.pair_weights
        cur = phase_loss_cfg.curriculum
        phase_config = {
            # Pair construction
            'k': ps.type_similarity.k if ps and ps.type_similarity else 16,
            'min_overlap': ps.ysfc_overlap.min_overlap if ps and ps.ysfc_overlap else 3,
            'min_pairs': ps.min_pairs if ps else 5,
            'include_self': ps.include_self if ps else True,
            'sigma': pw.sigma if pw else 5.0,
            'self_pair_weight': pw.self_pair_weight if pw else 1.0,
            # Loss
            'weight': phase_loss_cfg.weight if phase_loss_cfg.weight is not None else 1.0,
            'tau_ref': phase_loss_cfg.tau_ref if phase_loss_cfg.tau_ref is not None else 0.1,
            'tau_learned': phase_loss_cfg.tau_learned if phase_loss_cfg.tau_learned is not None else 0.1,
            'min_valid_per_row': phase_loss_cfg.min_valid_per_row if phase_loss_cfg.min_valid_per_row is not None else 2,
            'self_similarity_weight': phase_loss_cfg.self_similarity_weight if phase_loss_cfg.self_similarity_weight is not None else 1.0,
            'cross_pixel_weight': phase_loss_cfg.cross_pixel_weight if phase_loss_cfg.cross_pixel_weight is not None else 1.0,
            # Curriculum
            'curriculum_start_epoch': cur.start_epoch if cur else 10,
            'curriculum_ramp_epochs': cur.ramp_epochs if cur else 10,
        }
        logger.info(
            f"Phase loss enabled: sampler={phase_anchor_pop}, "
            f"k={phase_config['k']}, min_overlap={phase_config['min_overlap']}, "
            f"min_pairs={phase_config['min_pairs']}, sigma={phase_config['sigma']}, "
            f"tau_ref={phase_config['tau_ref']}, tau_learned={phase_config['tau_learned']}, "
            f"weight={phase_config['weight']}, "
            f"curriculum=[start={phase_config['curriculum_start_epoch']}, "
            f"ramp={phase_config['curriculum_ramp_epochs']}]"
        )
    else:
        logger.info("Phase pair construction disabled (no soft_neighborhood_phase loss in config)")

    # --- Phase spread ranking loss setup ---
    spread_loss_cfg = bindings_config.get_loss('phase_spread_ranking')
    spread_config = None
    if spread_loss_cfg is not None:
        cur_s = spread_loss_cfg.curriculum
        spread_config = {
            'weight': spread_loss_cfg.weight if spread_loss_cfg.weight is not None else 0.5,
            'margin': spread_loss_cfg.margin if spread_loss_cfg.margin is not None else 0.1,
            'delta': spread_loss_cfg.delta if spread_loss_cfg.delta is not None else 0.5,
            'curriculum_start_epoch': cur_s.start_epoch if cur_s else 30,
            'curriculum_ramp_epochs': cur_s.ramp_epochs if cur_s else 10,
        }
        logger.info(
            f"Phase spread ranking loss enabled: weight={spread_config['weight']}, "
            f"margin={spread_config['margin']}, delta={spread_config['delta']}, "
            f"curriculum=[start={spread_config['curriculum_start_epoch']}, "
            f"ramp={spread_config['curriculum_ramp_epochs']}]"
        )

    # --- Phase recovery discrimination loss setup ---
    recovery_disc_loss_cfg = bindings_config.get_loss('phase_recovery_discrimination')
    recovery_disc_config = None
    if recovery_disc_loss_cfg is not None:
        cur_rd = recovery_disc_loss_cfg.curriculum
        recovery_disc_config = {
            'weight': recovery_disc_loss_cfg.weight if recovery_disc_loss_cfg.weight is not None else 1.0,
            'margin': recovery_disc_loss_cfg.margin if recovery_disc_loss_cfg.margin is not None else 0.5,
            'low_ysfc_max': recovery_disc_loss_cfg.low_ysfc_max if recovery_disc_loss_cfg.low_ysfc_max is not None else 1.0,
            'high_ysfc_min': recovery_disc_loss_cfg.high_ysfc_min if recovery_disc_loss_cfg.high_ysfc_min is not None else 5.0,
            'curriculum_start_epoch': cur_rd.start_epoch if cur_rd else 30,
            'curriculum_ramp_epochs': cur_rd.ramp_epochs if cur_rd else 10,
        }
        logger.info(
            f"Phase recovery discrimination loss enabled: weight={recovery_disc_config['weight']}, "
            f"margin={recovery_disc_config['margin']}, "
            f"low_ysfc_max={recovery_disc_config['low_ysfc_max']}, "
            f"high_ysfc_min={recovery_disc_config['high_ysfc_min']}, "
            f"curriculum=[start={recovery_disc_config['curriculum_start_epoch']}, "
            f"ramp={recovery_disc_config['curriculum_ramp_epochs']}]"
        )

    # --- EVT soft neighbourhood loss setup ---
    # The EVT-stratified anchor sampler is built whenever the EVT loss config
    # exists, regardless of loss weight. This allows EVT-stratified anchor
    # sampling for the spectral kNN even when the EVT loss itself is disabled.
    evt_metric = None
    evt_sampler = None
    evt_loss_cfg = bindings_config.get_loss('soft_neighborhood_evt')
    if evt_loss_cfg is not None:
        evt_anchor_pop = (
            evt_loss_cfg.anchor_population
            if evt_loss_cfg.anchor_population
            else 'grid-plus-supplement-evt'
        )
        evt_sampler = build_anchor_sampler(bindings_config, evt_anchor_pop)

        if (evt_loss_cfg.weight or 0.0) > 0.0:
            # EVT code counts come from the shared stats file, at the path:
            #   stats["evt_class"]["static_categorical.evt"]["counts"]
            # Keys are string codes, values are integer pixel counts.
            evt_code_counts = (
                feature_builder.stats
                .get("evt_class", {})
                .get("static_categorical.evt", {})
                .get("counts", {})
            )
            if not evt_code_counts:
                raise ValueError(
                    "EVT code counts not found in stats file. "
                    "Run example_compute_stats.py to compute stats first."
                )
            evt_metric = EvtDiffusionMetric(
                confusion_csv=evt_loss_cfg.confusion_matrix_path,
                code_counts=evt_code_counts,
                min_count=evt_loss_cfg.min_count or 100,
                min_confusion_samples=evt_loss_cfg.min_confusion_samples or 30,
                diffusion_steps=evt_loss_cfg.diffusion_steps or 2,
                laplace_smoothing=evt_loss_cfg.laplace_smoothing or 0.0,
                binary_threshold=evt_loss_cfg.binary_threshold or 0.0,
            ).to(device)
            loss_config['evt_weight'] = evt_loss_cfg.weight
            loss_config['evt_tau_ref'] = evt_loss_cfg.tau_ref or 0.5
            loss_config['evt_tau_learned'] = evt_loss_cfg.tau_learned or 0.5
            # Restrict inverse-frequency weighting to valid EVT codes only.
            for spec in evt_sampler.weight_specs:
                if spec.transform == 'inverse-frequency':
                    spec.valid_values = evt_metric.valid_codes
            logger.info(
                f"EVT soft neighbourhood loss enabled: "
                f"{evt_metric.n_codes} codes, "
                f"diffusion_steps={evt_loss_cfg.diffusion_steps or 2}, "
                f"min_count={evt_loss_cfg.min_count or 100}, "
                f"weight={loss_config['evt_weight']}, "
                f"tau_ref={loss_config['evt_tau_ref']}, "
                f"tau_learned={loss_config['evt_tau_learned']}, "
                f"anchor_population={evt_anchor_pop}"
            )
        else:
            logger.info(
                f"EVT loss disabled (weight=0) but EVT-stratified anchor sampler "
                f"active for spectral kNN (population={evt_anchor_pop})"
            )
    else:
        logger.info("EVT soft neighbourhood loss disabled (not in bindings config)")

    # Create scheduler with optional warmup.
    # Must be after phase_config is built so the two-phase branch can read
    # curriculum_start_epoch from it.
    scheduler_config = training_config.scheduler
    total_steps = num_epochs * len(train_dataloader)
    eta_min_factor = scheduler_config.eta_min / lr  # express eta_min as a multiplier on peak lr

    def _cosine(start_val, end_val, progress):
        """Cosine interpolation from start_val to end_val over [0, 1]."""
        return end_val + (start_val - end_val) * 0.5 * (1.0 + np.cos(np.pi * progress))

    if start_epoch > 0:
        # Resumed run: fresh cosine from resume_lr → eta_min over remaining epochs.
        # No warmup or phase re-warmup needed — the model is already well-trained.
        remaining_steps = (num_epochs - start_epoch) * len(train_dataloader)
        eta_min_factor_resume = scheduler_config.eta_min / resume_lr
        logger.info(
            f"Resume scheduler: cosine from lr={resume_lr:.3e} to "
            f"eta_min={scheduler_config.eta_min:.1e} over "
            f"{num_epochs - start_epoch} epochs ({remaining_steps} steps)"
        )

        def lr_lambda(step):
            progress = step / max(remaining_steps, 1)
            return _cosine(1.0, eta_min_factor_resume, min(progress, 1.0))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_config.warmup.enabled:
        warmup_steps = scheduler_config.warmup.epochs * len(train_dataloader)
        phase_warmup_cfg = getattr(scheduler_config, 'phase_warmup', None)

        if (
            phase_warmup_cfg is not None
            and phase_warmup_cfg.enabled
            and phase_config is not None
        ):
            # Two-phase LR schedule to accommodate the phase-loss curriculum:
            #
            #  Segment 1 — initial warmup (0 → warmup_steps):
            #    LR rises linearly 0 → peak. Prevents large updates while weights
            #    are uninitialized.
            #
            #  Segment 2 — first cosine (warmup_steps → phase_start_step):
            #    LR decays along the full-range cosine (as if running to total_steps).
            #    Phase loss is zero during this window; spectral/spatial losses train
            #    freely.
            #
            #  Segment 3 — phase re-warmup (phase_start_step → phase_warmup_end_step):
            #    Phase loss enters. AdamW's variance estimates (v_t) for phase
            #    parameters are zero at this point — bias correction makes the first
            #    update a unit-norm step regardless of gradient magnitude, so a high
            #    LR here causes overshooting. To counteract this:
            #      a) LR drops immediately to start_factor × lr at phase_start_step,
            #         giving low-LR steps to let v_t accumulate accurate estimates.
            #      b) LR then ramps linearly to peak_factor × lr over phase_warmup.epochs,
            #         mirroring the initial warmup but scoped to the phase-loss entry.
            #
            #  Segment 4 — second cosine (phase_warmup_end_step → total_steps):
            #    LR decays from peak_factor × lr down to eta_min over the remainder
            #    of training (~165 epochs for a 200-epoch run).
            #
            # curriculum_start_epoch is the epoch where cw is first evaluated, but
            # cw = (epoch - start_epoch) / ramp = 0 when epoch == start_epoch exactly.
            # Phase gradients first appear at start_epoch + 1, so that is where the
            # LR drop must land.
            phase_start_epoch = phase_config['curriculum_start_epoch'] + 1
            phase_start_step = phase_start_epoch * len(train_dataloader)
            phase_warmup_end_step = (
                phase_start_step + phase_warmup_cfg.epochs * len(train_dataloader)
            )
            start_factor = phase_warmup_cfg.start_factor  # immediate drop on phase entry
            second_peak = phase_warmup_cfg.peak_factor    # ramp target, as multiplier on peak lr

            logger.info(
                f"Using two-phase cosine schedule: "
                f"warmup={scheduler_config.warmup.epochs} epochs, "
                f"phase re-warmup at epoch {phase_start_epoch} "
                f"for {phase_warmup_cfg.epochs} epochs "
                f"(start_factor={start_factor}, peak_factor={second_peak})"
            )

            def lr_lambda(step):
                if step < warmup_steps:
                    # Segment 1: linear warmup 0 → peak
                    return max(step / warmup_steps, 1e-8)
                elif step < phase_start_step:
                    # Segment 2: cosine decay (full-range) while phase loss is silent
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return _cosine(1.0, eta_min_factor, progress)
                elif step < phase_warmup_end_step:
                    # Segment 3: immediate drop to start_factor, then linear ramp to
                    # peak_factor. The low starting LR lets AdamW's v_t accumulate
                    # before taking large steps with the new phase gradients.
                    ramp_progress = (step - phase_start_step) / (phase_warmup_end_step - phase_start_step)
                    return start_factor + (second_peak - start_factor) * ramp_progress
                else:
                    # Segment 4: cosine decay from peak_factor to eta_min
                    progress = (step - phase_warmup_end_step) / (total_steps - phase_warmup_end_step)
                    return _cosine(second_peak, eta_min_factor, progress)

        else:
            # Standard single-phase: linear warmup + cosine annealing
            logger.info(f"Using cosine scheduler with {warmup_steps} warmup steps")

            def lr_lambda(step):
                if step < warmup_steps:
                    return max(step / warmup_steps, 1e-8)
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return _cosine(1.0, eta_min_factor, progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info(f"Using cosine annealing scheduler: eta_min={scheduler_config.eta_min}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=scheduler_config.eta_min,
        )

    # Create output directories
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    logger.info(f"Checkpoint dir: {ckpt_dir}")
    logger.info(f"Log dir: {log_dir}")

    # Save experiment artifacts for reproducibility
    shutil.copy2(bindings_path, experiment_dir / Path(bindings_path).name)
    shutil.copy2(args.training, experiment_dir / Path(args.training).name)
    shutil.copy2(model_config_path, experiment_dir / Path(model_config_path).name)
    shutil.copy2(RepresentationModel.source_file(), experiment_dir / "representation.py")
    logger.info(f"Saved config and model source to {experiment_dir}")

    # Pre-extract input dropout schedule config (scalar or dict) for the epoch loop.
    input_dropout_schedule_cfg = model_config.get("type_encoder", {}).get("input_dropout", 0.0)

    # Pre-extract spatial smoothing curriculum config.
    smoothing_cur_cfg = raw_training_config.get("spatial_smoothing_curriculum", {})
    smoothing_cur_enabled = smoothing_cur_cfg.get("enabled", False)
    smoothing_freeze_until = int(smoothing_cur_cfg.get("freeze_until_epoch", 20))
    smoothing_ramp_epochs = int(smoothing_cur_cfg.get("ramp_epochs", 30))
    if smoothing_cur_enabled:
        logger.info(
            f"Spatial smoothing curriculum: gate=1.0 (identity) for epochs 0–{smoothing_freeze_until - 1}, "
            f"linearly released to 0.0 over epochs {smoothing_freeze_until}–"
            f"{smoothing_freeze_until + smoothing_ramp_epochs - 1}"
        )

    # Tracks (monitor_val, path) for top-k checkpoint pruning.
    saved_ckpts: list = []

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
    for epoch in range(start_epoch, num_epochs):
        train_dataset.on_epoch_start()  # Reshuffle patches

        # Apply scheduled input dropout rate for this epoch.
        input_dropout_rate = compute_input_dropout_rate(
            input_dropout_schedule_cfg, epoch, num_epochs
        )
        model.set_input_dropout_rate(input_dropout_rate)
        if input_dropout_rate > 0.0:
            logger.debug(f"Epoch {epoch}: input_dropout_rate={input_dropout_rate:.4f}")

        # Apply spatial smoothing curriculum: lock gate open early so the model
        # can't use smoothing as a shortcut before z_type develops spectral structure.
        if smoothing_cur_enabled:
            if epoch < smoothing_freeze_until:
                min_gate = 1.0
            elif epoch < smoothing_freeze_until + smoothing_ramp_epochs:
                progress = (epoch - smoothing_freeze_until) / smoothing_ramp_epochs
                min_gate = 1.0 - progress
            else:
                min_gate = 0.0
            model.set_spatial_min_gate(min_gate)
            logger.debug(f"Epoch {epoch}: spatial min_gate={min_gate:.3f}")

        train_stats = train_epoch(
          train_dataloader, feature_builder, model,
          optimizer, scheduler, device, loss_config, epoch, num_epochs,
          phase_sampler=phase_sampler, phase_config=phase_config,
          spread_config=spread_config, recovery_disc_config=recovery_disc_config,
          evt_metric=evt_metric, evt_sampler=evt_sampler,
        )

        val_stats = validate_epoch(
          val_dataloader, feature_builder, model,
          device, loss_config,
          phase_sampler=phase_sampler, phase_config=phase_config,
          spread_config=spread_config, recovery_disc_config=recovery_disc_config,
          epoch=epoch, evt_metric=evt_metric, evt_sampler=evt_sampler,
        )

        logger.info(f"Epoch {epoch+1}/{num_epochs} complete")
        logger.info(
            f"  Train: {train_stats['loss']:.4f} "
            f"spec={train_stats['spectral_loss']:.4f} spat={train_stats['spatial_loss']:.4f} "
            f"phase={train_stats['phase_loss']:.4f} spr={train_stats.get('phase_spread_loss', 0.0):.4f} "
            f"rdisc={train_stats.get('phase_recovery_disc_loss', 0.0):.4f} "
            f"vcr={train_stats['vcr_loss']:.4f} "
            f"pvcr={train_stats['phase_vcr_loss']:.4f} evt={train_stats['evt_loss']:.4f}"
        )
        logger.info(
            f"  Val:   {val_stats['loss']:.4f} "
            f"spec={val_stats['spectral_loss']:.4f} spat={val_stats['spatial_loss']:.4f} "
            f"phase={val_stats['phase_loss']:.4f} spr={val_stats.get('phase_spread_loss', 0.0):.4f} "
            f"rdisc={val_stats.get('phase_recovery_disc_loss', 0.0):.4f} "
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
        def fmt_stats(s: dict) -> str:
            return f"mean={s['mean']:.3f}, std={s['std']:.3f}, [q25={s['q25']:.3f}, q50={s['q50']:.3f}, q75={s['q75']:.3f}]"

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
            eff_confusers = f"{2.718 ** train_stats.get('spatial_loss', 0.0):.1f}"
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
        elif pls:
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

        # Flat metrics dict — keys match the monitor strings used in the YAML.
        epoch_metrics = {
            "train/loss_total":         train_stats['loss'],
            "train/loss_spectral":      train_stats['spectral_loss'],
            "train/loss_spatial":       train_stats['spatial_loss'],
            "train/loss_phase":              train_stats['phase_loss'],
            "train/loss_phase_spread":       train_stats.get('phase_spread_loss', 0.0),
            "train/loss_phase_recovery_disc": train_stats.get('phase_recovery_disc_loss', 0.0),
            "train/loss_vcr":                train_stats['vcr_loss'],
            "train/loss_phase_vcr":          train_stats['phase_vcr_loss'],
            "val/loss_total":                val_stats['loss'],
            "val/loss_spectral":             val_stats['spectral_loss'],
            "val/loss_spatial":              val_stats['spatial_loss'],
            "val/loss_phase":                val_stats['phase_loss'],
            "val/loss_phase_spread":         val_stats.get('phase_spread_loss', 0.0),
            "val/loss_phase_recovery_disc":  val_stats.get('phase_recovery_disc_loss', 0.0),
            "val/loss_vcr":                  val_stats['vcr_loss'],
            "val/loss_phase_vcr":            val_stats['phase_vcr_loss'],
        }

        # Type-leakage metrics (train only — val computed separately below)
        for prefix, stats in [("train", train_stats), ("val", val_stats)]:
            tls = stats.get('type_leakage_stats')
            if tls is not None:
                epoch_metrics[f"{prefix}/phase_type_leakage_cov"]  = tls['cross_cov_frob']
                epoch_metrics[f"{prefix}/phase_type_leakage_r2_mean"] = tls['r2_mean']
                epoch_metrics[f"{prefix}/phase_type_leakage_r2_max"]  = tls['r2_max']

        # Checkpoint state dict (shared by periodic and last saves).
        ckpt_state = {
            'epoch': epoch + 1,
            'model_version': RepresentationModel.VERSION,
            'model_config': model_config,
            'type_in_channels': model.type_in_channels,
            'phase_in_channels': model.phase_in_channels,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            **{k: v for k, v in epoch_metrics.items()},
        }

        ckpt_cfg = training_config.run.checkpoint
        monitor_key = ckpt_cfg.monitor
        if monitor_key not in epoch_metrics:
            raise KeyError(
                f"Checkpoint monitor '{monitor_key}' not found in epoch_metrics. "
                f"Available keys: {list(epoch_metrics.keys())}"
            )
        monitor_val = epoch_metrics[monitor_key]

        # Always save 'last' checkpoint (overwrites each epoch).
        if ckpt_cfg.save_last:
            last_path = ckpt_dir / "encoder_last.pt"
            torch.save(ckpt_state, last_path)
            logger.info(f"Saved last checkpoint to {last_path}")

        # Periodic save (every nth epoch, never pruned).
        if (epoch + 1) % ckpt_cfg.save_every_n_epochs == 0:
            ckpt_path = ckpt_dir / f"encoder_epoch_{epoch+1:03d}.pt"
            torch.save(ckpt_state, ckpt_path)
            logger.info(
                f"Saved periodic checkpoint to {ckpt_path} "
                f"({monitor_key}={monitor_val:.4f})"
            )

        # Top-k save: only track best after monitor_start_epoch so pre-curriculum
        # epochs (where phase loss is zero) can't be selected as the best checkpoint.
        # saved_ckpts is sorted best-first (index 0 = rank 1).
        # NaN-safe sort: map non-finite values to the worst possible sentinel so they
        # sort to the tail and can be displaced by any finite value.
        reverse = (ckpt_cfg.mode == "max")
        nan_sentinel = float('-inf') if reverse else float('inf')
        saved_ckpts.sort(key=lambda x: x[0] if math.isfinite(x[0]) else nan_sentinel, reverse=reverse)
        worst_val_in_top_k = saved_ckpts[-1][0] if len(saved_ckpts) >= ckpt_cfg.save_top_k else None
        if worst_val_in_top_k is not None and not math.isfinite(worst_val_in_top_k):
            worst_val_in_top_k = nan_sentinel
        is_better = (
            math.isfinite(monitor_val)
            and (
                worst_val_in_top_k is None
                or (ckpt_cfg.mode == "min" and monitor_val < worst_val_in_top_k)
                or (ckpt_cfg.mode == "max" and monitor_val > worst_val_in_top_k)
            )
        )
        if is_better and epoch >= ckpt_cfg.monitor_start_epoch:
            # Save under a temporary name; will be renamed with rank below.
            tmp_path = ckpt_dir / f"encoder_best_epoch_{epoch+1:03d}.pt"
            torch.save(ckpt_state, tmp_path)
            saved_ckpts.append((monitor_val, tmp_path))
            saved_ckpts.sort(key=lambda x: x[0], reverse=reverse)

            # Prune worst entry if over top-k.
            while len(saved_ckpts) > ckpt_cfg.save_top_k:
                worst_val, worst_path = saved_ckpts.pop()
                if worst_path.exists():
                    worst_path.unlink()
                    logger.info(
                        f"Removed checkpoint {worst_path.name} "
                        f"({monitor_key}={worst_val:.4f}, outside top-{ckpt_cfg.save_top_k})"
                    )

            # Rename all top-k files to reflect current rank (rank 1 = best).
            # Use temp names first to avoid collisions during rename.
            tmp_renames = []
            for rank, (val, old_path) in enumerate(saved_ckpts, 1):
                ep = old_path.stem.split("_")[-1]  # e.g. '042'
                new_name = ckpt_dir / f"encoder_best_{rank}_epoch_{ep}.pt"
                tmp_name = ckpt_dir / f"_tmp_rank_{rank}_{ep}.pt"
                old_path.rename(tmp_name)
                tmp_renames.append((rank, val, tmp_name, new_name))
            saved_ckpts = []
            for rank, val, tmp_name, new_name in tmp_renames:
                tmp_name.rename(new_name)
                saved_ckpts.append((val, new_name))
            logger.info(f"Updated top-{ckpt_cfg.save_top_k} checkpoints:")
            for rank, (val, path) in enumerate(saved_ckpts, 1):
                logger.info(f"  #{rank}: {path.name} ({monitor_key}={val:.4f})")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
