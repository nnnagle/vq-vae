#!/usr/bin/env python3
"""
Minimal training script for representation learning.

This script trains a simple encoder using InfoNCE contrastive loss:
- Input: features.ccdc_history (16 channels)
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
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from data.sampling import sample_anchors_grid_plus_supplement
from data.sampling.anchor_sampling import AnchorSampler, build_anchor_sampler
from models import RepresentationModel
from losses import contrastive_loss, pairs_with_spatial_constraint
from losses.phase_pairs import build_phase_pairs
from utils import (
    compute_spatial_distances,
    extract_at_locations,
    extract_temporal_at_locations,
    spatial_knn_pairs,
    spatial_negative_pairs,
)

logger = logging.getLogger(__name__)

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
    n_valid = 0
    total_spectral_pos_pairs = 0
    total_spectral_neg_pairs = 0
    total_spatial_pos_pairs = 0
    total_spatial_neg_pairs = 0

    # Collectors for distribution logging (accumulated across samples)
    all_gate_values = []
    all_pos_weights = []
    all_neg_weights = []

    # Phase pair stats accumulators
    all_phase_pair_stats = []

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
        encoder_feature = feature_builder.build_feature('ccdc_history', sample)
        spec_dist_feature = feature_builder.build_feature('infonce_type_spectral', sample)

        # Convert to tensors
        encoder_data = torch.from_numpy(encoder_feature.data).float().to(device)
        spec_dist_data = torch.from_numpy(spec_dist_feature.data).float().to(device)
        mask = torch.from_numpy(encoder_feature.mask).to(device)

        # Also apply distance feature mask
        spec_dist_mask = torch.from_numpy(spec_dist_feature.mask).to(device)
        combined_mask = mask & spec_dist_mask

        # Sample anchor locations
        anchors = sample_anchors_grid_plus_supplement(
            combined_mask,
            stride=config.get('stride', 16),
            border=config.get('border', 16),
            jitter_radius=jitter_radius,
            supplement_n=config.get('supplement_n', 104),
        )

        if anchors.shape[0] < 10:
            continue

        # Extract features at anchor locations for spectral loss
        encoder_at_anchors = extract_at_locations(encoder_data, anchors)
        spec_dist_at_anchors = extract_at_locations(spec_dist_data, anchors)

        # Compute distances for spectral loss
        # Mahalanobis transform already applied by FeatureBuilder, so L2 here = Mahalanobis
        spec_feat_distances = torch.cdist(spec_dist_at_anchors, spec_dist_at_anchors)
        spatial_distances = compute_spatial_distances(anchors)

        # Generate pairs for spectral loss
        spectral_pos_pairs, spectral_neg_pairs = pairs_with_spatial_constraint(
            spec_feat_distances,
            spatial_distances,
            positive_k=config.get('positive_k', 16),
            positive_min_spatial=config.get('positive_min_spatial', 4.0),
            negative_quantile_low=config.get('negative_quantile_low', 0.5),
            negative_quantile_high=config.get('negative_quantile_high', 0.75),
            negative_min_spatial=config.get('negative_min_spatial', 8.0),
        )

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
            all_pos_weights.append(pos_weights.detach())

        if spatial_neg_pairs.numel() > 0:
            dneg = pair_l2(spec_dist_unique, spatial_neg_pairs)
            neg_weights = (1.0 - torch.exp(-dneg / tau)).clamp(min=min_w, max=1.0)
            all_neg_weights.append(neg_weights.detach())

        # Check if we have valid pairs for both losses
        has_spectral = spectral_pos_pairs.shape[0] > 0 and spectral_neg_pairs.shape[0] > 0
        has_spatial = spatial_pos_pairs.shape[0] > 0 and spatial_neg_pairs.shape[0] > 0

        if not has_spectral and not has_spatial:
            continue

        # Encode the full patch for efficient embedding extraction
        # encoder_data: [C, H, W] -> [1, C, H, W] -> model -> [1, D, H, W]
        encoder_input_full = encoder_data.unsqueeze(0)
        z_full, gate = model(encoder_input_full, return_gate=True)  # [1, D, H, W] each
        z_full = z_full.squeeze(0)  # [D, H, W]
        gate = gate.squeeze(0)  # [D, H, W]

        # Collect gate values (flatten to 1D for stats)
        all_gate_values.append(gate.detach().flatten())

        # Extract embeddings at anchor locations for spectral loss
        z_anchors = extract_at_locations(z_full, anchors)  # [num_anchors, D]

        # Compute spectral loss
        spectral_loss_val = torch.tensor(0.0, device=device)
        if has_spectral:
            spectral_loss_val = contrastive_loss(
                z_anchors,
                spectral_pos_pairs,
                spectral_neg_pairs,
                temperature=config.get('temperature', 0.07),
                similarity='l2',
            )

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
                similarity='l2',
            )

        # Combine losses with weights
        spectral_weight = config.get('spectral_loss_weight', 1.0)
        spatial_weight = config.get('spatial_loss_weight', 1.0)
        loss = spectral_weight * spectral_loss_val + spatial_weight * spatial_loss_val

        # Skip if loss is NaN or Inf (numerical instability)
        if not torch.isfinite(loss):
            logger.warning(f"Skipping sample with non-finite loss: {loss.item()}")
            continue

        # Accumulate: keep as tensor for training (backward), use .item() for validation
        if training:
            total_loss += loss
            total_spectral_loss += spectral_loss_val
            total_spatial_loss += spatial_loss_val
        else:
            total_loss += loss.item()
            total_spectral_loss += spectral_loss_val.item()
            total_spatial_loss += spatial_loss_val.item()
        n_valid += 1
        total_spectral_pos_pairs += spectral_pos_pairs.shape[0]
        total_spectral_neg_pairs += spectral_neg_pairs.shape[0]
        total_spatial_pos_pairs += spatial_pos_pairs.shape[0]
        total_spatial_neg_pairs += spatial_neg_pairs.shape[0]

        # --- Phase pair construction (no loss yet, just pairs + logging) ---
        if phase_sampler is not None and phase_config is not None:
            # Build ysfc feature via FeatureBuilder
            ysfc_feature = feature_builder.build_feature('ysfc', sample)
            ysfc_data = torch.from_numpy(ysfc_feature.data).float().to(device)
            # ysfc_data: [1, T, H, W] (single channel)

            # Combined mask for phase anchors: encoder mask AND ysfc validity
            ysfc_mask = torch.from_numpy(ysfc_feature.mask).to(device)
            # ysfc_mask is [T, H, W] for temporal; collapse to [H, W]
            if ysfc_mask.ndim == 3:
                ysfc_spatial_mask = ysfc_mask.all(dim=0)  # valid across all timesteps
            else:
                ysfc_spatial_mask = ysfc_mask
            phase_mask = combined_mask & ysfc_spatial_mask

            # Sample separate anchors for phase loss (CPU — weights are numpy-derived)
            phase_anchors = phase_sampler(
                phase_mask.cpu(), training=training, sample=sample
            )

            if phase_anchors.shape[0] >= 10:
                # Extract spectral features at phase anchors (for kNN + weights)
                phase_spec_at_anchors = extract_at_locations(
                    spec_dist_data, phase_anchors
                )  # [N_phase, C]

                # Extract ysfc time series at phase anchors
                # ysfc_data is [1, T, H, W]; squeeze channel dim for extraction
                ysfc_at_anchors = extract_temporal_at_locations(
                    ysfc_data, phase_anchors
                )  # [N_phase, T, 1]
                ysfc_at_anchors = ysfc_at_anchors.squeeze(-1)  # [N_phase, T]

                # Build pairs
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

    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    empty_phase_stats = {
        'n_anchors': 0, 'n_anchors_surviving': 0,
        'n_candidates': 0, 'n_after_overlap': 0,
        'n_self_pairs': 0, 'n_total_pairs': 0,
        'overlap_mean': 0.0, 'overlap_min': 0,
        'weight_mean': 0.0, 'weight_std': 0.0,
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

    if n_valid == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0, 'n_valid': 0,
            'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
            'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'phase_pair_stats': empty_phase_stats,
        }

    # Average losses over valid samples in batch
    mean_loss = total_loss / n_valid
    mean_spectral_loss = total_spectral_loss / n_valid
    mean_spatial_loss = total_spatial_loss / n_valid

    if training:
        # Final NaN check before backward
        if not torch.isfinite(mean_loss):
            logger.warning(f"Skipping batch with non-finite mean loss: {mean_loss.item()}")
            return {
                'loss': float('nan'), 'spectral_loss': float('nan'), 'spatial_loss': float('nan'),
                'n_valid': 0,
                'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
                'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
                'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
                'neg_weight_stats': empty_stats,
                'phase_pair_stats': empty_phase_stats,
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

    # Compute distribution statistics for gate values and weights
    def compute_stats(tensors: list[torch.Tensor]) -> dict:
        """Compute summary stats from list of tensors."""
        if not tensors:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
        combined = torch.cat(tensors)
        return {
            'mean': combined.mean().item(),
            'std': combined.std().item(),
            'min': combined.min().item(),
            'max': combined.max().item(),
            'q25': torch.quantile(combined, 0.25).item(),
            'q50': torch.quantile(combined, 0.50).item(),
            'q75': torch.quantile(combined, 0.75).item(),
        }

    return {
        'loss': mean_loss,
        'spectral_loss': mean_spectral_loss,
        'spatial_loss': mean_spatial_loss,
        'n_valid': n_valid,
        'spectral_pos_pairs': total_spectral_pos_pairs // n_valid if n_valid > 0 else 0,
        'spectral_neg_pairs': total_spectral_neg_pairs // n_valid if n_valid > 0 else 0,
        'spatial_pos_pairs': total_spatial_pos_pairs // n_valid if n_valid > 0 else 0,
        'spatial_neg_pairs': total_spatial_neg_pairs // n_valid if n_valid > 0 else 0,
        'gate_stats': compute_stats(all_gate_values),
        'pos_weight_stats': compute_stats(all_pos_weights),
        'neg_weight_stats': compute_stats(all_neg_weights),
        'phase_pair_stats': aggregate_phase_stats(all_phase_pair_stats),
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
) -> dict:
    """Run training on entire training set for one epoch."""
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_batches = 0

    # Keep last batch stats for epoch-level distribution logging
    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    last_gate_stats = empty_stats
    last_pos_weight_stats = empty_stats
    last_neg_weight_stats = empty_stats
    last_phase_pair_stats = None

    for batch_idx, batch in enumerate(train_dataloader):
        stats = process_batch(
            batch, feature_builder, model, device, config,
            training=True, optimizer=optimizer,
            phase_sampler=phase_sampler, phase_config=phase_config,
        )

        scheduler.step()

        if stats['n_valid'] > 0:
            total_loss += stats['loss']
            total_spectral_loss += stats['spectral_loss']
            total_spatial_loss += stats['spatial_loss']
            total_batches += 1

            # Update distribution stats from last valid batch
            last_gate_stats = stats['gate_stats']
            last_pos_weight_stats = stats['pos_weight_stats']
            last_neg_weight_stats = stats['neg_weight_stats']
            last_phase_pair_stats = stats.get('phase_pair_stats')

            if batch_idx % log_interval == 0:
                phase_msg = ""
                ps = stats.get('phase_pair_stats')
                if ps and ps['n_anchors'] > 0:
                    phase_msg = (
                        f" | Phase: {ps['n_total_pairs']:.0f} pairs "
                        f"({ps['n_anchors_surviving']:.0f}/{ps['n_anchors']:.0f} anchors, "
                        f"overlap={ps['overlap_mean']:.1f}, "
                        f"w={ps['weight_mean']:.3f}±{ps['weight_std']:.3f})"
                    )

                logger.info(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{len(train_dataloader)} | "
                    f"Loss: {stats['loss']:.4f} (spec: {stats['spectral_loss']:.4f}, spat: {stats['spatial_loss']:.4f}) | "
                    f"Pairs(+/-): spec {stats['spectral_pos_pairs']}/{stats['spectral_neg_pairs']}, "
                    f"spat {stats['spatial_pos_pairs']}/{stats['spatial_neg_pairs']} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    f"{phase_msg}"
                )

    if total_batches == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0, 'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'phase_pair_stats': None,
        }

    return {
        'loss': total_loss / total_batches,
        'spectral_loss': total_spectral_loss / total_batches,
        'spatial_loss': total_spatial_loss / total_batches,
        'batches': total_batches,
        'gate_stats': last_gate_stats,
        'pos_weight_stats': last_pos_weight_stats,
        'neg_weight_stats': last_neg_weight_stats,
        'phase_pair_stats': last_phase_pair_stats,
    }

def validate_epoch(
    val_dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    config: dict,
    phase_sampler: AnchorSampler | None = None,
    phase_config: dict | None = None,
) -> dict:
    """Run validation on entire validation set."""
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_batches = 0

    # Keep last batch stats for epoch-level distribution logging
    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    last_gate_stats = empty_stats
    last_pos_weight_stats = empty_stats
    last_neg_weight_stats = empty_stats
    last_phase_pair_stats = None

    with torch.no_grad():
        for batch in val_dataloader:
            stats = process_batch(
                batch, feature_builder, model, device, config,
                training=False,
                phase_sampler=phase_sampler, phase_config=phase_config,
            )
            if stats['n_valid'] > 0:
                total_loss += stats['loss']
                total_spectral_loss += stats['spectral_loss']
                total_spatial_loss += stats['spatial_loss']
                total_batches += 1

                # Update distribution stats from last valid batch
                last_gate_stats = stats['gate_stats']
                last_pos_weight_stats = stats['pos_weight_stats']
                last_neg_weight_stats = stats['neg_weight_stats']
                last_phase_pair_stats = stats.get('phase_pair_stats')

    if total_batches == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0, 'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'phase_pair_stats': None,
        }

    return {
        'loss': total_loss / total_batches,
        'spectral_loss': total_spectral_loss / total_batches,
        'spatial_loss': total_spatial_loss / total_batches,
        'batches': total_batches,
        'gate_stats': last_gate_stats,
        'pos_weight_stats': last_pos_weight_stats,
        'neg_weight_stats': last_neg_weight_stats,
        'phase_pair_stats': last_phase_pair_stats,
    }

def main():
    parser = argparse.ArgumentParser(description='Train representation encoder')
    parser.add_argument(
        '--bindings',
        type=str,
        default='config/frl_binding_v1.yaml',
        help='Path to bindings config'
    )
    parser.add_argument(
        '--training',
        type=str,
        default='config/frl_training_v1.yaml',
        help='Path to training config'
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
    args = parser.parse_args()

    # Parse configs first to get defaults
    logger.info(f"Loading bindings config from {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    logger.info(f"Loading training config from {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

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
    train_dataset = ForestDatasetV2(
        bindings_config,
        split='train',
        patch_size=patch_size,
        min_aoi_fraction=0.3,
    )
    logger.info(f"Train dataset has {len(train_dataset)} patches")

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

    # Create model
    logger.info("Creating representation model...")
    model = RepresentationModel().to(device)
    logger.info(f"Model (v{RepresentationModel.VERSION}): {model}")

    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    spatial_params = sum(p.numel() for p in model.spatial_conv.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: encoder={encoder_params:,}, spatial={spatial_params:,}, total={encoder_params + spatial_params:,}")

    # Create optimizer
    logger.info(f"Creating optimizer: lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Create scheduler with optional warmup
    scheduler_config = training_config.scheduler
    total_steps = num_epochs * len(train_dataloader)

    if scheduler_config.warmup.enabled:
        warmup_steps = scheduler_config.warmup.epochs * len(train_dataloader)
        logger.info(f"Using cosine scheduler with {warmup_steps} warmup steps")

        # Linear warmup + cosine annealing
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return scheduler_config.eta_min / lr + (1 - scheduler_config.eta_min / lr) * (
                    0.5 * (1 + np.cos(np.pi * progress))
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info(f"Using cosine scheduler: T_max={scheduler_config.T_max}, eta_min={scheduler_config.eta_min}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.T_max * len(train_dataloader),
            eta_min=scheduler_config.eta_min,
        )

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

        # Training (from training config)
        'gradient_clip_enabled': training_config.training.gradient_clip.enabled,
        'gradient_clip_max_norm': training_config.training.gradient_clip.max_norm,
    }

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
        f"weights(spectral={loss_config['spectral_loss_weight']}, spatial={loss_config['spatial_loss_weight']})"
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

        # Extract pair construction params from parsed config
        ps = phase_loss_cfg.pair_strategy
        pw = phase_loss_cfg.pair_weights
        phase_config = {
            'k': ps.type_similarity.k if ps and ps.type_similarity else 16,
            'min_overlap': ps.ysfc_overlap.min_overlap if ps and ps.ysfc_overlap else 3,
            'min_pairs': ps.min_pairs if ps else 5,
            'include_self': ps.include_self if ps else True,
            'sigma': pw.sigma if pw else 5.0,
            'self_pair_weight': pw.self_pair_weight if pw else 1.0,
        }
        logger.info(
            f"Phase pair construction enabled: sampler={phase_anchor_pop}, "
            f"k={phase_config['k']}, min_overlap={phase_config['min_overlap']}, "
            f"min_pairs={phase_config['min_pairs']}, sigma={phase_config['sigma']}"
        )
    else:
        logger.info("Phase pair construction disabled (no soft_neighborhood_phase loss in config)")

    # Create output directories
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
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
    shutil.copy2(args.bindings, experiment_dir / Path(args.bindings).name)
    shutil.copy2(args.training, experiment_dir / Path(args.training).name)
    shutil.copy2(RepresentationModel.source_file(), experiment_dir / "representation.py")
    logger.info(f"Saved config and model source to {experiment_dir}")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_dataset.on_epoch_start()  # Reshuffle patches

        train_stats = train_epoch(
          train_dataloader, feature_builder, model,
          optimizer, scheduler, device, loss_config, epoch, num_epochs,
          phase_sampler=phase_sampler, phase_config=phase_config,
        )

        val_stats = validate_epoch(
          val_dataloader, feature_builder, model,
          device, loss_config,
          phase_sampler=phase_sampler, phase_config=phase_config,
        )

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} complete | "
            f"Train Loss: {train_stats['loss']:.4f} "
            f"(spec: {train_stats['spectral_loss']:.4f}, spat: {train_stats['spatial_loss']:.4f}) | "
            f"Val Loss: {val_stats['loss']:.4f} "
            f"(spec: {val_stats['spectral_loss']:.4f}, spat: {val_stats['spatial_loss']:.4f}) | "
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

        # Log phase pair construction stats
        ps = train_stats.get('phase_pair_stats')
        if ps and ps['n_anchors'] > 0:
            logger.info(
                f"  Phase pairs: {ps['n_total_pairs']:.0f} total "
                f"({ps['n_self_pairs']:.0f} self + {ps['n_total_pairs'] - ps['n_self_pairs']:.0f} cross) | "
                f"Anchors: {ps['n_anchors_surviving']:.0f}/{ps['n_anchors']:.0f} surviving | "
                f"kNN candidates: {ps['n_candidates']:.0f} -> overlap filter: {ps['n_after_overlap']:.0f} | "
                f"Overlap: mean={ps['overlap_mean']:.1f}, min={ps['overlap_min']} | "
                f"Weights: {ps['weight_mean']:.3f}±{ps['weight_std']:.3f}"
            )

        # Save checkpoint
        ckpt_path = ckpt_dir / f"encoder_epoch_{epoch+1:03d}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_version': RepresentationModel.VERSION,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_stats['loss'],
            'train_spectral_loss': train_stats['spectral_loss'],
            'train_spatial_loss': train_stats['spatial_loss'],
            'val_loss': val_stats['loss'],
            'val_spectral_loss': val_stats['spectral_loss'],
            'val_spatial_loss': val_stats['spatial_loss'],
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
