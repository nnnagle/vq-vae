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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from data.sampling import sample_anchors_grid_plus_supplement
from models import Conv2DEncoder, GatedResidualConv2D
from losses import contrastive_loss, pairs_with_spatial_constraint
from utils import (
    compute_spatial_distances,
    extract_at_locations,
    spatial_knn_pairs,
    spatial_negative_pairs,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    encoder: nn.Module,
    spatial_conv: nn.Module,
    device: torch.device,
    config: dict,
    training: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Process a single batch for training or validation.

    Args:
        batch: Batch from dataloader
        feature_builder: FeatureBuilder instance
        encoder: Encoder network
        spatial_conv: Gated spatial convolution module
        device: Device to use
        config: Loss config dict
        training: If True, run in training mode (gradients, optimizer step).
                  If False, run in eval mode (no gradients, no jitter).
        optimizer: Optimizer (required if training=True)

    Returns:
        Dict with loss values and stats
    """
    if training:
        if optimizer is None:
            raise ValueError("optimizer is required when training=True")
        encoder.train()
        spatial_conv.train()
        optimizer.zero_grad()
    else:
        encoder.eval()
        spatial_conv.eval()

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
        # encoder_data: [C, H, W] -> [1, C, H, W] -> conv -> [1, D, H, W]
        encoder_input_full = encoder_data.unsqueeze(0)
        h_full = encoder(encoder_input_full)  # [1, D, H, W]
        z_full, gate = spatial_conv(h_full, return_gate=True)  # [1, D, H, W] each
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

    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    if n_valid == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0, 'n_valid': 0,
            'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
            'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
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
            }

        # Backward
        mean_loss.backward()

        # Gradient clipping (both encoder and spatial_conv)
        if config.get('gradient_clip_enabled', True):
            all_params = list(encoder.parameters()) + list(spatial_conv.parameters())
            torch.nn.utils.clip_grad_norm_(
                all_params,
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
    }


def train_epoch(
    train_dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    spatial_conv: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: dict,
    epoch: int,
    num_epochs: int,
    log_interval: int = 10,
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

    for batch_idx, batch in enumerate(train_dataloader):
        stats = process_batch(
            batch, feature_builder, encoder, spatial_conv, device, config,
            training=True, optimizer=optimizer,
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

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{len(train_dataloader)} | "
                    f"Loss: {stats['loss']:.4f} (spec: {stats['spectral_loss']:.4f}, spat: {stats['spatial_loss']:.4f}) | "
                    f"Pairs(+/-): spec {stats['spectral_pos_pairs']}/{stats['spectral_neg_pairs']}, "
                    f"spat {stats['spatial_pos_pairs']}/{stats['spatial_neg_pairs']} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

    if total_batches == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0, 'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
        }

    return {
        'loss': total_loss / total_batches,
        'spectral_loss': total_spectral_loss / total_batches,
        'spatial_loss': total_spatial_loss / total_batches,
        'batches': total_batches,
        'gate_stats': last_gate_stats,
        'pos_weight_stats': last_pos_weight_stats,
        'neg_weight_stats': last_neg_weight_stats,
    }

def validate_epoch(
    val_dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    spatial_conv: nn.Module,
    device: torch.device,
    config: dict,
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

    with torch.no_grad():
        for batch in val_dataloader:
            stats = process_batch(
                batch, feature_builder, encoder, spatial_conv, device, config,
                training=False,
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

    if total_batches == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0, 'batches': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
        }

    return {
        'loss': total_loss / total_batches,
        'spectral_loss': total_spectral_loss / total_batches,
        'spatial_loss': total_spatial_loss / total_batches,
        'batches': total_batches,
        'gate_stats': last_gate_stats,
        'pos_weight_stats': last_pos_weight_stats,
        'neg_weight_stats': last_neg_weight_stats,
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

    # Create encoder
    logger.info("Creating encoder...")
    encoder = Conv2DEncoder(
        in_channels=16,  # ccdc_history has 16 channels
        channels=[128, 64],  # Two layers: 128, 64
        kernel_size=1,
        padding=0,
        dropout_rate=0.1,
        num_groups=8,  # GroupNorm with 8 groups
    ).to(device)

    logger.info(f"Encoder: {encoder}")

    # Create gated spatial convolution
    logger.info("Creating spatial convolution...")
    spatial_conv = GatedResidualConv2D(
        channels=64,  # matches encoder output
        num_layers=2,
        kernel_size=3,
        padding=1,
        gate_hidden=64,
        gate_kernel_size=1,
    ).to(device)

    logger.info(f"Spatial conv: {spatial_conv}")

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    spatial_params = sum(p.numel() for p in spatial_conv.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: encoder={encoder_params:,}, spatial={spatial_params:,}, total={encoder_params + spatial_params:,}")

    # Create optimizer with both encoder and spatial_conv parameters
    logger.info(f"Creating optimizer: lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(spatial_conv.parameters()),
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

    # Loss and training config
    loss_config = {
        # Sampling
        'stride': 16,
        'border': 16,
        'jitter_radius': 4,
        'supplement_n': 104,
        # Spectral InfoNCE loss
        'positive_k': 16,
        'positive_min_spatial': 4.0,
        'negative_quantile_low': 0.5,
        'negative_quantile_high': 0.75,
        'negative_min_spatial': 8.0,
        'temperature': 0.07,
        # Spatial InfoNCE loss (offset-grid approach)
        'spatial_positive_k': 4,
        'spatial_positive_max_dist': 8,  # max radius for positive neighbors
        'spatial_negative_min_dist': 96.0,  # min distance for negatives
        'spatial_negative_max_dist': 192.0,  # max distance for negatives
        'spatial_negatives_per_anchor': 16,  # number of negatives per anchor
        'spatial_spectral_tau': 2, # dist. metric on spectral dist, ideally ~ .5*sqrt(2*ndim)
        'spatial_min_w': .03,
        'spatial_temperature': 0.07,
        # Loss weights
        'spectral_loss_weight': 0.2,
        'spatial_loss_weight': 1.0,
        # Training
        'gradient_clip_enabled': training_config.training.gradient_clip.enabled,
        'gradient_clip_max_norm': training_config.training.gradient_clip.max_norm,
    }

    # Create output directories
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint dir: {ckpt_dir}")
    logger.info(f"Log dir: {log_dir}")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_dataset.on_epoch_start()  # Reshuffle patches

        train_stats = train_epoch(
          train_dataloader, feature_builder, encoder, spatial_conv,
          optimizer, scheduler, device, loss_config, epoch, num_epochs,
        )

        val_stats = validate_epoch(
          val_dataloader, feature_builder, encoder, spatial_conv,
          device, loss_config,
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

        # Save checkpoint
        ckpt_path = ckpt_dir / f"encoder_epoch_{epoch+1:03d}.pt"
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'spatial_conv_state_dict': spatial_conv.state_dict(),
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
