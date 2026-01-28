#!/usr/bin/env python3
"""
Minimal training script for representation learning.

This script trains a simple encoder using InfoNCE contrastive loss:
- Input: features.ccdc_history (15 channels)
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

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

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
from models import Conv2DEncoder
from losses import contrastive_loss, pairs_with_spatial_constraint
from utils import compute_spatial_distances, extract_at_locations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_one_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict:
    """
    Train on a single batch.

    Args:
        batch: Batch from dataloader
        feature_builder: FeatureBuilder instance
        encoder: Encoder network
        optimizer: Optimizer
        device: Device to use
        config: Loss config dict

    Returns:
        Dict with loss values and stats
    """
    encoder.train()
    optimizer.zero_grad()

    # Process each sample in batch (pair generation is per-patch)
    total_loss = 0.0
    n_valid = 0
    total_pos_pairs = 0
    total_neg_pairs = 0

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
        distance_feature = feature_builder.build_feature('infonce_type_spectral', sample)

        # Convert to tensors
        encoder_data = torch.from_numpy(encoder_feature.data).float().to(device)  # [C, H, W]
        distance_data = torch.from_numpy(distance_feature.data).float().to(device)  # [C, H, W]
        mask = torch.from_numpy(encoder_feature.mask).to(device)  # [H, W]

        # Also apply distance feature mask
        dist_mask = torch.from_numpy(distance_feature.mask).to(device)
        combined_mask = mask & dist_mask

        # Sample anchor locations
        anchors = sample_anchors_grid_plus_supplement(
            combined_mask,
            stride=config.get('stride', 16),
            border=config.get('border', 16),
            jitter_radius=config.get('jitter_radius', 4),
            supplement_n=config.get('supplement_n', 104),
        )

        if anchors.shape[0] < 10:
            # Skip if too few valid anchors
            continue

        # Extract features at anchor locations
        encoder_at_anchors = extract_at_locations(encoder_data, anchors)  # [N, 15]
        distance_at_anchors = extract_at_locations(distance_data, anchors)  # [N, 7]

        # Compute distances
        # Mahalanobis transform already applied by FeatureBuilder, so L2 here = Mahalanobis
        feature_distances = torch.cdist(distance_at_anchors, distance_at_anchors)
        spatial_distances = compute_spatial_distances(anchors)

        # Generate pairs
        pos_pairs, neg_pairs = pairs_with_spatial_constraint(
            feature_distances,
            spatial_distances,
            positive_k=config.get('positive_k', 16),
            positive_min_spatial=config.get('positive_min_spatial', 4.0),
            negative_quantile_low=config.get('negative_quantile_low', 0.5),
            negative_quantile_high=config.get('negative_quantile_high', 0.75),
            negative_min_spatial=config.get('negative_min_spatial', 8.0),
        )

        if pos_pairs.shape[0] == 0 or neg_pairs.shape[0] == 0:
            continue

        # Encode through network
        # Add batch and remove: [N, C] -> [1, C, 1, N] -> conv -> [1, D, 1, N] -> [N, D]
        encoder_input = encoder_at_anchors.T.unsqueeze(0).unsqueeze(2)  # [1, C, 1, N]
        z_type = encoder(encoder_input)  # [1, D, 1, N]
        z_type = z_type.squeeze(0).squeeze(1).T  # [N, D]

        # Compute loss
        loss = contrastive_loss(
            z_type,
            pos_pairs,
            neg_pairs,
            temperature=config.get('temperature', 0.07),
            similarity='l2',
        )

        # Skip if loss is NaN or Inf (numerical instability)
        if not torch.isfinite(loss):
            logger.warning(f"Skipping sample with non-finite loss: {loss.item()}")
            continue

        total_loss += loss
        n_valid += 1
        total_pos_pairs += pos_pairs.shape[0]
        total_neg_pairs += neg_pairs.shape[0]

    if n_valid == 0:
        return {'loss': 0.0, 'n_valid': 0, 'pos_pairs': 0, 'neg_pairs': 0}

    # Average loss over valid samples in batch
    # Note: contrastive_loss already averages over anchors within each sample,
    # so this averages across samples (standard mini-batch averaging)
    mean_loss = total_loss / n_valid

    # Final NaN check before backward
    if not torch.isfinite(mean_loss):
        logger.warning(f"Skipping batch with non-finite mean loss: {mean_loss.item()}")
        return {'loss': float('nan'), 'n_valid': 0, 'pos_pairs': 0, 'neg_pairs': 0}

    # Backward
    mean_loss.backward()

    # Gradient clipping
    if config.get('gradient_clip_enabled', True):
        torch.nn.utils.clip_grad_norm_(
            encoder.parameters(),
            max_norm=config.get('gradient_clip_max_norm', 1.0)
        )

    optimizer.step()

    return {
        'loss': mean_loss.item(),
        'n_valid': n_valid,
        'pos_pairs': total_pos_pairs // n_valid,
        'neg_pairs': total_neg_pairs // n_valid,
    }


@torch.no_grad()
def validate_one_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """
    Validate on a single batch (no gradient updates).

    Args:
        batch: Batch from dataloader
        feature_builder: FeatureBuilder instance
        encoder: Encoder network
        device: Device to use
        config: Loss config dict

    Returns:
        Dict with loss values and stats
    """
    encoder.eval()

    total_loss = 0.0
    n_valid = 0
    total_pos_pairs = 0
    total_neg_pairs = 0

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
        distance_feature = feature_builder.build_feature('infonce_type_spectral', sample)

        # Convert to tensors
        encoder_data = torch.from_numpy(encoder_feature.data).float().to(device)
        distance_data = torch.from_numpy(distance_feature.data).float().to(device)
        mask = torch.from_numpy(encoder_feature.mask).to(device)

        # Also apply distance feature mask
        dist_mask = torch.from_numpy(distance_feature.mask).to(device)
        combined_mask = mask & dist_mask

        # Sample anchor locations (no jitter for deterministic validation)
        anchors = sample_anchors_grid_plus_supplement(
            combined_mask,
            stride=config.get('stride', 16),
            border=config.get('border', 16),
            jitter_radius=0,  # No jitter for validation
            supplement_n=config.get('supplement_n', 104),
        )

        if anchors.shape[0] < 10:
            continue

        # Extract features at anchor locations
        encoder_at_anchors = extract_at_locations(encoder_data, anchors)
        distance_at_anchors = extract_at_locations(distance_data, anchors)

        # Compute distances
        feature_distances = torch.cdist(distance_at_anchors, distance_at_anchors)
        spatial_distances = compute_spatial_distances(anchors)

        # Generate pairs
        pos_pairs, neg_pairs = pairs_with_spatial_constraint(
            feature_distances,
            spatial_distances,
            positive_k=config.get('positive_k', 16),
            positive_min_spatial=config.get('positive_min_spatial', 4.0),
            negative_quantile_low=config.get('negative_quantile_low', 0.5),
            negative_quantile_high=config.get('negative_quantile_high', 0.75),
            negative_min_spatial=config.get('negative_min_spatial', 8.0),
        )

        if pos_pairs.shape[0] == 0 or neg_pairs.shape[0] == 0:
            continue

        # Encode through network
        encoder_input = encoder_at_anchors.T.unsqueeze(0).unsqueeze(2)
        z_type = encoder(encoder_input)
        z_type = z_type.squeeze(0).squeeze(1).T

        # Compute loss
        loss = contrastive_loss(
            z_type,
            pos_pairs,
            neg_pairs,
            temperature=config.get('temperature', 0.07),
            similarity='l2',
        )

        total_loss += loss.item()
        n_valid += 1
        total_pos_pairs += pos_pairs.shape[0]
        total_neg_pairs += neg_pairs.shape[0]

    if n_valid == 0:
        return {'loss': 0.0, 'n_valid': 0, 'pos_pairs': 0, 'neg_pairs': 0}

    return {
        'loss': total_loss / n_valid,
        'n_valid': n_valid,
        'pos_pairs': total_pos_pairs // n_valid,
        'neg_pairs': total_neg_pairs // n_valid,
    }


def validate_epoch(
    val_dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    device: torch.device,
    config: dict,
) -> float:
    """Run validation on entire validation set."""
    total_loss = 0.0
    total_batches = 0

    for batch in val_dataloader:
        stats = validate_one_batch(batch, feature_builder, encoder, device, config)
        if stats['n_valid'] > 0:
            total_loss += stats['loss']
            total_batches += 1

    if total_batches == 0:
        return 0.0

    return total_loss / total_batches


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
    checkpoint_dir = args.checkpoint_dir or training_config.run.ckpt_dir

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
        in_channels=15,  # ccdc_history has 15 channels
        channels=[128, 64],  # Two layers: 128, 64
        kernel_size=1,
        padding=0,
        dropout_rate=0.1,
        num_groups=8,  # GroupNorm with 8 groups
    ).to(device)

    logger.info(f"Encoder: {encoder}")
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Create optimizer
    logger.info(f"Creating optimizer: lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
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
        'stride': 16,
        'border': 16,
        'jitter_radius': 4,
        'supplement_n': 104,
        'positive_k': 16,
        'positive_min_spatial': 4.0,
        'negative_quantile_low': 0.5,
        'negative_quantile_high': 0.75,
        'negative_min_spatial': 8.0,
        'temperature': 0.07,
        'gradient_clip_enabled': training_config.training.gradient_clip.enabled,
        'gradient_clip_max_norm': training_config.training.gradient_clip.max_norm,
    }

    # Create checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_dataset.on_epoch_start()  # Reshuffle patches
        encoder.train()

        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            stats = train_one_batch(
                batch,
                feature_builder,
                encoder,
                optimizer,
                device,
                loss_config,
            )

            scheduler.step()

            if stats['n_valid'] > 0:
                epoch_loss += stats['loss']
                epoch_batches += 1

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Batch {batch_idx+1}/{len(train_dataloader)} | "
                        f"Loss: {stats['loss']:.4f} | "
                        f"Pos: {stats['pos_pairs']} | "
                        f"Neg: {stats['neg_pairs']} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

        train_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0

        # Validation
        val_loss = validate_epoch(val_dataloader, feature_builder, encoder, device, loss_config)

        logger.info(
            f"Epoch {epoch+1} complete | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save checkpoint
        ckpt_path = ckpt_dir / f"encoder_epoch_{epoch+1:03d}.pt"
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
