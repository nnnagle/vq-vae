#!/usr/bin/env python3
"""
Linear probe training script for evaluating learned representations.

This script trains a linear head on top of frozen encoder representations
to predict target metrics (static spectral indices from target_metrics feature).

The linear probe tests whether the encoder's learned representations
contain information predictive of the target metrics.

Usage:
    python training/train_linear_probe.py \
        --checkpoint checkpoints/encoder_epoch_050.pt \
        --bindings config/frl_binding_v1.yaml \
        --training config/frl_training_v1.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import Conv2DEncoder, LinearHead

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Target metrics from target_metrics feature (static [C,H,W])
TARGET_METRICS = [
    'static.mean_ndvi',
    'static.mean_ndmi',
    'static.mean_nbr',
    'static.mean_seasonal_amp_nir',
    'static.variance_ndvi',
]


@dataclass
class ProbeMetrics:
    """Metrics for linear probe evaluation."""
    mse_per_metric: Dict[str, float]
    r2_per_metric: Dict[str, float]
    mse_total: float
    r2_total: float
    n_samples: int


def compute_r2(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute R² (coefficient of determination) for masked predictions.

    Args:
        predictions: [N] predicted values
        targets: [N] target values
        mask: [N] boolean mask (True = valid)

    Returns:
        R² score (can be negative if predictions are worse than mean)
    """
    if mask.sum() == 0:
        return 0.0

    pred_masked = predictions[mask]
    tgt_masked = targets[mask]

    ss_res = ((pred_masked - tgt_masked) ** 2).sum()
    ss_tot = ((tgt_masked - tgt_masked.mean()) ** 2).sum()

    if ss_tot < 1e-8:
        return 0.0

    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    probe_head: nn.Module,
    device: torch.device,
    training: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """
    Process a single batch for training or validation.

    Args:
        batch: Batch from dataloader
        feature_builder: FeatureBuilder instance
        encoder: Frozen encoder network
        probe_head: Linear probe head (trainable)
        device: Device to use
        training: If True, run optimizer step
        optimizer: Optimizer (required if training=True)

    Returns:
        Tuple of (mse_per_metric, r2_per_metric, n_valid_pixels)
    """
    if training:
        if optimizer is None:
            raise ValueError("optimizer is required when training=True")
        probe_head.train()
        optimizer.zero_grad()
    else:
        probe_head.eval()

    # Encoder is always in eval mode (frozen)
    encoder.eval()

    batch_mse = {name: 0.0 for name in TARGET_METRICS}
    batch_ss_res = {name: 0.0 for name in TARGET_METRICS}
    batch_ss_tot = {name: 0.0 for name in TARGET_METRICS}
    batch_mean = {name: 0.0 for name in TARGET_METRICS}
    total_pixels = 0

    batch_size = len(batch['metadata'])

    # Collect all samples for batch processing
    encoder_inputs = []
    target_tensors = []
    masks = []

    for i in range(batch_size):
        # Extract single sample from batch
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != 'metadata'
        }
        sample['metadata'] = batch['metadata'][i]

        # Build encoder input feature (ccdc_history)
        encoder_feature = feature_builder.build_feature('ccdc_history', sample)

        # Build target feature (target_metrics) - static [C, H, W]
        target_feature = feature_builder.build_feature('target_metrics', sample)

        # Get encoder input [C, H, W]
        encoder_data = torch.from_numpy(encoder_feature.data).float()

        # Get target data [C, H, W]
        target_data = torch.from_numpy(target_feature.data).float()

        # Combined mask: encoder mask AND target mask
        encoder_mask = torch.from_numpy(encoder_feature.mask)
        target_mask = torch.from_numpy(target_feature.mask)
        combined_mask = encoder_mask & target_mask

        encoder_inputs.append(encoder_data)
        target_tensors.append(target_data)
        masks.append(combined_mask)

    # Stack into batches
    encoder_batch = torch.stack(encoder_inputs).to(device)  # [B, C_in, H, W]
    target_batch = torch.stack(target_tensors).to(device)   # [B, C_out, H, W]
    mask_batch = torch.stack(masks).to(device)              # [B, H, W]

    # Forward pass through frozen encoder
    with torch.no_grad():
        z = encoder(encoder_batch)  # [B, 64, H, W]

    # Forward pass through probe head
    if training:
        predictions = probe_head(z)  # [B, n_targets, H, W]
    else:
        with torch.no_grad():
            predictions = probe_head(z)

    # Compute loss (MSE on valid pixels only)
    # Expand mask to match predictions shape
    mask_expanded = mask_batch.unsqueeze(1).expand_as(predictions)  # [B, C, H, W]

    # Compute per-metric MSE and R²
    total_loss = 0.0
    n_valid = mask_batch.sum().item()

    if n_valid > 0:
        for c_idx, metric_name in enumerate(TARGET_METRICS):
            pred_c = predictions[:, c_idx, :, :]  # [B, H, W]
            tgt_c = target_batch[:, c_idx, :, :]  # [B, H, W]
            mask_c = mask_batch  # [B, H, W]

            # MSE for this metric
            mse_c = F.mse_loss(pred_c[mask_c], tgt_c[mask_c], reduction='mean')
            batch_mse[metric_name] = mse_c.item()
            total_loss = total_loss + mse_c

            # R² components (sum of squared residuals and total)
            batch_ss_res[metric_name] = ((pred_c[mask_c] - tgt_c[mask_c]) ** 2).sum().item()
            tgt_mean = tgt_c[mask_c].mean()
            batch_ss_tot[metric_name] = ((tgt_c[mask_c] - tgt_mean) ** 2).sum().item()
            batch_mean[metric_name] = tgt_mean.item()

        total_pixels = int(n_valid)

        if training:
            total_loss.backward()
            optimizer.step()

    # Compute R² per metric
    batch_r2 = {}
    for metric_name in TARGET_METRICS:
        if batch_ss_tot[metric_name] > 1e-8:
            batch_r2[metric_name] = 1 - (batch_ss_res[metric_name] / batch_ss_tot[metric_name])
        else:
            batch_r2[metric_name] = 0.0

    return batch_mse, batch_r2, total_pixels


def train_epoch(
    dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    probe_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    log_interval: int = 10,
) -> ProbeMetrics:
    """Run training for one epoch."""

    # Accumulators for epoch metrics
    epoch_mse = {name: 0.0 for name in TARGET_METRICS}
    epoch_ss_res = {name: 0.0 for name in TARGET_METRICS}
    epoch_ss_tot = {name: 0.0 for name in TARGET_METRICS}
    total_pixels = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        batch_mse, batch_r2, n_pixels = process_batch(
            batch, feature_builder, encoder, probe_head, device,
            training=True, optimizer=optimizer,
        )

        if n_pixels > 0:
            for name in TARGET_METRICS:
                epoch_mse[name] += batch_mse[name]
            total_pixels += n_pixels
            n_batches += 1

            if batch_idx % log_interval == 0:
                avg_mse = sum(batch_mse.values()) / len(TARGET_METRICS)
                avg_r2 = sum(batch_r2.values()) / len(TARGET_METRICS)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"MSE: {avg_mse:.4f} | R²: {avg_r2:.4f}"
                )

    # Compute epoch averages
    if n_batches > 0:
        for name in TARGET_METRICS:
            epoch_mse[name] /= n_batches

    # For R², we need to recompute over validation set properly
    # Training R² is approximate (batch-level)
    epoch_r2 = {name: 0.0 for name in TARGET_METRICS}

    mse_total = sum(epoch_mse.values()) / len(TARGET_METRICS)
    r2_total = sum(epoch_r2.values()) / len(TARGET_METRICS)

    return ProbeMetrics(
        mse_per_metric=epoch_mse,
        r2_per_metric=epoch_r2,
        mse_total=mse_total,
        r2_total=r2_total,
        n_samples=total_pixels,
    )


def validate_epoch(
    dataloader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    probe_head: nn.Module,
    device: torch.device,
) -> ProbeMetrics:
    """Run validation and compute proper R² across entire dataset."""

    # Accumulators
    epoch_mse = {name: 0.0 for name in TARGET_METRICS}
    epoch_ss_res = {name: 0.0 for name in TARGET_METRICS}
    all_targets = {name: [] for name in TARGET_METRICS}
    all_preds = {name: [] for name in TARGET_METRICS}
    total_pixels = 0
    n_batches = 0

    probe_head.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch_size = len(batch['metadata'])

            # Process each sample
            for i in range(batch_size):
                sample = {
                    key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
                    for key, val in batch.items()
                    if key != 'metadata'
                }
                sample['metadata'] = batch['metadata'][i]

                # Build features
                encoder_feature = feature_builder.build_feature('ccdc_history', sample)
                target_feature = feature_builder.build_feature('target_metrics', sample)

                # Prepare tensors - both are static [C, H, W]
                encoder_data = torch.from_numpy(encoder_feature.data).float().to(device)
                target_data = torch.from_numpy(target_feature.data).float().to(device)

                encoder_mask = torch.from_numpy(encoder_feature.mask).to(device)
                target_mask = torch.from_numpy(target_feature.mask).to(device)
                combined_mask = encoder_mask & target_mask

                if combined_mask.sum() == 0:
                    continue

                # Forward pass
                encoder_input = encoder_data.unsqueeze(0)  # [1, C, H, W]
                z = encoder(encoder_input)
                predictions = probe_head(z).squeeze(0)  # [C_out, H, W]

                # Collect per-metric stats
                for c_idx, metric_name in enumerate(TARGET_METRICS):
                    pred_c = predictions[c_idx][combined_mask]
                    tgt_c = target_data[c_idx][combined_mask]

                    mse_c = F.mse_loss(pred_c, tgt_c, reduction='mean').item()
                    epoch_mse[metric_name] += mse_c

                    # Store for R² computation
                    all_preds[metric_name].append(pred_c.cpu())
                    all_targets[metric_name].append(tgt_c.cpu())

                total_pixels += combined_mask.sum().item()
                n_batches += 1

    # Compute final metrics
    if n_batches > 0:
        for name in TARGET_METRICS:
            epoch_mse[name] /= n_batches

    # Compute R² over all collected predictions
    epoch_r2 = {}
    for metric_name in TARGET_METRICS:
        if all_preds[metric_name]:
            all_pred = torch.cat(all_preds[metric_name])
            all_tgt = torch.cat(all_targets[metric_name])

            ss_res = ((all_pred - all_tgt) ** 2).sum().item()
            ss_tot = ((all_tgt - all_tgt.mean()) ** 2).sum().item()

            if ss_tot > 1e-8:
                epoch_r2[metric_name] = 1 - (ss_res / ss_tot)
            else:
                epoch_r2[metric_name] = 0.0
        else:
            epoch_r2[metric_name] = 0.0

    mse_total = sum(epoch_mse.values()) / len(TARGET_METRICS) if epoch_mse else 0.0
    r2_total = sum(epoch_r2.values()) / len(TARGET_METRICS) if epoch_r2 else 0.0

    return ProbeMetrics(
        mse_per_metric=epoch_mse,
        r2_per_metric=epoch_r2,
        mse_total=mse_total,
        r2_total=r2_total,
        n_samples=total_pixels,
    )


def log_metrics(metrics: ProbeMetrics, prefix: str = ""):
    """Log per-metric MSE and R² values."""
    logger.info(f"{prefix} Per-metric results:")
    logger.info(f"{'Metric':<30} {'MSE':>10} {'R²':>10}")
    logger.info("-" * 52)
    for metric_name in TARGET_METRICS:
        mse = metrics.mse_per_metric.get(metric_name, 0.0)
        r2 = metrics.r2_per_metric.get(metric_name, 0.0)
        # Shorten metric name for display
        short_name = metric_name.replace('static.', '')
        logger.info(f"{short_name:<30} {mse:>10.4f} {r2:>10.4f}")
    logger.info("-" * 52)
    logger.info(f"{'Average':<30} {metrics.mse_total:>10.4f} {metrics.r2_total:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train linear probe on frozen encoder')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to encoder checkpoint'
    )
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
        default=20,
        help='Number of training epochs'
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
        default=1e-1,
        help='Learning rate for probe head'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save probe checkpoints'
    )
    args = parser.parse_args()

    # Parse configs
    logger.info(f"Loading bindings config from {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    logger.info(f"Loading training config from {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    # Apply config values with command-line overrides
    batch_size = args.batch_size or training_config.training.batch_size
    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Create datasets
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

    # Load encoder and freeze
    logger.info(f"Loading encoder from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    encoder = Conv2DEncoder(
        in_channels=16,  # ccdc_history has 16 channels
        channels=[128, 64],
        kernel_size=1,
        padding=0,
        dropout_rate=0.1,
        num_groups=8,
    ).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    logger.info("Encoder loaded and frozen")
    n_encoder_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder parameters: {n_encoder_params:,} (all frozen)")

    # Create linear probe head
    n_targets = len(TARGET_METRICS)
    probe_head = LinearHead(
        in_dim=64,  # encoder output dim
        out_dim=n_targets,
        activation='none',
    ).to(device)

    n_probe_params = sum(p.numel() for p in probe_head.parameters() if p.requires_grad)
    logger.info(f"Probe head parameters: {n_probe_params:,} (trainable)")
    logger.info(f"Predicting {n_targets} target metrics")

    # Create optimizer (only for probe head)
    optimizer = torch.optim.Adam(probe_head.parameters(), lr=args.lr)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / 'linear_probe'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info(f"Starting linear probe training for {args.epochs} epochs...")
    best_val_r2 = -float('inf')

    for epoch in range(args.epochs):
        train_dataset.on_epoch_start()

        # Train
        train_metrics = train_epoch(
            train_dataloader, feature_builder, encoder, probe_head,
            optimizer, device, epoch, args.epochs,
        )

        # Validate
        val_metrics = validate_epoch(
            val_dataloader, feature_builder, encoder, probe_head,
            device,
        )

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train MSE: {train_metrics.mse_total:.4f} | "
            f"Val MSE: {val_metrics.mse_total:.4f} | "
            f"Val R²: {val_metrics.r2_total:.4f}"
        )

        # Log detailed validation metrics
        log_metrics(val_metrics, prefix=f"Epoch {epoch+1} Val")

        # Save best model
        if val_metrics.r2_total > best_val_r2:
            best_val_r2 = val_metrics.r2_total
            best_path = output_dir / 'probe_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'probe_state_dict': probe_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_metrics.mse_total,
                'val_r2': val_metrics.r2_total,
                'val_mse_per_metric': val_metrics.mse_per_metric,
                'val_r2_per_metric': val_metrics.r2_per_metric,
                'target_metrics': TARGET_METRICS,
                'encoder_checkpoint': args.checkpoint,
            }, best_path)
            logger.info(f"Saved best model to {best_path} (R² = {best_val_r2:.4f})")

    # Save final model
    final_path = output_dir / f'probe_epoch_{args.epochs:03d}.pt'
    torch.save({
        'epoch': args.epochs,
        'probe_state_dict': probe_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mse': val_metrics.mse_total,
        'val_r2': val_metrics.r2_total,
        'val_mse_per_metric': val_metrics.mse_per_metric,
        'val_r2_per_metric': val_metrics.r2_per_metric,
        'target_metrics': TARGET_METRICS,
        'encoder_checkpoint': args.checkpoint,
    }, final_path)

    logger.info("Training complete!")
    logger.info(f"Best validation R²: {best_val_r2:.4f}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    log_metrics(val_metrics, prefix="Final")


if __name__ == '__main__':
    main()
