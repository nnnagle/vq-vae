#!/usr/bin/env python3
"""
Closed-form linear probe for evaluating learned representations.

What it does
- Loads a frozen encoder checkpoint.
- Runs the encoder over the TRAIN split to extract per-pixel embeddings z (D=64).
- Fits a linear probe (W,b) in CLOSED FORM via streaming ridge regression on valid pixels:
      argmin_{W,b} ||XW + b - Y||^2 + λ||W||^2
  without ever materializing the full design matrix X (millions of pixels).
- Evaluates on TRAIN and VAL splits, reporting per-metric MSE and R² (masked).

Why this is better than SGD for a linear probe
- Deterministic, fast, no LR/epochs/optimizer sensitivity.
- Convex solution (least squares / ridge) for a linear head + MSE.

Usage:
    python training/linear_probe_closed_form.py \
        --checkpoint checkpoints/encoder_epoch_050.pt \
        --bindings config/frl_binding_v1.yaml \
        --training config/frl_training_v1.yaml \
        --ridge-lambda 1e-3 \
        --max-batches-train 0 \
        --max-batches-eval 0

Notes / Assumptions (important)
- This script assumes the target_metrics feature channels are in the SAME ORDER as TARGET_METRICS.
  If that’s not true in your pipeline, you MUST map by channel name (see TODO in code).
- Masks are assumed boolean with True = valid.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from models import Conv2DEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("linear_probe_closed_form")


TARGET_METRICS = [
    "static.mean_ndvi",
    "static.mean_ndmi",
    "static.mean_nbr",
    "static.mean_seasonal_amp_nir",
    "static.variance_ndvi",
]


@dataclass
class ProbeMetrics:
    mse_per_metric: Dict[str, float]
    r2_per_metric: Dict[str, float]
    mse_total: float
    r2_total: float
    n_pixels: int


def _iter_batches(dataloader: DataLoader, max_batches: int):
    """Helper: iterate through dataloader with optional cap. max_batches=0 => no cap."""
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        yield batch


def extract_batch_tensors(
    batch: dict,
    feature_builder: FeatureBuilder,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build encoder inputs, targets, and mask for a batch.

    Returns:
        Ximg: [B, Cin, H, W]
        Yimg: [B, T,   H, W]
        M:    [B, H,   W] boolean mask (True=valid)
    """
    batch_size = len(batch["metadata"])
    encoder_inputs = []
    target_tensors = []
    masks = []

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f = feature_builder.build_feature("ccdc_history", sample)
        tgt_f = feature_builder.build_feature("target_metrics", sample)

        enc = torch.from_numpy(enc_f.data).float()
        tgt = torch.from_numpy(tgt_f.data).float()

        # TODO: if tgt channel order isn't guaranteed, map channels by name here.
        # Example pattern (depends on your Feature object):
        # names = tgt_f.channel_names
        # idx = [names.index(m) for m in TARGET_METRICS]
        # tgt = tgt[idx, :, :]

        m = torch.from_numpy(enc_f.mask) & torch.from_numpy(tgt_f.mask)

        encoder_inputs.append(enc)
        target_tensors.append(tgt)
        masks.append(m)

    Ximg = torch.stack(encoder_inputs).to(device)  # [B, Cin, H, W]
    Yimg = torch.stack(target_tensors).to(device)  # [B, T,   H, W]
    M = torch.stack(masks).to(device)              # [B, H,   W]
    return Ximg, Yimg, M


def fit_closed_form_ridge(
    train_loader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    device: torch.device,
    ridge_lambda: float = 1e-3,
    max_batches_train: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming ridge regression to fit linear probe:
        y = xW + b

    Returns:
        W: [D, T]
        b: [T]
    """
    encoder.eval()
    D = 64
    T = len(TARGET_METRICS)
    Da = D + 1  # augmented with bias term

    # Accumulate normal equations in float64 for numerical stability.
    A = torch.zeros((Da, Da), device=device, dtype=torch.float64)
    B = torch.zeros((Da, T), device=device, dtype=torch.float64)

    n_pixels_total = 0

    with torch.no_grad():
        for batch in _iter_batches(train_loader, max_batches_train):
            Ximg, Yimg, M = extract_batch_tensors(batch, feature_builder, device)
            z = encoder(Ximg)  # [B, D, H, W]

            # Flatten pixels
            z_perm = z.permute(0, 2, 3, 1).contiguous()     # [B, H, W, D]
            y_perm = Yimg.permute(0, 2, 3, 1).contiguous()  # [B, H, W, T]
            m_flat = M.reshape(-1)

            X = z_perm.reshape(-1, D)[m_flat]               # [N, D]
            Y = y_perm.reshape(-1, T)[m_flat]               # [N, T]

            if X.numel() == 0:
                continue

            n_pixels_total += X.shape[0]

            ones = torch.ones((X.shape[0], 1), device=device, dtype=X.dtype)
            Xaug = torch.cat([X, ones], dim=1).to(torch.float64)  # [N, D+1]
            Y = Y.to(torch.float64)

            A += Xaug.T @ Xaug
            B += Xaug.T @ Y

    if n_pixels_total == 0:
        raise RuntimeError("No valid pixels found in training data; cannot fit probe.")

    # Ridge penalty: penalize weights but not bias
    reg = torch.eye(Da, device=device, dtype=torch.float64) * ridge_lambda
    reg[-1, -1] = 0.0

    Wb = torch.linalg.solve(A + reg, B)  # [D+1, T]
    W = Wb[:-1, :].to(torch.float32)     # [D, T]
    b = Wb[-1, :].to(torch.float32)      # [T]

    logger.info(f"Fitted closed-form probe on {n_pixels_total:,} valid pixels (ridge_lambda={ridge_lambda:g}).")
    return W, b


def evaluate_probe(
    loader: DataLoader,
    feature_builder: FeatureBuilder,
    encoder: nn.Module,
    W: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    max_batches_eval: int = 0,
) -> ProbeMetrics:
    """
    Evaluate MSE and R² per metric (masked) across an entire split.
    """
    encoder.eval()
    D = W.shape[0]
    T = W.shape[1]
    assert T == len(TARGET_METRICS)

    # Accumulate SSE and SST per metric, plus counts for MSE.
    sse = {m: 0.0 for m in TARGET_METRICS}
    sst = {m: 0.0 for m in TARGET_METRICS}
    sum_y = {m: 0.0 for m in TARGET_METRICS}
    sum_y2 = {m: 0.0 for m in TARGET_METRICS}
    n = {m: 0 for m in TARGET_METRICS}
    total_pixels = 0

    with torch.no_grad():
        for batch in _iter_batches(loader, max_batches_eval):
            Ximg, Yimg, M = extract_batch_tensors(batch, feature_builder, device)
            z = encoder(Ximg)  # [B, D, H, W]

            # Predict: [B, T, H, W]
            pred = torch.einsum("bdhw,dt->bthw", z, W) + b.view(1, -1, 1, 1)

            # Flatten
            pred_perm = pred.permute(0, 2, 3, 1).contiguous()  # [B,H,W,T]
            y_perm = Yimg.permute(0, 2, 3, 1).contiguous()     # [B,H,W,T]
            m_flat = M.reshape(-1)

            P = pred_perm.reshape(-1, T)[m_flat]               # [N,T]
            Y = y_perm.reshape(-1, T)[m_flat]                  # [N,T]

            if Y.numel() == 0:
                continue

            total_pixels += Y.shape[0]

            # Accumulate sufficient statistics per metric
            # We’ll compute SSE directly, and compute SST using sums for stability:
            # SST = sum((y - mean)^2) = sum(y^2) - (sum(y)^2)/n
            for t_idx, metric in enumerate(TARGET_METRICS):
                yt = Y[:, t_idx]
                pt = P[:, t_idx]
                err = pt - yt

                sse[metric] += float((err * err).sum().item())
                sum_y[metric] += float(yt.sum().item())
                sum_y2[metric] += float((yt * yt).sum().item())
                n[metric] += int(yt.numel())

    mse_per = {}
    r2_per = {}

    for metric in TARGET_METRICS:
        if n[metric] == 0:
            mse_per[metric] = 0.0
            r2_per[metric] = 0.0
            continue

        mse_per[metric] = sse[metric] / n[metric]

        # SST from sums
        sst_val = max(0.0, sum_y2[metric] - (sum_y[metric] * sum_y[metric]) / n[metric])
        sst[metric] = sst_val

        if sst_val > 1e-8:
            r2_per[metric] = 1.0 - (sse[metric] / sst_val)
        else:
            r2_per[metric] = 0.0

    mse_total = float(np.mean([mse_per[m] for m in TARGET_METRICS]))
    r2_total = float(np.mean([r2_per[m] for m in TARGET_METRICS]))

    return ProbeMetrics(
        mse_per_metric=mse_per,
        r2_per_metric=r2_per,
        mse_total=mse_total,
        r2_total=r2_total,
        n_pixels=total_pixels,
    )


def log_metrics(metrics: ProbeMetrics, prefix: str):
    logger.info(f"{prefix} results (valid pixels: {metrics.n_pixels:,}):")
    logger.info(f"{'Metric':<28} {'MSE':>12} {'R²':>12}")
    logger.info("-" * 56)
    for m in TARGET_METRICS:
        short = m.replace("static.", "")
        logger.info(f"{short:<28} {metrics.mse_per_metric[m]:>12.6f} {metrics.r2_per_metric[m]:>12.6f}")
    logger.info("-" * 56)
    logger.info(f"{'Average':<28} {metrics.mse_total:>12.6f} {metrics.r2_total:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Closed-form linear probe (streaming ridge regression)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to encoder checkpoint (.pt)")
    parser.add_argument("--bindings", type=str, default="config/frl_binding_v1.yaml", help="Bindings YAML")
    parser.add_argument("--training", type=str, default="config/frl_training_v1.yaml", help="Training YAML")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge penalty λ (weights only)")
    parser.add_argument("--max-batches-train", type=int, default=0, help="Cap batches for fitting (0 = all)")
    parser.add_argument("--max-batches-eval", type=int, default=0, help="Cap batches for eval (0 = all)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save fitted probe (pt)")
    args = parser.parse_args()

    logger.info(f"Loading bindings config from {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    logger.info(f"Loading training config from {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    batch_size = args.batch_size or training_config.training.batch_size
    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    patch_size = training_config.sampling.patch_size

    logger.info("Creating train dataset...")
    train_dataset = ForestDatasetV2(
        bindings_config,
        split="train",
        patch_size=patch_size,
        min_aoi_fraction=0.3,
    )
    logger.info(f"Train dataset has {len(train_dataset)} patches")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # order doesn't matter for closed-form; deterministic is nice
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info("Creating validation dataset...")
    val_dataset = ForestDatasetV2(
        bindings_config,
        split="val",
        patch_size=patch_size,
        min_aoi_fraction=0.3,
    )
    logger.info(f"Validation dataset has {len(val_dataset)} patches")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info("Creating feature builder...")
    feature_builder = FeatureBuilder(bindings_config)

    logger.info(f"Loading encoder from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    encoder = Conv2DEncoder(
        in_channels=16,
        channels=[128, 64],
        kernel_size=1,
        padding=0,
        dropout_rate=0.1,
        num_groups=8,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    logger.info("Encoder loaded and frozen.")

    # Fit closed-form probe
    W, b = fit_closed_form_ridge(
        train_loader,
        feature_builder,
        encoder,
        device,
        ridge_lambda=args.ridge_lambda,
        max_batches_train=args.max_batches_train,
    )

    # Evaluate
    train_metrics = evaluate_probe(
        train_loader,
        feature_builder,
        encoder,
        W,
        b,
        device,
        max_batches_eval=args.max_batches_eval,
    )
    val_metrics = evaluate_probe(
        val_loader,
        feature_builder,
        encoder,
        W,
        b,
        device,
        max_batches_eval=args.max_batches_eval,
    )

    log_metrics(train_metrics, prefix="TRAIN")
    log_metrics(val_metrics, prefix="VAL")

    # Save fitted probe
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.checkpoint).parent / "linear_probe_closed_form.pt"

    torch.save(
        {
            "W": W.cpu(),
            "b": b.cpu(),
            "ridge_lambda": args.ridge_lambda,
            "target_metrics": TARGET_METRICS,
            "encoder_checkpoint": args.checkpoint,
            "train_mse_total": train_metrics.mse_total,
            "train_r2_total": train_metrics.r2_total,
            "val_mse_total": val_metrics.mse_total,
            "val_r2_total": val_metrics.r2_total,
            "val_mse_per_metric": val_metrics.mse_per_metric,
            "val_r2_per_metric": val_metrics.r2_per_metric,
        },
        out_path,
    )
    logger.info(f"Saved fitted probe to {out_path}")


if __name__ == "__main__":
    main()
