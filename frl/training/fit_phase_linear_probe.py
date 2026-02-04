#!/usr/bin/env python3
"""
Closed-form linear probe for evaluating phase + type representations.

What it does
- Loads a frozen RepresentationModel checkpoint (type + phase pathways).
- For every patch, runs the type encoder densely to get z_type [B, 64, H, W],
  then the phase encoder densely to get z_phase [B, 12, T, H, W].
- Concatenates [z_type broadcast across T; z_phase] → [B, 76, T, H, W].
- Fits a linear probe (W, b) in closed form via streaming ridge regression
  against the soft_neighborhood_phase target [B, 7, T, H, W] (Mahalanobis-
  whitened spectral time-series).
- Only pixels whose spatial location is ≥ ``halo`` pixels from the patch
  edge contribute, avoiding boundary artefacts from the spatial conv.
- Reports per-channel MSE and R² in both whitened and original data scale.

Denormalization
- The soft_neighborhood_phase feature is produced by: center (subtract per-
  channel mean), then whiten (W_whiten @ centered).  The inverse is:
  (1) colour: inv(W_whiten) @ whitened, then (2) add back channel means.

Usage:
    python training/fit_phase_linear_probe.py \\
        --checkpoint checkpoints/encoder_epoch_050.pt \\
        --bindings config/frl_binding_v1.yaml \\
        --training config/frl_training_v1.yaml \\
        --ridge-lambda 1e-3 \\
        --halo 16

Notes
- The dense ``forward_phase`` method is used (not ``forward_phase_at_locations``)
  because the probe needs predictions at every valid pixel.
- z_type is detached before being fed to the phase pathway (same convention as
  training: the type encoder is not trained through the phase pathway).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase_linear_probe")


PHASE_TARGET_FEATURE = "soft_neighborhood_phase"
PHASE_INPUT_FEATURE = "phase_ls8"

PHASE_TARGET_CHANNELS = [
    "annual.evi2_summer_p95",
    "annual.nbr_annual_min",
    "annual.nbr_summer_p95",
    "annual.ndmi_summer_mean",
    "annual.ndvi_amplitude",
    "annual.ndvi_summer_p95",
    "annual.ndvi_winter_max",
]


# ---------------------------------------------------------------------------
# Denormalization helpers
# ---------------------------------------------------------------------------

def _build_denorm_params(
    feature_builder: FeatureBuilder,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the inverse-whitening matrix and channel-mean vector.

    The forward transform in FeatureBuilder is:
        centered = data - mean            (per channel)
        whitened = W_whiten @ centered    (W_whiten = cholesky(cov_inv))

    The inverse is:
        centered = inv(W_whiten) @ whitened
        original = centered + mean

    Returns:
        W_inv: [C, C] inverse whitening matrix (float64 for precision).
        means: [C]    per-channel means (float64).
    """
    W_whiten = feature_builder._get_whitening_matrix(PHASE_TARGET_FEATURE)
    if W_whiten is None:
        raise RuntimeError(
            f"No whitening matrix found for feature '{PHASE_TARGET_FEATURE}'. "
            "Ensure the stats JSON contains a covariance entry for this feature."
        )

    W_inv = np.linalg.inv(W_whiten).astype(np.float64)

    feature_config = feature_builder.config.get_feature(PHASE_TARGET_FEATURE)
    means = np.array(
        feature_builder._get_channel_means(PHASE_TARGET_FEATURE, feature_config),
        dtype=np.float64,
    )

    return W_inv, means


def denormalize(
    whitened: torch.Tensor,
    W_inv: torch.Tensor,
    means: torch.Tensor,
) -> torch.Tensor:
    """Map predictions from whitened space back to original data scale.

    Args:
        whitened: [..., C] predictions in whitened space.
        W_inv: [C, C] inverse whitening matrix.
        means: [C] channel means.

    Returns:
        original: [..., C] predictions on the original data scale.
    """
    # whitened @ W_inv.T  =>  inv(W_whiten) @ whitened  (row-vector convention)
    centered = whitened @ W_inv.T
    return centered + means


# ---------------------------------------------------------------------------
# Halo mask
# ---------------------------------------------------------------------------

def _halo_mask(H: int, W: int, halo: int, device: torch.device) -> torch.Tensor:
    """Return a [H, W] bool mask that is True for pixels ≥ halo from all edges."""
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    if 2 * halo >= H or 2 * halo >= W:
        return mask  # patch too small — nothing survives
    mask[halo : H - halo, halo : W - halo] = True
    return mask


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_phase_batch_tensors(
    batch: dict,
    feature_builder: FeatureBuilder,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build encoder inputs, phase inputs, targets, and mask for a batch.

    Returns:
        Ximg:       [B, 16, H, W]        ccdc_history encoder input
        Xphase:     [B, 8, T, H, W]      phase_ls8 temporal input
        Yimg:       [B, 7, T, H, W]      soft_neighborhood_phase target (whitened)
        M:          [B, H, W]            boolean mask (True = valid, all features)
    """
    batch_size = len(batch["metadata"])
    encoder_inputs: List[torch.Tensor] = []
    phase_inputs: List[torch.Tensor] = []
    target_tensors: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f = feature_builder.build_feature("ccdc_history", sample)
        phase_f = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
        tgt_f = feature_builder.build_feature(PHASE_TARGET_FEATURE, sample)

        enc = torch.from_numpy(enc_f.data).float()        # [16, H, W]
        phase = torch.from_numpy(phase_f.data).float()    # [8, T, H, W]
        tgt = torch.from_numpy(tgt_f.data).float()        # [7, T, H, W]

        # Combine masks: encoder spatial mask AND target temporal mask
        # enc_f.mask: [H, W], tgt_f.mask: [T, H, W]
        # Collapse target temporal mask to spatial: valid only if ALL timesteps valid
        tgt_mask_spatial = torch.from_numpy(tgt_f.mask).all(dim=0)  # [H, W]
        phase_mask_spatial = torch.from_numpy(phase_f.mask).all(dim=0)  # [H, W]
        m = (
            torch.from_numpy(enc_f.mask)
            & tgt_mask_spatial
            & phase_mask_spatial
        )

        encoder_inputs.append(enc)
        phase_inputs.append(phase)
        target_tensors.append(tgt)
        masks.append(m)

    Ximg = torch.stack(encoder_inputs).to(device)      # [B, 16, H, W]
    Xphase = torch.stack(phase_inputs).to(device)      # [B, 8, T, H, W]
    Yimg = torch.stack(target_tensors).to(device)      # [B, 7, T, H, W]
    M = torch.stack(masks).to(device)                  # [B, H, W]
    return Ximg, Xphase, Yimg, M


# ---------------------------------------------------------------------------
# Streaming ridge regression
# ---------------------------------------------------------------------------

@dataclass
class PhaseProbeMetrics:
    mse_per_channel: Dict[str, float]
    r2_per_channel: Dict[str, float]
    mse_total: float
    r2_total: float
    n_observations: int
    # Optionally populated with denormalized-scale metrics
    mse_per_channel_original: Optional[Dict[str, float]] = None
    r2_per_channel_original: Optional[Dict[str, float]] = None
    mse_total_original: Optional[float] = None
    r2_total_original: Optional[float] = None


def _iter_batches(dataloader: DataLoader, max_batches: int):
    """Iterate with optional cap. max_batches=0 => no cap."""
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        yield batch


def fit_phase_probe(
    train_loader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    ridge_lambda: float = 1e-3,
    halo: int = 16,
    max_batches_train: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming ridge regression for the phase linear probe.

    The design matrix X is the concatenation of:
        z_type  [B, 64, H, W]  broadcast to [B, 64, T, H, W]
        z_phase [B, 12, T, H, W]
    giving D = 76 features per (pixel, timestep) observation.

    The target Y is soft_neighborhood_phase [B, 7, T, H, W].

    Only (pixel, timestep) observations that satisfy the combined data mask
    AND the halo constraint contribute.

    Returns:
        W: [D, C]  = [76, 7]
        b: [C]     = [7]
    """
    model.eval()
    D_type = 64
    D_phase = 12
    D = D_type + D_phase            # 76
    C = len(PHASE_TARGET_CHANNELS)  # 7
    Da = D + 1                      # augmented with bias

    A = torch.zeros((Da, Da), device=device, dtype=torch.float64)
    B_mat = torch.zeros((Da, C), device=device, dtype=torch.float64)
    n_obs_total = 0

    with torch.no_grad():
        for batch in _iter_batches(train_loader, max_batches_train):
            Ximg, Xphase, Yimg, M = extract_phase_batch_tensors(
                batch, feature_builder, device,
            )
            Bsz, _, H, W = Ximg.shape
            T = Xphase.shape[2]

            # Type encoder (dense)
            z_type = model(Ximg)  # [B, 64, H, W]

            # Phase encoder (dense, stop-grad on z_type)
            z_phase = model.forward_phase(Xphase, z_type.detach())  # [B, 12, T, H, W]

            # Broadcast z_type across T
            z_type_t = z_type.unsqueeze(2).expand(-1, -1, T, -1, -1)  # [B, 64, T, H, W]

            # Concatenate → [B, 76, T, H, W]
            z_full = torch.cat([z_type_t, z_phase], dim=1)

            # Apply halo mask
            halo_m = _halo_mask(H, W, halo, device)  # [H, W]
            full_mask = M & halo_m.unsqueeze(0)       # [B, H, W]

            # Flatten to [B, T, H, W, D] and [B, T, H, W, C]
            z_perm = z_full.permute(0, 2, 3, 4, 1).contiguous()    # [B, T, H, W, 76]
            y_perm = Yimg.permute(0, 2, 3, 4, 1).contiguous()      # [B, T, H, W, 7]

            # Broadcast mask: [B, H, W] → [B, T, H, W]
            m_t = full_mask.unsqueeze(1).expand(-1, T, -1, -1)     # [B, T, H, W]
            m_flat = m_t.reshape(-1)

            X = z_perm.reshape(-1, D)[m_flat]   # [N, 76]
            Y = y_perm.reshape(-1, C)[m_flat]   # [N, 7]

            if X.numel() == 0:
                continue

            n_obs_total += X.shape[0]

            ones = torch.ones((X.shape[0], 1), device=device, dtype=X.dtype)
            Xaug = torch.cat([X, ones], dim=1).to(torch.float64)  # [N, 77]
            Y64 = Y.to(torch.float64)

            A += Xaug.T @ Xaug       # [77, 77]
            B_mat += Xaug.T @ Y64    # [77, 7]

    if n_obs_total == 0:
        raise RuntimeError("No valid observations found in training data; cannot fit probe.")

    # Ridge penalty (not on bias)
    reg = torch.eye(Da, device=device, dtype=torch.float64) * ridge_lambda
    reg[-1, -1] = 0.0

    Wb = torch.linalg.solve(A + reg, B_mat)  # [77, 7]
    W = Wb[:-1, :].to(torch.float32)          # [76, 7]
    b = Wb[-1, :].to(torch.float32)           # [7]

    logger.info(
        f"Fitted phase probe on {n_obs_total:,} observations "
        f"(ridge_lambda={ridge_lambda:g}, halo={halo})."
    )
    return W, b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_phase_probe(
    loader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    W: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    halo: int = 16,
    max_batches_eval: int = 0,
    W_inv: Optional[torch.Tensor] = None,
    channel_means: Optional[torch.Tensor] = None,
) -> PhaseProbeMetrics:
    """Evaluate MSE and R² per channel (whitened space, and optionally original scale).

    Args:
        W_inv: [C, C] inverse whitening matrix (torch, float32). If provided
            together with *channel_means*, metrics in original data scale are
            also computed.
        channel_means: [C] per-channel means (torch, float32).
    """
    model.eval()
    D = W.shape[0]   # 76
    C = W.shape[1]   # 7

    compute_original = W_inv is not None and channel_means is not None

    # Accumulators — whitened space
    sse_w = torch.zeros(C, device=device, dtype=torch.float64)
    sum_y_w = torch.zeros(C, device=device, dtype=torch.float64)
    sum_y2_w = torch.zeros(C, device=device, dtype=torch.float64)

    # Accumulators — original scale
    sse_o = torch.zeros(C, device=device, dtype=torch.float64)
    sum_y_o = torch.zeros(C, device=device, dtype=torch.float64)
    sum_y2_o = torch.zeros(C, device=device, dtype=torch.float64)

    n_obs = 0

    with torch.no_grad():
        for batch in _iter_batches(loader, max_batches_eval):
            Ximg, Xphase, Yimg, M = extract_phase_batch_tensors(
                batch, feature_builder, device,
            )
            Bsz, _, H, W_sp = Ximg.shape
            T = Xphase.shape[2]

            z_type = model(Ximg)
            z_phase = model.forward_phase(Xphase, z_type.detach())

            z_type_t = z_type.unsqueeze(2).expand(-1, -1, T, -1, -1)
            z_full = torch.cat([z_type_t, z_phase], dim=1)

            halo_m = _halo_mask(H, W_sp, halo, device)
            full_mask = M & halo_m.unsqueeze(0)

            # Predict in whitened space: [B, C, T, H, W]
            # z_full is [B, D, T, H, W]; W is [D, C]
            pred_w = torch.einsum("bdthw,dc->bcthw", z_full, W) + b.view(1, -1, 1, 1, 1)

            # Flatten
            pred_perm = pred_w.permute(0, 2, 3, 4, 1).contiguous()   # [B,T,H,W,C]
            y_perm = Yimg.permute(0, 2, 3, 4, 1).contiguous()        # [B,T,H,W,C]
            m_t = full_mask.unsqueeze(1).expand(-1, T, -1, -1)
            m_flat = m_t.reshape(-1)

            P = pred_perm.reshape(-1, C)[m_flat].to(torch.float64)   # [N, C]
            Y = y_perm.reshape(-1, C)[m_flat].to(torch.float64)      # [N, C]

            if Y.numel() == 0:
                continue

            n_obs += Y.shape[0]

            err_w = P - Y
            sse_w += (err_w * err_w).sum(dim=0)
            sum_y_w += Y.sum(dim=0)
            sum_y2_w += (Y * Y).sum(dim=0)

            if compute_original:
                P_o = denormalize(P.float(), W_inv, channel_means).double()
                Y_o = denormalize(Y.float(), W_inv, channel_means).double()
                err_o = P_o - Y_o
                sse_o += (err_o * err_o).sum(dim=0)
                sum_y_o += Y_o.sum(dim=0)
                sum_y2_o += (Y_o * Y_o).sum(dim=0)

    # Compute metrics
    def _compute_metrics(sse, sum_y, sum_y2, n):
        mse_per = {}
        r2_per = {}
        for c_idx, ch in enumerate(PHASE_TARGET_CHANNELS):
            if n == 0:
                mse_per[ch] = 0.0
                r2_per[ch] = 0.0
                continue
            mse_per[ch] = float(sse[c_idx].item()) / n
            sst_val = max(0.0, float(sum_y2[c_idx].item()) - float(sum_y[c_idx].item()) ** 2 / n)
            if sst_val > 1e-8:
                r2_per[ch] = 1.0 - float(sse[c_idx].item()) / sst_val
            else:
                r2_per[ch] = 0.0
        mse_total = float(np.mean(list(mse_per.values())))
        r2_total = float(np.mean(list(r2_per.values())))
        return mse_per, r2_per, mse_total, r2_total

    mse_w, r2_w, mse_wt, r2_wt = _compute_metrics(sse_w, sum_y_w, sum_y2_w, n_obs)

    mse_o = r2_o = mse_ot = r2_ot = None
    if compute_original:
        mse_o, r2_o, mse_ot, r2_ot = _compute_metrics(sse_o, sum_y_o, sum_y2_o, n_obs)

    return PhaseProbeMetrics(
        mse_per_channel=mse_w,
        r2_per_channel=r2_w,
        mse_total=mse_wt,
        r2_total=r2_wt,
        n_observations=n_obs,
        mse_per_channel_original=mse_o,
        r2_per_channel_original=r2_o,
        mse_total_original=mse_ot,
        r2_total_original=r2_ot,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metrics(metrics: PhaseProbeMetrics, prefix: str):
    logger.info(f"{prefix} results (observations: {metrics.n_observations:,}):")

    logger.info(f"  Whitened space:")
    logger.info(f"  {'Channel':<30} {'MSE':>12} {'R²':>12}")
    logger.info(f"  {'-' * 56}")
    for ch in PHASE_TARGET_CHANNELS:
        short = ch.replace("annual.", "")
        logger.info(
            f"  {short:<30} "
            f"{metrics.mse_per_channel[ch]:>12.6f} "
            f"{metrics.r2_per_channel[ch]:>12.6f}"
        )
    logger.info(f"  {'-' * 56}")
    logger.info(f"  {'Average':<30} {metrics.mse_total:>12.6f} {metrics.r2_total:>12.6f}")

    if metrics.mse_per_channel_original is not None:
        logger.info(f"  Original data scale:")
        logger.info(f"  {'Channel':<30} {'MSE':>12} {'R²':>12}")
        logger.info(f"  {'-' * 56}")
        for ch in PHASE_TARGET_CHANNELS:
            short = ch.replace("annual.", "")
            logger.info(
                f"  {short:<30} "
                f"{metrics.mse_per_channel_original[ch]:>12.6f} "
                f"{metrics.r2_per_channel_original[ch]:>12.6f}"
            )
        logger.info(f"  {'-' * 56}")
        logger.info(
            f"  {'Average':<30} "
            f"{metrics.mse_total_original:>12.6f} "
            f"{metrics.r2_total_original:>12.6f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Closed-form phase linear probe (streaming ridge regression)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to encoder checkpoint (.pt)")
    parser.add_argument("--bindings", type=str, default="config/frl_binding_v1.yaml", help="Bindings YAML")
    parser.add_argument("--training", type=str, default="config/frl_training_v1.yaml", help="Training YAML")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge penalty λ (weights only)")
    parser.add_argument("--halo", type=int, default=16, help="Pixels from edge to exclude (default: 16)")
    parser.add_argument("--max-batches-train", type=int, default=0, help="Cap batches for fitting (0 = all)")
    parser.add_argument("--max-batches-eval", type=int, default=0, help="Cap batches for eval (0 = all)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save fitted probe (.pt)")
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
        shuffle=False,
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

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)

    # Build denormalization parameters
    W_inv_np, means_np = _build_denorm_params(feature_builder)
    W_inv_t = torch.from_numpy(W_inv_np).float().to(device)
    means_t = torch.from_numpy(means_np).float().to(device)

    logger.info(
        f"Denormalization: W_inv {W_inv_t.shape}, means {means_t.shape} "
        f"(channels: {PHASE_TARGET_CHANNELS})"
    )

    # Fit
    W, b_bias = fit_phase_probe(
        train_loader,
        feature_builder,
        model,
        device,
        ridge_lambda=args.ridge_lambda,
        halo=args.halo,
        max_batches_train=args.max_batches_train,
    )

    # Evaluate
    train_metrics = evaluate_phase_probe(
        train_loader, feature_builder, model, W, b_bias, device,
        halo=args.halo,
        max_batches_eval=args.max_batches_eval,
        W_inv=W_inv_t,
        channel_means=means_t,
    )
    val_metrics = evaluate_phase_probe(
        val_loader, feature_builder, model, W, b_bias, device,
        halo=args.halo,
        max_batches_eval=args.max_batches_eval,
        W_inv=W_inv_t,
        channel_means=means_t,
    )

    log_metrics(train_metrics, prefix="TRAIN")
    log_metrics(val_metrics, prefix="VAL")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.checkpoint).parent / "phase_linear_probe.pt"

    torch.save(
        {
            "W": W.cpu(),
            "b": b_bias.cpu(),
            "ridge_lambda": args.ridge_lambda,
            "halo": args.halo,
            "target_feature": PHASE_TARGET_FEATURE,
            "target_channels": PHASE_TARGET_CHANNELS,
            "input_dim_type": 64,
            "input_dim_phase": 12,
            "encoder_checkpoint": args.checkpoint,
            # Denormalization params
            "denorm_W_inv": torch.from_numpy(W_inv_np).float(),
            "denorm_means": torch.from_numpy(means_np).float(),
            # Summary metrics
            "train_mse_total": train_metrics.mse_total,
            "train_r2_total": train_metrics.r2_total,
            "val_mse_total": val_metrics.mse_total,
            "val_r2_total": val_metrics.r2_total,
            "val_mse_per_channel": val_metrics.mse_per_channel,
            "val_r2_per_channel": val_metrics.r2_per_channel,
            "val_mse_total_original": val_metrics.mse_total_original,
            "val_r2_total_original": val_metrics.r2_total_original,
            "val_mse_per_channel_original": val_metrics.mse_per_channel_original,
            "val_r2_per_channel_original": val_metrics.r2_per_channel_original,
        },
        out_path,
    )
    logger.info(f"Saved phase probe to {out_path}")


if __name__ == "__main__":
    main()
