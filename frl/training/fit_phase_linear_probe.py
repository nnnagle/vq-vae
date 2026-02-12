#!/usr/bin/env python3
"""
Closed-form linear probe for evaluating phase + type representations.

What it does
- Loads a frozen RepresentationModel checkpoint (type + phase pathways).
- For every patch, runs the type encoder densely to get z_type [B, 64, H, W],
  then the phase encoder densely to get z_phase [B, 12, T, H, W].
- Builds a design matrix from z_type (64), z_phase (12), and their full
  element-wise interaction z_type ⊗ z_phase (64×12 = 768), giving D = 844
  features per (pixel, timestep) observation.
- Fits a linear probe (W, b) in closed form via streaming ridge regression
  against the soft_neighborhood_phase target [B, 7, T, H, W] in the
  **transformed + normalized (Mahalanobis-whitened) space**.
- The interaction terms capture the multiplicative type×phase structure that
  FiLM conditioning creates in the encoder.
- Only pixels whose spatial location is ≥ ``halo`` pixels from the patch
  edge contribute, avoiding boundary artefacts from the spatial conv.
- Reports per-channel R² in both the normalized space and (after inverting
  the whitening, centering, and per-channel transforms) the original data
  scale.

Targets
- The soft_neighborhood_phase feature is extracted with full normalization
  (transform + Mahalanobis whitening), so the probe fits in the space the
  model actually optimises.  Predictions are then projected back to the
  original spectral units via the inverse pipeline for interpretability.

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
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from data.loaders.transforms import inverse_transform
from models import RepresentationModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase_linear_probe")


PHASE_TARGET_FEATURE = "soft_neighborhood_phase"
PHASE_INPUT_FEATURE = "phase_ccdc"

PHASE_TARGET_CHANNELS = [
    "annual.red",
    "annual.nir",
    "annual.nbr",
    "annual.ndmi",
    "annual.spectral_velocity",
    "annual.seas_amp_swir1",
    "annual.seas_amp_swir2",
]


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

    Targets are extracted in the **transformed + normalized** space (the
    same space the model's loss operates in).

    The mask restricts to pixels that are:
      - inside the AOI (``static_mask.aoi``)
      - classified as forest (``static_mask.forest``)
      - valid across all features and timesteps

    Returns:
        Ximg:       [B, 16, H, W]        ccdc_history encoder input
        Xphase:     [B, 8, T, H, W]      phase_ccdc temporal input
        Yimg:       [B, 7, T, H, W]      soft_neighborhood_phase target (normalized)
        M:          [B, H, W]            boolean mask (True = valid)
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

        # AOI and forest masks from static_mask group
        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]  # [C, H, W]
        aoi_idx = sm_names.index("aoi")
        forest_idx = sm_names.index("forest")
        aoi_mask = torch.from_numpy(sm_data[aoi_idx]).bool()        # [H, W]
        forest_mask = torch.from_numpy(sm_data[forest_idx]).bool()  # [H, W]

        m = (
            torch.from_numpy(enc_f.mask)
            & tgt_mask_spatial
            & phase_mask_spatial
            & aoi_mask
            & forest_mask
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

def _spearman_rho2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Spearman's rank correlation squared (ρ²) for 1-D tensors."""
    n = pred.shape[0]
    if n < 2:
        return 0.0
    pred_ranks = pred.double().argsort().argsort().double()
    tgt_ranks = target.double().argsort().argsort().double()
    p = pred_ranks - pred_ranks.mean()
    t = tgt_ranks - tgt_ranks.mean()
    num = (p * t).sum()
    den = torch.sqrt((p * p).sum() * (t * t).sum())
    if den < 1e-12:
        return 0.0
    rho = num / den
    return float(rho.item() ** 2)


@dataclass
class PhaseProbeMetrics:
    """Metrics in both normalized and original data scales."""
    # Normalized (transformed + whitened) space
    mse_per_channel: Dict[str, float]
    r2_per_channel: Dict[str, float]
    spearman_rho2_per_channel: Dict[str, float]
    mse_total: float
    r2_total: float
    spearman_rho2_total: float
    # Original (un-normalized, un-transformed) space
    mse_per_channel_original: Dict[str, float]
    r2_per_channel_original: Dict[str, float]
    spearman_rho2_per_channel_original: Dict[str, float]
    mse_total_original: float
    r2_total_original: float
    spearman_rho2_total_original: float
    n_observations: int


def _iter_batches(dataloader: DataLoader, max_batches: int):
    """Iterate with optional cap. max_batches=0 => no cap."""
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        yield batch


def _build_design_matrix(
    z_type_flat: torch.Tensor,
    z_phase_flat: torch.Tensor,
    design: str = "full",
) -> torch.Tensor:
    """Build the design matrix according to *design*.

    Args:
        z_type_flat:  [N, 64]
        z_phase_flat: [N, 12]
        design: one of
            ``"full"``       – [z_type, z_phase, z_type ⊗ z_phase]  (844 cols)
            ``"type-only"``  – [z_type]                               (64 cols)
            ``"phase-only"`` – [z_phase]                              (12 cols)

    Returns:
        X: [N, D]
    """
    if design == "type-only":
        return z_type_flat
    if design == "phase-only":
        return z_phase_flat

    # full: main effects + interaction
    interaction = (
        z_type_flat.unsqueeze(2) * z_phase_flat.unsqueeze(1)
    ).reshape(z_type_flat.shape[0], -1)
    return torch.cat([z_type_flat, z_phase_flat, interaction], dim=1)


def _design_dim(design: str) -> int:
    """Return the number of columns produced by *design*."""
    if design == "type-only":
        return D_TYPE
    if design == "phase-only":
        return D_PHASE
    return D_TOTAL


DESIGN_CHOICES = ("full", "type-only", "phase-only")


D_TYPE = 64
D_PHASE = 12
D_INTERACTION = D_TYPE * D_PHASE  # 768
D_TOTAL = D_TYPE + D_PHASE + D_INTERACTION  # 844


def fit_phase_probe(
    train_loader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    ridge_lambda: float = 1e-3,
    halo: int = 16,
    max_batches_train: int = 0,
    design: str = "full",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming ridge regression for the phase linear probe.

    Args:
        design: ``"full"`` (type + phase + interaction, 844 cols),
            ``"type-only"`` (64 cols), or ``"phase-only"`` (12 cols).

    Normal equations are accumulated on CPU in float64.

    Returns:
        W: [D, C]
        b: [C]
    """
    model.eval()
    D = _design_dim(design)
    C = len(PHASE_TARGET_CHANNELS)  # 7
    Da = D + 1

    # Accumulate on CPU — the matrices are small, CPU RAM is plentiful
    A = torch.zeros((Da, Da), dtype=torch.float64)
    B_mat = torch.zeros((Da, C), dtype=torch.float64)
    n_obs_total = 0

    with torch.no_grad():
        for batch in _iter_batches(train_loader, max_batches_train):
            Ximg, Xphase, Yimg, M = extract_phase_batch_tensors(
                batch, feature_builder, device,
            )
            Bsz, _, H, W_sp = Ximg.shape
            T = Xphase.shape[2]

            # Type encoder (dense)
            z_type = model(Ximg)  # [B, 64, H, W]

            # Phase encoder (dense, stop-grad on z_type)
            z_phase = model.forward_phase(Xphase, z_type.detach())  # [B, 12, T, H, W]

            # Apply halo mask
            halo_m = _halo_mask(H, W_sp, halo, device)  # [H, W]
            full_mask = M & halo_m.unsqueeze(0)          # [B, H, W]

            # Flatten z_type: [B, 64, H, W] → [B, H, W, 64], mask, broadcast T
            zt = z_type.permute(0, 2, 3, 1).contiguous()            # [B, H, W, 64]
            zp = z_phase.permute(0, 2, 3, 4, 1).contiguous()       # [B, T, H, W, 12]
            y_perm = Yimg.permute(0, 2, 3, 4, 1).contiguous()      # [B, T, H, W, 7]

            # Broadcast mask: [B, H, W] → [B, T, H, W]
            m_t = full_mask.unsqueeze(1).expand(-1, T, -1, -1)     # [B, T, H, W]
            m_flat = m_t.reshape(-1)

            # Broadcast z_type across T: [B, H, W, 64] → [B, T, H, W, 64]
            zt_t = zt.unsqueeze(1).expand(-1, T, -1, -1, -1)       # [B, T, H, W, 64]

            zt_flat = zt_t.reshape(-1, D_TYPE)[m_flat]              # [N, 64]
            zp_flat = zp.reshape(-1, D_PHASE)[m_flat]               # [N, 12]
            Y = y_perm.reshape(-1, C)[m_flat]                       # [N, 7]

            if zt_flat.shape[0] == 0:
                continue

            n_obs_total += zt_flat.shape[0]

            # Move to CPU before building design matrix
            zt_cpu = zt_flat.cpu()
            zp_cpu = zp_flat.cpu()
            X = _build_design_matrix(zt_cpu, zp_cpu, design)       # [N, D]
            ones = torch.ones((X.shape[0], 1), dtype=X.dtype)
            Xaug = torch.cat([X, ones], dim=1).to(torch.float64)   # [N, 845]
            Y64 = Y.to(dtype=torch.float64, device="cpu")

            A += Xaug.T @ Xaug       # [845, 845]
            B_mat += Xaug.T @ Y64    # [845, 7]

    if n_obs_total == 0:
        raise RuntimeError("No valid observations found in training data; cannot fit probe.")

    # Ridge penalty (not on bias)
    reg = torch.eye(Da, dtype=torch.float64) * ridge_lambda
    reg[-1, -1] = 0.0

    Wb = torch.linalg.solve(A + reg, B_mat)  # [845, 7]
    W = Wb[:-1, :].to(torch.float32)          # [844, 7]
    b = Wb[-1, :].to(torch.float32)           # [7]

    logger.info(
        f"Fitted phase probe on {n_obs_total:,} observations "
        f"(design={design}, D={D}, ridge_lambda={ridge_lambda:g}, halo={halo})."
    )
    return W, b


# ---------------------------------------------------------------------------
# Inverse normalization pipeline
# ---------------------------------------------------------------------------

def _build_inverse_normalization(
    feature_builder: FeatureBuilder,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Pre-compute the objects needed to invert the Mahalanobis pipeline.

    Returns:
        inv_whitening: [C, C]  inverse of the whitening matrix
        means:         [C]     per-channel means used for centering
        transform_specs: list of per-channel transform specs (str/dict/None)
    """
    feature_config = feature_builder.config.get_feature(PHASE_TARGET_FEATURE)
    C = len(PHASE_TARGET_CHANNELS)

    # Whitening matrix and its inverse
    whitening = feature_builder._get_whitening_matrix(PHASE_TARGET_FEATURE)
    if whitening is not None:
        inv_whitening = np.linalg.inv(whitening).astype(np.float32)
    else:
        inv_whitening = np.eye(C, dtype=np.float32)

    # Per-channel means
    means = np.array(
        feature_builder._get_channel_means(PHASE_TARGET_FEATURE, feature_config),
        dtype=np.float32,
    )

    # Per-channel transform specs
    channel_names = list(feature_config.channels.keys())
    transform_specs = [
        feature_config.channels[ch].transform for ch in channel_names
    ]

    return inv_whitening, means, transform_specs


def _invert_to_original_scale(
    predictions: torch.Tensor,
    inv_whitening: np.ndarray,
    means: np.ndarray,
    transform_specs: list,
) -> torch.Tensor:
    """Map predictions from normalized space back to original data scale.

    Steps (reverse of the Mahalanobis pipeline):
      1. Un-whiten:    x_centered = inv_whitening @ x_whitened
      2. Un-center:    x_transformed = x_centered + mean
      3. Un-transform: x_original = inverse_transform(x_transformed)

    Args:
        predictions: [N, C] tensor in normalized space (float64 or float32)
        inv_whitening: [C, C] inverse whitening matrix
        means: [C] channel means
        transform_specs: per-channel transform specs

    Returns:
        [N, C] tensor in original data scale (same dtype as input)
    """
    dtype = predictions.dtype
    P = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    P = P.astype(np.float64)

    # 1. Un-whiten: [N, C] @ [C, C]^T  (inv_whitening is lower-triangular)
    P = P @ inv_whitening.astype(np.float64).T

    # 2. Un-center
    P = P + means.astype(np.float64)

    # 3. Per-channel inverse transform
    for c_idx, spec in enumerate(transform_specs):
        if spec is not None:
            P[:, c_idx] = inverse_transform(P[:, c_idx], spec)

    return torch.from_numpy(P).to(dtype)


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
    design: str = "full",
) -> PhaseProbeMetrics:
    """Evaluate MSE and R² in both normalized and original data scales.

    Predictions are made in the normalized (whitened) space, then inverted
    back to the original spectral units for a second set of metrics.
    """
    model.eval()
    C = W.shape[1]   # 7

    # W and b live on CPU (returned from fit), keep accumulators on CPU too
    W_cpu = W.cpu()
    b_cpu = b.cpu()

    # Pre-compute inverse-normalization objects
    inv_whitening, means, transform_specs = _build_inverse_normalization(
        feature_builder,
    )

    # Accumulators — normalized space
    sse = torch.zeros(C, dtype=torch.float64)
    sum_y = torch.zeros(C, dtype=torch.float64)
    sum_y2 = torch.zeros(C, dtype=torch.float64)

    # Accumulators — original space
    sse_orig = torch.zeros(C, dtype=torch.float64)
    sum_y_orig = torch.zeros(C, dtype=torch.float64)
    sum_y2_orig = torch.zeros(C, dtype=torch.float64)

    # Collect per-channel predictions/targets for Spearman ρ²
    all_preds_norm = [[] for _ in range(C)]
    all_targets_norm = [[] for _ in range(C)]
    all_preds_orig = [[] for _ in range(C)]
    all_targets_orig = [[] for _ in range(C)]

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

            halo_m = _halo_mask(H, W_sp, halo, device)
            full_mask = M & halo_m.unsqueeze(0)

            zt = z_type.permute(0, 2, 3, 1).contiguous()
            zp = z_phase.permute(0, 2, 3, 4, 1).contiguous()
            y_perm = Yimg.permute(0, 2, 3, 4, 1).contiguous()

            m_t = full_mask.unsqueeze(1).expand(-1, T, -1, -1)
            m_flat = m_t.reshape(-1)

            zt_t = zt.unsqueeze(1).expand(-1, T, -1, -1, -1)
            zt_flat = zt_t.reshape(-1, D_TYPE)[m_flat]
            zp_flat = zp.reshape(-1, D_PHASE)[m_flat]
            Y_norm = y_perm.reshape(-1, C)[m_flat]

            if zt_flat.shape[0] == 0:
                continue

            n_obs += zt_flat.shape[0]

            # Build design matrix on GPU, predict on CPU
            X = _build_design_matrix(zt_flat, zp_flat, design).cpu().float()  # [N, D]
            Y_norm = Y_norm.cpu().to(torch.float64)

            P_norm = (X @ W_cpu + b_cpu).to(torch.float64)  # [N, 7]

            # --- Normalized-space accumulators ---
            err = P_norm - Y_norm
            sse += (err * err).sum(dim=0)
            sum_y += Y_norm.sum(dim=0)
            sum_y2 += (Y_norm * Y_norm).sum(dim=0)

            for c_idx in range(C):
                all_preds_norm[c_idx].append(P_norm[:, c_idx].float())
                all_targets_norm[c_idx].append(Y_norm[:, c_idx].float())

            # --- Original-space accumulators ---
            P_orig = _invert_to_original_scale(
                P_norm, inv_whitening, means, transform_specs,
            )
            Y_orig = _invert_to_original_scale(
                Y_norm, inv_whitening, means, transform_specs,
            )
            err_orig = P_orig - Y_orig
            sse_orig += (err_orig * err_orig).sum(dim=0)
            sum_y_orig += Y_orig.sum(dim=0)
            sum_y2_orig += (Y_orig * Y_orig).sum(dim=0)

            for c_idx in range(C):
                all_preds_orig[c_idx].append(P_orig[:, c_idx].float())
                all_targets_orig[c_idx].append(Y_orig[:, c_idx].float())

    def _compute_mse_r2(sse_acc, sum_y_acc, sum_y2_acc, n):
        mse_per: Dict[str, float] = {}
        r2_per: Dict[str, float] = {}
        for c_idx, ch in enumerate(PHASE_TARGET_CHANNELS):
            if n == 0:
                mse_per[ch] = 0.0
                r2_per[ch] = 0.0
                continue
            mse_per[ch] = float(sse_acc[c_idx].item()) / n
            sst_val = max(
                0.0,
                float(sum_y2_acc[c_idx].item())
                - float(sum_y_acc[c_idx].item()) ** 2 / n,
            )
            if sst_val > 1e-8:
                r2_per[ch] = 1.0 - float(sse_acc[c_idx].item()) / sst_val
            else:
                r2_per[ch] = 0.0
        return mse_per, r2_per

    def _compute_spearman(all_p, all_t):
        spearman_per: Dict[str, float] = {}
        for c_idx, ch in enumerate(PHASE_TARGET_CHANNELS):
            if all_p[c_idx]:
                p_cat = torch.cat(all_p[c_idx])
                t_cat = torch.cat(all_t[c_idx])
                spearman_per[ch] = _spearman_rho2(p_cat, t_cat)
            else:
                spearman_per[ch] = 0.0
        return spearman_per

    mse_per, r2_per = _compute_mse_r2(sse, sum_y, sum_y2, n_obs)
    mse_per_orig, r2_per_orig = _compute_mse_r2(
        sse_orig, sum_y_orig, sum_y2_orig, n_obs,
    )
    spearman_per = _compute_spearman(all_preds_norm, all_targets_norm)
    spearman_per_orig = _compute_spearman(all_preds_orig, all_targets_orig)

    return PhaseProbeMetrics(
        mse_per_channel=mse_per,
        r2_per_channel=r2_per,
        spearman_rho2_per_channel=spearman_per,
        mse_total=float(np.mean(list(mse_per.values()))),
        r2_total=float(np.mean(list(r2_per.values()))),
        spearman_rho2_total=float(np.mean(list(spearman_per.values()))),
        mse_per_channel_original=mse_per_orig,
        r2_per_channel_original=r2_per_orig,
        spearman_rho2_per_channel_original=spearman_per_orig,
        mse_total_original=float(np.mean(list(mse_per_orig.values()))),
        r2_total_original=float(np.mean(list(r2_per_orig.values()))),
        spearman_rho2_total_original=float(np.mean(list(spearman_per_orig.values()))),
        n_observations=n_obs,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metrics(metrics: PhaseProbeMetrics, prefix: str):
    logger.info(f"{prefix} results (observations: {metrics.n_observations:,}):")

    # Normalized space
    logger.info(f"  [Normalized space]")
    logger.info(f"  {'Channel':<30} {'MSE':>12} {'R²':>12} {'ρ²':>12}")
    logger.info(f"  {'-' * 70}")
    for ch in PHASE_TARGET_CHANNELS:
        short = ch.replace("annual.", "")
        logger.info(
            f"  {short:<30} "
            f"{metrics.mse_per_channel[ch]:>12.6f} "
            f"{metrics.r2_per_channel[ch]:>12.6f} "
            f"{metrics.spearman_rho2_per_channel[ch]:>12.6f}"
        )
    logger.info(f"  {'-' * 70}")
    logger.info(
        f"  {'Average':<30} {metrics.mse_total:>12.6f} "
        f"{metrics.r2_total:>12.6f} "
        f"{metrics.spearman_rho2_total:>12.6f}"
    )

    # Original space
    logger.info(f"  [Original scale]")
    logger.info(f"  {'Channel':<30} {'MSE':>12} {'R²':>12} {'ρ²':>12}")
    logger.info(f"  {'-' * 70}")
    for ch in PHASE_TARGET_CHANNELS:
        short = ch.replace("annual.", "")
        logger.info(
            f"  {short:<30} "
            f"{metrics.mse_per_channel_original[ch]:>12.6f} "
            f"{metrics.r2_per_channel_original[ch]:>12.6f} "
            f"{metrics.spearman_rho2_per_channel_original[ch]:>12.6f}"
        )
    logger.info(f"  {'-' * 70}")
    logger.info(
        f"  {'Average':<30} {metrics.mse_total_original:>12.6f} "
        f"{metrics.r2_total_original:>12.6f} "
        f"{metrics.spearman_rho2_total_original:>12.6f}"
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
    parser.add_argument(
        "--design", type=str, default="full", choices=DESIGN_CHOICES,
        help="Design matrix: full (type+phase+interaction), type-only, phase-only",
    )
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

    logger.info(f"Design matrix: {args.design} (D={_design_dim(args.design)})")

    # Fit
    W, b_bias = fit_phase_probe(
        train_loader,
        feature_builder,
        model,
        device,
        ridge_lambda=args.ridge_lambda,
        halo=args.halo,
        max_batches_train=args.max_batches_train,
        design=args.design,
    )

    # Evaluate
    train_metrics = evaluate_phase_probe(
        train_loader, feature_builder, model, W, b_bias, device,
        halo=args.halo,
        max_batches_eval=args.max_batches_eval,
        design=args.design,
    )
    val_metrics = evaluate_phase_probe(
        val_loader, feature_builder, model, W, b_bias, device,
        halo=args.halo,
        max_batches_eval=args.max_batches_eval,
        design=args.design,
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
            "design": args.design,
            "input_dim": _design_dim(args.design),
            "encoder_checkpoint": args.checkpoint,
            # Normalized space
            "train_mse_total": train_metrics.mse_total,
            "train_r2_total": train_metrics.r2_total,
            "train_spearman_rho2_total": train_metrics.spearman_rho2_total,
            "val_mse_total": val_metrics.mse_total,
            "val_r2_total": val_metrics.r2_total,
            "val_spearman_rho2_total": val_metrics.spearman_rho2_total,
            "val_mse_per_channel": val_metrics.mse_per_channel,
            "val_r2_per_channel": val_metrics.r2_per_channel,
            "val_spearman_rho2_per_channel": val_metrics.spearman_rho2_per_channel,
            # Original scale
            "train_mse_total_original": train_metrics.mse_total_original,
            "train_r2_total_original": train_metrics.r2_total_original,
            "train_spearman_rho2_total_original": train_metrics.spearman_rho2_total_original,
            "val_mse_total_original": val_metrics.mse_total_original,
            "val_r2_total_original": val_metrics.r2_total_original,
            "val_spearman_rho2_total_original": val_metrics.spearman_rho2_total_original,
            "val_mse_per_channel_original": val_metrics.mse_per_channel_original,
            "val_r2_per_channel_original": val_metrics.r2_per_channel_original,
            "val_spearman_rho2_per_channel_original": val_metrics.spearman_rho2_per_channel_original,
        },
        out_path,
    )
    logger.info(f"Saved phase probe to {out_path}")


if __name__ == "__main__":
    main()
