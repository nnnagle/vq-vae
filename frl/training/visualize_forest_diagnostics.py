#!/usr/bin/env python3
"""
Diagnostic forest map: spatial time-series for patches with the most
recently-disturbed forest (ysfc_min < 10).

For each of the 7 soft_neighborhood_phase target channels, produces a figure
with spatial maps at each timestep, showing both observed and predicted values
from the phase linear probe, masked to AOI & forest pixels.

Patches are selected by ranking all test patches by the number of forest
pixels with ysfc_min < 10 (recently-disturbed forest) and taking the top N.

Usage:
    python training/visualize_forest_diagnostics.py \\
        --checkpoint runs/.../checkpoints/encoder_epoch_050.pt \\
        --probe      runs/.../checkpoints/phase_linear_probe.pt \\
        --bindings   config/frl_binding_v1.yaml \\
        --training   config/frl_training_v1.yaml \\
        --num-patches 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel

from training.fit_phase_linear_probe import (
    PHASE_TARGET_FEATURE,
    PHASE_INPUT_FEATURE,
    _get_target_channels,
    ProbePreprocessor,
    _build_design_matrix,
    _build_inverse_normalization,
    _invert_to_original_scale,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("forest_diagnostics")


# ---------------------------------------------------------------------------
# Patch access helper
# ---------------------------------------------------------------------------

def _load_patch(dataset: ForestDatasetV2, patch_idx: int) -> dict:
    """Load a single patch by its raw index in dataset.patches."""
    saved = dataset._current_indices
    dataset._current_indices = list(range(len(dataset.patches)))
    sample = dataset[patch_idx]
    dataset._current_indices = saved
    return sample


# ---------------------------------------------------------------------------
# Patch ranking by ysfc
# ---------------------------------------------------------------------------

def rank_patches_by_ysfc(
    dataset: ForestDatasetV2,
) -> List[Tuple[int, int]]:
    """Count forest pixels with ysfc_min < 10 per patch.

    Returns:
        List of (patch_idx, count) sorted descending by count.
    """
    counts: List[Tuple[int, int]] = []
    n_patches = len(dataset.patches)
    logger.info(
        f"Scanning {n_patches} patches for ysfc_min < 10 forest pixels..."
    )

    for patch_idx in range(n_patches):
        sample = _load_patch(dataset, patch_idx)

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]
        aoi_mask = sm_data[sm_names.index("aoi")] > 0
        forest_mask = sm_data[sm_names.index("forest")] > 0

        st_names = sample["metadata"]["channel_names"]["static"]
        st_data = sample["static"]
        ysfc_min = st_data[st_names.index("ysfc_min")]

        valid = aoi_mask & forest_mask & np.isfinite(ysfc_min)
        count = int((valid & (ysfc_min < 10)).sum())
        counts.append((patch_idx, count))

        if (patch_idx + 1) % 20 == 0 or patch_idx == n_patches - 1:
            logger.info(f"  Scanned {patch_idx + 1}/{n_patches} patches")

    counts.sort(key=lambda x: x[1], reverse=True)
    return counts


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_phase_diagnostics(
    dataset: ForestDatasetV2,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    W: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    patch_indices: List[int],
    design: str = "full",
    preprocessor: "ProbePreprocessor | None" = None,
    target_channels: "List[str] | None" = None,
) -> List[Dict[str, np.ndarray]]:
    """Run encoder + phase probe on selected patches.

    Returns list of dicts with observed and predicted [T, H, W] arrays
    in original (back-transformed) scale, plus masks and ysfc_min.

    If *preprocessor* is provided, the design matrix is standardised (and
    optionally PCA-compressed) before prediction.
    """
    model.eval()

    inv_whitening, means, transform_specs = _build_inverse_normalization(
        feature_builder,
    )

    if target_channels is None:
        target_channels = _get_target_channels(feature_builder)
    C = len(target_channels)
    W_cpu = W.cpu().float()
    b_cpu = b.cpu().float()

    results: List[Dict[str, np.ndarray]] = []

    with torch.no_grad():
        for patch_idx in patch_indices:
            sample = _load_patch(dataset, patch_idx)

            # Build features
            enc_f = feature_builder.build_feature("ccdc_history", sample)
            phase_f = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
            tgt_f = feature_builder.build_feature(PHASE_TARGET_FEATURE, sample)

            enc_tensor = (
                torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)
            )  # [1, 16, H, W]
            phase_tensor = (
                torch.from_numpy(phase_f.data).float().unsqueeze(0).to(device)
            )  # [1, 8, T, H, W]

            T = phase_tensor.shape[2]
            H = enc_tensor.shape[2]
            W_sp = enc_tensor.shape[3]

            # Type encoder
            z_type = model(enc_tensor)  # [1, 64, H, W]

            # Phase encoder (stop-grad on z_type)
            z_phase = model.forward_phase(
                phase_tensor, z_type.detach(),
            )  # [1, 12, T, H, W]

            # Move to CPU for prediction
            zt = z_type.squeeze(0).permute(1, 2, 0).cpu()       # [H, W, 64]
            zp = z_phase.squeeze(0).permute(1, 2, 3, 0).cpu()   # [T, H, W, 12]

            # Predict per-timestep to manage memory
            pred_norm_list = []
            for t in range(T):
                zt_flat = zt.reshape(-1, model.z_type_dim)    # [H*W, d_type]
                zp_flat = zp[t].reshape(-1, model.z_phase_dim)  # [H*W, d_phase]
                if preprocessor is not None:
                    X = preprocessor.transform(zt_flat, zp_flat)
                else:
                    X = _build_design_matrix(zt_flat, zp_flat, design)
                pred_t = (X @ W_cpu + b_cpu)            # [H*W, 7]
                pred_norm_list.append(pred_t.reshape(H, W_sp, C))

            pred_norm = torch.stack(pred_norm_list)  # [T, H, W, 7]

            # Target (already normalized by feature builder)
            tgt_norm = torch.from_numpy(tgt_f.data).float()  # [7, T, H, W]
            tgt_norm_thwc = tgt_norm.permute(1, 2, 3, 0)     # [T, H, W, 7]

            # Invert to original scale
            pred_orig = _invert_to_original_scale(
                pred_norm.reshape(-1, C), inv_whitening, means, transform_specs,
            ).reshape(T, H, W_sp, C).permute(3, 0, 1, 2).numpy()  # [7, T, H, W]

            tgt_orig = _invert_to_original_scale(
                tgt_norm_thwc.reshape(-1, C),
                inv_whitening, means, transform_specs,
            ).reshape(T, H, W_sp, C).permute(3, 0, 1, 2).numpy()  # [7, T, H, W]

            # Masks
            sm_names = sample["metadata"]["channel_names"]["static_mask"]
            sm_data = sample["static_mask"]
            aoi_mask = sm_data[sm_names.index("aoi")] > 0
            forest_mask = sm_data[sm_names.index("forest")] > 0

            tgt_mask_spatial = (
                tgt_f.mask.all(axis=0) if tgt_f.mask.ndim == 3
                else tgt_f.mask
            )
            phase_mask_spatial = (
                phase_f.mask.all(axis=0) if phase_f.mask.ndim == 3
                else phase_f.mask
            )

            combined_mask = (
                aoi_mask & forest_mask & enc_f.mask
                & tgt_mask_spatial & phase_mask_spatial
            )

            # ysfc_min for annotation
            st_names = sample["metadata"]["channel_names"]["static"]
            st_data = sample["static"]
            ysfc_min = st_data[st_names.index("ysfc_min")]

            record: Dict[str, np.ndarray] = {
                "aoi_mask": aoi_mask,
                "forest_mask": forest_mask,
                "combined_mask": combined_mask,
                "ysfc_min": ysfc_min,
                "patch_idx": patch_idx,
                "T": T,
            }
            for c_idx, ch in enumerate(target_channels):
                record[f"target_{ch}"] = tgt_orig[c_idx]   # [T, H, W]
                record[f"pred_{ch}"] = pred_orig[c_idx]     # [T, H, W]

            results.append(record)

            ysfc_count = int(
                (
                    (aoi_mask & forest_mask & np.isfinite(ysfc_min))
                    & (ysfc_min < 10)
                ).sum()
            )

            # Diagnostic: quantify temporal variation in z_phase and predictions
            zp_np = zp.numpy()  # [T, H, W, 12]
            mask_np = combined_mask
            zp_masked = zp_np[:, mask_np, :]  # [T, N_valid, 12]
            zp_temporal_std = zp_masked.std(axis=0).mean()
            zp_spatial_std = zp_masked.std(axis=1).mean()
            logger.info(
                f"  Patch {patch_idx}: T={T}, H={H}, W={W_sp}, "
                f"mask={combined_mask.sum()}, ysfc<10={ysfc_count}"
            )
            logger.info(
                f"    z_phase temporal std (mean over pixels): {zp_temporal_std:.6f}, "
                f"spatial std (mean over timesteps): {zp_spatial_std:.6f}"
            )
            for c_idx, ch in enumerate(target_channels):
                pred_ch = pred_orig[c_idx][:, mask_np]  # [T, N_valid]
                tgt_ch = tgt_orig[c_idx][:, mask_np]
                pred_t_std = pred_ch.std(axis=0).mean()
                pred_s_std = pred_ch.std(axis=1).mean()
                tgt_t_std = tgt_ch.std(axis=0).mean()
                tgt_s_std = tgt_ch.std(axis=1).mean()
                logger.info(
                    f"    {ch}: pred temporal_std={pred_t_std:.6f} "
                    f"spatial_std={pred_s_std:.6f} | "
                    f"obs temporal_std={tgt_t_std:.6f} "
                    f"spatial_std={tgt_s_std:.6f}"
                )

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_masked(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    """Return a masked array where *mask==False* is masked out."""
    return np.ma.array(arr, mask=~mask)


def plot_variable_timeseries(
    records: List[Dict[str, np.ndarray]],
    channel_ref: str,
    output_path: Path,
    years: List[int],
) -> None:
    """Spatial maps across timesteps for one variable.

    Layout: rows = patch x (obs, pred), columns = timestep (year).
    """
    n_patches = len(records)
    T = records[0]["T"]

    n_rows = n_patches * 2
    n_cols = T

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.8 * n_cols, 1.8 * n_rows),
        squeeze=False,
    )

    # Global colour limits from all observed values (masked)
    all_vals: List[np.ndarray] = []
    for rec in records:
        arr = rec[f"target_{channel_ref}"]  # [T, H, W]
        mask = rec["combined_mask"]         # [H, W]
        for t in range(T):
            vals = arr[t][mask]
            if vals.size > 0:
                all_vals.append(vals)

    if all_vals:
        combined_vals = np.concatenate(all_vals)
        vmin = float(np.nanpercentile(combined_vals, 2))
        vmax = float(np.nanpercentile(combined_vals, 98))
    else:
        vmin, vmax = 0.0, 1.0

    short_name = channel_ref.replace("annual.", "")

    im = None
    for p_idx, rec in enumerate(records):
        mask = rec["combined_mask"]
        obs = rec[f"target_{channel_ref}"]   # [T, H, W]
        pred = rec[f"pred_{channel_ref}"]    # [T, H, W]

        obs_row = p_idx * 2
        pred_row = obs_row + 1

        for t in range(T):
            ax_obs = axes[obs_row, t]
            ax_pred = axes[pred_row, t]

            im = ax_obs.imshow(
                _make_masked(obs[t], mask),
                vmin=vmin, vmax=vmax, cmap="viridis",
                interpolation="nearest",
            )
            ax_pred.imshow(
                _make_masked(pred[t], mask),
                vmin=vmin, vmax=vmax, cmap="viridis",
                interpolation="nearest",
            )

            ax_obs.set_xticks([])
            ax_obs.set_yticks([])
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            # Year label on first patch row only
            if p_idx == 0:
                yr = years[t] if t < len(years) else t
                ax_obs.set_title(str(yr), fontsize=7)

        # Row labels on left column
        ysfc_count = int(
            (
                (rec["aoi_mask"] & rec["forest_mask"]
                 & np.isfinite(rec["ysfc_min"]))
                & (rec["ysfc_min"] < 10)
            ).sum()
        )
        axes[obs_row, 0].set_ylabel(
            f"P{rec['patch_idx']} Obs\n(n={ysfc_count})", fontsize=6,
        )
        axes[pred_row, 0].set_ylabel("Pred", fontsize=6)

    fig.suptitle(
        f"{short_name}  (observed vs predicted, original scale)", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])

    if im is not None:
        cbar_ax = fig.add_axes([0.95, 0.08, 0.012, 0.84])
        fig.colorbar(im, cax=cbar_ax)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def plot_variable_anomaly(
    records: List[Dict[str, np.ndarray]],
    channel_ref: str,
    output_path: Path,
    years: List[int],
) -> None:
    """Temporal-anomaly maps: deviation from each pixel's temporal mean.

    Uses per-pixel demeaning so the color scale highlights temporal change
    rather than being dominated by spatial variation.
    Layout: rows = patch x (obs anomaly, pred anomaly), columns = year.
    """
    n_patches = len(records)
    T = records[0]["T"]

    n_rows = n_patches * 2
    n_cols = T

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.8 * n_cols, 1.8 * n_rows),
        squeeze=False,
    )

    # Compute anomalies and global colour limits
    obs_anomalies = []
    pred_anomalies = []
    all_anom_vals: List[np.ndarray] = []

    for rec in records:
        mask = rec["combined_mask"]
        obs = rec[f"target_{channel_ref}"]   # [T, H, W]
        pred = rec[f"pred_{channel_ref}"]    # [T, H, W]

        # Temporal mean per pixel (masked)
        obs_stack = np.where(mask[None, :, :], obs, np.nan)
        pred_stack = np.where(mask[None, :, :], pred, np.nan)

        obs_mean = np.nanmean(obs_stack, axis=0, keepdims=True)   # [1, H, W]
        pred_mean = np.nanmean(pred_stack, axis=0, keepdims=True)

        obs_anom = obs_stack - obs_mean    # [T, H, W]
        pred_anom = pred_stack - pred_mean

        obs_anomalies.append(obs_anom)
        pred_anomalies.append(pred_anom)

        for t in range(T):
            vals = obs_anom[t][mask]
            if vals.size > 0:
                all_anom_vals.append(vals)
            vals = pred_anom[t][mask]
            if vals.size > 0:
                all_anom_vals.append(vals)

    if all_anom_vals:
        combined = np.concatenate(all_anom_vals)
        vlim = float(np.nanpercentile(np.abs(combined), 98))
    else:
        vlim = 1.0
    vmin, vmax = -vlim, vlim

    short_name = channel_ref.replace("annual.", "")

    im = None
    for p_idx, rec in enumerate(records):
        mask = rec["combined_mask"]
        obs_anom = obs_anomalies[p_idx]
        pred_anom = pred_anomalies[p_idx]

        obs_row = p_idx * 2
        pred_row = obs_row + 1

        for t in range(T):
            ax_obs = axes[obs_row, t]
            ax_pred = axes[pred_row, t]

            im = ax_obs.imshow(
                _make_masked(obs_anom[t], mask),
                vmin=vmin, vmax=vmax, cmap="RdBu_r",
                interpolation="nearest",
            )
            ax_pred.imshow(
                _make_masked(pred_anom[t], mask),
                vmin=vmin, vmax=vmax, cmap="RdBu_r",
                interpolation="nearest",
            )

            ax_obs.set_xticks([])
            ax_obs.set_yticks([])
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])

            if p_idx == 0:
                yr = years[t] if t < len(years) else t
                ax_obs.set_title(str(yr), fontsize=7)

        axes[obs_row, 0].set_ylabel(
            f"P{rec['patch_idx']} Obs\nanom", fontsize=6,
        )
        axes[pred_row, 0].set_ylabel("Pred\nanom", fontsize=6)

    fig.suptitle(
        f"{short_name}  (temporal anomaly: deviation from pixel mean)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])

    if im is not None:
        cbar_ax = fig.add_axes([0.95, 0.08, 0.012, 0.84])
        fig.colorbar(im, cax=cbar_ax)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def plot_ysfc_map(
    records: List[Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """ysfc_min spatial maps for the selected patches."""
    n_patches = len(records)

    fig, axes = plt.subplots(
        1, n_patches, figsize=(4 * n_patches, 3.5), squeeze=False,
    )

    im = None
    for i, rec in enumerate(records):
        mask = rec["combined_mask"]
        ysfc = _make_masked(rec["ysfc_min"], mask)

        ax = axes[0, i]
        im = ax.imshow(
            ysfc, cmap="RdYlGn", vmin=0, vmax=40, interpolation="nearest",
        )
        ysfc_count = int(
            (
                (rec["aoi_mask"] & rec["forest_mask"]
                 & np.isfinite(rec["ysfc_min"]))
                & (rec["ysfc_min"] < 10)
            ).sum()
        )
        ax.set_title(
            f"Patch {rec['patch_idx']} (ysfc<10: {ysfc_count})", fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("ysfc_min (years since fast change)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.84])
        fig.colorbar(im, cax=cbar_ax, label="years")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic forest map: spatial time-series for "
        "patches with the most recently-disturbed forest.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to encoder checkpoint (.pt)",
    )
    parser.add_argument(
        "--probe", type=str, default=None,
        help="Path to saved phase linear probe (.pt). "
        "Defaults to phase_linear_probe.pt next to checkpoint.",
    )
    parser.add_argument(
        "--bindings", type=str, default="config/frl_binding_v1.yaml",
        help="Bindings YAML",
    )
    parser.add_argument(
        "--training", type=str, default="config/frl_training_v1.yaml",
        help="Training YAML",
    )
    parser.add_argument(
        "--num-patches", type=int, default=4,
        help="Number of top patches to visualise (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: <experiment>/diagnostics)",
    )
    args = parser.parse_args()

    # ---- configs ----------------------------------------------------------
    logger.info(f"Loading bindings config from {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    logger.info(f"Loading training config from {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    patch_size = training_config.sampling.patch_size
    years = bindings_config.time_window.years

    # ---- output dir -------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_path = Path(args.checkpoint)
        experiment_dir = ckpt_path.parent.parent
        output_dir = experiment_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ---- model + probe ----------------------------------------------------
    logger.info(f"Loading encoder from {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(
        args.checkpoint, device=device, freeze=True,
    )

    probe_path = args.probe or str(
        Path(args.checkpoint).parent / "phase_linear_probe.pt"
    )
    logger.info(f"Loading phase linear probe from {probe_path}")
    probe_ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
    W = probe_ckpt["W"]   # [D, 7]
    b = probe_ckpt["b"]   # [7]
    design = probe_ckpt.get("design", "full")

    # Load preprocessor (standardisation + PCA) if present
    if "preprocessor_mean" in probe_ckpt:
        preprocessor = ProbePreprocessor.from_dict(probe_ckpt)
        logger.info(
            f"Probe preprocessor: design={preprocessor.design}, "
            f"pca_k={preprocessor.interaction_pca_k}, "
            f"output_dim={preprocessor.output_dim}"
        )
    else:
        preprocessor = None
        logger.info("No preprocessor in probe checkpoint; using raw features.")

    # ---- feature builder (needed early for target_channels) ---------------
    feature_builder = FeatureBuilder(bindings_config)

    target_channels: List[str] = probe_ckpt.get(
        "target_channels", _get_target_channels(feature_builder)
    )

    logger.info(
        f"Probe: design={design}, W shape={list(W.shape)}, "
        f"target channels={target_channels}"
    )

    # ---- dataset ----------------------------------------------------------
    logger.info("Creating test dataset...")
    test_dataset = ForestDatasetV2(
        bindings_config,
        split="test",
        patch_size=patch_size,
        min_aoi_fraction=0.3,
    )
    n_test = len(test_dataset.patches)
    logger.info(f"Test dataset has {n_test} patches")

    # ---- rank patches by ysfc_min < 10 -----------------------------------
    ranked = rank_patches_by_ysfc(test_dataset)
    top_n = min(args.num_patches, len(ranked))
    selected = ranked[:top_n]
    selected_indices = [idx for idx, _ in selected]

    logger.info(f"Top {top_n} patches by ysfc_min < 10 forest pixel count:")
    for idx, count in selected:
        logger.info(f"  Patch {idx}: {count} pixels")

    # ---- collect data -----------------------------------------------------
    logger.info("Running inference on selected patches...")
    records = collect_phase_diagnostics(
        test_dataset, feature_builder, model, W, b,
        device, selected_indices, design=design,
        preprocessor=preprocessor, target_channels=target_channels,
    )
    logger.info(f"Collected data for {len(records)} patches")

    # ---- plot -------------------------------------------------------------
    logger.info("Generating diagnostic figures...")

    # ysfc_min overview
    plot_ysfc_map(records, output_dir / "forest_diag_ysfc_min.png")

    # One sheet per target variable: absolute values and temporal anomalies
    for ch in target_channels:
        safe_name = ch.replace(".", "_")
        out_path = output_dir / f"forest_diag_{safe_name}.png"
        plot_variable_timeseries(records, ch, out_path, years=years)

        anom_path = output_dir / f"forest_diag_{safe_name}_anomaly.png"
        plot_variable_anomaly(records, ch, anom_path, years=years)

    logger.info("Done.")


if __name__ == "__main__":
    main()
