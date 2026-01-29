#!/usr/bin/env python3
"""
Visualize linear-probe predictions and gate values on test patches.

Produces diagnostic figures for a trained representation model:
- One sheet per target variable: observed (back-transformed) vs predicted,
  tiled across a random sample of test patches, masked to AOI & forest.
- One sheet for the spatial gate (mean across channels), tiled likewise.

Outputs are written to ``<experiment_dir>/diagnostics/``.

Usage:
    python training/visualize_test_patches.py \
        --checkpoint runs/frl_v0_exp001/checkpoints/encoder_epoch_050.pt \
        --probe      runs/frl_v0_exp001/checkpoints/linear_probe_closed_form.pt \
        --bindings   config/frl_binding_v1.yaml \
        --training   config/frl_training_v1.yaml \
        --num-patches 16 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
logger = logging.getLogger("visualize_test_patches")


# ---------------------------------------------------------------------------
# Inverse normalisation helpers
# ---------------------------------------------------------------------------

def inverse_robust_iqr(
    normalized: np.ndarray,
    q25: float,
    q50: float,
    q75: float,
) -> np.ndarray:
    """Invert robust_iqr normalisation: original = normalised * IQR + median."""
    iqr = q75 - q25
    if iqr < 1e-8:
        iqr = 1.0
    return normalized * iqr + q50


def inverse_zscore(
    normalized: np.ndarray,
    mean: float,
    sd: float,
) -> np.ndarray:
    """Invert zscore normalisation: original = normalised * sd + mean."""
    if sd < 1e-8:
        sd = 1.0
    return normalized * sd + mean


def back_transform_channel(
    normalized: np.ndarray,
    norm_type: str,
    stats: Dict[str, float],
) -> np.ndarray:
    """Back-transform a normalised channel to its original scale.

    Supports ``robust_iqr`` and ``zscore``.  Other types pass through unchanged.
    """
    if norm_type == "robust_iqr":
        return inverse_robust_iqr(
            normalized,
            q25=stats.get("q25", 0.0),
            q50=stats.get("q50", 0.0),
            q75=stats.get("q75", 1.0),
        )
    elif norm_type == "zscore":
        return inverse_zscore(
            normalized,
            mean=stats.get("mean", 0.0),
            sd=stats.get("sd", 1.0),
        )
    # identity / clamp / unknown → leave as-is
    return normalized


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_patch_data(
    dataset: ForestDatasetV2,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    W: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    target_metrics: List[str],
    patch_indices: List[int],
) -> List[Dict[str, np.ndarray]]:
    """Run the encoder + linear probe on selected test patches.

    Returns a list of dicts, one per patch, each containing:
        target_<metric>  : [H, W] back-transformed observed values
        pred_<metric>    : [H, W] back-transformed predicted values
        gate_mean        : [H, W] gate averaged over channels
        aoi_mask         : [H, W] bool
        forest_mask      : [H, W] bool
        combined_mask    : [H, W] bool  (aoi & forest & valid-data)
    """
    model.eval()

    # Normalization info for back-transform
    target_feature_config = feature_builder.config.get_feature("target_metrics")
    channel_refs = list(target_feature_config.channels.keys())
    norm_types: Dict[str, str] = {}
    for cref in channel_refs:
        ch_cfg = target_feature_config.channels[cref]
        norm_types[cref] = ch_cfg.norm or "identity"

    D = W.shape[0]
    T = W.shape[1]

    results = []

    with torch.no_grad():
        for raw_idx in patch_indices:
            # Directly index the underlying patch list (bypass epoch shuffling)
            saved = dataset._current_indices
            dataset._current_indices = list(range(len(dataset.patches)))
            sample = dataset[raw_idx]
            dataset._current_indices = saved

            # Build features
            enc_f = feature_builder.build_feature("ccdc_history", sample)
            tgt_f = feature_builder.build_feature("target_metrics", sample)

            enc_tensor = torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)

            # Forward pass
            z, gate = model(enc_tensor, return_gate=True)
            z = z.squeeze(0)          # [D, H, W]
            gate = gate.squeeze(0)    # [D, H, W]

            # Linear probe prediction  (normalised space)
            # z: [D,H,W]  W: [D,T]  -> pred: [T,H,W]
            z_np = z.cpu().numpy()                      # [D,H,W]
            pred_norm = np.einsum("dhw,dt->thw", z_np, W.cpu().numpy()) + b.cpu().numpy()[:, None, None]

            tgt_norm = tgt_f.data  # [C,H,W] (already normalised by feature_builder)
            tgt_mask = tgt_f.mask  # [H,W]

            # Masks from raw sample
            meta_ch = sample["metadata"]["channel_names"]
            sm_data = sample["static_mask"]  # [C,H,W]
            sm_names = meta_ch["static_mask"]

            aoi_idx = sm_names.index("aoi")
            forest_idx = sm_names.index("forest")
            aoi_mask = sm_data[aoi_idx] > 0       # [H,W]
            forest_mask = sm_data[forest_idx] > 0  # [H,W]

            combined_mask = aoi_mask & forest_mask & tgt_mask & enc_f.mask

            # Back-transform each target metric channel
            record: Dict[str, np.ndarray] = {
                "gate_mean": gate.cpu().numpy().mean(axis=0),  # [H,W]
                "aoi_mask": aoi_mask,
                "forest_mask": forest_mask,
                "combined_mask": combined_mask,
            }
            for c_idx, cref in enumerate(channel_refs):
                stats = feature_builder._get_channel_stats("target_metrics", cref)
                ntype = norm_types[cref]

                obs_orig = back_transform_channel(tgt_norm[c_idx], ntype, stats)
                pred_orig = back_transform_channel(pred_norm[c_idx], ntype, stats)

                record[f"target_{cref}"] = obs_orig
                record[f"pred_{cref}"] = pred_orig

            results.append(record)

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_masked(arr: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    """Return a masked array where *mask==False* is masked out."""
    return np.ma.array(arr, mask=~mask)


def plot_variable_sheet(
    records: List[Dict[str, np.ndarray]],
    channel_ref: str,
    output_path: Path,
    max_cols: int = 4,
) -> None:
    """Create a single figure with observed / predicted tiles for one variable.

    Layout: each patch gets two columns (observed, predicted) in a grid.
    """
    n_patches = len(records)
    n_cols = min(max_cols, n_patches)
    n_rows = int(np.ceil(n_patches / n_cols))

    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows * 2),
        squeeze=False,
    )

    # Compute global colour limits from all observed values (masked)
    all_obs = []
    for rec in records:
        arr = rec[f"target_{channel_ref}"]
        mask = rec["combined_mask"]
        vals = arr[mask]
        if vals.size > 0:
            all_obs.append(vals)
    if all_obs:
        combined_vals = np.concatenate(all_obs)
        vmin = float(np.nanpercentile(combined_vals, 2))
        vmax = float(np.nanpercentile(combined_vals, 98))
    else:
        vmin, vmax = 0.0, 1.0

    short_name = channel_ref.replace("static.", "")

    for i, rec in enumerate(records):
        row_block = i // n_cols
        col = i % n_cols

        obs_row = row_block * 2
        pred_row = obs_row + 1

        mask = rec["combined_mask"]
        obs = _make_masked(rec[f"target_{channel_ref}"], mask)
        pred = _make_masked(rec[f"pred_{channel_ref}"], mask)

        ax_obs = axes[obs_row, col]
        ax_pred = axes[pred_row, col]

        im_obs = ax_obs.imshow(obs, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
        ax_obs.set_title(f"Obs #{i}", fontsize=8)
        ax_obs.set_xticks([])
        ax_obs.set_yticks([])

        im_pred = ax_pred.imshow(pred, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
        ax_pred.set_title(f"Pred #{i}", fontsize=8)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

    # Turn off unused axes
    for r in range(n_rows * 2):
        for c in range(n_cols):
            idx = (r // 2) * n_cols + c
            if idx >= n_patches:
                axes[r, c].axis("off")

    fig.suptitle(f"{short_name}  (observed vs predicted, original scale)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    # Shared colour bar
    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.84])
    fig.colorbar(im_obs, cax=cbar_ax)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def plot_gate_sheet(
    records: List[Dict[str, np.ndarray]],
    output_path: Path,
    max_cols: int = 4,
) -> None:
    """Create a figure showing mean gate values for each patch."""
    n_patches = len(records)
    n_cols = min(max_cols, n_patches)
    n_rows = int(np.ceil(n_patches / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    # Compute data-driven colour limits from masked gate values
    all_gate = []
    for rec in records:
        vals = rec["gate_mean"][rec["combined_mask"]]
        if vals.size > 0:
            all_gate.append(vals)
    if all_gate:
        combined_gate = np.concatenate(all_gate)
        vmin = float(np.nanpercentile(combined_gate, 2))
        vmax = float(np.nanpercentile(combined_gate, 98))
        # Small guard so the range is never degenerate
        if vmax - vmin < 0.01:
            mid = (vmin + vmax) / 2
            vmin, vmax = mid - 0.05, mid + 0.05
    else:
        vmin, vmax = 0.0, 1.0

    for i, rec in enumerate(records):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        mask = rec["combined_mask"]
        gate = _make_masked(rec["gate_mean"], mask)

        im = ax.imshow(gate, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", interpolation="nearest")
        ax.set_title(f"Patch #{i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused axes
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= n_patches:
                axes[r, c].axis("off")

    fig.suptitle(
        f"Spatial gate (channel-mean, range [{vmin:.2f}, {vmax:.2f}])",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.84])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize linear-probe predictions and gate values on test patches."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to encoder checkpoint (.pt)")
    parser.add_argument("--probe", type=str, default=None,
                        help="Path to saved linear-probe (.pt). "
                             "Defaults to linear_probe_closed_form.pt next to checkpoint.")
    parser.add_argument("--bindings", type=str, default="config/frl_binding_v1.yaml",
                        help="Bindings YAML")
    parser.add_argument("--training", type=str, default="config/frl_training_v1.yaml",
                        help="Training YAML")
    parser.add_argument("--num-patches", type=int, default=16,
                        help="Number of test patches to visualise")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for patch selection")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. cuda:0 or cpu)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: <experiment>/diagnostics)")
    args = parser.parse_args()

    # ---- configs ----------------------------------------------------------
    logger.info(f"Loading bindings config from {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    logger.info(f"Loading training config from {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    patch_size = training_config.sampling.patch_size

    # ---- output dir -------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Derive from checkpoint path: go up to experiment dir, add diagnostics/
        ckpt_path = Path(args.checkpoint)
        experiment_dir = ckpt_path.parent.parent  # checkpoints/ -> experiment/
        output_dir = experiment_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ---- model + probe ----------------------------------------------------
    logger.info(f"Loading encoder from {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)

    probe_path = args.probe or str(Path(args.checkpoint).parent / "linear_probe_closed_form.pt")
    logger.info(f"Loading linear probe from {probe_path}")
    probe_ckpt = torch.load(probe_path, map_location=device, weights_only=False)
    W = probe_ckpt["W"]  # [D, T]
    b = probe_ckpt["b"]  # [T]
    target_metrics: List[str] = probe_ckpt["target_metrics"]
    logger.info(f"Probe target metrics ({len(target_metrics)}): {target_metrics}")

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

    # ---- select patches ---------------------------------------------------
    rng = random.Random(args.seed)
    n_select = min(args.num_patches, n_test)
    selected_indices = sorted(rng.sample(range(n_test), n_select))
    logger.info(f"Selected {n_select} patches for visualisation: {selected_indices}")

    # ---- feature builder --------------------------------------------------
    feature_builder = FeatureBuilder(bindings_config)

    # ---- collect data -----------------------------------------------------
    logger.info("Running inference on selected test patches...")
    records = collect_patch_data(
        test_dataset,
        feature_builder,
        model,
        W, b,
        device,
        target_metrics,
        selected_indices,
    )
    logger.info(f"Collected data for {len(records)} patches")

    # ---- plot -------------------------------------------------------------
    logger.info("Generating diagnostic figures...")

    # One sheet per target variable
    for cref in target_metrics:
        safe_name = cref.replace(".", "_")
        out_path = output_dir / f"test_{safe_name}.png"
        plot_variable_sheet(records, cref, out_path)

    # Gate sheet
    plot_gate_sheet(records, output_dir / "test_gate_mean.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
