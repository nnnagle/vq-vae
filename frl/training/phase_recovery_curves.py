#!/usr/bin/env python3
"""
Per-EVT post-disturbance recovery curves for predicted NBR.

For each of the top-20 EVT classes (by pixel count), this script produces a
box-plot panel showing predicted NBR (y-axis, z-score normalized) vs. years
since fire/change — ysfc (x-axis), drawn from the phase linear probe output.

A rising curve from low post-disturbance NBR (ysfc~0) to higher NBR at
ysfc~20 indicates the phase embedding has learned recovery dynamics.

Outputs (all in --output-dir):
  recovery_curves.png       4×5 grid of box-plot panels (one per EVT class)
  nbr_by_ysfc_by_evt.csv   Per-EVT, per-bin median/quartile statistics

Usage:
    PYTHONPATH=. python training/phase_recovery_curves.py \\
        --checkpoint runs/.../encoder_epoch.pt \\
        --probe      runs/.../phase_linear_probe.pt \\
        --training   config/frl_training_v1.yaml \\
        --bindings   config/frl_binding_v1.yaml \\
        --evt-map    ../data/LF2024_EVT.csv \\
        --top-k-evt  20 \\
        --output-dir runs/.../recovery_curves/
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel

from training.fit_phase_linear_probe import (
    PHASE_INPUT_FEATURE,
    PHASE_TARGET_FEATURE,
    ProbePreprocessor,
    _get_target_channels,
    _halo_mask,
)
from utils.sampling import ReservoirSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase_recovery_curves")

# ysfc bins (years since fire/change).  Right endpoint is exclusive.
YSFC_BINS = [
    (0,  1),
    (1,  2),
    (2,  3),
    (3,  5),
    (5,  8),
    (8,  13),
    (13, 20),
    (20, 31),
]
YSFC_BIN_LABELS = ["0", "1", "2", "3–4", "5–7", "8–12", "13–19", "20–30"]


# ---------------------------------------------------------------------------
# Per-EVT reservoir (wraps ReservoirSampler from fit_gmm_clusters)
# ---------------------------------------------------------------------------

class EvtReservoir:
    """Per-EVT reservoir of (ysfc, pred_nbr, obs_nbr) triples.

    Internally stores a ``ReservoirSampler(capacity, dim=3)`` per EVT class.
    Columns: [0]=ysfc, [1]=pred_nbr, [2]=obs_nbr.
    """
    _DIM = 3  # [ysfc, pred_nbr, obs_nbr]

    def __init__(self, max_per_evt: int, seed: int = 42) -> None:
        self.max_per_evt = max_per_evt
        self.seed = seed
        self._samplers: Dict[int, ReservoirSampler] = {}
        # track total pixel×timestep observations per EVT (before sampling)
        self._n_seen: Dict[int, int] = defaultdict(int)

    def _get_or_create(self, evt_code: int) -> ReservoirSampler:
        if evt_code not in self._samplers:
            self._samplers[evt_code] = ReservoirSampler(
                capacity=self.max_per_evt, dim=self._DIM, seed=self.seed
            )
        return self._samplers[evt_code]

    def add_batch(
        self,
        evt_codes: np.ndarray,   # [N] int32
        ysfc: np.ndarray,        # [N] float32
        pred_nbr: np.ndarray,    # [N] float32
        obs_nbr: np.ndarray,     # [N] float32
    ) -> None:
        unique_evts = np.unique(evt_codes)
        for e in unique_evts:
            mask = evt_codes == e
            vecs = np.stack(
                [ysfc[mask], pred_nbr[mask], obs_nbr[mask]], axis=1
            ).astype(np.float32)
            self._n_seen[int(e)] += int(mask.sum())
            self._get_or_create(int(e)).add(vecs)

    def get(self, evt_code: int) -> Optional[np.ndarray]:
        """Return [N, 3] array (ysfc, pred_nbr, obs_nbr) or None."""
        s = self._samplers.get(evt_code)
        return s.get() if s is not None else None

    def all_evt_codes(self) -> List[int]:
        return list(self._samplers.keys())

    def n_seen(self, evt_code: int) -> int:
        return self._n_seen.get(evt_code, 0)

    def n_total(self) -> int:
        return sum(self._n_seen.values())

    def pixel_counts(self) -> Dict[int, int]:
        """n_seen per EVT class (total pixel×timestep observations)."""
        return dict(self._n_seen)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    reservoir: EvtReservoir,
    device: torch.device,
    halo: int,
    enc_feature_name: str,
    probe_W: torch.Tensor,        # [D, C] float64 CPU
    probe_b: torch.Tensor,        # [C]    float64 CPU
    preprocessor: ProbePreprocessor,
    nbr_ch_idx: int,
    max_ysfc: float,
) -> None:
    """Forward pass per sample; accumulate (ysfc, pred_nbr, obs_nbr) per EVT."""
    batch_size = len(batch["metadata"])

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f    = feature_builder.build_feature(enc_feature_name, sample)
        phase_f  = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
        evt_f    = feature_builder.build_feature("evt_class", sample)
        ysfc_f   = feature_builder.build_feature("ysfc", sample)
        tgt_f    = feature_builder.build_feature(PHASE_TARGET_FEATURE, sample)

        H, W = enc_f.data.shape[-2], enc_f.data.shape[-1]
        T    = phase_f.data.shape[1]

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data  = sample["static_mask"]
        aoi_mask    = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
        forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
        enc_mask    = torch.from_numpy(enc_f.mask).bool()
        phase_mask_spatial = torch.from_numpy(phase_f.mask).all(dim=0)

        evt_grid  = evt_f.data[0].astype(np.float32)     # [H, W]
        evt_valid = torch.from_numpy(np.isfinite(evt_grid) & (evt_grid > 0))

        halo_m = _halo_mask(H, W, halo, torch.device("cpu"))
        spatial_mask = (
            enc_mask & aoi_mask & forest_mask & phase_mask_spatial
            & evt_valid & halo_m
        )
        if not spatial_mask.any():
            continue

        # ysfc: [1, T, H, W] identity-norm (raw years)
        ysfc_grid   = ysfc_f.data[0]                               # [T, H, W]
        ysfc_spatial = torch.from_numpy(ysfc_f.mask[0]).bool()    # [T, H, W]

        # target NBR: [12, T, H, W]
        tgt_nbr_grid  = tgt_f.data[nbr_ch_idx]                    # [T, H, W]
        tgt_mask_spatial = torch.from_numpy(tgt_f.mask).all(dim=0) # [T, H, W]

        # Forward pass (single sample to avoid OOM)
        enc_tensor   = torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)
        phase_tensor = torch.from_numpy(phase_f.data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            z_type  = model(enc_tensor)                               # [1, 64, H, W]
            z_type_det = z_type.detach()
            z_phase = model.forward_phase(phase_tensor, z_type_det)   # [1, 12, T, H, W]

        zt_all = z_type.squeeze(0).permute(1, 2, 0).reshape(-1, model.z_type_dim).cpu()   # [H*W, 64]
        zp_all = z_phase.squeeze(0).permute(2, 3, 0, 1).reshape(-1, model.z_phase_dim, T).cpu()  # [H*W, 12, T]

        evt_flat = evt_grid.reshape(-1).astype(np.int32)  # [H*W]

        for t in range(T):
            # Per-timestep combined mask
            t_mask = (
                spatial_mask
                & ysfc_spatial[t]
                & tgt_mask_spatial[t]
            )
            # Filter ysfc range: 0 <= ysfc <= max_ysfc
            ysfc_t_grid = torch.from_numpy(ysfc_grid[t])
            t_mask = t_mask & (ysfc_t_grid >= 0) & (ysfc_t_grid <= max_ysfc)

            if not t_mask.any():
                continue

            mask_flat = t_mask.reshape(-1)

            zt_px = zt_all[mask_flat]              # [N, 64]
            zp_px = zp_all[mask_flat, :, t]        # [N, 12]
            evt_px = evt_flat[mask_flat.numpy()]   # [N] int32
            ysfc_px = ysfc_grid[t].reshape(-1)[mask_flat.numpy()].astype(np.float32)  # [N]
            obs_px  = tgt_nbr_grid[t].reshape(-1)[mask_flat.numpy()].astype(np.float32)  # [N]

            X_t     = preprocessor.transform(zt_px, zp_px)           # [N, D]
            pred_t  = (X_t.double() @ probe_W) + probe_b             # [N, n_ch]
            pred_nbr_px = pred_t[:, nbr_ch_idx].float().numpy()      # [N]

            reservoir.add_batch(evt_px, ysfc_px, pred_nbr_px, obs_px)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(
    reservoir: EvtReservoir,
    top_evt_codes: List[int],
    evt_code_to_label: Dict[int, str],
    out_path: Path,
) -> None:
    """Write per-EVT, per-bin quartile statistics to CSV."""
    rows = []
    for code in top_evt_codes:
        data = reservoir.get(code)
        if data is None:
            continue
        name = evt_code_to_label.get(code, f"EVT_{code}")
        ysfc_vals    = data[:, 0]
        pred_vals    = data[:, 1]
        obs_vals     = data[:, 2]
        for (lo, hi), label in zip(YSFC_BINS, YSFC_BIN_LABELS):
            bin_mask = (ysfc_vals >= lo) & (ysfc_vals < hi)
            n = int(bin_mask.sum())
            if n == 0:
                continue
            p = pred_vals[bin_mask]
            o = obs_vals[bin_mask]
            rows.append({
                "evt_code":       code,
                "evt_name":       name,
                "ysfc_bin":       label,
                "n_samples":      n,
                "pred_nbr_q25":   float(np.percentile(p, 25)),
                "pred_nbr_median":float(np.median(p)),
                "pred_nbr_q75":   float(np.percentile(p, 75)),
                "obs_nbr_median": float(np.median(o)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_recovery_curves(
    reservoir: EvtReservoir,
    top_evt_codes: List[int],
    evt_code_to_label: Dict[int, str],
    out_path: Path,
    min_bin_samples: int = 5,
) -> None:
    """4-column grid of recovery-curve box plots, one panel per top EVT class."""
    n_evt = len(top_evt_codes)
    ncols = 4
    nrows = (n_evt + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.5),
        sharey=False,
    )
    axes_list = list(np.array(axes).flat)  # always a plain list, regardless of shape

    # Collect global y-range from all data for a shared reference
    all_preds: List[float] = []
    for code in top_evt_codes:
        d = reservoir.get(code)
        if d is not None:
            all_preds.extend(d[:, 1].tolist())
    if all_preds:
        p5  = np.percentile(all_preds, 2)
        p95 = np.percentile(all_preds, 98)
        pad = (p95 - p5) * 0.1
        y_lo, y_hi = p5 - pad, p95 + pad
    else:
        y_lo, y_hi = -3, 3

    for ax, code in zip(axes_list, top_evt_codes):
        data = reservoir.get(code)
        name = evt_code_to_label.get(code, f"EVT_{code}")
        n_total = reservoir.n_seen(code)

        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{code}: {name[:28]}", fontsize=7)
            continue

        ysfc_vals = data[:, 0]
        pred_vals = data[:, 1]
        obs_vals  = data[:, 2]

        box_data: List[np.ndarray] = []
        obs_medians: List[Optional[float]] = []
        valid_positions: List[int] = []

        for b_idx, (lo, hi) in enumerate(YSFC_BINS):
            mask = (ysfc_vals >= lo) & (ysfc_vals < hi)
            n = int(mask.sum())
            if n >= min_bin_samples:
                box_data.append(pred_vals[mask])
                obs_medians.append(float(np.median(obs_vals[mask])))
                valid_positions.append(b_idx)
            else:
                box_data.append(None)
                obs_medians.append(None)
                valid_positions.append(b_idx)

        # Only render non-None bins
        non_empty = [i for i, d in enumerate(box_data) if d is not None]
        if not non_empty:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{code}: {name[:28]}", fontsize=7)
            continue

        bp = ax.boxplot(
            [box_data[i] for i in non_empty],
            positions=non_empty,
            widths=0.55,
            patch_artist=True,
            boxprops=dict(facecolor="#5b9bd5", alpha=0.7, linewidth=0.8),
            medianprops=dict(color="navy", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=1.5, alpha=0.3, color="steelblue"),
            showfliers=True,
        )
        _ = bp  # suppress unused-variable lint

        # Observed NBR median line
        obs_x = [i for i in non_empty if obs_medians[i] is not None]
        obs_y = [obs_medians[i] for i in obs_x]
        if obs_x:
            ax.plot(obs_x, obs_y, "o--", color="#e07b2a", linewidth=1.0,
                    markersize=3.5, label="obs median", zorder=5)

        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":", alpha=0.7)
        ax.set_xlim(-0.6, len(YSFC_BINS) - 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(range(len(YSFC_BINS)))
        ax.set_xticklabels(YSFC_BIN_LABELS, rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"{code}: {name[:28]}\n(n={n_total:,})",
            fontsize=7, pad=2,
        )
        ax.set_xlabel("ysfc (years)", fontsize=6)
        ax.set_ylabel("Predicted NBR (z-score)", fontsize=6)

        if obs_x:
            ax.legend(fontsize=5, loc="upper left", framealpha=0.5)

    # Hide unused axes
    for ax in axes_list[n_evt:]:
        ax.axis("off")

    fig.suptitle(
        "Post-disturbance recovery: predicted NBR vs. years since fire/change\n"
        "Top-20 EVT classes  |  blue boxes = predicted NBR  |  orange = observed median",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-EVT post-disturbance recovery curves: predicted NBR vs. "
            "years since fire/change (ysfc), top-20 EVT classes."
        )
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--probe", required=True,
                        help="Path to phase_linear_probe.pt checkpoint")
    parser.add_argument("--training", default="config/frl_training_v1.yaml")
    parser.add_argument("--bindings", default="config/frl_binding_v1.yaml")
    parser.add_argument(
        "--evt-map", default="../data/LF2024_EVT.csv",
        help="CSV crosswalk with VALUE/EVT_NAME columns (LANDFIRE LF2024 format)",
    )
    parser.add_argument("--top-k-evt", type=int, default=20,
                        help="Number of top EVT classes to plot (default: 20)")
    parser.add_argument("--max-ysfc", type=float, default=30.0,
                        help="Exclude pixels with ysfc > this value (default: 30)")
    parser.add_argument("--max-samples-per-evt", type=int, default=10_000,
                        help="Reservoir cap per EVT class (default: 10000)")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--halo", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Cap batches processed (0 = all; for quick tests)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Config ---
    bindings_config = DatasetBindingsParser(args.bindings).parse()
    training_config = TrainingConfigParser(args.training).parse()

    device_str = args.device or training_config.hardware.device
    device     = torch.device(device_str)
    batch_size = args.batch_size or training_config.training.batch_size
    patch_size = training_config.sampling.patch_size
    logger.info(f"Device: {device}  |  batch_size: {batch_size}")

    # --- EVT crosswalk ---
    evt_code_to_label: Dict[int, str] = {}
    if args.evt_map and Path(args.evt_map).exists():
        xwalk  = pd.read_csv(args.evt_map)
        cols   = xwalk.columns.tolist()
        code_col = "VALUE" if "VALUE" in cols else "Value"
        name_col = "EVT_NAME" if "EVT_NAME" in cols else "Description"
        evt_code_to_label = {
            int(row[code_col]): str(row[name_col])
            for _, row in xwalk.iterrows()
            if pd.notna(row[code_col]) and int(row[code_col]) > 0
        }
        logger.info(f"Loaded {len(evt_code_to_label)} EVT codes from {args.evt_map}")

    # --- Output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).parent / "recovery_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Dataset ---
    logger.info(f"Creating {args.split} dataset...")
    dataset = ForestDatasetV2(
        bindings_config,
        split=args.split,
        patch_size=patch_size,
        min_aoi_fraction=0.3,
    )
    logger.info(f"{args.split} dataset: {len(dataset)} patches")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    feature_builder  = FeatureBuilder(bindings_config)
    enc_feature_name = training_config.model_input.type_encoder_feature
    logger.info(f"Using encoder feature: '{enc_feature_name}'")

    # --- Model ---
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)
    logger.info(f"z_type_dim={model.z_type_dim}  z_phase_dim={model.z_phase_dim}")

    # --- Probe ---
    logger.info(f"Loading phase probe: {args.probe}")
    probe_ckpt  = torch.load(args.probe, map_location="cpu", weights_only=False)
    probe_W     = probe_ckpt["W"].to(torch.float64)
    probe_b     = probe_ckpt["b"].to(torch.float64)
    preprocessor = ProbePreprocessor.from_dict(probe_ckpt)

    target_channels = _get_target_channels(feature_builder)
    try:
        nbr_ch_idx = target_channels.index("annual.nbr")
    except ValueError:
        raise RuntimeError(
            f"'annual.nbr' not found in target channels: {target_channels}"
        )
    logger.info(f"NBR channel index: {nbr_ch_idx}  (of {len(target_channels)} targets)")

    # --- Reservoir ---
    reservoir = EvtReservoir(max_per_evt=args.max_samples_per_evt, seed=args.seed)

    # --- Streaming pass ---
    n_batches  = len(loader)
    max_batches = args.max_batches

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        process_batch(
            batch, feature_builder, model, reservoir,
            device=device,
            halo=args.halo,
            enc_feature_name=enc_feature_name,
            probe_W=probe_W,
            probe_b=probe_b,
            preprocessor=preprocessor,
            nbr_ch_idx=nbr_ch_idx,
            max_ysfc=args.max_ysfc,
        )

        if batch_idx % 5 == 0:
            logger.info(
                f"  Batch {batch_idx:4d}/{n_batches}  |  "
                f"EVT classes: {len(reservoir.all_evt_codes()):4d}  |  "
                f"observations: {reservoir.n_total():,}"
            )

    logger.info(
        f"Streaming complete: {reservoir.n_total():,} total observations  |  "
        f"{len(reservoir.all_evt_codes())} EVT classes"
    )

    if reservoir.n_total() == 0:
        raise RuntimeError("No valid observations — check masks and split.")

    # --- Top-K EVT by observation count ---
    sorted_by_count = sorted(
        reservoir.pixel_counts().items(), key=lambda x: x[1], reverse=True
    )
    top_k = min(args.top_k_evt, len(sorted_by_count))
    top_evt_codes = [code for code, _ in sorted_by_count[:top_k]]

    logger.info(f"\nTop {top_k} EVT classes by observation count:")
    for code, n in sorted_by_count[:top_k]:
        label = evt_code_to_label.get(code, f"EVT_{code}")
        logger.info(f"  {code:6d}  {n:10,}  {label}")

    # --- Save outputs ---
    save_csv(
        reservoir, top_evt_codes, evt_code_to_label,
        out_path=out_dir / "nbr_by_ysfc_by_evt.csv",
    )
    plot_recovery_curves(
        reservoir, top_evt_codes, evt_code_to_label,
        out_path=out_dir / "recovery_curves.png",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
