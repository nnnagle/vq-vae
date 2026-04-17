#!/usr/bin/env python3
"""
Frequency distributions of ysfc (years since fire/change) for the top-20 EVT classes.

A data-level diagnostic — no model checkpoint required.  For each of the top-20
EVT classes (by pixel×timestep observation count), produces a bar-chart histogram
of ysfc values, revealing how much disturbance history is present in each forest type.

Outputs (all in --output-dir):
  ysfc_histograms.png    4×5 grid of histogram panels (one per EVT class)
  ysfc_by_evt.csv        Per-EVT, per-bin counts and summary statistics

Usage:
    PYTHONPATH=frl python frl/training/ysfc_evt_histograms.py \\
        --bindings frl/config/frl_binding_v1.yaml \\
        --training frl/config/frl_training_v1.yaml \\
        --evt-map  ../data/LF2024_EVT.csv \\
        --top-k-evt 20 \\
        --split train \\
        --output-dir runs/.../ysfc_histograms/
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
from training.fit_phase_linear_probe import _halo_mask
from utils.sampling import ReservoirSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ysfc_evt_histograms")

# ysfc bins (years since fire/change).  Right endpoint is exclusive.
YSFC_BINS: List[Tuple[int, int]] = [
    (0,  1),
    (1,  2),
    (2,  3),
    (3,  5),
    (5,  8),
    (8,  13),
    (13, 20),
    (20, 31),
    (31, 41),
]
YSFC_BIN_LABELS: List[str] = [
    "0", "1", "2", "3–4", "5–7", "8–12", "13–19", "20–30", "31–40",
]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    enc_feature_name: str,
    reservoirs: Dict[int, ReservoirSampler],
    n_seen: Dict[int, int],
    halo: int,
    max_ysfc: float,
    max_per_evt: int,
    seed: int,
) -> None:
    """Extract (ysfc, evt_code) pairs per pixel×timestep and add to reservoirs."""
    batch_size = len(batch["metadata"])

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f  = feature_builder.build_feature(enc_feature_name, sample)
        evt_f  = feature_builder.build_feature("evt_class", sample)
        ysfc_f = feature_builder.build_feature("ysfc", sample)

        H, W = enc_f.data.shape[-2], enc_f.data.shape[-1]

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data  = sample["static_mask"]
        aoi_mask    = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
        forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
        enc_mask    = torch.from_numpy(enc_f.mask).bool()

        evt_grid  = evt_f.data[0].astype(np.float32)       # [H, W]
        evt_valid = torch.from_numpy(np.isfinite(evt_grid) & (evt_grid > 0))

        halo_m = _halo_mask(H, W, halo, torch.device("cpu"))

        spatial_mask = enc_mask & aoi_mask & forest_mask & evt_valid & halo_m
        if not spatial_mask.any():
            continue

        evt_flat  = evt_grid.reshape(-1).astype(np.int32)   # [H*W]

        # ysfc: [1, T, H, W] → [T, H, W] raw years
        ysfc_grid     = ysfc_f.data[0]                              # [T, H, W]
        ysfc_mask_all = torch.from_numpy(ysfc_f.mask[0]).bool()    # [T, H, W]
        T = ysfc_grid.shape[0]

        for t in range(T):
            ysfc_t = torch.from_numpy(ysfc_grid[t])  # [H, W]
            t_mask = (
                spatial_mask
                & ysfc_mask_all[t]
                & (ysfc_t >= 0)
                & (ysfc_t <= max_ysfc)
            )
            if not t_mask.any():
                continue

            mask_flat = t_mask.reshape(-1)
            evt_px  = evt_flat[mask_flat.numpy()]                           # [N]
            ysfc_px = ysfc_grid[t].reshape(-1)[mask_flat.numpy()].astype(np.float32)  # [N]

            for code in np.unique(evt_px):
                code = int(code)
                sel  = evt_px == code
                vals = ysfc_px[sel][:, None]  # [N, 1] required by ReservoirSampler
                n_seen[code] += int(sel.sum())
                if code not in reservoirs:
                    reservoirs[code] = ReservoirSampler(
                        capacity=max_per_evt, dim=1, seed=seed
                    )
                reservoirs[code].add(vals)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(
    reservoirs: Dict[int, ReservoirSampler],
    top_evt_codes: List[int],
    evt_code_to_label: Dict[int, str],
    n_seen: Dict[int, int],
    max_ysfc: float,
    out_path: Path,
) -> None:
    rows = []
    for code in top_evt_codes:
        name = evt_code_to_label.get(code, f"EVT_{code}")
        sampler = reservoirs.get(code)
        if sampler is None or sampler.filled == 0:
            continue
        vals = sampler.get()[:, 0]  # [N]
        total = n_seen[code]
        for (lo, hi), label in zip(YSFC_BINS, YSFC_BIN_LABELS):
            if lo >= max_ysfc:
                break
            count = int(((vals >= lo) & (vals < hi)).sum())
            width = hi - lo
            rows.append({
                "evt_code":        code,
                "evt_name":        name,
                "bin_label":       label,
                "bin_lo":          lo,
                "bin_hi":          hi,
                "bin_width":       width,
                "count":           count,
                "density_per_year": count / width,
                "total_count":     total,
                "fraction":        count / total if total > 0 else 0.0,
                "mean_ysfc":       float(vals.mean()),
                "median_ysfc":     float(np.median(vals)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(
    reservoirs: Dict[int, ReservoirSampler],
    top_evt_codes: List[int],
    evt_code_to_label: Dict[int, str],
    n_seen: Dict[int, int],
    max_ysfc: float,
    out_path: Path,
) -> None:
    n_cols = 5
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    axes_list = axes.flatten().tolist()

    active_bins = [(lo, hi) for lo, hi in YSFC_BINS if lo < max_ysfc]
    active_labels = YSFC_BIN_LABELS[: len(active_bins)]
    x = np.arange(len(active_bins))

    for idx, code in enumerate(top_evt_codes):
        ax   = axes_list[idx]
        name = evt_code_to_label.get(code, f"EVT_{code}")
        sampler = reservoirs.get(code)

        if sampler is None or sampler.filled == 0:
            ax.axis("off")
            continue

        vals = sampler.get()[:, 0]  # [N] ysfc values
        n_total = n_seen[code]

        counts = np.array([
            int(((vals >= lo) & (vals < hi)).sum())
            for lo, hi in active_bins
        ], dtype=np.float64)
        widths = np.array([hi - lo for lo, hi in active_bins], dtype=np.float64)
        density = counts / widths  # counts per year

        ax.bar(x, density, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(active_labels, rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"{code}: {name[:30]}\n(n={n_total:,} obs)",
            fontsize=7, pad=2,
        )
        ax.set_xlabel("ysfc (years)", fontsize=6)
        ax.set_ylabel("Count per year (sampled)", fontsize=6)

    for ax in axes_list[len(top_evt_codes):]:
        ax.axis("off")

    fig.suptitle(
        "ysfc distribution by EVT class  |  Top-20 EVT classes by observation count\n"
        "(reservoir-sampled; y-axis = sampled count ÷ bin width, comparable across bins)",
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
            "Histogram of ysfc (years since fire/change) for the top-K EVT classes. "
            "Data-level diagnostic — no model checkpoint required."
        )
    )
    parser.add_argument("--training", default="config/frl_training_v1.yaml")
    parser.add_argument("--bindings", default="config/frl_binding_v1.yaml")
    parser.add_argument(
        "--evt-map", default="../data/LF2024_EVT.csv",
        help="CSV crosswalk with VALUE/EVT_NAME columns (LANDFIRE LF2024 format)",
    )
    parser.add_argument("--top-k-evt", type=int, default=20,
                        help="Number of top EVT classes to plot (default: 20)")
    parser.add_argument("--max-ysfc", type=float, default=40.0,
                        help="Exclude pixels with ysfc > this value (default: 40)")
    parser.add_argument("--max-samples-per-evt", type=int, default=50_000,
                        help="Reservoir cap per EVT class (default: 50000)")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--halo", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Cap batches processed (0 = all; for quick tests)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Config ---
    bindings_config = DatasetBindingsParser(args.bindings).parse()
    training_config = TrainingConfigParser(args.training).parse()

    batch_size = args.batch_size or training_config.training.batch_size
    patch_size = training_config.sampling.patch_size

    # --- EVT crosswalk ---
    evt_code_to_label: Dict[int, str] = {}
    if args.evt_map and Path(args.evt_map).exists():
        xwalk    = pd.read_csv(args.evt_map)
        cols     = xwalk.columns.tolist()
        code_col = "VALUE" if "VALUE" in cols else "Value"
        name_col = "EVT_NAME" if "EVT_NAME" in cols else "Description"
        evt_code_to_label = {
            int(row[code_col]): str(row[name_col])
            for _, row in xwalk.iterrows()
            if pd.notna(row[code_col]) and int(row[code_col]) > 0
        }
        logger.info(f"Loaded {len(evt_code_to_label)} EVT codes from {args.evt_map}")

    # --- Output directory ---
    out_dir = Path(args.output_dir)
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
    logger.info(f"Using encoder feature for masking: '{enc_feature_name}'")

    # Per-EVT reservoirs and observation counters
    reservoirs: Dict[int, ReservoirSampler] = {}
    n_seen: Dict[int, int] = defaultdict(int)

    # --- Streaming pass ---
    n_batches   = len(loader)
    max_batches = args.max_batches

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        process_batch(
            batch=batch,
            feature_builder=feature_builder,
            enc_feature_name=enc_feature_name,
            reservoirs=reservoirs,
            n_seen=n_seen,
            halo=args.halo,
            max_ysfc=args.max_ysfc,
            max_per_evt=args.max_samples_per_evt,
            seed=args.seed,
        )

        if batch_idx % 10 == 0:
            total_obs = sum(n_seen.values())
            logger.info(
                f"  Batch {batch_idx:4d}/{n_batches}  |  "
                f"EVT classes: {len(n_seen):4d}  |  "
                f"observations: {total_obs:,}"
            )

    total_obs = sum(n_seen.values())
    logger.info(
        f"Streaming complete: {total_obs:,} total observations  |  "
        f"{len(n_seen)} EVT classes"
    )

    if total_obs == 0:
        raise RuntimeError("No valid observations — check masks and split.")

    # --- Top-K EVT by observation count ---
    sorted_by_count = sorted(n_seen.items(), key=lambda x: x[1], reverse=True)
    top_k = min(args.top_k_evt, len(sorted_by_count))
    top_evt_codes = [code for code, _ in sorted_by_count[:top_k]]

    logger.info(f"\nTop {top_k} EVT classes by observation count:")
    for code, n in sorted_by_count[:top_k]:
        label = evt_code_to_label.get(code, f"EVT_{code}")
        logger.info(f"  {code:6d}  {n:10,}  {label}")

    # --- Save outputs ---
    save_csv(
        reservoirs, top_evt_codes, evt_code_to_label, n_seen,
        max_ysfc=args.max_ysfc,
        out_path=out_dir / "ysfc_by_evt.csv",
    )
    plot_histograms(
        reservoirs, top_evt_codes, evt_code_to_label, n_seen,
        max_ysfc=args.max_ysfc,
        out_path=out_dir / "ysfc_histograms.png",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
