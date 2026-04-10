#!/usr/bin/env python3
"""Compare GMM cluster assignments against exogenous EVT vegetation type labels.

For each valid pixel (aoi ∩ forest mask) in the chosen split this script:
  1. Extracts z_type embeddings via the frozen encoder.
  2. Assigns a GMM cluster label.
  3. Records the co-occurring EVT code.
  4. Accumulates a full contingency table [K_gmm × N_evt] and a
     reservoir sample of (cluster, evt) pairs for sklearn metrics.

Outputs
-------
  contingency_raw.csv        Raw pixel counts  [K_gmm × top_K_evt]
  contingency_heatmap.png    Row-normalised heatmap (seaborn)
  metrics.json               NMI, homogeneity, completeness, V-measure, purity

Usage
-----
    python frl/training/compare_gmm_evt.py \\
        --checkpoint runs/checkpoints/model.pt \\
        --gmm        runs/gmm/gmm_k20_diag.pkl \\
        --training   frl/config/frl_training_v1.yaml \\
        --bindings   frl/config/frl_binding_v1.yaml \\
        --evt-map    path/to/evt_crosswalk.csv \\
        --top-k-evt  15 \\
        --output-dir runs/gmm_evt_analysis/

    python -m training.compare_gmm_evt \\
        --checkpoint runs/frl_v0_exp005/checkpoints/encoder_best_1_epoch_197.pt \\
        --training   runs/frl_v0_exp005/frl_training_v1.yaml \\
        --bindings   runs/frl_v0_exp005/frl_binding_v1.yaml \\
        --evt-map    ../data/LANDFIRE_EVT_v1_4_0_classes.csv \\
        --top-k-evt  15 \\
        --output-dir runs/frl_v0_exp005/gmm_evt_analysis \\
        --gmm        runs/frl_v0_exp005/checkpoints/gmm_k20_diag.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("compare_gmm_evt")


# ---------------------------------------------------------------------------
# Reservoir sampler (same Algorithm R used in fit_gmm_clusters.py)
# ---------------------------------------------------------------------------

class ReservoirSampler:
    """Fixed-capacity uniform random sample of 2-column (cluster, evt) pairs."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty((capacity, 2), dtype=np.int32)
        self.n_seen = 0

    def add(self, cluster_labels: np.ndarray, evt_codes: np.ndarray) -> None:
        """Add a batch of (cluster, evt) pairs."""
        n = cluster_labels.shape[0]
        pairs = np.stack([cluster_labels, evt_codes], axis=1).astype(np.int32)
        for i in range(n):
            self.n_seen += 1
            if self.n_seen <= self.capacity:
                self.buffer[self.n_seen - 1] = pairs[i]
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.capacity:
                    self.buffer[j] = pairs[i]

    @property
    def filled(self) -> int:
        return min(self.n_seen, self.capacity)

    def get(self) -> np.ndarray:
        """Return sampled pairs as [N, 2] (cluster, evt) int32 array."""
        return self.buffer[: self.filled]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    gmm,
    device: torch.device,
    enc_feature_name: str = "type_encoder_input",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run encoder + GMM on one batch; return (cluster_labels, evt_codes) for
    valid pixels, or None if the batch contains no valid pixels."""
    batch_size = len(batch["metadata"])
    all_clusters: list[np.ndarray] = []
    all_evt: list[np.ndarray] = []

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        # --- encoder inputs ---
        enc_f = feature_builder.build_feature(enc_feature_name, sample)
        enc = torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)  # [1, C, H, W]

        # --- spatial masks: aoi ∩ forest ∩ encoder validity ---
        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]
        aoi_mask = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
        forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
        enc_mask = torch.from_numpy(enc_f.mask).bool()  # [H, W]
        base_mask = enc_mask & aoi_mask & forest_mask

        # --- EVT codes ---
        evt_f = feature_builder.build_feature("evt_class", sample)
        # evt_f.data: [1, H, W], raw codes passed through identity norm (float)
        evt_grid = evt_f.data[0].astype(np.float32)  # [H, W]
        evt_valid = np.isfinite(evt_grid) & (evt_grid > 0)
        combined_mask = base_mask & torch.from_numpy(evt_valid)  # [H, W]

        if not combined_mask.any():
            continue

        # --- forward pass (type pathway only) ---
        with torch.no_grad():
            z = model(enc).cpu().squeeze(0)  # [D, H, W]

        D = z.shape[0]
        z_flat = z.permute(1, 2, 0).reshape(-1, D).numpy().astype(np.float32)
        mask_flat = combined_mask.reshape(-1).numpy()

        z_valid = z_flat[mask_flat]
        evt_valid_codes = evt_grid.reshape(-1)[mask_flat].astype(np.int32)

        if z_valid.shape[0] == 0:
            continue

        cluster_labels = gmm.predict(z_valid).astype(np.int32)
        all_clusters.append(cluster_labels)
        all_evt.append(evt_valid_codes)

    if not all_clusters:
        return None

    return np.concatenate(all_clusters), np.concatenate(all_evt)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(cluster_labels: np.ndarray, evt_labels: np.ndarray) -> dict:
    """Compute NMI, homogeneity, completeness, V-measure, and purity."""
    try:
        from sklearn.metrics import (
            normalized_mutual_info_score,
            homogeneity_completeness_v_measure,
        )
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    nmi = normalized_mutual_info_score(evt_labels, cluster_labels, average_method="arithmetic")
    hom, comp, vm = homogeneity_completeness_v_measure(evt_labels, cluster_labels)

    # Purity: for each cluster, fraction of its majority EVT class
    n = len(cluster_labels)
    k_vals = np.unique(cluster_labels)
    majority_sum = 0
    for k in k_vals:
        mask = cluster_labels == k
        if mask.sum() == 0:
            continue
        evt_in_cluster = evt_labels[mask]
        counts = np.bincount(evt_in_cluster - evt_in_cluster.min())
        majority_sum += counts.max()
    purity = majority_sum / n

    return {
        "nmi": float(nmi),
        "homogeneity": float(hom),
        "completeness": float(comp),
        "v_measure": float(vm),
        "purity": float(purity),
        "n_pixels_metrics": int(n),
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    contingency: np.ndarray,
    evt_labels: list[str],
    n_components: int,
    out_path: Path,
) -> None:
    """Row-normalised heatmap: each row sums to 1 (fraction of cluster in each EVT)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sums = contingency.sum(axis=1, keepdims=True).clip(min=1)
    normed = contingency / row_sums  # [K, E]

    # Truncate long labels for display
    short_labels = [(lbl[:42] + "…") if len(lbl) > 42 else lbl for lbl in evt_labels]

    fig_h = max(5, n_components * 0.45)
    fig_w = max(10, len(evt_labels) * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Annotate only if the grid is small enough to be readable
    annotate = n_components <= 30 and len(evt_labels) <= 20

    sns.heatmap(
        normed,
        ax=ax,
        xticklabels=short_labels,
        yticklabels=[f"C{k}" for k in range(n_components)],
        cmap="YlOrRd",
        vmin=0,
        vmax=normed.max(),
        linewidths=0.3,
        linecolor="lightgrey",
        cbar_kws={"label": "Fraction of cluster pixels"},
        annot=annotate,
        fmt=".2f",
    )
    ax.set_xlabel("EVT Class")
    ax.set_ylabel("GMM Cluster")
    ax.set_title(
        f"GMM Clusters ({n_components}) vs EVT (top {len(evt_labels)} classes)\n"
        "Row-normalised: each row = fraction of that cluster's pixels per EVT class"
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GMM cluster labels against EVT vegetation type classification."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--gmm", required=True, help="Path to fitted GMM .pkl file")
    parser.add_argument("--training", default="config/frl_training_v1.yaml")
    parser.add_argument("--bindings", default="config/frl_binding_v1.yaml")
    parser.add_argument(
        "--evt-map", default=None,
        help="CSV crosswalk with columns Value,Color,Description (LANDFIRE format)",
    )
    parser.add_argument(
        "--top-k-evt", type=int, default=15,
        help="Number of top EVT classes (by pixel count) to show in heatmap (default: 15)",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--max-reservoir", type=int, default=500_000,
        help="Max (cluster, evt) pairs to reservoir-sample for sklearn metrics (default: 500000)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Config ---
    logger.info(f"Loading bindings: {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()
    logger.info(f"Loading training config: {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    batch_size = args.batch_size or training_config.training.batch_size
    patch_size = training_config.sampling.patch_size
    logger.info(f"Device: {device}  |  batch_size: {batch_size}")

    # --- EVT crosswalk ---
    evt_code_to_label: dict[int, str] = {}
    if args.evt_map:
        xwalk = pd.read_csv(args.evt_map, dtype={"Value": int})
        evt_code_to_label = {
            int(row["Value"]): str(row["Description"])
            for _, row in xwalk.iterrows()
        }
        logger.info(f"Loaded {len(evt_code_to_label)} EVT codes from {args.evt_map}")

    # --- Output directory ---
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.gmm).parent / "evt_comparison"
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

    feature_builder = FeatureBuilder(bindings_config)

    # --- Model ---
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)
    logger.info(f"z_type_dim = {model.z_type_dim}")
    enc_feature_name = training_config.model_input.type_encoder_feature
    logger.info(f"Using encoder feature: '{enc_feature_name}'")

    # --- GMM ---
    logger.info(f"Loading GMM: {args.gmm}")
    with open(args.gmm, "rb") as f:
        gmm_payload = pickle.load(f)
    gmm = gmm_payload["gmm"]
    n_components = gmm_payload["n_components"]
    logger.info(f"GMM: K={n_components}, covariance_type={gmm_payload['covariance_type']}")

    # --- Streaming pass ---
    # Full contingency: counts[gmm_k][evt_code] — small dict, always kept in full
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # Reservoir for sklearn metrics (bounded memory)
    reservoir = ReservoirSampler(capacity=args.max_reservoir, seed=args.seed)
    total_pixels = 0

    for batch_idx, batch in enumerate(loader):
        result = process_batch(batch, feature_builder, model, gmm, device, enc_feature_name)
        if result is None:
            continue
        cluster_labels, evt_codes = result

        # Accumulate contingency
        for cl, ev in zip(cluster_labels.tolist(), evt_codes.tolist()):
            counts[cl][ev] += 1

        # Reservoir sample for metrics
        reservoir.add(cluster_labels, evt_codes)
        total_pixels += len(cluster_labels)

        if batch_idx % 50 == 0:
            logger.info(
                f"  Batch {batch_idx:4d}/{len(loader)}  |  "
                f"pixels: {total_pixels:,}  |  reservoir: {reservoir.filled:,}"
            )

    logger.info(f"Streaming complete: {total_pixels:,} valid pixels processed")

    if total_pixels == 0:
        raise RuntimeError("No valid pixels found — check masks and split.")

    # --- EVT frequency ranking ---
    evt_totals: dict[int, int] = defaultdict(int)
    for cluster_counts in counts.values():
        for evt_code, n in cluster_counts.items():
            evt_totals[evt_code] += n

    sorted_evt = sorted(evt_totals.items(), key=lambda x: x[1], reverse=True)
    top_k = min(args.top_k_evt, len(sorted_evt))
    top_evt_codes = [code for code, _ in sorted_evt[:top_k]]

    logger.info(f"\nTop {top_k} EVT codes by pixel count (of {len(sorted_evt)} observed):")
    cumulative = 0
    for code, n in sorted_evt[:top_k]:
        label = evt_code_to_label.get(code, f"EVT_{code}")
        cumulative += n
        logger.info(f"  {code:6d}  {n:9,}  ({100*n/total_pixels:.1f}%)  {label}")
    logger.info(
        f"  Top-{top_k} covers {100*cumulative/total_pixels:.1f}% of all valid pixels"
    )

    # --- Contingency matrix [K, top_k] ---
    contingency = np.zeros((n_components, top_k), dtype=np.int64)
    for k in range(n_components):
        for col_idx, evt_code in enumerate(top_evt_codes):
            contingency[k, col_idx] = counts[k].get(evt_code, 0)

    evt_col_labels = [evt_code_to_label.get(c, f"EVT_{c}") for c in top_evt_codes]

    # --- Save CSV ---
    col_headers = [f"{c}: {evt_code_to_label.get(c, c)}" for c in top_evt_codes]
    df = pd.DataFrame(
        contingency,
        index=[f"C{k}" for k in range(n_components)],
        columns=col_headers,
    )
    csv_path = out_dir / "contingency_raw.csv"
    df.to_csv(csv_path)
    logger.info(f"Contingency table saved to {csv_path}")

    # --- Metrics (from reservoir sample, covers all EVT codes) ---
    logger.info(f"Computing clustering metrics on {reservoir.filled:,} reservoir pixels...")
    pairs = reservoir.get()
    # Re-encode EVT codes to dense integers for sklearn
    cluster_sample = pairs[:, 0]
    evt_sample = pairs[:, 1]
    unique_evt = np.unique(evt_sample)
    evt_remap = {v: i for i, v in enumerate(unique_evt)}
    evt_dense = np.array([evt_remap[v] for v in evt_sample], dtype=np.int32)

    metrics = compute_metrics(cluster_sample, evt_dense)
    metrics.update({
        "gmm_pkl": str(args.gmm),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_gmm_components": n_components,
        "n_pixels_total": total_pixels,
        "n_evt_codes_observed": len(sorted_evt),
        "top_k_evt_shown": top_k,
    })

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(
        f"\n  NMI          = {metrics['nmi']:.4f}\n"
        f"  Homogeneity  = {metrics['homogeneity']:.4f}  "
        f"(each GMM cluster ≈ single EVT type)\n"
        f"  Completeness = {metrics['completeness']:.4f}  "
        f"(each EVT type ≈ single GMM cluster)\n"
        f"  V-measure    = {metrics['v_measure']:.4f}\n"
        f"  Purity       = {metrics['purity']:.4f}"
    )

    # --- Heatmap ---
    plot_heatmap(contingency, evt_col_labels, n_components, out_dir / "contingency_heatmap.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
