#!/usr/bin/env python3
"""
Hierarchical post-hoc landscape categorization: forest type × recovery phase.

Algorithm
---------
1. Stream TRAIN patches through the frozen encoder to collect, per valid forest pixel:
   - z_type [64-dim]:   what kind of forest
   - phase_summary [36-dim]: disturbed centroid (12) + recovered centroid (12) + overall
     mean (12) of z_phase, using ysfc ≤ 1 for disturbed and ysfc ≥ 5 for recovered.
     Pixels without observed disturbed or recovered timesteps fall back to overall mean
     for that slot.

2. BIC-curve GMM sweep on z_type to select K_type* (the number of forest type clusters).

3. Assign type cluster labels to all sampled pixels.

4. For each type cluster k, BIC-curve GMM sweep on its phase_summary vectors over
   K_phase ∈ {1, …, max_phase_k} (hard cap: 5).
   K_phase = 1 means the type is non-dynamic (no meaningful temporal structure).
   K_phase ≥ 2 means it is dynamic (e.g. disturbed / recovering / mature sub-categories).

Outputs (saved to --output-dir)
-------------------------------
  type_gmm.pkl              Final K_type* GMM on z_type [64-dim]
  bic_curve_type.png        BIC vs K for type sweep
  phase_gmm_{k}.pkl         Phase GMM for type cluster k (all clusters, K_phase ≥ 1)
  bic_curve_phase_{k}.png   BIC vs K for phase sweep of cluster k
  taxonomy.json             Hierarchy: cluster_id → {n_phase, is_dynamic, ...}
  dynamic_scores.csv        Per-cluster temporal spread diagnostic

Usage
-----
    python training/fit_landscape_categories.py \\
        --checkpoint runs/checkpoints/model.pt \\
        --training   config/frl_training_v1.yaml

    # Custom type K range and output directory
    python training/fit_landscape_categories.py \\
        --checkpoint runs/checkpoints/model.pt \\
        --training   config/frl_training_v1.yaml \\
        --output-dir runs/taxonomy/v1 \\
        --k-type-min 10 --k-type-max 100 --k-type-step 10

Notes
-----
- The BIC sweep uses n_init=1 for speed; only the final refit uses --n-init.
- Phase GMMs are skipped for clusters with fewer than --min-cluster-pixels valid pixels.
- ysfc values are obtained from the "ysfc" feature (shape [1, T, H, W]).
  Timesteps outside the ysfc feature mask are treated as NaN (no recovery observation).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel
from utils.sampling import ReservoirSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fit_landscape_categories")

PHASE_INPUT_FEATURE = "phase_ccdc"
YSFC_FEATURE = "ysfc"

# ysfc thresholds matching the recovery discrimination loss defaults
LOW_YSFC_MAX: float = 1.0   # disturbed: ysfc ∈ {0, 1}
HIGH_YSFC_MIN: float = 5.0  # recovered: ysfc ≥ 5


# ---------------------------------------------------------------------------
# Phase summary
# ---------------------------------------------------------------------------

def _compute_phase_summary(
    z_phase: torch.Tensor,
    ysfc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute phase summary and per-pixel temporal variance.

    Args:
        z_phase: [N, T, D]  D = z_phase_dim (12)
        ysfc:    [N, T]     float, NaN where ysfc is not observed

    Returns:
        phase_summary: [N, 36] = concat(disturbed_centroid, recovered_centroid, overall_mean)
            Pixels lacking disturbed or recovered timesteps fall back to overall_mean.
        temporal_var: [N] mean across channels of per-timestep variance (z_phase.var(dim=T))
    """
    overall_mean = z_phase.mean(dim=1)  # [N, D]
    temporal_var = z_phase.var(dim=1).mean(dim=-1)  # [N]  (var over T, mean over D)

    def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
        w = mask.float().unsqueeze(-1)          # [N, T, 1]
        s = (z_phase * w).sum(dim=1)            # [N, D]
        c = w.sum(dim=1).clamp(min=1.0)         # [N, 1]
        centroid = s / c                         # [N, D]
        has_any = mask.any(dim=1, keepdim=True)  # [N, 1]
        return torch.where(has_any, centroid, overall_mean)

    valid = ysfc.isfinite()
    disturbed_centroid = _masked_mean(valid & (ysfc <= LOW_YSFC_MAX))
    recovered_centroid = _masked_mean(valid & (ysfc >= HIGH_YSFC_MIN))

    phase_summary = torch.cat([disturbed_centroid, recovered_centroid, overall_mean], dim=1)
    return phase_summary, temporal_var


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    enc_feature_name: str,
) -> Optional[np.ndarray]:
    """Run both pathways on one batch and return combined pixel vectors.

    Processes one patch at a time through the phase encoder to keep GPU
    memory bounded (one [1, C, T, H, W] tensor rather than the full batch).

    Returns:
        float32 array [N, 101] for valid pixels across all patches, or None.
    """
    all_pixels: List[np.ndarray] = []

    for i in range(len(batch["metadata"])):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f = feature_builder.build_feature(enc_feature_name, sample)
        phase_f = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
        ysfc_f = feature_builder.build_feature(YSFC_FEATURE, sample)

        enc = torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)   # [1, C, H, W]
        phase = torch.from_numpy(phase_f.data).float().unsqueeze(0).to(device)  # [1, 8, T, H, W]

        ysfc_data = ysfc_f.data[0].astype(np.float32)       # [T, H, W]
        ysfc_data = np.where(ysfc_f.mask, ysfc_data, np.nan)
        ysfc = torch.from_numpy(ysfc_data).float()           # [T, H, W]

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]
        aoi_mask = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
        forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
        phase_mask = torch.from_numpy(phase_f.mask).all(dim=0)

        m = torch.from_numpy(enc_f.mask) & phase_mask & aoi_mask & forest_mask  # [H, W]

        with torch.no_grad():
            z_type = model(enc)                                   # [1, 64, H, W]
            z_phase = model.forward_phase(phase, z_type.detach()) # [1, 12, T, H, W]

        T = phase.shape[2]
        H, W = m.shape
        m_flat = m.reshape(-1)  # [H*W]

        # [H*W, D] → valid pixels
        z_type_pix = z_type[0].permute(1, 2, 0).reshape(-1, z_type.shape[1])
        z_phase_pix = z_phase[0].permute(2, 3, 0, 1).reshape(-1, T, z_phase.shape[1])
        ysfc_pix = ysfc.permute(1, 2, 0).reshape(-1, T)

        z_type_v = z_type_pix[m_flat].cpu()
        z_phase_v = z_phase_pix[m_flat].cpu()
        ysfc_v = ysfc_pix[m_flat]

        if z_type_v.shape[0] == 0:
            continue

        phase_summary, temporal_var = _compute_phase_summary(z_phase_v, ysfc_v)
        combined = torch.cat([z_type_v, phase_summary, temporal_var.unsqueeze(1)], dim=1)
        all_pixels.append(combined.numpy().astype(np.float32))

    if not all_pixels:
        return None
    return np.concatenate(all_pixels, axis=0)


# ---------------------------------------------------------------------------
# GMM sweep helpers (silhouette for type, BIC for phase)
# ---------------------------------------------------------------------------

def _bic_sweep(
    X: np.ndarray,
    k_values: List[int],
    covariance_type: str,
    n_init_sweep: int,
    n_init_final: int,
    max_iter: int,
    seed: int,
) -> Tuple[int, object, Dict[int, float]]:
    """Fit GMMs for each K in k_values, return (best_k, best_gmm, bic_dict).

    The sweep uses n_init=1 for speed; the winner is refit with n_init_final.
    """
    from sklearn.mixture import GaussianMixture

    bic_dict: Dict[int, float] = {}
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init_sweep,
            max_iter=max_iter,
            random_state=seed,
        )
        gmm.fit(X)
        bic_dict[k] = float(gmm.bic(X))
        logger.debug(f"  K={k:3d}  BIC={bic_dict[k]:.1f}  converged={gmm.converged_}")

    best_k = min(bic_dict, key=bic_dict.__getitem__)

    logger.info(f"BIC selected K={best_k} (BIC={bic_dict[best_k]:.1f}); refitting...")
    best_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=covariance_type,
        n_init=n_init_final,
        max_iter=max_iter,
        random_state=seed,
    )
    best_gmm.fit(X)
    if not best_gmm.converged_:
        logger.warning(f"  Final GMM (K={best_k}) did not converge.")
    return best_k, best_gmm, bic_dict


def _silhouette_sweep(
    X: np.ndarray,
    k_values: List[int],
    covariance_type: str,
    n_init_sweep: int,
    n_init_final: int,
    max_iter: int,
    seed: int,
    n_silhouette_samples: int = 20_000,
) -> Tuple[int, object, Dict[int, float]]:
    """Fit GMMs for each K, score by average silhouette; return (best_k, best_gmm, scores).

    Average silhouette measures cluster *separation* rather than model fit, so it
    naturally penalises both under-clustering (poor separation) and over-clustering
    (many points near cluster boundaries).  A score near 1 = well-separated; near 0
    = overlapping clusters.

    Silhouette is O(N²) — we subsample to n_silhouette_samples for speed.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    idx_sil = rng.choice(len(X), size=min(n_silhouette_samples, len(X)), replace=False)
    X_sil = X[idx_sil]

    scores: Dict[int, float] = {}
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init_sweep,
            max_iter=max_iter,
            random_state=seed,
        )
        gmm.fit(X)
        labels_sil = gmm.predict(X_sil)
        if len(np.unique(labels_sil)) < 2:
            scores[k] = -1.0  # degenerate: all points in one cluster
        else:
            scores[k] = float(silhouette_score(X_sil, labels_sil, metric="euclidean"))
        logger.info(f"  K={k:3d}  silhouette={scores[k]:.4f}  converged={gmm.converged_}")

    best_k = max(scores, key=scores.__getitem__)

    logger.info(f"Silhouette selected K={best_k} (score={scores[best_k]:.4f}); refitting...")
    best_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=covariance_type,
        n_init=n_init_final,
        max_iter=max_iter,
        random_state=seed,
    )
    best_gmm.fit(X)
    if not best_gmm.converged_:
        logger.warning(f"  Final GMM (K={best_k}) did not converge.")
    return best_k, best_gmm, scores


def _save_score_plot(
    scores: Dict[int, float],
    best_k: int,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")
        return

    ks = sorted(scores)
    vals = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, vals, "o-", ms=4, lw=1.5)
    ax.axvline(best_k, color="red", ls="--", lw=1, label=f"K*={best_k}")
    ax.set_xlabel("K (number of components)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_bic_plot(
    bic_dict: Dict[int, float],
    best_k: int,
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping BIC plot.")
        return

    ks = sorted(bic_dict)
    bics = [bic_dict[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, bics, "o-", ms=4, lw=1.5)
    ax.axvline(best_k, color="red", ls="--", lw=1, label=f"K*={best_k}")
    ax.set_xlabel("K (number of components)")
    ax.set_ylabel("BIC")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Variability diagnostic
# ---------------------------------------------------------------------------

def _compute_variability_stats(
    X_phase: np.ndarray,
    temporal_var: np.ndarray,
    type_labels: np.ndarray,
    n_type: int,
) -> Dict[int, dict]:
    """Per-cluster temporal and spatial variability stats.

    temporal_var[n]: mean-across-channels variance of z_phase over T timesteps for pixel n.
    X_phase[n, 24:36]: overall_mean block — spatial spread captures between-pixel diversity.

    temporal_fraction = var_temporal / (var_temporal + var_spatial)
        → 1.0: dynamics are mostly within-pixel temporal change
        → 0.0: pixels are stable but spatially heterogeneous within the cluster
    """
    overall_mean_block = X_phase[:, 24:36]  # [N, 12]
    stats: Dict[int, dict] = {}
    for k in range(n_type):
        mask = type_labels == k
        tv = temporal_var[mask]
        om = overall_mean_block[mask]
        var_t = float(tv.mean()) if len(tv) > 0 else 0.0
        var_s = float(om.var(axis=0).mean()) if len(om) > 1 else 0.0
        denom = var_t + var_s
        stats[k] = {
            "n_pixels": int(mask.sum()),
            "mean_temporal_var": var_t,
            "median_temporal_var": float(np.median(tv)) if len(tv) > 0 else 0.0,
            "q25_temporal_var": float(np.percentile(tv, 25)) if len(tv) > 0 else 0.0,
            "q75_temporal_var": float(np.percentile(tv, 75)) if len(tv) > 0 else 0.0,
            "q90_temporal_var": float(np.percentile(tv, 90)) if len(tv) > 0 else 0.0,
            "spatial_spread": var_s,
            "temporal_fraction": float(var_t / denom) if denom > 0 else 0.0,
        }
    return stats


def _generate_variability_diagnostic(
    temporal_var: np.ndarray,
    type_labels: np.ndarray,
    phase_labels: Dict[int, np.ndarray],
    var_stats: Dict[int, dict],
    taxonomy: Dict[int, dict],
    out_dir: Path,
) -> None:
    """Save variability CSV and figures.

    type_variability.png — violin plots of per-pixel temporal_var per type cluster,
        sorted by temporal_fraction.  Clusters whose phase GMM split them into
        K_phase > 1 sub-groups show each sub-group overlaid in colour.

    variability_summary.csv — one row per type cluster with variability stats.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available; skipping diagnostic plots.")
        return

    # --- CSV ------------------------------------------------------------------
    with open(out_dir / "variability_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster", "n_pixels", "k_phase", "is_dynamic",
            "mean_temporal_var", "median_temporal_var",
            "q25_temporal_var", "q75_temporal_var", "q90_temporal_var",
            "spatial_spread", "temporal_fraction",
        ])
        for k, s in var_stats.items():
            t = taxonomy.get(k, {})
            writer.writerow([
                k, s["n_pixels"], t.get("k_phase", 1), t.get("is_dynamic", False),
                f"{s['mean_temporal_var']:.6f}", f"{s['median_temporal_var']:.6f}",
                f"{s['q25_temporal_var']:.6f}", f"{s['q75_temporal_var']:.6f}",
                f"{s['q90_temporal_var']:.6f}",
                f"{s['spatial_spread']:.6f}", f"{s['temporal_fraction']:.4f}",
            ])
    logger.info("Saved variability_summary.csv")

    # --- Sort clusters by temporal_fraction -----------------------------------
    order = sorted(var_stats, key=lambda k: var_stats[k]["temporal_fraction"])
    n_clusters = len(order)

    # Colour palette for phase sub-clusters (up to 5)
    phase_colours = ["#4e9af1", "#f4a261", "#2a9d8f", "#e76f51", "#8ecae6"]

    # --- Figure: one violin per type cluster, before/after overlay -----------
    fig_h = max(4, n_clusters * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    yticks, yticklabels = [], []

    for row_idx, k in enumerate(order):
        tv_k = temporal_var[type_labels == k]
        tf = var_stats[k]["temporal_fraction"]
        k_phase = taxonomy.get(k, {}).get("k_phase", 1)

        # Background violin — full cluster (grey)
        vp = ax.violinplot(
            [tv_k], positions=[row_idx], vert=False,
            showmedians=True, showextrema=False, widths=0.7,
        )
        for pc in vp["bodies"]:
            pc.set_facecolor("#cccccc")
            pc.set_alpha(0.5)
        vp["cmedians"].set_color("#888888")

        # Phase sub-cluster violins (coloured, narrower)
        ph_labels_k = phase_labels.get(k)
        if ph_labels_k is not None and k_phase > 1:
            for j in range(k_phase):
                tv_kj = tv_k[ph_labels_k == j]
                if len(tv_kj) < 5:
                    continue
                vp2 = ax.violinplot(
                    [tv_kj], positions=[row_idx], vert=False,
                    showmedians=True, showextrema=False, widths=0.5,
                )
                col = phase_colours[j % len(phase_colours)]
                for pc in vp2["bodies"]:
                    pc.set_facecolor(col)
                    pc.set_alpha(0.6)
                vp2["cmedians"].set_color(col)

        label = f"C{k}  tf={tf:.2f}  K_φ={k_phase}"
        yticks.append(row_idx)
        yticklabels.append(label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_xlabel("Per-pixel temporal variance of z_phase  (mean over channels)")
    ax.set_title(
        "Interannual variability by type cluster\n"
        "Grey = full cluster · Coloured = phase sub-clusters (sorted by temporal fraction ↑)"
    )

    # Legend
    legend_elements = [Patch(facecolor="#cccccc", alpha=0.6, label="Full type cluster")]
    for j in range(min(3, max((taxonomy.get(k, {}).get("k_phase", 1) for k in order), default=1))):
        legend_elements.append(Patch(facecolor=phase_colours[j], alpha=0.7, label=f"Phase sub-cluster {j}"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "type_variability.png", dpi=150)
    plt.close(fig)
    logger.info("Saved type_variability.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical GMM categorization: forest type × recovery phase."
    )
    parser.add_argument("--checkpoint", required=True, help="RepresentationModel checkpoint (.pt)")
    parser.add_argument("--training", default="config/frl_training_v1.yaml",
                        help="Training config YAML")
    parser.add_argument("--bindings", default="config/frl_binding_v1.yaml",
                        help="Bindings config YAML (default: config/frl_binding_v1.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <checkpoint_dir>/taxonomy/)")
    parser.add_argument("--k-type-min", type=int, default=5)
    parser.add_argument("--k-type-max", type=int, default=50)
    parser.add_argument("--k-type-step", type=int, default=5)
    parser.add_argument("--k-phase-max", type=int, default=5,
                        help="Hard cap on phase sub-categories (default: 5)")
    parser.add_argument("--covariance-type", default="diag",
                        choices=["diag", "full", "tied", "spherical"])
    parser.add_argument("--max-pixels", type=int, default=500_000,
                        help="Reservoir capacity for type GMM fitting (default: 500000)")
    parser.add_argument("--min-cluster-pixels", type=int, default=1_000,
                        help="Skip phase GMM for clusters smaller than this (default: 1000)")
    parser.add_argument("--dynamic-var-quantile", type=int, default=90,
                        help="Percentile of temporal_var used to assess tail activity (default: 90)")
    parser.add_argument("--dynamic-var-threshold", type=float, default=0.25,
                        help="A type is dynamic if its --dynamic-var-quantile exceeds this (default: 0.25)")
    parser.add_argument("--n-init", type=int, default=3,
                        help="n_init for final GMM refits (default: 3)")
    parser.add_argument("--n-init-sweep", type=int, default=1,
                        help="n_init during BIC sweep (default: 1)")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # --- Config ---------------------------------------------------------------
    training_config = TrainingConfigParser(args.training).parse()

    logger.info(f"Bindings: {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    batch_size = args.batch_size or training_config.training.batch_size
    patch_size = training_config.sampling.patch_size
    enc_feature_name = training_config.model_input.type_encoder_feature

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}  batch_size: {batch_size}  output: {out_dir}")

    # --- Dataset --------------------------------------------------------------
    train_dataset = ForestDatasetV2(
        bindings_config, split="train", patch_size=patch_size, min_aoi_fraction=0.3,
    )
    logger.info(f"Train dataset: {len(train_dataset)} patches")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    feature_builder = FeatureBuilder(bindings_config)

    # --- Model ----------------------------------------------------------------
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)
    z_type_dim = model.z_type_dim      # 64
    z_phase_dim = model.z_phase_dim    # 12
    combined_dim = z_type_dim + 3 * z_phase_dim + 1  # 64 + 36 + 1 = 101
    logger.info(f"z_type_dim={z_type_dim}  z_phase_dim={z_phase_dim}  encoder_feature='{enc_feature_name}'")

    # --- Reservoir sampling pass ----------------------------------------------
    logger.info(f"Collecting up to {args.max_pixels:,} pixels (z_type + phase_summary)...")
    sampler = ReservoirSampler(capacity=args.max_pixels, dim=combined_dim, seed=args.seed)

    for batch_idx, batch in enumerate(train_loader):
        pixels = extract_batch(batch, feature_builder, model, device, enc_feature_name)
        if pixels is not None:
            sampler.add(pixels)

        if batch_idx % 50 == 0:
            logger.info(
                f"  Batch {batch_idx:4d}/{len(train_loader)}  "
                f"seen={sampler.n_seen:,}  buffer={sampler.filled:,}"
            )

    buf = sampler.get()                          # [N, 101]
    X_type = buf[:, :z_type_dim]                 # [N, 64]
    X_phase = buf[:, z_type_dim:z_type_dim + 3 * z_phase_dim]  # [N, 36]
    temporal_var = buf[:, -1]                    # [N]  per-pixel temporal variance
    logger.info(f"Reservoir: {buf.shape[0]:,} pixels  (total seen: {sampler.n_seen:,})")

    # --- Type GMM silhouette sweep --------------------------------------------
    k_type_values = list(range(args.k_type_min, args.k_type_max + 1, args.k_type_step))
    if not k_type_values:
        raise ValueError(f"Empty k-type range: [{args.k_type_min}, {args.k_type_max}, {args.k_type_step}]")

    logger.info(f"Type silhouette sweep: K ∈ {k_type_values}")
    k_type_star, gmm_type, type_silhouette = _silhouette_sweep(
        X_type, k_type_values,
        covariance_type=args.covariance_type,
        n_init_sweep=args.n_init_sweep,
        n_init_final=args.n_init,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    logger.info(f"Type GMM: K*={k_type_star}  converged={gmm_type.converged_}")

    _save_score_plot(type_silhouette, k_type_star,
                     "Average silhouette score", f"Type silhouette sweep (K*={k_type_star})",
                     out_dir / "silhouette_curve_type.png")

    type_gmm_payload = {
        "gmm": gmm_type,
        "n_components": k_type_star,
        "covariance_type": args.covariance_type,
        "z_type_dim": z_type_dim,
        "n_pixels_fit": X_type.shape[0],
        "n_pixels_seen": sampler.n_seen,
        "silhouette": float(type_silhouette[k_type_star]),
        "silhouette_curve": type_silhouette,
        "converged": bool(gmm_type.converged_),
        "encoder_checkpoint": str(args.checkpoint),
        "seed": args.seed,
    }
    with open(out_dir / "type_gmm.pkl", "wb") as f:
        pickle.dump(type_gmm_payload, f, protocol=5)
    logger.info(f"Saved type_gmm.pkl  (K={k_type_star})")

    # --- Assign type labels ---------------------------------------------------
    type_labels = gmm_type.predict(X_type)  # [N]  int array
    cluster_sizes = {k: int((type_labels == k).sum()) for k in range(k_type_star)}
    logger.info("Cluster sizes: " + "  ".join(f"C{k}:{sz}" for k, sz in cluster_sizes.items()))

    # --- Variability stats (before phase discretization) ----------------------
    var_stats = _compute_variability_stats(X_phase, temporal_var, type_labels, k_type_star)

    # --- Phase GMM per type cluster -------------------------------------------
    # A type is dynamic if the --dynamic-var-quantile of its per-pixel temporal_var
    # exceeds --dynamic-var-threshold.  Stable types get K_phase=1 (no GMM).
    # Dynamic types: silhouette sweep over K=2,3,4 to pick K_phase.
    k_phase_dynamic = [2, 3, 4]
    taxonomy: Dict[int, dict] = {}
    phase_labels: Dict[int, np.ndarray] = {}
    phase_gmms: Dict[int, object] = {}  # fitted GMM for each dynamic cluster

    logger.info(
        f"Phase classification: dynamic if q{args.dynamic_var_quantile}(temporal_var) "
        f"> {args.dynamic_var_threshold}"
    )

    for k in range(k_type_star):
        sel = X_phase[type_labels == k]      # [n_k, 36]
        tv_k = temporal_var[type_labels == k]  # [n_k]
        n_k = sel.shape[0]
        q_tail = float(np.percentile(tv_k, args.dynamic_var_quantile)) if n_k > 0 else 0.0
        is_dynamic = q_tail > args.dynamic_var_threshold

        logger.info(
            f"  Cluster {k}: n={n_k}  "
            f"q{args.dynamic_var_quantile}_temporal_var={q_tail:.4f}  "
            f"dynamic={is_dynamic}"
        )

        if n_k < args.min_cluster_pixels or not is_dynamic:
            reason = "too few pixels" if n_k < args.min_cluster_pixels else "stable"
            logger.info(f"    → K_phase=1 ({reason})")
            taxonomy[k] = {
                "n_type_pixels": n_k,
                "k_phase": 1,
                "is_dynamic": False,
                f"q{args.dynamic_var_quantile}_temporal_var": q_tail,
                "phase_gmm_skipped": True,
            }
            continue

        # Dynamic: silhouette sweep over K=2,3,4
        k_phase_available = [kp for kp in k_phase_dynamic if kp <= n_k]
        if not k_phase_available:
            k_phase_available = [2]

        k_phase_star, gmm_phase, phase_sil = _silhouette_sweep(
            sel, k_phase_available,
            covariance_type=args.covariance_type,
            n_init_sweep=args.n_init_sweep,
            n_init_final=args.n_init,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        logger.info(
            f"    → K_phase={k_phase_star} "
            f"(silhouette={phase_sil[k_phase_star]:.4f})  converged={gmm_phase.converged_}"
        )

        phase_labels[k] = gmm_phase.predict(sel)
        phase_gmms[k] = gmm_phase

        _save_score_plot(
            phase_sil, k_phase_star,
            "Average silhouette score",
            f"Cluster {k} phase silhouette (K_phase*={k_phase_star})",
            out_dir / f"silhouette_curve_phase_{k}.png",
        )

        phase_gmm_payload = {
            "gmm": gmm_phase,
            "type_cluster": k,
            "k_phase": k_phase_star,
            "is_dynamic": True,
            "covariance_type": args.covariance_type,
            "z_type_dim": z_type_dim,
            "z_phase_dim": z_phase_dim,
            "phase_summary_dim": 3 * z_phase_dim,
            "n_pixels_fit": n_k,
            "silhouette": float(phase_sil[k_phase_star]),
            "silhouette_curve": phase_sil,
            "converged": bool(gmm_phase.converged_),
            f"q{args.dynamic_var_quantile}_temporal_var": q_tail,
        }
        with open(out_dir / f"phase_gmm_{k}.pkl", "wb") as f:
            pickle.dump(phase_gmm_payload, f, protocol=5)

        taxonomy[k] = {
            "n_type_pixels": n_k,
            "k_phase": k_phase_star,
            "is_dynamic": True,
            f"q{args.dynamic_var_quantile}_temporal_var": q_tail,
            "phase_silhouette": float(phase_sil[k_phase_star]),
            "phase_gmm_skipped": False,
        }

    # --- Taxonomy summary -----------------------------------------------------
    n_dynamic = sum(1 for v in taxonomy.values() if v["is_dynamic"])
    n_nondynamic = k_type_star - n_dynamic
    logger.info(f"Taxonomy: {k_type_star} type clusters  "
                f"({n_dynamic} dynamic, {n_nondynamic} non-dynamic)")

    taxonomy_out = {
        "k_type": k_type_star,
        "n_dynamic_clusters": n_dynamic,
        "n_nondynamic_clusters": n_nondynamic,
        "dynamic_var_quantile": args.dynamic_var_quantile,
        "dynamic_var_threshold": args.dynamic_var_threshold,
        "low_ysfc_max": LOW_YSFC_MAX,
        "high_ysfc_min": HIGH_YSFC_MIN,
        "encoder_checkpoint": str(args.checkpoint),
        "covariance_type": args.covariance_type,
        "n_pixels_fit": int(X_type.shape[0]),
        "clusters": {str(k): v for k, v in taxonomy.items()},
    }
    with open(out_dir / "taxonomy.json", "w") as f:
        json.dump(taxonomy_out, f, indent=2)
    logger.info(f"Saved taxonomy.json → {out_dir / 'taxonomy.json'}")

    # Summary table
    logger.info("\nFinal taxonomy:")
    q_key = f"q{args.dynamic_var_quantile}_temporal_var"
    logger.info(f"  {'Cluster':>7}  {'N pixels':>10}  {'K_phase':>7}  {'Dynamic':>7}  {f'q{args.dynamic_var_quantile}_tvar':>12}")
    for k in range(k_type_star):
        t = taxonomy[k]
        logger.info(f"  {k:>7}  {t['n_type_pixels']:>10,}  {t['k_phase']:>7}  "
                    f"{'yes' if t['is_dynamic'] else 'no':>7}  {t.get(q_key, 0):.4f}")

    # --- Variability diagnostic -----------------------------------------------
    _generate_variability_diagnostic(
        temporal_var=temporal_var,
        type_labels=type_labels,
        phase_labels=phase_labels,
        var_stats=var_stats,
        taxonomy=taxonomy,
        out_dir=out_dir,
    )

    # --- ysfc-by-phase diagnostic (second streaming pass) --------------------
    _run_ysfc_diagnostic(
        train_loader=train_loader,
        feature_builder=feature_builder,
        model=model,
        device=device,
        enc_feature_name=enc_feature_name,
        gmm_type=gmm_type,
        phase_gmms=phase_gmms,
        taxonomy=taxonomy,
        z_type_dim=z_type_dim,
        z_phase_dim=z_phase_dim,
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# ysfc-by-phase diagnostic (second pass)
# ---------------------------------------------------------------------------

def _run_ysfc_diagnostic(
    train_loader: DataLoader,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    enc_feature_name: str,
    gmm_type,
    phase_gmms: Dict[int, object],
    taxonomy: Dict[int, dict],
    z_type_dim: int,
    z_phase_dim: int,
    out_dir: Path,
    max_per_group: int = 20_000,
) -> None:
    """Second streaming pass: collect ysfc values per (type_cluster, phase_sub_cluster).

    For each dynamic type cluster k and phase sub-cluster j, accumulates the ysfc
    value of every valid (pixel, timestep) whose pixel is assigned to (k, j).
    Produces a grid figure: rows = dynamic clusters, columns = phase sub-clusters,
    each cell a violin of ysfc (years since fast change).

    ysfc = 0  →  disturbance year
    ysfc = 1  →  one year post-disturbance
    ysfc ≥ 5  →  model considers "recovered"
    NaN       →  no disturbance observed in study window (stable/never disturbed)
    """
    dynamic_clusters = {k: v for k, v in taxonomy.items() if v["is_dynamic"]}
    if not dynamic_clusters:
        logger.info("No dynamic clusters — skipping ysfc diagnostic.")
        return

    # Storage: (cluster_k, phase_j) → list of valid ysfc floats
    ysfc_store: Dict[Tuple[int, int], List[float]] = {
        (k, j): [] for k, t in dynamic_clusters.items() for j in range(t["k_phase"])
    }
    # Also track total pixels per group to report NaN fraction
    n_pixels_store: Dict[Tuple[int, int], int] = {key: 0 for key in ysfc_store}

    logger.info("ysfc diagnostic: second pass through training data...")

    for batch_idx, batch in enumerate(train_loader):
        for i in range(len(batch["metadata"])):
            sample = {
                key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
                for key, val in batch.items()
                if key != "metadata"
            }
            sample["metadata"] = batch["metadata"][i]

            enc_f   = feature_builder.build_feature(enc_feature_name, sample)
            phase_f = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
            ysfc_f  = feature_builder.build_feature(YSFC_FEATURE, sample)

            enc   = torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)
            phase = torch.from_numpy(phase_f.data).float().unsqueeze(0).to(device)

            ysfc_data = ysfc_f.data[0].astype(np.float32)
            ysfc_data = np.where(ysfc_f.mask, ysfc_data, np.nan)
            ysfc_t = torch.from_numpy(ysfc_data).float()   # [T, H, W]

            sm_names    = sample["metadata"]["channel_names"]["static_mask"]
            sm_data     = sample["static_mask"]
            aoi_mask    = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
            forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
            phase_mask  = torch.from_numpy(phase_f.mask).all(dim=0)
            m = torch.from_numpy(enc_f.mask) & phase_mask & aoi_mask & forest_mask

            with torch.no_grad():
                z_type = model(enc)   # [1, 64, H, W]

            T = phase.shape[2]
            m_flat = m.reshape(-1)

            z_type_pix = z_type[0].permute(1, 2, 0).reshape(-1, z_type_dim).cpu().numpy()
            z_type_v   = z_type_pix[m_flat.numpy()]   # [N_valid, 64]

            if z_type_v.shape[0] == 0:
                continue

            type_labels_v = gmm_type.predict(z_type_v)   # [N_valid]
            has_dynamic   = any(taxonomy.get(int(k), {}).get("is_dynamic", False)
                                for k in type_labels_v)
            if not has_dynamic:
                continue

            # Run phase encoder for all valid pixels (one patch at a time)
            with torch.no_grad():
                z_phase = model.forward_phase(phase, z_type.detach())   # [1, 12, T, H, W]

            z_phase_pix = z_phase[0].permute(2, 3, 0, 1).reshape(-1, T, z_phase_dim).cpu()
            ysfc_pix    = ysfc_t.permute(1, 2, 0).reshape(-1, T)        # [H*W, T]

            z_phase_v = z_phase_pix[m_flat]    # [N_valid, T, 12]
            ysfc_v    = ysfc_pix[m_flat]       # [N_valid, T]

            # Process each dynamic cluster's pixels in one vectorised block
            for k, t in dynamic_clusters.items():
                cluster_mask = type_labels_v == k
                if not cluster_mask.any():
                    continue

                zp_k   = z_phase_v[cluster_mask]   # [N_k, T, 12]
                yf_k   = ysfc_v[cluster_mask]      # [N_k, T]

                phase_summary_k, _ = _compute_phase_summary(zp_k, yf_k)  # [N_k, 36]
                phase_labels_k     = phase_gmms[k].predict(phase_summary_k.numpy())

                for j in range(t["k_phase"]):
                    key = (k, j)
                    pix_j = phase_labels_k == j
                    n_pixels_store[key] += int(pix_j.sum())

                    # Collect valid (non-NaN) ysfc timestep values
                    store = ysfc_store[key]
                    if len(store) >= max_per_group:
                        continue
                    yf_j = yf_k[pix_j]   # [N_j, T]
                    for n in range(yf_j.shape[0]):
                        if len(store) >= max_per_group:
                            break
                        valid_vals = yf_j[n][yf_j[n].isfinite()].tolist()
                        store.extend(valid_vals)

        if batch_idx % 50 == 0:
            logger.info(f"  ysfc pass: batch {batch_idx}/{len(train_loader)}")

    # --- Plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping ysfc plot.")
        return

    n_dyn     = len(dynamic_clusters)
    k_phase_max = max(t["k_phase"] for t in dynamic_clusters.values())
    phase_colours = ["#4e9af1", "#f4a261", "#2a9d8f", "#e76f51"]

    fig, axes = plt.subplots(
        n_dyn, k_phase_max,
        figsize=(3 * k_phase_max, 2.5 * n_dyn),
        squeeze=False,
        sharey=True,
    )

    for row, (k, t) in enumerate(sorted(dynamic_clusters.items())):
        k_phase = t["k_phase"]
        for j in range(k_phase_max):
            ax = axes[row, j]
            if j >= k_phase:
                ax.set_visible(False)
                continue

            key   = (k, j)
            vals  = np.array(ysfc_store[key], dtype=np.float32)
            n_pix = n_pixels_store[key]
            n_valid_ysfc = int(np.isfinite(vals).sum()) if len(vals) > 0 else 0
            nan_frac = 1.0 - n_valid_ysfc / max(len(vals), 1)

            if len(vals) >= 5:
                vp = ax.violinplot(vals[np.isfinite(vals)], showmedians=True, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(phase_colours[j % len(phase_colours)])
                    pc.set_alpha(0.7)
                vp["cmedians"].set_color("black")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

            ax.set_title(f"C{k} · Phase {j}\nn={n_pix:,}  NaN={nan_frac:.0%}",
                         fontsize=8)
            ax.set_xlabel("ysfc" if row == n_dyn - 1 else "")
            ax.set_ylabel("years since\nfast change" if j == 0 else "")
            ax.set_xticks([])

    fig.suptitle(
        "ysfc distribution by type cluster and phase sub-cluster\n"
        "(NaN = pixel never disturbed in study window)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ysfc_by_phase.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ysfc_by_phase.png")


if __name__ == "__main__":
    main()

