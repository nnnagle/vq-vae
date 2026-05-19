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
) -> torch.Tensor:
    """Compute 36-dim phase summary per pixel.

    Args:
        z_phase: [N, T, D]  D = z_phase_dim (12)
        ysfc:    [N, T]     float, NaN where ysfc is not observed

    Returns:
        [N, 36] = concat(disturbed_centroid, recovered_centroid, overall_mean)
        where each block is [N, D].  Pixels lacking disturbed or recovered
        timesteps fall back to overall_mean for that block.
    """
    overall_mean = z_phase.mean(dim=1)  # [N, D]

    def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
        # mask: [N, T] bool
        w = mask.float().unsqueeze(-1)          # [N, T, 1]
        s = (z_phase * w).sum(dim=1)            # [N, D]
        c = w.sum(dim=1).clamp(min=1.0)         # [N, 1]
        centroid = s / c                         # [N, D]
        has_any = mask.any(dim=1, keepdim=True)  # [N, 1]
        return torch.where(has_any, centroid, overall_mean)

    valid = ysfc.isfinite()
    disturbed_centroid = _masked_mean(valid & (ysfc <= LOW_YSFC_MAX))
    recovered_centroid = _masked_mean(valid & (ysfc >= HIGH_YSFC_MIN))

    return torch.cat([disturbed_centroid, recovered_centroid, overall_mean], dim=1)  # [N, 36]


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

    Returns:
        float32 array [N, 64+36] for valid pixels, or None if none.
    """
    batch_size = len(batch["metadata"])
    encoder_inputs: List[torch.Tensor] = []
    phase_inputs: List[torch.Tensor] = []
    ysfc_arrays: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f = feature_builder.build_feature(enc_feature_name, sample)
        phase_f = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
        ysfc_f = feature_builder.build_feature(YSFC_FEATURE, sample)

        enc = torch.from_numpy(enc_f.data).float()           # [C, H, W]
        phase = torch.from_numpy(phase_f.data).float()       # [8, T, H, W]

        # ysfc: [1, T, H, W]; mask invalid timesteps → NaN so isfinite() works later
        ysfc_data = ysfc_f.data[0].astype(np.float32)        # [T, H, W]
        ysfc_mask = ysfc_f.mask                               # [T, H, W] bool
        ysfc_data = np.where(ysfc_mask, ysfc_data, np.nan)   # NaN where unobserved
        ysfc_t = torch.from_numpy(ysfc_data).float()          # [T, H, W]

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]
        aoi_mask = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
        forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()

        # Collapse phase mask to spatial: valid if ALL timesteps valid (consistent
        # with how phase probe masks pixels — require complete temporal coverage).
        phase_mask_spatial = torch.from_numpy(phase_f.mask).all(dim=0)  # [H, W]

        m = (
            torch.from_numpy(enc_f.mask)
            & phase_mask_spatial
            & aoi_mask
            & forest_mask
        )

        encoder_inputs.append(enc)
        phase_inputs.append(phase)
        ysfc_arrays.append(ysfc_t)
        masks.append(m)

    Ximg = torch.stack(encoder_inputs).to(device)    # [B, C, H, W]
    Xphase = torch.stack(phase_inputs).to(device)    # [B, 8, T, H, W]
    Xysfc = torch.stack(ysfc_arrays)                 # [B, T, H, W]  (kept on CPU)
    M = torch.stack(masks)                            # [B, H, W]

    with torch.no_grad():
        z_type = model(Ximg)                                    # [B, 64, H, W]
        z_phase = model.forward_phase(Xphase, z_type.detach())  # [B, 12, T, H, W]

    B, D_type, H, W = z_type.shape
    _, D_phase, T, _, _ = z_phase.shape

    m_flat = M.reshape(-1)  # [B*H*W]

    # z_type: [B, D, H, W] → [B*H*W, D]
    z_type_pix = z_type.permute(0, 2, 3, 1).reshape(-1, D_type)

    # z_phase: [B, D, T, H, W] → [B, H, W, T, D] → [B*H*W, T, D]
    z_phase_pix = z_phase.permute(0, 3, 4, 2, 1).reshape(-1, T, D_phase)

    # ysfc: [B, T, H, W] → [B, H, W, T] → [B*H*W, T]
    ysfc_pix = Xysfc.permute(0, 2, 3, 1).reshape(-1, T)

    # Select valid pixels
    z_type_v = z_type_pix[m_flat]     # [N, 64]
    z_phase_v = z_phase_pix[m_flat]   # [N, T, 12]
    ysfc_v = ysfc_pix[m_flat]         # [N, T]

    if z_type_v.shape[0] == 0:
        return None

    phase_summary = _compute_phase_summary(z_phase_v.cpu(), ysfc_v)  # [N, 36]
    combined = torch.cat([z_type_v.cpu(), phase_summary], dim=1)     # [N, 100]
    return combined.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# BIC sweep helpers
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
# Dynamic-score diagnostic
# ---------------------------------------------------------------------------

def _dynamic_scores(
    X_phase_summary: np.ndarray,
    labels: np.ndarray,
    n_type: int,
) -> Dict[int, float]:
    """Per-cluster mean temporal spread in z_phase embedding.

    Uses the standard deviation across the phase_summary's overall_mean block
    (last 12 dims) as a proxy for temporal spread diversity within each cluster.
    Higher scores → more heterogeneous temporal dynamics.

    Returns dict: cluster_id → score.
    """
    scores: Dict[int, float] = {}
    overall_mean_block = X_phase_summary[:, 24:36]  # [N, 12] — the overall_mean part
    for k in range(n_type):
        sel = overall_mean_block[labels == k]
        scores[k] = float(np.std(sel, axis=0).mean()) if sel.shape[0] > 1 else 0.0
    return scores


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
    combined_dim = z_type_dim + 3 * z_phase_dim  # 64 + 36 = 100
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

    buf = sampler.get()                          # [N, 100]
    X_type = buf[:, :z_type_dim]                 # [N, 64]
    X_phase = buf[:, z_type_dim:]                # [N, 36]
    logger.info(f"Reservoir: {buf.shape[0]:,} pixels  (total seen: {sampler.n_seen:,})")

    # --- Type GMM BIC sweep ---------------------------------------------------
    k_type_values = list(range(args.k_type_min, args.k_type_max + 1, args.k_type_step))
    if not k_type_values:
        raise ValueError(f"Empty k-type range: [{args.k_type_min}, {args.k_type_max}, {args.k_type_step}]")

    logger.info(f"Type BIC sweep: K ∈ {k_type_values}")
    k_type_star, gmm_type, type_bic = _bic_sweep(
        X_type, k_type_values,
        covariance_type=args.covariance_type,
        n_init_sweep=args.n_init_sweep,
        n_init_final=args.n_init,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    logger.info(f"Type GMM: K*={k_type_star}  converged={gmm_type.converged_}")

    _save_bic_plot(type_bic, k_type_star, f"Type BIC sweep (K*={k_type_star})",
                   out_dir / "bic_curve_type.png")

    type_gmm_payload = {
        "gmm": gmm_type,
        "n_components": k_type_star,
        "covariance_type": args.covariance_type,
        "z_type_dim": z_type_dim,
        "n_pixels_fit": X_type.shape[0],
        "n_pixels_seen": sampler.n_seen,
        "bic": float(gmm_type.bic(X_type)),
        "aic": float(gmm_type.aic(X_type)),
        "bic_curve": type_bic,
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

    # --- Dynamic-score diagnostic ---------------------------------------------
    dyn_scores = _dynamic_scores(X_phase, type_labels, k_type_star)

    dyn_csv_path = out_dir / "dynamic_scores.csv"
    with open(dyn_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "n_pixels", "dynamic_score"])
        for k in range(k_type_star):
            writer.writerow([k, cluster_sizes[k], f"{dyn_scores[k]:.6f}"])
    logger.info(f"Saved dynamic_scores.csv")

    # --- Phase GMM per type cluster -------------------------------------------
    k_phase_values = list(range(1, args.k_phase_max + 1))  # 1..5
    taxonomy: Dict[int, dict] = {}

    for k in range(k_type_star):
        sel = X_phase[type_labels == k]   # [n_k, 36]
        n_k = sel.shape[0]
        logger.info(f"  Cluster {k}: n={n_k}  dynamic_score={dyn_scores[k]:.4f}")

        if n_k < args.min_cluster_pixels:
            logger.warning(f"    Too few pixels ({n_k} < {args.min_cluster_pixels}); assigning K_phase=1.")
            taxonomy[k] = {"n_type_pixels": n_k, "k_phase": 1, "is_dynamic": False,
                           "dynamic_score": dyn_scores[k], "phase_bic_skipped": True}
            continue

        # Limit K_phase candidates by data: can't have more components than pixels
        k_phase_available = [kp for kp in k_phase_values if kp <= n_k]
        if len(k_phase_available) < 2:
            k_phase_available = [1]

        k_phase_star, gmm_phase, phase_bic = _bic_sweep(
            sel, k_phase_available,
            covariance_type=args.covariance_type,
            n_init_sweep=args.n_init_sweep,
            n_init_final=args.n_init,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        is_dynamic = k_phase_star > 1
        logger.info(f"    K_phase*={k_phase_star}  is_dynamic={is_dynamic}  converged={gmm_phase.converged_}")

        _save_bic_plot(phase_bic, k_phase_star,
                       f"Cluster {k} phase BIC (K_phase*={k_phase_star}, dynamic={is_dynamic})",
                       out_dir / f"bic_curve_phase_{k}.png")

        phase_gmm_payload = {
            "gmm": gmm_phase,
            "type_cluster": k,
            "k_phase": k_phase_star,
            "is_dynamic": is_dynamic,
            "covariance_type": args.covariance_type,
            "z_type_dim": z_type_dim,
            "z_phase_dim": z_phase_dim,
            "phase_summary_dim": 3 * z_phase_dim,
            "n_pixels_fit": n_k,
            "bic": float(gmm_phase.bic(sel)),
            "bic_curve": phase_bic,
            "converged": bool(gmm_phase.converged_),
            "dynamic_score": dyn_scores[k],
        }
        with open(out_dir / f"phase_gmm_{k}.pkl", "wb") as f:
            pickle.dump(phase_gmm_payload, f, protocol=5)

        taxonomy[k] = {
            "n_type_pixels": n_k,
            "k_phase": k_phase_star,
            "is_dynamic": is_dynamic,
            "dynamic_score": dyn_scores[k],
            "phase_bic": float(gmm_phase.bic(sel)),
            "phase_bic_skipped": False,
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
        "k_phase_max": args.k_phase_max,
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
    logger.info(f"  {'Cluster':>7}  {'N pixels':>10}  {'K_phase':>7}  {'Dynamic':>7}  {'Dyn score':>10}")
    for k in range(k_type_star):
        t = taxonomy[k]
        logger.info(f"  {k:>7}  {t['n_type_pixels']:>10,}  {t['k_phase']:>7}  "
                    f"{'yes' if t['is_dynamic'] else 'no':>7}  {t['dynamic_score']:>10.4f}")


if __name__ == "__main__":
    main()
