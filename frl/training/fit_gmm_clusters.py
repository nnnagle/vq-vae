#!/usr/bin/env python3
"""
Fit a Gaussian Mixture Model on z_type embeddings extracted from a trained
RepresentationModel checkpoint.

What it does
------------
- Loads a frozen encoder from a checkpoint.
- Streams the TRAIN split through the type pathway to extract per-pixel z_type
  embeddings (shape [B, 64, H, W]).
- Applies AOI + forest masks, then reservoir-samples up to --max-pixels valid
  pixels into an in-memory buffer.
- Fits sklearn GaussianMixture (diagonal covariance by default) on the buffer.
- Reports BIC, AIC, and per-component weight summary.
- Saves the fitted GMM and metadata to a .pkl file.

Usage
-----
    python training/fit_gmm_clusters.py \\
        --checkpoint runs/checkpoints/model.pt \\
        --training   config/frl_training_v1.yaml \\
        --bindings   config/frl_binding_v1.yaml

    # Override defaults
    python training/fit_gmm_clusters.py \\
        --checkpoint runs/checkpoints/model.pt \\
        --n-components 50 \\
        --covariance-type diag \\
        --max-pixels 500000 \\
        --n-init 5 \\
        --seed 42 \\
        --output runs/gmm/gmm_k50.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
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
logger = logging.getLogger("fit_gmm_clusters")


# ---------------------------------------------------------------------------
# Reservoir sampling (Algorithm R — Vitter 1985)
# Maintains a uniform random sample of exactly `capacity` items from a stream
# of unknown length without storing the full stream.
# ---------------------------------------------------------------------------

class ReservoirSampler:
    """Fixed-capacity uniform random sample from a stream.

    Args:
        capacity: Maximum number of items to keep.
        dim: Feature dimension of each item.
        seed: Random seed for reproducibility.
    """

    def __init__(self, capacity: int, dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty((capacity, dim), dtype=np.float32)
        self.n_seen = 0  # total items offered so far

    def add(self, vectors: np.ndarray) -> None:
        """Add a batch of vectors to the reservoir.

        Args:
            vectors: [N, dim] float32 array.
        """
        n = vectors.shape[0]
        for i in range(n):
            self.n_seen += 1
            if self.n_seen <= self.capacity:
                self.buffer[self.n_seen - 1] = vectors[i]
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.capacity:
                    self.buffer[j] = vectors[i]

    @property
    def filled(self) -> int:
        """Number of valid rows currently in the buffer."""
        return min(self.n_seen, self.capacity)

    def get(self) -> np.ndarray:
        """Return the sampled buffer as [filled, dim]."""
        return self.buffer[: self.filled]


# ---------------------------------------------------------------------------
# Batch extraction — mirrors fit_linear_probe.py
# ---------------------------------------------------------------------------

def extract_embeddings_from_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
) -> np.ndarray | None:
    """Run the type pathway on one batch and return valid masked pixels.

    Args:
        batch: Collated batch dict from ForestDatasetV2.
        feature_builder: Builds normalised encoder inputs.
        model: Frozen RepresentationModel.
        device: Computation device.

    Returns:
        Float32 array of shape [N, z_type_dim] containing valid pixels,
        or None if the batch has no valid pixels.
    """
    batch_size = len(batch["metadata"])
    encoder_inputs = []
    masks = []

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f = feature_builder.build_feature("ccdc_history", sample)
        enc = torch.from_numpy(enc_f.data).float()

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]  # [C, H, W]
        aoi_idx = sm_names.index("aoi")
        forest_idx = sm_names.index("forest")
        aoi_mask = torch.from_numpy(sm_data[aoi_idx]).bool()
        forest_mask = torch.from_numpy(sm_data[forest_idx]).bool()

        m = torch.from_numpy(enc_f.mask) & aoi_mask & forest_mask  # [H, W]

        encoder_inputs.append(enc)
        masks.append(m)

    Ximg = torch.stack(encoder_inputs).to(device)  # [B, Cin, H, W]
    M = torch.stack(masks)                         # [B, H, W]

    with torch.no_grad():
        z = model(Ximg).cpu()  # [B, z_type_dim, H, W]

    # Flatten to pixels and apply mask
    B, D, H, W = z.shape
    z_perm = z.permute(0, 2, 3, 1).reshape(-1, D)   # [B*H*W, D]
    m_flat = M.reshape(-1)                           # [B*H*W]

    valid = z_perm[m_flat].numpy().astype(np.float32)
    return valid if valid.shape[0] > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit GMM on z_type embeddings from a trained RepresentationModel."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--training", default="config/frl_training_v1.yaml", help="Training config YAML")
    parser.add_argument("--bindings", default="config/frl_binding_v1.yaml", help="Bindings config YAML")
    parser.add_argument("--n-components", type=int, default=50, help="Number of GMM components (default: 50)")
    parser.add_argument("--covariance-type", default="diag",
                        choices=["diag", "full", "tied", "spherical"],
                        help="GMM covariance structure (default: diag)")
    parser.add_argument("--max-pixels", type=int, default=500_000,
                        help="Max pixels to reservoir-sample for fitting (default: 500000)")
    parser.add_argument("--n-init", type=int, default=3,
                        help="Number of random GMM initialisations (default: 3)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max EM iterations per init (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0, cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for .pkl file (default: <checkpoint_dir>/gmm_k<n>.pkl)")
    args = parser.parse_args()

    # --- Config ---
    logger.info(f"Loading bindings config: {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()

    logger.info(f"Loading training config: {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    batch_size = args.batch_size or training_config.training.batch_size
    patch_size = training_config.sampling.patch_size

    logger.info(f"Device: {device}  |  batch_size: {batch_size}  |  patch_size: {patch_size}")

    # --- Dataset (train split only) ---
    logger.info("Creating train dataset...")
    train_dataset = ForestDatasetV2(
        bindings_config,
        split="train",
        patch_size=patch_size,
        min_aoi_fraction=0.3,
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

    # --- Model ---
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)
    z_dim = model.z_type_dim
    logger.info(f"z_type_dim = {z_dim}")

    # --- Reservoir sampling pass ---
    logger.info(f"Reservoir sampling up to {args.max_pixels:,} valid pixels...")
    sampler = ReservoirSampler(capacity=args.max_pixels, dim=z_dim, seed=args.seed)

    for batch_idx, batch in enumerate(train_loader):
        valid_pixels = extract_embeddings_from_batch(batch, feature_builder, model, device)
        if valid_pixels is not None:
            sampler.add(valid_pixels)

        if batch_idx % 50 == 0:
            logger.info(
                f"  Batch {batch_idx:4d}/{len(train_loader)}  |  "
                f"pixels seen: {sampler.n_seen:,}  |  buffer filled: {sampler.filled:,}"
            )

        # Once we've seen enough pixels to fill the reservoir many times over,
        # further streaming only makes small corrections. Stop early if the user
        # wants to limit runtime (currently we always run the full dataset for
        # an unbiased sample, but log progress so the user can interrupt).

    X = sampler.get()
    logger.info(f"Reservoir complete: {X.shape[0]:,} pixels sampled  (total seen: {sampler.n_seen:,})")

    if X.shape[0] < args.n_components:
        raise RuntimeError(
            f"Only {X.shape[0]} valid pixels collected, but n_components={args.n_components}. "
            "Lower --n-components or check masks."
        )

    # --- Fit GMM ---
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    logger.info(
        f"Fitting GaussianMixture(n_components={args.n_components}, "
        f"covariance_type='{args.covariance_type}', "
        f"n_init={args.n_init}, max_iter={args.max_iter}, "
        f"random_state={args.seed}) on {X.shape[0]:,} pixels..."
    )
    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=1,
        verbose_interval=10,
    )
    gmm.fit(X)

    if not gmm.converged_:
        logger.warning("GMM did not converge — consider increasing --max-iter or --n-init.")
    else:
        logger.info("GMM converged.")

    # --- Diagnostics ---
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    logger.info(f"BIC: {bic:.1f}  |  AIC: {aic:.1f}")

    weights = gmm.weights_  # [K]
    sizes = (weights * X.shape[0]).astype(int)
    logger.info("Component weight summary (sorted by weight, descending):")
    order = np.argsort(weights)[::-1]
    for rank, k in enumerate(order[:10]):
        logger.info(f"  rank {rank+1:2d}  component {k:3d}  weight={weights[k]:.4f}  "
                    f"~{sizes[k]:,} pixels")
    if args.n_components > 10:
        smallest = order[-5:]
        logger.info("  ... smallest 5 components:")
        for k in smallest:
            logger.info(f"       component {k:3d}  weight={weights[k]:.4f}  ~{sizes[k]:,} pixels")

    # --- Save ---
    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_dir = Path(args.checkpoint).parent
        out_path = ckpt_dir / f"gmm_k{args.n_components}_{args.covariance_type}.pkl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gmm": gmm,
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
        "z_type_dim": z_dim,
        "n_pixels_fit": X.shape[0],
        "n_pixels_seen": sampler.n_seen,
        "bic": bic,
        "aic": aic,
        "converged": bool(gmm.converged_),
        "encoder_checkpoint": str(args.checkpoint),
        "seed": args.seed,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=5)
    logger.info(f"Saved GMM to {out_path}")


if __name__ == "__main__":
    main()
