# src/training/trainer.py
#
# High-level trainer for ConvVAE + PatchDataset.
#
# Responsibilities:
#   - build datasets / dataloaders (including spatial splits + debug window),
#   - build model, optimizer, normalizer, categorical encoder,
#   - run epoch-based train/val loops (using src.training.loops),
#   - manage a simple "best checkpoint" on validation loss,
#   - run a final reconstruction diagnostic pass.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import yaml
import csv 

import torch
from torch.utils.data import DataLoader
from torch import optim
import xarray as xr

from src.data.normalization import build_normalizer_from_zarr
from src.data.patch_dataset import PatchDataset
from src.data.categorical_meta import infer_categorical_meta_from_zarr

from src.models.categorical_embedding import CategoricalEmbeddingEncoder
from src.models.vae_v0 import ConvVAE
from src.training.categorical_eval import print_categorical_histograms
from src.training.loops import train_one_epoch, eval_one_epoch


@dataclass
class TrainConfig:
    """
    Lightweight training configuration for ConvVAE + PatchDataset.

    Args:
        zarr_path:
            Path to the Zarr cube (VA cube).
        patch_size:
            Spatial patch size in pixels (e.g., 256).
        batch_size:
            Batch size per DataLoader.
        num_epochs:
            Number of training epochs.
        lr:
            Learning rate for the optimizer.
        beta:
            KL weight for the VAE loss.
        lambda_cat:
            Weight on the categorical reconstruction loss.
        debug_window:
            If True, restrict training to a small spatial window.
        debug_window_origin:
            (y0, x0) in pixels for the debug window (used if debug_window=True).
        debug_window_size:
            (height, width) in pixels for the debug window.
        debug_block_dims:
            (block_height, block_width) used when debug_window=True.
        full_block_dims:
            (block_height, block_width) used when debug_window=False.
        num_workers:
            Number of DataLoader workers.
        pin_memory:
            Whether to pin memory in DataLoaders.
        ckpt_dir:
            Directory in which to store checkpoints.
    """
    zarr_path: str
    patch_size: int = 256
    batch_size: int = 2
    num_epochs: int = 5
    lr: float = 1e-4
    beta: float = 0.1
    lambda_cat: float = 1.0

    debug_window: bool = True
    debug_window_origin: Tuple[int, int] = (256 * 10, 256 * 20)   # (y0, x0)
    debug_window_size: Tuple[int, int] = (1024, 1024)             # (H, W)
    debug_block_dims: Tuple[int, int] = (1, 1)                    # (H_blocks, W_blocks)

    full_block_dims: Tuple[int, int] = (7, 7)

    num_workers: int = 0
    pin_memory: bool = True

    run_root: str = "runs"              # parent directory for all experiments
    experiment_name: str = "vae_v0"     # subdirectory per experiment
    ckpt_dir: str = "checkpoints"       # subdirectory inside run_dir for checkpoints


    @staticmethod
    def from_yaml(path: str | Path) -> "TrainConfig":
        """
        Load TrainConfig from a YAML file and return a populated dataclass.
        """
        path = Path(path)
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # Coerce some known scalars to the right types in case YAML had quotes.
        for key in ("lr", "beta", "lambda_cat"):
            if key in cfg:
                cfg[key] = float(cfg[key])

        # Tuples: allow YAML lists and convert
        for key in ("debug_window_origin", "debug_window_size",
                    "debug_block_dims", "full_block_dims"):
            if key in cfg:
                cfg[key] = tuple(cfg[key])

        return TrainConfig(**cfg)


class Trainer:
    """
    Trainer for ConvVAE + PatchDataset.

    This class encapsulates the end-to-end training process:
      - dataset construction (train/val/test),
      - model + optimizer setup,
      - epoch-based training/validation loop,
      - checkpointing,
      - final reconstruction diagnostics.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # Resolve paths
        self.zarr_path = Path(cfg.zarr_path)
        self.run_root = Path(cfg.run_root)
        self.run_dir = self.run_root / cfg.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory lives *inside* run_dir
        self.ckpt_dir = self.run_dir / cfg.ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Trainer] run_dir: {self.run_dir}")
        print(f"[Trainer] ckpt_dir: {self.ckpt_dir}")

        # Save a copy of the config into the run directory for reproducibility
        cfg_out = self.run_dir / "config_resolved.yaml"
        with open(cfg_out, "w") as f:
            yaml.safe_dump(self.cfg.__dict__, f)
        print(f"[Trainer] Saved resolved config to {cfg_out}")
        
        
        # Decide on spatial split parameters and debug window
        if cfg.debug_window:
            self.window_origin = cfg.debug_window_origin
            self.window_size = cfg.debug_window_size
            self.block_height, self.block_width = cfg.debug_block_dims
            print(
                f"[Trainer] DEBUG WINDOW ENABLED @ origin={self.window_origin}, "
                f"size={self.window_size}, block_dims={cfg.debug_block_dims}"
            )
        else:
            self.window_origin = None
            self.window_size = None
            self.block_height, self.block_width = cfg.full_block_dims
            print(
                f"[Trainer] DEBUG WINDOW DISABLED — using full domain with "
                f"block_dims={cfg.full_block_dims}"
            )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        # Build data + model stack
        self._build_data_and_normalizer()
        self._build_model_and_optimizer()

        # Book-keeping
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_data_and_normalizer(self):
        """
        Build:
          - xarray.Dataset from Zarr,
          - normalizer,
          - categorical encoder (if present),
          - train/val/test PatchDatasets and DataLoaders.
        """
        # Open Zarr
        ds = xr.open_zarr(self.zarr_path)
        self.ds = ds

        # Normalizer from Zarr metadata
        self.normalizer = build_normalizer_from_zarr(
            ds,
            group="continuous",
            feature_dim="feature_continuous",
            enable=True,
        )

        # Optional categorical encoder
        if "categorical" in ds:
            feature_ids, num_classes, emb_dims = infer_categorical_meta_from_zarr(ds)
            self.feature_ids = feature_ids
            self.num_classes = num_classes
            self.emb_dims = emb_dims

            self.cat_encoder = CategoricalEmbeddingEncoder(
                feature_ids=feature_ids,
                num_classes=num_classes,
                emb_dims=emb_dims,
            )
            print("[Trainer] categorical features:", feature_ids)
            print("[Trainer] categorical embedding out_channels:", self.cat_encoder.out_channels)
        else:
            self.cat_encoder = None
            self.feature_ids = None
            self.num_classes = None
            self.emb_dims = None
            print("[Trainer] no categorical group; running continuous-only.")

        # Build PatchDatasets with spatial splits
        ps = self.cfg.patch_size

        self.train_dataset = PatchDataset(
            self.zarr_path,
            patch_size=ps,
            split="train",
            block_width=self.block_width,
            block_height=self.block_height,
            window_origin=self.window_origin,
            window_size=self.window_size,
        )
        self.val_dataset = PatchDataset(
            self.zarr_path,
            patch_size=ps,
            split="val",
            block_width=self.block_width,
            block_height=self.block_height,
            window_origin=self.window_origin,
            window_size=self.window_size,
        )
        self.test_dataset = PatchDataset(
            self.zarr_path,
            patch_size=ps,
            split="test",
            block_width=self.block_width,
            block_height=self.block_height,
            window_origin=self.window_origin,
            window_size=self.window_size,
        )

        print(
            f"[Trainer] dataset sizes: "
            f"train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, "
            f"test={len(self.test_dataset)}"
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        # Peek at one TRAIN batch to determine input shape
        batch0 = next(iter(self.train_loader))
        x_cont0 = batch0["x_cont"]   # [B, T, C_cont, H, W]
        x_cat0  = batch0["x_cat"]    # [B, T, C_cat,  H, W] or None
        B, T, C_cont, H, W = x_cont0.shape

        C_emb = (
            self.cat_encoder.out_channels
            if (self.cat_encoder is not None and x_cat0 is not None)
            else 0
        )

        C_all = C_cont + C_emb
        in_channels = T * C_all
        print(f"[Trainer] Batch shape: {x_cont0.shape}  -> in_channels={in_channels}")
        print(f"[Trainer] C_cont={C_cont}, C_emb={C_emb}, T={T} -> in_channels={in_channels}")

        self.B0 = B
        self.T = T
        self.C_cont = C_cont
        self.C_all = C_all
        self.in_channels = in_channels

    def _build_model_and_optimizer(self):
        """
        Build ConvVAE model and optimizer, and move everything to the target device.
        """
        self.model = ConvVAE(in_channels=self.in_channels, latent_dim=128).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        if self.cat_encoder is not None:
            self.cat_encoder = self.cat_encoder.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """
        Run the full training loop:
          - train/val epochs,
          - best-checkpoint saving,
          - metrics logging to CSV,
          - final reconstruction diagnostics.
        """
        print(
            f"[Trainer] Starting training for {self.cfg.num_epochs} epochs "
            f"(beta={self.cfg.beta}, lambda_cat={self.cfg.lambda_cat})"
        )

        # Path to metrics CSV inside the run directory
        metrics_path = self.run_dir / "metrics.csv"

        # Open once for the whole run and write a simple header
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "split", "loss", "cont_recon", "cat_loss", "kl"]
            )

            for epoch in range(self.cfg.num_epochs):
                # ------------------------------
                # Training epoch
                # ------------------------------
                train_metrics = train_one_epoch(
                    model=self.model,
                    train_loader=self.train_loader,
                    optimizer=self.optimizer,
                    device=self.device,
                    normalizer=self.normalizer,
                    cat_encoder=self.cat_encoder,
                    beta=self.cfg.beta,
                    lambda_cat=self.cfg.lambda_cat,
                )

                # ------------------------------
                # Validation epoch
                # ------------------------------
                val_metrics = eval_one_epoch(
                    model=self.model,
                    val_loader=self.val_loader,
                    device=self.device,
                    normalizer=self.normalizer,
                    cat_encoder=self.cat_encoder,
                    beta=self.cfg.beta,
                    lambda_cat=self.cfg.lambda_cat,
                )

                # Console summary (same as before)
                print(
                    f"[epoch {epoch:03d}] "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"train_cont={train_metrics['cont_recon']:.4f}  "
                    f"train_cat={train_metrics['cat_loss']:.4f}  "
                    f"train_kl={train_metrics['kl']:.4f}  |  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_cont={val_metrics['cont_recon']:.4f}  "
                    f"val_cat={val_metrics['cat_loss']:.4f}  "
                    f"val_kl={val_metrics['kl']:.4f}"
                )

                # ------------------------------
                # CSV logging
                # ------------------------------
                writer.writerow(
                    [
                        epoch,
                        "train",
                        train_metrics["loss"],
                        train_metrics["cont_recon"],
                        train_metrics["cat_loss"],
                        train_metrics["kl"],
                    ]
                )
                writer.writerow(
                    [
                        epoch,
                        "val",
                        val_metrics["loss"],
                        val_metrics["cont_recon"],
                        val_metrics["cat_loss"],
                        val_metrics["kl"],
                    ]
                )

                # ------------------------------
                # Best-checkpoint update
                # ------------------------------
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    ckpt_path = self.ckpt_dir / "vae_v0_best.pt"
                    torch.save(
                        {
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "beta": self.cfg.beta,
                            "lambda_cat": self.cfg.lambda_cat,
                        },
                        ckpt_path,
                    )
                    print(
                        f"[Trainer] Saved new best model at epoch {epoch} "
                        f"(val_loss={self.best_val_loss:.4f}) -> {ckpt_path}"
                    )

        print(f"[Trainer] Metrics written to {metrics_path}")

        # ------------------------------------------------------------------
        # Final diagnostics after training
        # ------------------------------------------------------------------
        self._final_reconstruction_diagnostics()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _final_reconstruction_diagnostics(self):
        """
        Run a small reconstruction diagnostic on a single VAL batch:
          - summarize normalized + physical stats,
          - optionally print categorical histograms.
        """
        self.model.eval()

        try:
            val_batch = next(iter(self.val_loader))
        except StopIteration:
            print("[Trainer] val_loader is empty; skipping reconstruction diagnostics.")
            return

        x_cont = val_batch["x_cont"]    # [B, T, C_cont, H, W]
        x_cat  = val_batch["x_cat"]     # [B, T, C_cat, H, W] or None

        x_cont = x_cont.to(self.device)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)

        B, T, C_cont, H, W = x_cont.shape

        # forward preprocessing (same as training)
        x_bt = x_cont.view(B * T, C_cont, H, W)
        x_bt_norm = self.normalizer(x_bt)
        x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)

        if self.cat_encoder is not None and x_cat is not None:
            x_cat_emb = self.cat_encoder(x_cat)
            x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
        else:
            x_all = x_cont_norm

        B, T, C_all, H, W = x_all.shape
        x_flat = x_all.view(B, T * C_all, H, W)

        recon, mu, logvar = self.model(x_flat)
        recon_full = recon.view(B, T, C_all, H, W)

        # continuous back-transform
        recon_cont = recon_full[:, :, :C_cont, ...]
        recon_bt = recon_cont.reshape(B * T, C_cont, H, W)
        recon_phys_bt = self.normalizer.unnormalize(recon_bt)
        recon_phys = recon_phys_bt.view(B, T, C_cont, H, W)

        x_phys_bt = self.normalizer.unnormalize(x_bt_norm)
        x_phys = x_phys_bt.view(B, T, C_cont, H, W)

        # Flatten one example for summary
        x0_norm = x_flat[0].cpu().numpy()
        r0_norm = recon[0].cpu().numpy()

        x0_phys = x_phys[0].cpu().numpy()
        r0_phys = recon_phys[0].cpu().numpy()

        print("NORMALIZED (VAL example):")
        print("  input min/max:", x0_norm.min(), x0_norm.max())
        print("  recon min/max:", r0_norm.min(), r0_norm.max())
        print("  input mean/std:", x0_norm.mean(), x0_norm.std())
        print("  recon mean/std:", r0_norm.mean(), r0_norm.std())

        print("\nPHYSICAL (VAL example):")
        print("  input min/max:", x0_phys.min(), x0_phys.max())
        print("  recon min/max:", r0_phys.min(), r0_phys.max())
        print("  input mean/std:", x0_phys.mean(), x0_phys.std())
        print("  recon mean/std:", r0_phys.mean(), r0_phys.std())

        if (
            (self.cat_encoder is not None)
            and (x_cat is not None)
            and (self.num_classes is not None)
        ):
            print_categorical_histograms(
                recon_full=recon_full,
                x_cat=x_cat,
                cat_encoder=self.cat_encoder,
                num_classes=self.num_classes,
                C_cont=C_cont,
            )
        else:
            print("[Trainer] skipping categorical histograms (no categorical encoder/data).")
