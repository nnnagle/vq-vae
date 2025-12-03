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
# @dataclass class SchedulerConfig:
# @dataclass class OptimizerConfig
# @dataclass class TrainConfig:
# @dataclass class BetaScheduleConfig
# class Trainer



from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, List
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
#from src.models.vae_v0 import ConvVAE
from src.models.forest_trajectory_ae import ForestTrajectoryAE
from src.training.categorical_eval import print_categorical_histograms
from src.training.loops import train_one_epoch, eval_one_epoch

@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.

    Supported:
      - name="none"
      - name="cosine": uses T_max_epochs, eta_min
      - name="step": uses milestones, gamma
    """
    name: str = "none"
    T_max_epochs: Optional[int] = None
    eta_min: float = 1e-6

    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.1

@dataclass
class OptimizerConfig:
    """
    Optimizer configuration (currently Adam-only in code).
    """
    name: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class BetaScheduleConfig:
    """
    Beta (KL weight) annealing configuration.
    """
    enabled: bool = False
    schedule_type: str = "linear"  # ["linear", "cosine"]
    start_epoch: int = 0
    end_epoch: int = 0
    start_value: float = 0.1
    end_value: float = 0.1
    
@dataclass
class TrainConfig:
    """
    Lightweight training configuration for ConvVAE + PatchDataset.

    Args:
        zarr_path:
            Path to the Zarr cube (VA cube).
        time_steps:
            Number of temporal steps T in the input trajectories.
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
    # --- Data / model ------------------------------------------------------
    zarr_path: str
    time_steps: int = 10
    patch_size: int = 256
    batch_size: int = 2
    num_epochs: int = 5
    
    # --- Loss weighting ----------------------------------------------------
    beta: float = 0.1
    lambda_cat: float = 1.0
    beta_schedule: BetaScheduleConfig = field(default_factory=BetaScheduleConfig)

    # --- Optimization ------------------------------------------------------
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # --- Spatial / debug ---------------------------------------------------
    debug_window: bool = True
    debug_window_origin: Tuple[int, int] = (256 * 10, 256 * 20)   # (y0, x0)
    debug_window_size: Tuple[int, int] = (1024, 1024)             # (H, W)
    debug_block_dims: Tuple[int, int] = (1, 1)                    # (H_blocks, W_blocks)
    full_block_dims: Tuple[int, int] = (7, 7)

    # --- DataLoader --------------------------------------------------------
    num_workers: int = 0
    pin_memory: bool = True

    # --- Run management ----------------------------------------------------
    run_root: str = "runs"
    experiment_name: str = "vae_v0"
    ckpt_dir: str = "checkpoints"


    @staticmethod
    def from_yaml(path: str | Path) -> "TrainConfig":
        """
        Load TrainConfig from a YAML file with the structured schema.
        """
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Convert lists -> tuples for spatial fields
        for key in (
            "debug_window_origin",
            "debug_window_size",
            "debug_block_dims",
            "full_block_dims",
        ):
            if key in raw and isinstance(raw[key], list):
                raw[key] = tuple(raw[key])

        # Optimizer block
        opt_raw = raw.get("optimizer", {})
        sched_raw = opt_raw.get("scheduler", {})

        scheduler = SchedulerConfig(**sched_raw) if sched_raw else SchedulerConfig()
        optimizer = OptimizerConfig(
            name=opt_raw.get("name", "adam"),
            lr=float(opt_raw.get("lr", 1e-4)),
            weight_decay=float(opt_raw.get("weight_decay", 0.0)),
            scheduler=scheduler,
        )

        # Beta schedule block
        bs_raw = raw.get("beta_schedule", {})
        beta_schedule = BetaScheduleConfig(
            enabled=bool(bs_raw.get("enabled", False)),
            schedule_type=bs_raw.get("schedule_type", "linear"),
            start_epoch=int(bs_raw.get("start_epoch", 0)),
            end_epoch=int(bs_raw.get("end_epoch", raw.get("num_epochs", 20))),
            start_value=float(bs_raw.get("start_value", raw.get("beta", 0.1))),
            end_value=float(bs_raw.get("end_value", raw.get("beta", 0.1))),
        )

        # Top-level scalars
        cfg = TrainConfig(
            zarr_path=raw["zarr_path"],
            time_steps=int(raw.get("time_steps")),
            patch_size=int(raw.get("patch_size", 256)),
            batch_size=int(raw.get("batch_size", 2)),   # or 4 if you prefer
            num_epochs=int(raw.get("num_epochs", 5)),   # match your defaults
            beta=float(raw.get("beta", 0.1)),
            lambda_cat=float(raw.get("lambda_cat", 1.0)),
            beta_schedule=beta_schedule,
            optimizer=optimizer,
            debug_window=bool(raw.get("debug_window", True)),
            debug_window_origin=raw.get("debug_window_origin", (256 * 10, 256 * 20)),
            debug_window_size=raw.get("debug_window_size", (1024, 1024)),
            debug_block_dims=raw.get("debug_block_dims", (1, 1)),
            full_block_dims=raw.get("full_block_dims", (7, 7)),
            num_workers=int(raw.get("num_workers", 0)),
            pin_memory=bool(raw.get("pin_memory", True)),
            run_root=raw.get("run_root", "runs"),
            experiment_name=raw.get("experiment_name", "vae_v0"),
            ckpt_dir=raw.get("ckpt_dir", "checkpoints"),
        )

        return cfg

class BetaScheduler:
    """
    Simple epoch-based beta scheduler.

    Supports:
      - 'linear'
      - 'cosine'
    """

    def __init__(self, start_epoch, end_epoch, start_value, end_value, schedule_type="linear"):
        assert end_epoch >= start_epoch, "end_epoch must be >= start_epoch"
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_value = start_value
        self.end_value = end_value
        self.schedule_type = schedule_type.lower()

    def __call__(self, epoch: int) -> float:
        # Before the ramp
        if epoch <= self.start_epoch:
            return self.start_value

        # After the ramp
        if epoch >= self.end_epoch:
            return self.end_value

        # Fraction in [0, 1]
        t = (epoch - self.start_epoch) / max(1, (self.end_epoch - self.start_epoch))

        if self.schedule_type == "cosine":
            import math
            cos_t = 0.5 * (1.0 - math.cos(math.pi * t))
            return self.start_value + (self.end_value - self.start_value) * cos_t

        # Default: linear
        return self.start_value + (self.end_value - self.start_value) * t


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
            yaml.safe_dump(asdict(self.cfg), f)
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
        # ------------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------------
        # The trainer already computed self.in_channels = T * C_all
        # So we also need time_steps. You already have it in config.
        time_steps = self.cfg.time_steps   # or wherever T lives in your YAML
        
        self.model = ForestTrajectoryAE(
            in_channels=self.in_channels,
            time_steps=time_steps,
            feature_channels=32,     # or make configurable later
            temporal_hidden=64,      # or make configurable later
        ).to(self.device)

        # Keep categorical encoder behavior unchanged
        if self.cat_encoder is not None:
            self.cat_encoder = self.cat_encoder.to(self.device)

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        opt_cfg = self.cfg.optimizer
        if opt_cfg.name.lower() != "adam":
            raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )


        # -------------------------------------------------------------
        # LR Scheduler
        # -------------------------------------------------------------
        sched_cfg = opt_cfg.scheduler
        sched_name = sched_cfg.name.lower()

        self.scheduler = None

        if sched_name == "none":
            print("[Trainer] No LR scheduler.")

        elif sched_name == "cosine":
            # Use T_max_epochs if provided, otherwise num_epochs
            T_max = sched_cfg.T_max_epochs or self.cfg.num_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=sched_cfg.eta_min,
            )
            print(
                f"[Trainer] LR Scheduler: CosineAnnealingLR("
                f"T_max={T_max}, eta_min={sched_cfg.eta_min})"
            )

        elif sched_name == "step":
            milestones = sched_cfg.milestones
            if not milestones:
                # If user didn't supply, default to a single step at 70% of training
                milestones = [int(0.7 * self.cfg.num_epochs)]

            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=sched_cfg.gamma,
            )
            print(
                f"[Trainer] LR Scheduler: MultiStepLR("
                f"milestones={milestones}, gamma={sched_cfg.gamma})"
            )

        else:
            raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")

        # -------------------------------------------------------------
        # Beta scheduler (KL weight annealing)
        # -------------------------------------------------------------
        bs = self.cfg.beta_schedule

        if bs.enabled:
            self.beta_scheduler = BetaScheduler(
                start_epoch=bs.start_epoch,
                end_epoch=bs.end_epoch,
                start_value=bs.start_value,
                end_value=bs.end_value,
                schedule_type=bs.schedule_type,
            )
            print(
                f"[Trainer] BetaScheduler enabled: "
                f"{bs.schedule_type} from {bs.start_value} → {bs.end_value} "
                f"over epochs {bs.start_epoch}–{bs.end_epoch}"
            )
        else:
            self.beta_scheduler = None
            print(f"[Trainer] BetaScheduler disabled (constant beta={self.cfg.beta}).")


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
            f"(initial beta={self.cfg.beta}, lambda_cat={self.cfg.lambda_cat})"
        )

        # Path to metrics CSV inside the run directory
        metrics_path = self.run_dir / "metrics.csv"

        # Open once for the whole run and write a simple header
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "split", "loss", "cont_recon", "cat_loss", "kl", "beta", "lr"]
            )

            for epoch in range(self.cfg.num_epochs):
                # ---- Determine beta for this epoch ----
                if self.beta_scheduler is not None:
                    beta = self.beta_scheduler(epoch)
                else:
                    beta = self.cfg.beta

            
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
                    beta=beta,
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
                    beta=beta,
                    lambda_cat=self.cfg.lambda_cat,
                )
                
                # ---- Current LR ----
                current_lr = self.optimizer.param_groups[0]["lr"]

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
                        beta, 
                        current_lr,
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
                        beta, 
                        current_lr,
                    ]
                )
                
                # ---- LR scheduler step (epoch-wise schedulers) ----
                if self.scheduler is not None:
                    self.scheduler.step()

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
                            "beta": beta,
                            "lambda_cat": self.cfg.lambda_cat,
                        },
                        ckpt_path,
                    )
                    print(
                        f"[Trainer] Saved new best model at epoch {epoch} "
                        f"(val_loss={self.best_val_loss:.4f}) -> {ckpt_path}"
                    )

        print(f"[Trainer] Metrics written to {metrics_path}")
        print("[Trainer] Training complete. Loading best checkpoint for diagnostics (if present).")
        best_ckpt = self.ckpt_dir / "vae_v0_best.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(state["model_state"])
            print(f"[Trainer] Loaded best checkpoint from {best_ckpt}")
        else:
            print("[Trainer] No best checkpoint found; using final-epoch weights.")

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
        Run a small reconstruction diagnostic on a single VAL batch.

        This method:
          - pulls one batch from the validation loader,
          - runs it through the *current* model (ideally best checkpoint),
          - summarizes normalized and physical-space stats for a single example,
          - optionally prints categorical reconstruction histograms,
          - optionally writes a simple spacetime reconstruction grid to disk.

        Notes
        -----
        - This should be called after training, ideally after reloading the
          best checkpoint on validation loss.
        - The goal here is *sanity checking*, not full evaluation.
        """

        # Put model in eval mode and ensure no gradients are tracked
        self.model.eval()

        # ------------------------------------------------------------------
        # 1. Get a single validation batch
        # ------------------------------------------------------------------
        try:
            val_batch = next(iter(self.val_loader))
        except StopIteration:
            print("[Trainer] val_loader is empty; skipping reconstruction diagnostics.")
            return

        # Continuous features: [B, T, C_cont, H, W]
        x_cont = val_batch["x_cont"].to(self.device)

        # Categorical features: [B, T, C_cat, H, W] or None
        # Some datasets may not have categorical channels.
        x_cat = val_batch.get("x_cat", None)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)

        # Optional AOI mask: [B, H, W] (bool), for masking outside-forest pixels
        aoi = val_batch.get("aoi", None)
        if aoi is not None:
            aoi = aoi.to(self.device)

        B, T, C_cont, H, W = x_cont.shape
        print("[Trainer] Running reconstruction diagnostics on VAL batch:")
        print(f"  x_cont shape: B={B}, T={T}, C_cont={C_cont}, H={H}, W={W}")

        # ------------------------------------------------------------------
        # 2. Normalize continuous inputs (same as training)
        # ------------------------------------------------------------------
        # Training path is: [B, T, C_cont, H, W] -> [B*T, C_cont, H, W]
        x_bt = x_cont.view(B * T, C_cont, H, W)

        # Apply normalizer to continuous channels only
        # Result is still [B*T, C_cont, H, W] but in normalized space.
        x_bt_norm = self.normalizer(x_bt)

        # Reshape back to [B, T, C_cont, H, W]
        x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)

        # ------------------------------------------------------------------
        # 3. Concatenate categorical embeddings, if present
        # ------------------------------------------------------------------
        if self.cat_encoder is not None and x_cat is not None:
            # cat_encoder is expected to map [B, T, C_cat, H, W] ->
            # [B, T, C_emb, H, W]
            x_cat_emb = self.cat_encoder(x_cat)
            # Concatenate along the channel (feature) dimension
            x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
        else:
            x_all = x_cont_norm

        B, T, C_all, H, W = x_all.shape
        print(f"  x_all shape (cont + cat_emb): B={B}, T={T}, C_all={C_all}")

        # ------------------------------------------------------------------
        # 4. Flatten time + channel for the ConvVAE input and run forward
        # ------------------------------------------------------------------
        # Model expects [B, C_in, H, W]; here we treat time × channels as C_in.
        x_flat = x_all.view(B, T * C_all, H, W)

        # Forward pass through model: returns recon in same flattened shape
        recon, mu, logvar = self.model(x_flat)

        # Bring recon back to [B, T, C_all, H, W]
        recon_full = recon.view(B, T, C_all, H, W)

        # ------------------------------------------------------------------
        # 5. Slice out continuous channels and unnormalize
        # ------------------------------------------------------------------
        # Continuous outputs: first C_cont channels
        recon_cont = recon_full[:, :, :C_cont, ...]           # [B, T, C_cont, H, W]

        # For unnormalizing, go back to [B*T, C_cont, H, W]
        recon_bt = recon_cont.reshape(B * T, C_cont, H, W)

        # Unnormalize reconstructions to physical units
        recon_phys_bt = self.normalizer.unnormalize(recon_bt)
        recon_phys = recon_phys_bt.view(B, T, C_cont, H, W)

        # Also unnormalize the (already normalized) inputs for comparison
        x_phys_bt = self.normalizer.unnormalize(x_bt_norm)
        x_phys = x_phys_bt.view(B, T, C_cont, H, W)

        # ------------------------------------------------------------------
        # 6. Print summary stats for a single example (b=0)
        # ------------------------------------------------------------------
        # Normalized view: use x_bt_norm / recon_cont
        x0_norm = x_bt_norm.view(B, T, C_cont, H, W)[0].detach().cpu().numpy()
        r0_norm = recon_cont[0].detach().cpu().numpy()

        # Physical view: use x_phys / recon_phys
        x0_phys = x_phys[0].detach().cpu().numpy()
        r0_phys = recon_phys[0].detach().cpu().numpy()

        print("\n=== FINAL RECONSTRUCTION DIAGNOSTICS (VAL) ===")
        print("NORMALIZED (VAL example b=0):")
        print("  input   min/max:", x0_norm.min(), x0_norm.max())
        print("  recon   min/max:", r0_norm.min(), r0_norm.max())
        print("  input   mean/std:", x0_norm.mean(), x0_norm.std())
        print("  recon   mean/std:", r0_norm.mean(), r0_norm.std())

        print("\nPHYSICAL (VAL example b=0):")
        print("  input   min/max:", x0_phys.min(), x0_phys.max())
        print("  recon   min/max:", r0_phys.min(), r0_phys.max())
        print("  input   mean/std:", x0_phys.mean(), x0_phys.std())
        print("  recon   mean/std:", r0_phys.mean(), r0_phys.std())

        # ------------------------------------------------------------------
        # 7. Optional categorical reconstruction histograms
        # ------------------------------------------------------------------
        if (
            (self.cat_encoder is not None)
            and (x_cat is not None)
            and (self.num_classes is not None)
        ):
            # print_categorical_histograms expects the *full* recon tensor
            # including continuous + categorical-derived channels.
            print_categorical_histograms(
                recon_full=recon_full,
                x_cat=x_cat,
                cat_encoder=self.cat_encoder,
                num_classes=self.num_classes,
                C_cont=C_cont,
            )
        else:
            print("[Trainer] Skipping categorical histograms "
                  "(no categorical encoder/data).")

        # ------------------------------------------------------------------
        # 8. Optional spatial–temporal visualization to disk
        # ------------------------------------------------------------------
        # This is intentionally wrapped in a try/except so diagnostics do not
        # crash training if plotting fails (e.g., missing dependency).
        try:
            from src.diagnostics.visualization import save_spacetime_recon_grid

            diagnostics_dir = self.run_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            # Choose a feature to visualize. Here we pick index 0, and try to
            # recover its name from the normalizer metadata if available.
            feat_idx = 0
            feat_name = None
            if hasattr(self.normalizer, "feature_ids"):
                if 0 <= feat_idx < len(self.normalizer.feature_ids):
                    feat_name = self.normalizer.feature_ids[feat_idx]

            out_path = diagnostics_dir / f"val_recon_spacetime_feat{feat_idx}.png"

            print(f"[Trainer] Writing spacetime recon grid to: {out_path}")

            save_spacetime_recon_grid(
                x_phys=x_phys.detach().cpu(),
                recon_phys=recon_phys.detach().cpu(),
                aoi=aoi.detach().cpu() if aoi is not None else None,
                out_path=out_path,
                feature_idx=feat_idx,
                feature_name=feat_name,
                max_patches=4,
            )
        except Exception as e:
            print(f"[Trainer] Warning: failed to write spacetime recon grid: {e}")

 
