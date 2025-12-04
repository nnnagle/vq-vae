# src/training/loops.py
#
# Epoch-level train/eval helpers for ConvVAE + PatchDataset.
#
# These functions know:
#   - how to preprocess batches (normalization + embeddings),
#   - how to run the VAE forward pass,
#   - how to compute and aggregate losses.
#
# They do NOT:
#   - build datasets,
#   - construct models/optimizers,
#   - manage checkpoints or configs.

from __future__ import annotations

from typing import Dict, Any

import torch
from torch import Tensor
from src.training.losses import (
    aoi_masked_vae_loss,
    categorical_recon_loss_from_embeddings,
    final_frame_weighted_mse, delta_from_final_mse_all,
    temporal_derivative_mse
)


# Global loss weights for temporal terms
LAMBDA_DELTA = 10.0      # delta-from-final weight
LAMBDA_DERIV = 10.0       # temporal-derivative weight (tune as you like)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalizer,
    cat_encoder,
    beta: float,
    lambda_cat: float,
) -> Dict[str, float]:
    """
    Single training epoch over the TRAIN split.

    Args:
        model:
            ConvVAE / ForestTrajectoryAE model on the correct device.
        train_loader:
            DataLoader over the training PatchDataset split. Each batch is a
            dict with keys "x_cont", "x_cat" (optional), "x_mask" (optional),
            "aoi".
        optimizer:
            Optimizer instance (e.g., Adam) configured for `model.parameters()`.
        device:
            Torch device (cpu or cuda) on which computations will run.
        normalizer:
            Normalizer with __call__(x) -> normalized, and .unnormalize(x)
            methods. Used for continuous features.
        cat_encoder:
            CategoricalEmbeddingEncoder instance, or None if no categorical
            features are present.
        beta:
            Weight on the KL term in the VAE loss.
        lambda_cat:
            Weight on the categorical reconstruction loss.

    Returns:
        dict:
            Dictionary with averaged metrics over the epoch:
              - "loss"
              - "cont_recon"
              - "delta_loss"
              - "deriv_loss"
              - "cat_loss"
              - "kl"
    """
    model.train()

    total_loss = 0.0
    total_cont = 0.0
    total_delta = 0.0
    total_deriv = 0.0
    total_cat = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in train_loader:
        x_cont = batch["x_cont"]             # [B, T, C_cont, H, W]
        x_cat  = batch.get("x_cat", None)    # [B, T, C_cat, H, W] or None
        aoi    = batch["aoi"]                # [B, H, W]

        B, T, C_cont, H, W = x_cont.shape

        # Move to device
        x_cont = x_cont.to(device, non_blocking=True)
        if x_cat is not None:
            x_cat = x_cat.to(device, non_blocking=True)
        aoi = aoi.to(device, non_blocking=True)    # [B, H, W]

        # Collapse time into batch to apply normalizer
        x_bt = x_cont.view(B * T, C_cont, H, W)    # [B*T, C_cont, H, W]
        x_bt_norm = normalizer(x_bt)               # normalized
        x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)

        # Optional categorical embeddings
        if cat_encoder is not None and x_cat is not None:
            x_cat_emb = cat_encoder(x_cat)         # [B, T, C_emb, H, W]
            x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
        else:
            x_all = x_cont_norm

        B, T, C_all, H, W = x_all.shape

        # Flatten time × channels for model
        x_flat_all = x_all.view(B, T * C_all, H, W)          # [B, T*C_all, H, W]

        # AOI -> [B,1,H,W]
        if aoi.dim() == 3:
            aoi_mask = aoi.unsqueeze(1)
        else:
            aoi_mask = aoi

        # Forward
        optimizer.zero_grad()
        recon, mu, logvar = model(x_flat_all)     # recon is in normalized space

        recon_full = recon.view(B, T, C_all, H, W)
        recon_cont = recon_full[:, :, :C_cont, ...]           # [B,T,C_cont,H,W]

        # -------------------------------
        # Continuous reconstruction loss:
        #   - final frame upweighted
        #   - AOI masking
        # -------------------------------
        w_final = 2.0  # tune as needed
        recon_loss = final_frame_weighted_mse(
            recon_full=recon_cont,
            x_full=x_cont_norm,    # normalized continuous inputs [B,T,C_cont,H,W]
            w_final=w_final,
            aoi_mask=aoi_mask,     # [B,1,H,W]
        )

        # -------------------------------
        # Delta-from-final loss on band 0
        # -------------------------------
        lambda_delta_final = LAMBDA_DELTA
        delta_loss = delta_from_final_mse_all(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            aoi_mask=aoi_mask,          # [B,1,H,W]
            channel_indices=[0],        # or None -> all continuous channels
            change_thresh=0.05,
        )

        # -------------------------------
        # Temporal derivative loss on band 0
        # -------------------------------
        lambda_deriv = LAMBDA_DERIV
        deriv_loss = temporal_derivative_mse(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            aoi_mask=aoi_mask,
            channel_indices=[0],      # or None for all continuous channels
            change_thresh=0.05,
        )

        # -------------------------------
        # KL term (for ForestTrajectoryAE mu/logvar ~ 0)
        # -------------------------------
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weighted = beta * kl

        # -------------------------------
        # Categorical reconstruction loss
        # -------------------------------
        cat_loss = categorical_recon_loss_from_embeddings(
            recon_full=recon_full,
            x_cat=x_cat,
            cat_encoder=cat_encoder,
            C_cont=C_cont,
        )
        if not torch.isfinite(cat_loss):
            print("[DEBUG] non-finite cat_loss in train_one_epoch, setting to zero.")
            cat_loss = torch.tensor(0.0, device=recon.device)

        # -------------------------------
        # Total loss
        # -------------------------------
        loss = (
            recon_loss
            + lambda_delta_final * delta_loss
            + lambda_deriv * deriv_loss
            + kl_weighted
            + lambda_cat * cat_loss
        )

        # Backprop + update
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += float(loss.item())
        total_cont += float(recon_loss.item())
        total_delta += float(delta_loss.item())
        total_deriv += float(deriv_loss.item())
        total_cat  += float(cat_loss.item())
        total_kl   += float(kl.item())
        n_batches  += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "cont_recon": float("nan"),
            "delta_loss": float("nan"),
            "deriv_loss": float("nan"),
            "cat_loss": float("nan"),
            "kl": float("nan"),
        }

    return {
        "loss": total_loss / n_batches,
        "cont_recon": total_cont / n_batches,
        "delta_loss": total_delta / n_batches,
        "deriv_loss": total_deriv / n_batches,
        "cat_loss": total_cat / n_batches,
        "kl": total_kl / n_batches,
    }


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    normalizer,
    cat_encoder,
    beta: float,
    lambda_cat: float,
) -> Dict[str, float]:
    """
    Single evaluation epoch over the VAL split.

    Mirrors train_one_epoch, but:
      - uses model.eval()
      - no optimizer or gradients

    Returns:
        dict:
            Same keys as train_one_epoch:
              - "loss"
              - "cont_recon"
              - "delta_loss"
              - "deriv_loss"
              - "cat_loss"
              - "kl"
    """
    model.eval()

    total_loss = 0.0
    total_cont = 0.0
    total_cat = 0.0
    total_delta = 0.0
    total_deriv = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in val_loader:
        x_cont = batch["x_cont"]             # [B, T, C_cont, H, W]
        x_cat  = batch.get("x_cat", None)    # [B, T, C_cat, H, W] or None
        aoi    = batch["aoi"]                # [B, H, W]

        B, T, C_cont, H, W = x_cont.shape

        # Move to device
        x_cont = x_cont.to(device, non_blocking=True)
        if x_cat is not None:
            x_cat = x_cat.to(device, non_blocking=True)
        aoi = aoi.to(device, non_blocking=True)

        # Normalize continuous
        x_bt = x_cont.view(B * T, C_cont, H, W)
        x_bt_norm = normalizer(x_bt)
        x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)

        # Optional categorical embeddings
        if cat_encoder is not None and x_cat is not None:
            x_cat_emb = cat_encoder(x_cat)         # [B,T,C_emb,H,W]
            x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
        else:
            x_all = x_cont_norm

        B, T, C_all, H, W = x_all.shape
        x_flat_all = x_all.view(B, T * C_all, H, W)

        # AOI -> [B,1,H,W]
        if aoi.dim() == 3:
            aoi_mask = aoi.unsqueeze(1)
        else:
            aoi_mask = aoi

        # Forward
        recon, mu, logvar = model(x_flat_all)     # recon in normalized space

        recon_full = recon.view(B, T, C_all, H, W)
        recon_cont = recon_full[:, :, :C_cont, ...]   # [B,T,C_cont,H,W]

        # -------------------------------
        # Continuous reconstruction loss:
        #   - final frame upweighted
        #   - AOI masking
        # -------------------------------
        w_final = 2.0
        recon_loss = final_frame_weighted_mse(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            w_final=w_final,
            aoi_mask=aoi_mask,          # [B,1,H,W]
        )

        # -------------------------------
        # Delta-from-final loss on band 0
        # -------------------------------
        lambda_delta_final = LAMBDA_DELTA
        delta_loss = delta_from_final_mse_all(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            aoi_mask=aoi_mask,
            channel_indices=[0],   # or None for all continuous channels
            change_thresh=0.05,
        )

        # -------------------------------
        # Temporal derivative loss on band 0
        # -------------------------------
        lambda_deriv = LAMBDA_DERIV
        deriv_loss = temporal_derivative_mse(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            aoi_mask=aoi_mask,
            channel_indices=[0],
            change_thresh=0.05,
        )

        # -------------------------------
        # KL term
        # -------------------------------
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weighted = beta * kl

        # -------------------------------
        # Categorical reconstruction loss
        # -------------------------------
        cat_loss = categorical_recon_loss_from_embeddings(
            recon_full=recon_full,
            x_cat=x_cat,
            cat_encoder=cat_encoder,
            C_cont=C_cont,
        )
        if not torch.isfinite(cat_loss):
            print("[DEBUG] non-finite cat_loss in eval_one_epoch, setting to zero.")
            cat_loss = torch.tensor(0.0, device=recon.device)

        # -------------------------------
        # Total loss (same structure as train_one_epoch)
        # -------------------------------
        loss = (
            recon_loss
            + lambda_delta_final * delta_loss
            + lambda_deriv * deriv_loss
            + kl_weighted
            + lambda_cat * cat_loss
        )

        total_loss  += float(loss.item())
        total_cont  += float(recon_loss.item())
        total_delta += float(delta_loss.item())
        total_deriv += float(deriv_loss.item())
        total_cat   += float(cat_loss.item())
        total_kl    += float(kl.item())
        n_batches   += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "cont_recon": float("nan"),
            "delta_loss": float("nan"),
            "deriv_loss": float("nan"),
            "cat_loss": float("nan"),
            "kl": float("nan"),
        }

    return {
        "loss": total_loss / n_batches,
        "cont_recon": total_cont / n_batches,
        "delta_loss": total_delta / n_batches,
        "deriv_loss": total_deriv / n_batches,
        "cat_loss": total_cat / n_batches,
        "kl": total_kl / n_batches,
    }
