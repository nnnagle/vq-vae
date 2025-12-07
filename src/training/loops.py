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

from typing import Dict

import torch
from torch import Tensor

from src.training.losses import ForestTrajectoryVAELoss
from src.training.loss_config import ForestTrajectoryLossConfig


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalizer,
    cat_encoder,
    loss_cfg: ForestTrajectoryLossConfig,
) -> Dict[str, float]:
    """
    Single training epoch over the TRAIN split.

    Args
    ----
    model : torch.nn.Module
    train_loader : DataLoader
    optimizer : torch.optim.Optimizer
    device : torch.device
    normalizer : callable
        Normalization module/function applied to continuous inputs.
    cat_encoder : CategoricalEmbeddingEncoder or None
    loss_cfg : ForestTrajectoryLossConfig
        Configuration for all loss weights and options.

    Returns
    -------
    dict with averaged metrics over the epoch:
      - "loss"
      - "cont_recon"
      - "delta_loss"
      - "deriv_loss"
      - "spatial_grad_loss"
      - "cat_loss"
      - "kl"
    """
    model.train()

    loss_fn = ForestTrajectoryVAELoss(loss_cfg)

    total_loss = 0.0
    total_cont = 0.0
    total_delta = 0.0
    total_deriv = 0.0
    total_spatial = 0.0
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

        loss_dict = loss_fn(
            recon_full=recon_full,
            x_cont_norm=x_cont_norm,
            x_cat=x_cat,
            mu=mu,
            logvar=logvar,
            aoi_mask=aoi_mask,
            C_cont=C_cont,
            cat_encoder=cat_encoder,
        )

        loss = loss_dict["loss"]
        recon_loss = loss_dict["cont_recon"]
        delta_loss = loss_dict["delta_loss"]
        deriv_loss = loss_dict["deriv_loss"]
        spatial_loss = loss_dict.get(
            "spatial_grad_loss",
            torch.tensor(0.0, device=device),
        )
        cat_loss = loss_dict["cat_loss"]
        kl = loss_dict["kl"]

        # Backprop + update
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss    += float(loss.item())
        total_cont    += float(recon_loss.item())
        total_delta   += float(delta_loss.item())
        total_deriv   += float(deriv_loss.item())
        total_spatial += float(spatial_loss.item())
        total_cat     += float(cat_loss.item())
        total_kl      += float(kl.item())
        n_batches     += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "cont_recon": float("nan"),
            "delta_loss": float("nan"),
            "deriv_loss": float("nan"),
            "spatial_grad_loss": float("nan"),
            "cat_loss": float("nan"),
            "kl": float("nan"),
        }

    return {
        "loss": total_loss / n_batches,
        "cont_recon": total_cont / n_batches,
        "delta_loss": total_delta / n_batches,
        "deriv_loss": total_deriv / n_batches,
        "spatial_grad_loss": total_spatial / n_batches,
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
    loss_cfg: ForestTrajectoryLossConfig,
) -> Dict[str, float]:
    """
    Single evaluation epoch over the VAL split.

    Mirrors train_one_epoch, but:
      - uses model.eval()
      - no optimizer or gradients
    """
    model.eval()

    loss_fn = ForestTrajectoryVAELoss(loss_cfg)

    total_loss = 0.0
    total_cont = 0.0
    total_cat = 0.0
    total_delta = 0.0
    total_deriv = 0.0
    total_spatial = 0.0
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
            x_cat_emb = cat_encoder(x_cat)         # [B, T, C_emb, H, W]
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

        loss_dict = loss_fn(
            recon_full=recon_full,
            x_cont_norm=x_cont_norm,
            x_cat=x_cat,
            mu=mu,
            logvar=logvar,
            aoi_mask=aoi_mask,
            C_cont=C_cont,
            cat_encoder=cat_encoder,
        )

        loss = loss_dict["loss"]
        recon_loss = loss_dict["cont_recon"]
        delta_loss = loss_dict["delta_loss"]
        deriv_loss = loss_dict["deriv_loss"]
        spatial_loss = loss_dict.get(
            "spatial_grad_loss",
            torch.tensor(0.0, device=device),
        )
        cat_loss = loss_dict["cat_loss"]
        kl = loss_dict["kl"]

        total_loss    += float(loss.item())
        total_cont    += float(recon_loss.item())
        total_delta   += float(delta_loss.item())
        total_deriv   += float(deriv_loss.item())
        total_spatial += float(spatial_loss.item())
        total_cat     += float(cat_loss.item())
        total_kl      += float(kl.item())
        n_batches     += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "cont_recon": float("nan"),
            "delta_loss": float("nan"),
            "deriv_loss": float("nan"),
            "spatial_grad_loss": float("nan"),
            "cat_loss": float("nan"),
            "kl": float("nan"),
        }

    return {
        "loss": total_loss / n_batches,
        "cont_recon": total_cont / n_batches,
        "delta_loss": total_delta / n_batches,
        "deriv_loss": total_deriv / n_batches,
        "spatial_grad_loss": total_spatial / n_batches,
        "cat_loss": total_cat / n_batches,
        "kl": total_kl / n_batches,
    }
