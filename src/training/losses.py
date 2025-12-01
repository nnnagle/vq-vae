# src/training/losses.py

import torch
import torch.nn.functional as F
from src.models.categorical_embedding import CategoricalEmbeddingEncoder


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """
    Standard VAE loss = reconstruction loss + β * KL divergence.

    Args
    ----
    recon : torch.Tensor
        Reconstructed input [B, C_in, H, W].
    x : torch.Tensor
        Original input [B, C_in, H, W].
    mu : torch.Tensor
        Latent mean [B, latent_dim].
    logvar : torch.Tensor
        Latent log-variance [B, latent_dim].
    beta : float
        Weight on the KL term. beta=1.0 → standard VAE;
        beta < 1.0 → softer regularization.

    Returns
    -------
    total_loss : torch.Tensor (scalar)
    recon_loss : torch.Tensor (scalar)
    kl : torch.Tensor (scalar)
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


def aoi_masked_vae_loss(recon, x, mu, logvar, aoi_mask, beta: float = 1.0):
    if aoi_mask.dim() == 3:
        aoi_mask = aoi_mask.unsqueeze(1)
    aoi_mask = aoi_mask.bool()
    mask = aoi_mask.expand(-1, x.size(1), -1, -1)

    diff = (recon - x) * mask
    valid = mask.sum()

    if valid == 0:
        recon_loss = torch.tensor(0.0, device=x.device)
    else:
        recon_loss = (diff ** 2).sum() / valid

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


DEBUG_CAT_LOSS = False # flip to False when you're happy


def categorical_recon_loss_from_embeddings(
    recon_full: torch.Tensor,                 # [B, T, C_all, H, W]
    x_cat: torch.Tensor | None,               # [B, T, C_cat, H, W]
    cat_encoder: CategoricalEmbeddingEncoder | None,
    C_cont: int,
) -> torch.Tensor:
    """
    Categorical reconstruction loss using decoded embeddings.

    Debug version: prints per-feature histograms and basic stats
    when non-finite logits or NaN loss conditions are detected.
    """
    if cat_encoder is None or x_cat is None:
        return torch.tensor(0.0, device=recon_full.device)

    B, T, C_all, H, W = recon_full.shape
    device = recon_full.device

    loss_sum = 0.0
    feat_count = 0
    offset = C_cont  # first C_cont channels are continuous

    for c_idx, fid in enumerate(cat_encoder.feature_ids):
        D = cat_encoder.emb_dims[fid]

        # predicted embedding for this feature: [B, T, D, H, W]
        recon_emb = recon_full[:, :, offset : offset + D, ...]
        offset += D

        # ground truth integer codes: [B, T, H, W]
        codes = x_cat[:, :, c_idx, ...].long().to(device)
        pad_idx = cat_encoder.missing_index[fid]

        # map negative codes (e.g. -1) to pad_idx
        codes = codes.clone()
        codes[codes < 0] = pad_idx

        # flatten embeddings and targets
        recon_flat = (
            recon_emb.permute(0, 1, 3, 4, 2)  # [B,T,H,W,D]
            .reshape(-1, D)                   # [N,D]
        )
        targets = codes.reshape(-1)           # [N]

        # mask out padding / missing
        valid_mask = (targets != pad_idx)
        if not valid_mask.any():
            if DEBUG_CAT_LOSS:
                print(f"[DEBUG cat] feature={fid!r}: all targets are pad_idx in this batch; skipping.")
            continue

        logits = recon_flat @ cat_encoder.embs[fid].weight.t()  # [N, K+1]

        logits_valid = logits[valid_mask]   # [N_valid, K+1]
        targets_valid = targets[valid_mask] # [N_valid]

        # --- DEBUG: print target histogram + logits stats if requested ---
        if DEBUG_CAT_LOSS:
            with torch.no_grad():
                tv_cpu = targets_valid.detach().cpu()
                # histogram over classes 0..(K+pad)
                num_classes = cat_encoder.embs[fid].weight.shape[0]
                hist = torch.bincount(tv_cpu, minlength=num_classes)
                # basic logits stats
                lv = logits_valid.detach()
                finite_mask = torch.isfinite(lv)
                frac_finite = finite_mask.float().mean().item()
                lv_finite = lv[finite_mask]
                if lv_finite.numel() > 0:
                    min_logit = lv_finite.min().item()
                    max_logit = lv_finite.max().item()
                else:
                    min_logit = float("nan")
                    max_logit = float("nan")
                print(
                    f"[DEBUG cat] feature={fid!r} "
                    f"N_valid={targets_valid.numel()} "
                    f"hist={hist.tolist()} "
                    f"logits_finite={frac_finite:.3f} "
                    f"logit_min={min_logit:.3e} "
                    f"logit_max={max_logit:.3e}"
                )

        # safety: if logits are non-finite, skip this feature for this batch
        if not torch.isfinite(logits_valid).all():
            if DEBUG_CAT_LOSS:
                print(f"[DEBUG cat] feature={fid!r}: non-finite logits detected; skipping feature in this batch.")
            continue

        feat_loss = F.cross_entropy(logits_valid, targets_valid)

        # another safety check
        if not torch.isfinite(feat_loss):
            if DEBUG_CAT_LOSS:
                print(f"[DEBUG cat] feature={fid!r}: non-finite feat_loss={feat_loss.item()} (NaN/Inf); skipping.")
            continue

        loss_sum = loss_sum + feat_loss
        feat_count += 1

    if feat_count == 0:
        if DEBUG_CAT_LOSS:
            print("[DEBUG cat] feat_count == 0 for this batch; returning 0.0 cat loss.")
        return torch.tensor(0.0, device=device)

    return loss_sum / feat_count
