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

def final_frame_weighted_mse(
    recon_full: torch.Tensor,
    x_full: torch.Tensor,
    w_final: float = 2.0,
    aoi_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean-squared error over [B,T,C,H,W], but with the final timestep
    weighted more heavily.

    Args
    ----
    recon_full, x_full : [B, T, C, H, W]
        Reconstructed and target tensors in (typically) normalized space.

    w_final : float
        Relative weight on the final timestep (T-1). For example, w_final=2.0
        makes the final frame contribute twice as much as any one earlier frame.

    aoi_mask : [B, H, W] or [B, 1, H, W], optional
        Boolean mask where True indicates valid area. If provided, the loss
        is averaged only over valid pixels.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    assert recon_full.shape == x_full.shape, "Shape mismatch recon_full/x_full"
    B, T, C, H, W = x_full.shape

    diff2 = (recon_full - x_full) ** 2  # [B, T, C, H, W]

    if aoi_mask is not None:
        if aoi_mask.dim() == 3:
            aoi_mask = aoi_mask.unsqueeze(1)  # [B, 1, H, W]
        elif aoi_mask.dim() == 4:
            pass  # [B, 1, H, W]
        else:
            raise ValueError(f"Unexpected aoi_mask shape: {aoi_mask.shape}")
        # Broadcast to [B, T, C, H, W]
        mask = aoi_mask.bool().unsqueeze(1).expand(-1, T, C, -1, -1)
        diff2 = diff2 * mask

        valid_early = mask[:, :-1].sum()
        valid_final = mask[:, -1:].sum()
    else:
        valid_early = diff2[:, :-1].numel()
        valid_final = diff2[:, -1:].numel()

    # Avoid zero-division if mask wipes everything
    if valid_early > 0:
        err_early = diff2[:, :-1].sum() / valid_early
    else:
        err_early = torch.tensor(0.0, device=x_full.device)

    if valid_final > 0:
        err_final = diff2[:, -1:].sum() / valid_final
    else:
        err_final = torch.tensor(0.0, device=x_full.device)

    loss = (err_early + w_final * err_final) / (1.0 + w_final)
    return loss


def delta_from_final_mse_all(
    recon_full: torch.Tensor,
    x_full: torch.Tensor,
    aoi_mask: torch.Tensor | None = None,
    channel_indices: list[int] | None = None,
    change_thresh: float = 0.05,
) -> torch.Tensor:
    """
    MSE between deltas relative to the final timestep for a set of channels:

        (x(t) - x(T-1)) vs (recon(t) - recon(T-1))

    BUT we only average over pixels where the *input* delta magnitude
    exceeds `change_thresh` (in normalized units). This focuses the loss
    on genuinely changing pixels instead of drowning them in stable forest.

    recon_full, x_full : [B, T, C, H, W]
    aoi_mask : [B,H,W] or [B,1,H,W]
    channel_indices : list[int] or None (None -> all channels)
    change_thresh : float
        Threshold on |dx_in| (L2 over channels) to consider a pixel "changing".
    """
    assert recon_full.shape == x_full.shape, "Shape mismatch recon_full/x_full"
    B, T, C, H, W = x_full.shape
    device = x_full.device

    if channel_indices is None:
        x_sel = x_full             # [B,T,C,H,W]
        r_sel = recon_full
    else:
        x_sel = x_full[:, :, channel_indices]      # [B,T,C_sel,H,W]
        r_sel = recon_full[:, :, channel_indices]

    # Anchor at final timestep
    xT = x_sel[:, -1:]   # [B,1,C_sel,H,W]
    rT = r_sel[:, -1:]

    dx = x_sel - xT      # [B,T,C_sel,H,W]
    dr = r_sel - rT

    diff2 = (dx - dr) ** 2

    # Magnitude of input delta across channels, max over time
    # -> where does *anything* change?
    dx_mag = dx.pow(2).sum(dim=2).sqrt()     # [B,T,H,W]
    dx_mag_max = dx_mag.max(dim=1).values    # [B,H,W]

    change_mask = dx_mag_max > change_thresh   # [B,H,W]

    if aoi_mask is not None:
        if aoi_mask.dim() == 3:
            aoi_mask = aoi_mask.unsqueeze(1)  # [B,1,H,W]
        elif aoi_mask.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected aoi_mask shape: {aoi_mask.shape}")
        aoi_mask = aoi_mask.bool().squeeze(1)   # [B,H,W]
        change_mask = change_mask & aoi_mask

    # Broadcast change mask to [B,T,C_sel,H,W]
    if change_mask.any():
        mask = change_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,H,W]
        mask = mask.expand_as(diff2)                  # [B,T,C_sel,H,W]
        diff2 = diff2 * mask
        valid = mask.sum()
    else:
        return torch.tensor(0.0, device=device)

    if valid == 0:
        return torch.tensor(0.0, device=device)

    return diff2.sum() / valid
  
  
def temporal_derivative_mse(
    recon_full: torch.Tensor,
    x_full: torch.Tensor,
    aoi_mask: torch.Tensor | None = None,
    channel_indices: list[int] | None = None,
    change_thresh: float = 0.05,
) -> torch.Tensor:
    """
    MSE between temporal derivatives:

        (x(t) - x(t-1)) vs (recon(t) - recon(t-1))

    Focuses on timesteps/pixels where the *input* derivative magnitude
    exceeds `change_thresh`, so we emphasize actual change events.

    Args
    ----
    recon_full, x_full : [B, T, C, H, W]  (normalized space)
    aoi_mask           : [B,H,W] or [B,1,H,W], optional
    channel_indices    : list[int] or None (None -> all continuous channels)
    change_thresh      : float threshold in normalized units.
    """
    assert recon_full.shape == x_full.shape, "Shape mismatch recon_full/x_full"
    B, T, C, H, W = x_full.shape
    device = x_full.device

    if channel_indices is None:
        x_sel = x_full              # [B,T,C,H,W]
        r_sel = recon_full
    else:
        x_sel = x_full[:, :, channel_indices]      # [B,T,C_sel,H,W]
        r_sel = recon_full[:, :, channel_indices]

    # Temporal derivatives along t
    dx = x_sel[:, 1:] - x_sel[:, :-1]   # [B,T-1,C_sel,H,W]
    dr = r_sel[:, 1:] - r_sel[:, :-1]   # [B,T-1,C_sel,H,W]

    diff2 = (dx - dr) ** 2

    # Magnitude of input derivative across channels
    dx_mag = dx.pow(2).sum(dim=2).sqrt()    # [B,T-1,H,W]
    change_mask = dx_mag > change_thresh    # [B,T-1,H,W]

    if aoi_mask is not None:
        if aoi_mask.dim() == 3:
            aoi_mask = aoi_mask.unsqueeze(1)    # [B,1,H,W]
        elif aoi_mask.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected aoi_mask shape: {aoi_mask.shape}")
        aoi_mask = aoi_mask.bool().squeeze(1)   # [B,H,W]
        change_mask = change_mask & aoi_mask.unsqueeze(1)  # [B,T-1,H,W]

    if not change_mask.any():
        return torch.tensor(0.0, device=device)

    # Broadcast change mask to derivative tensor
    mask = change_mask.unsqueeze(2)            # [B,T-1,1,H,W]
    mask = mask.expand_as(diff2)               # [B,T-1,C_sel,H,W]
    diff2 = diff2 * mask
    valid = mask.sum()

    if valid == 0:
        return torch.tensor(0.0, device=device)

    return diff2.sum() / valid
  
class ForestTrajectoryVAELoss:
    """
    Bundle all the per-batch loss logic for the forest trajectory VAE.

    Computes:
      - continuous recon loss with final-frame weighting
      - delta-from-final loss
      - temporal derivative loss
      - KL
      - categorical recon loss (via embeddings)

    and returns both the total and individual components.
    """

    def __init__(
        self,
            beta: float,
            lambda_cat: float,
            lambda_delta: float = 10.0,
            lambda_deriv: float = 10.0,
            w_final: float = 2.0,  # Extra weight on last year
            channel_indices: list[int] | None = None, # Continuous Channels with time loss
            change_thresh: float = 0.05, # time deltas less than this incur no loss
    ):
        self.beta = beta
        self.lambda_cat = lambda_cat
        self.lambda_delta = lambda_delta
        self.lambda_deriv = lambda_deriv
        self.w_final = w_final
        self.channel_indices = channel_indices if channel_indices is not None else [0]
        self.change_thresh = change_thresh

    def __call__(
        self,
        recon_full: torch.Tensor,   # [B, T, C_all, H, W]
        x_cont_norm: torch.Tensor,  # [B, T, C_cont, H, W]
        x_cat: torch.Tensor | None,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        aoi_mask: torch.Tensor,     # [B, 1, H, W] or [B, H, W]
        C_cont: int,
        cat_encoder: CategoricalEmbeddingEncoder | None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss terms for a single batch.

        Returns a dict of scalar tensors:
          - "loss"
          - "cont_recon"
          - "delta_loss"
          - "deriv_loss"
          - "cat_loss"
          - "kl"
        """
        # ---------------------------------------
        # Continuous reconstruction loss
        # ---------------------------------------
        recon_cont = recon_full[:, :, :C_cont, ...]  # [B,T,C_cont,H,W]
        recon_loss = final_frame_weighted_mse(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            w_final=self.w_final,
            aoi_mask=aoi_mask,
        )

        # ---------------------------------------
        # Delta-from-final loss
        # ---------------------------------------
        delta_loss = delta_from_final_mse_all(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            aoi_mask=aoi_mask,
            channel_indices=self.channel_indices,
            change_thresh=self.change_thresh,
        )

        # ---------------------------------------
        # Temporal derivative loss
        # ---------------------------------------
        deriv_loss = temporal_derivative_mse(
            recon_full=recon_cont,
            x_full=x_cont_norm,
            aoi_mask=aoi_mask,
            channel_indices=self.channel_indices,
            change_thresh=self.change_thresh,
        )

        # ---------------------------------------
        # KL
        # ---------------------------------------
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weighted = self.beta * kl

        # ---------------------------------------
        # Categorical reconstruction loss
        # ---------------------------------------
        cat_loss = categorical_recon_loss_from_embeddings(
            recon_full=recon_full,
            x_cat=x_cat,
            cat_encoder=cat_encoder,
            C_cont=C_cont,
        )

        # Handle non-finite cat_loss here so callers don't duplicate logic
        if not torch.isfinite(cat_loss):
            if DEBUG_CAT_LOSS:
                print(
                    "[DEBUG] non-finite cat_loss in ForestTrajectoryVAELoss; "
                    "setting to zero."
                )
            cat_loss = torch.tensor(0.0, device=recon_full.device)

        # ---------------------------------------
        # Total loss
        # ---------------------------------------
        loss = (
            recon_loss
            + self.lambda_delta * delta_loss
            + self.lambda_deriv * deriv_loss
            + kl_weighted
            + self.lambda_cat * cat_loss
        )

        return {
            "loss": loss,
            "cont_recon": recon_loss,
            "delta_loss": delta_loss,
            "deriv_loss": deriv_loss,
            "cat_loss": cat_loss,
            "kl": kl,
        }
