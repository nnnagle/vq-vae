# src/training/categorical_eval.py

from __future__ import annotations

from typing import Dict, Tuple

import torch

from src.models.categorical_embedding import CategoricalEmbeddingEncoder


def _iter_categorical_embeddings(
    recon_full: torch.Tensor,                 # [B, T, C_all, H, W]
    cat_encoder: CategoricalEmbeddingEncoder,
    C_cont: int,
):
    """
    Yield (fid, recon_emb) for each categorical feature in the order
    used by the encoder.

    recon_emb is [B, T, D, H, W] where D = emb_dims[fid].
    This matches the slicing convention used in categorical_recon_loss_from_embeddings.
    """
    B, T, C_all, H, W = recon_full.shape
    offset = C_cont  # first C_cont channels are continuous

    for fid in cat_encoder.feature_ids:
        D = cat_encoder.emb_dims[fid]  # embedding dim for this feature
        recon_emb = recon_full[:, :, offset : offset + D, ...]  # [B, T, D, H, W]
        offset += D
        yield fid, recon_emb


def print_categorical_histograms(
    recon_full: torch.Tensor,                 # [B, T, C_all, H, W]
    x_cat: torch.Tensor,                      # [B, T, C_cat, H, W]
    cat_encoder: CategoricalEmbeddingEncoder,
    num_classes: Dict[str, int],             # fid -> K (no pad)
    C_cont: int,
) -> None:
    """
    Print original vs reconstructed histograms for each categorical feature.

    Uses the same embedding → logits machinery as categorical_recon_loss_from_embeddings:
      logits = recon_emb_flat @ cat_encoder.embs[fid].weight.T

    Assumptions:
      - x_cat[..., c_idx, ...] contains integer codes for feature fid = feature_ids[c_idx]
      - codes >= 0 are valid, codes < 0 are missing / ignored in histograms
      - cat_encoder.embs[fid].weight has shape [K+1, D], where the extra row is pad/missing
      - num_classes[fid] = K (number of "real" classes 0..K-1)
    """
    if cat_encoder is None or x_cat is None:
        print("[INFO] no categorical encoder or data; skipping categorical histograms.")
        return

    device = recon_full.device
    B, T, C_cat, H, W = x_cat.shape

    print("\n[CATEGORICAL RECONSTRUCTION HISTOGRAMS]")

    # Map fid -> channel index in x_cat
    fid_to_cidx = {fid: idx for idx, fid in enumerate(cat_encoder.feature_ids)}

    for fid, recon_emb in _iter_categorical_embeddings(
        recon_full=recon_full,
        cat_encoder=cat_encoder,
        C_cont=C_cont,
    ):
        c_idx = fid_to_cidx.get(fid, None)
        if c_idx is None or c_idx >= C_cat:
            print(f"  feature '{fid}': no matching channel in x_cat, skipping.")
            continue

        # ground-truth integer codes: [B, T, H, W]
        codes = x_cat[:, :, c_idx, ...].long().to(device)
        codes_flat = codes.reshape(-1)  # [N]

        # valid data positions: codes >= 0
        valid_data_mask = (codes_flat >= 0)
        if not valid_data_mask.any():
            print(f"  feature '{fid}': all values are missing in this batch, skipping.")
            continue

        # predicted embeddings: [B, T, D, H, W] -> [N, D]
        recon_flat = (
            recon_emb.permute(0, 1, 3, 4, 2)  # [B,T,H,W,D]
            .reshape(-1, recon_emb.shape[2])  # [N,D]
        )

        # embedding matrix for this feature: [K+1, D]
        embs = cat_encoder.embs[fid].weight   # [K+1, D]
        logits = recon_flat @ embs.t()        # [N, K+1]

        # predicted class indices: [N]
        preds_flat = logits.argmax(dim=1)

        # number of "real" classes (excluding pad/missing index)
        K = int(num_classes[fid])

        # mask positions where:
        #  - original is non-missing (>=0)
        #  - predicted class in [0, K-1] (ignore pad index, etc.)
        mask = valid_data_mask & (preds_flat < K)
        if not mask.any():
            print(f"  feature '{fid}': no valid positions after masking, skipping.")
            continue

        true_valid = codes_flat[mask].cpu()
        pred_valid = preds_flat[mask].cpu()

        # histograms over 0..K-1
        orig_counts = torch.bincount(true_valid, minlength=K).numpy()
        recon_counts = torch.bincount(pred_valid, minlength=K).numpy()

        print(f"\n  feature '{fid}' (K={K})")
        for cls in range(K):
            print(
                f"    class {cls:2d}: "
                f"orig={int(orig_counts[cls])}  "
                f"recon={int(recon_counts[cls])}"
            )
