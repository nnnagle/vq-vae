"""
vqvae/model.py
---------------
Defines the neural architecture for the Vector Quantized Variational Autoencoder (VQ-VAE),
including encoder, decoder, and vector quantization modules.

Purpose
    - Provide reusable PyTorch modules for training and inference of the VQ-VAE.
    - Expose modular components (Encoder, Decoder, Quantizer) for use in training, decoding,
      and novel-data prediction pipelines.
    - Serve as the central definition of model architecture, separating logic from training scripts.

Used by
    - scripts/train_vqvae.py : main training entrypoint; instantiates VQVAE and runs optimization.
    - scripts/decode_codebook.py : loads a trained model to decode code indices into feature space.
    - scripts/predict.py : performs inference on novel data using pretrained checkpoints.
    - vqvae/api.py : wraps model methods for high-level encoding, decoding, and prediction.

Design notes
    - Encoder/decoder operate on mixed continuous and categorical features plus NAIP imagery.
    - Quantizer uses nearest-neighbor lookup with straight-through gradient estimation.
    - Model forward() returns both reconstruction outputs and latent indices for VQ monitoring.
    - Checkpoints are expected to include `state_dict` and `model_kwargs` for reproducible loading.

Assistant guidance
    When extending or refactoring:
        - Keep class interfaces stable: `encode`, `quantize`, and `decode` are externally used.
        - Do not import training-specific utilities (logging, schedulers) here—keep this pure model code.
        - Preserve tensor shape conventions: `[B, T, D]` for temporal embeddings, `[B, C, H, W]` for imagery.
        - Maintain device-agnostic code (`.to(device)` only in higher-level scripts).
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch, torch.nn as nn, torch.nn.functional as F
from .codebook_manager import CodebookManager

class MixedInputEncoder(nn.Module):
    """
    Inputs:
      - cont: [B,T,C_cont] (z-scored; NaNs possible)
      - cat:  [B,T,C_cat]  (dense IDs; used by embeddings)
      - naip: [B,Bn,3,3]   (single-band canopy height patch; Bn==1)
    Output:
      - z_e:  [B,T,D]      (pre-quantization latent)
    """
    def __init__(self, cont_dim: int, cat_dims: List[int], naip_bands: int,
                 emb_dim: int, cat_emb_dim: int = 6, hidden: int = 128):
        super().__init__()
        # per-categorical Embedding tables
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(v, cat_emb_dim) if v > 0 else None for v in cat_dims]
        )

        # tiny CNN for NAIP patch (bands-first; here naip_bands==1)
        self.naip_cnn = nn.Sequential(
            nn.Conv2d(in_channels=naip_bands, out_channels=hidden, kernel_size=3),  # -> [B,hidden,1,1]
            nn.ReLU(),
        )
        self.naip_proj = nn.Linear(hidden, hidden)

        # projections for other inputs
        self.cont_proj = nn.Linear(max(cont_dim, 1), hidden) if cont_dim > 0 else None
        self.cat_dim_total = sum(cat_emb_dim for v in cat_dims if v > 0)
        self.cat_proj = nn.Linear(max(self.cat_dim_total, 1), hidden) if self.cat_dim_total > 0 else None

        # fuse to latent
        fuse_in = hidden + (hidden if self.cont_proj else 0) + (hidden if self.cat_proj else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # T inferred from cont if present; else from cat
        if batch["cont"].ndim == 3:
            B, T, _ = batch["cont"].shape
        else:
            B, T, _ = batch["cat"].shape

        # NAIP context (shared across T)
        naip = torch.nan_to_num(batch["naip"], nan=0.0, posinf=0.0, neginf=0.0)
        naip_feat = self.naip_cnn(naip).flatten(1)  # [B,hidden]
        naip_feat = self.naip_proj(naip_feat)
        naip_rep = naip_feat.unsqueeze(1).expand(-1, T, -1) # [B,T,hidden]

        # Continuous path
        if self.cont_proj is not None:
            cont = torch.nan_to_num(batch["cont"], nan=0.0, posinf=0.0, neginf=0.0)
            cont_h = self.cont_proj(cont)                    # [B,T,hidden]
        else:
            cont_h = torch.zeros(B, T, 0, device=naip_rep.device)

        # Categorical path
        if self.cat_proj is not None and self.cat_dim_total > 0:
            cat_ids = batch["cat"]                           # [B,T,K]
            cat_embs = []
            col = 0
            for emb in self.cat_embs:
                if emb is None:
                    continue
                cat_embs.append(emb(cat_ids[..., col]))     # [B,T,cat_emb_dim]
                col += 1
            if cat_embs:
                cat_concat = torch.cat(cat_embs, dim=-1)
            else:
                cat_concat = torch.zeros(B, T, 0, device=naip_rep.device)
            cat_h = self.cat_proj(cat_concat)               # [B,T,hidden]
        else:
            cat_h = torch.zeros(B, T, 0, device=naip_rep.device)

        fused = torch.cat([naip_rep, cont_h, cat_h], dim=-1)
        z_e = self.fuse(fused)                               # [B,T,D]
        return z_e

class VectorQuantizerST(nn.Module):
    """Straight-Through Vector Quantizer (ST-VQ)."""
    def __init__(self, codebook_size: int, emb_dim: int, beta: float = 0.25, 
                 codebook_manager: Optional[CodebookManager] = None):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.codebook = nn.Parameter(torch.randn(codebook_size, emb_dim) * 0.05)
        self.codebook_manager = codebook_manager
        
        print(f"Quantizer choice at build: st")

    def forward(self, z_e: torch.Tensor):
        # flatten
        z = z_e.reshape(-1, self.emb_dim)                    # [N,D]
        z_sq = (z ** 2).sum(dim=1, keepdim=True)             # [N,1]
        e_sq = (self.codebook ** 2).sum(dim=1)               # [K]
        distances = z_sq + e_sq.unsqueeze(0) - 2 * (z @ self.codebook.t())
        indices = torch.argmin(distances, dim=1)             # [N]
        z_q = F.embedding(indices, self.codebook).view_as(z_e)

        # VQ losses
        commitment = F.mse_loss(z_e.detach(), z_q, reduction="mean")
        codebook = F.mse_loss(z_e, z_q.detach(), reduction="mean")
        z_q_st = z_e + (z_q - z_e).detach()
        loss_vq = codebook + self.beta * commitment

        # --- CodebookManager integration ---
        if self.training and self.codebook_manager is not None:
            self.codebook_manager.push_reservoir(z)
            self.codebook_manager.observe_assignments(indices)
            self.codebook_manager.maybe_reseed(self.codebook)

        # Perplexity (always scalar)
        if self.codebook_manager is not None:
            perplexity = self.codebook_manager.stats()["perplexity"]
        else:
            with torch.no_grad():
                one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
                avg_probs = one_hot.mean(dim=0)
                perplexity = torch.exp(- (avg_probs * (avg_probs + 1e-12).log()).sum()).item()

        return z_q_st, indices.view(z_e.shape[:-1]), loss_vq, float(perplexity)

# put this next to VectorQuantizer in model.py

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) updates.
    Matches the ST interface:
      forward(z_e) -> (z_q_st, indices, vq_loss, perplexity)

    z_e: [B, T, D] (any leading shape, last dim = emb_dim)
    """
    def __init__(self, codebook_size: int, emb_dim: int, decay: float = 0.99, eps: float = 1e-5, beta: float = 0.25,
                 codebook_manager: Optional[CodebookManager] = None):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta
        self.codebook_manager = codebook_manager

        # codebook parameters (not updated by gradient; we assign with EMA)
        embed = torch.randn(codebook_size, emb_dim) * 0.05
        self.codebook = nn.Parameter(embed, requires_grad=False)

        # EMA statistics
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", torch.zeros(codebook_size, emb_dim))
        
        print(f"Quantizer choice at build: ema, ema_decay={decay}")

    def forward(self, z_e: torch.Tensor):
        D = self.emb_dim
        z = z_e.reshape(-1, D)  # [N, D]

        # nearest neighbors
        z_sq = (z ** 2).sum(dim=1, keepdim=True)              # [N,1]
        e_sq = (self.codebook ** 2).sum(dim=1)                # [K]
        distances = z_sq + e_sq.unsqueeze(0) - 2 * (z @ self.codebook.t())
        indices = torch.argmin(distances, dim=1)              # [N]
        z_q = self.codebook.index_select(0, indices).view_as(z_e)

        # commitment loss (no codebook loss—EMA handles updates)
        commitment = F.mse_loss(z_e, z_q.detach(), reduction="mean")
        loss_vq = self.beta * commitment

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # EMA updates
        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).to(z.dtype)
            cluster_size_batch = one_hot.sum(dim=0)
            embed_sum_batch = one_hot.t() @ z

            self.ema_cluster_size.mul_(self.decay).add_(cluster_size_batch, alpha=1.0 - self.decay)
            self.ema_embed_sum.mul_(self.decay).add_(embed_sum_batch, alpha=1.0 - self.decay)

            n = self.ema_cluster_size + self.eps * self.codebook_size
            embed_normalized = self.ema_embed_sum / n.unsqueeze(1)
            self.codebook.copy_(embed_normalized)

            # Update manager if provided
            if self.codebook_manager is not None:
                self.codebook_manager.push_reservoir(z)
                self.codebook_manager.observe_assignments(indices)
                self.codebook_manager.maybe_reseed(self.codebook)

        # Scalar perplexity
        if self.codebook_manager is not None:
            perplexity = self.codebook_manager.stats()["perplexity"]
        else:
            with torch.no_grad():
                avg_probs = one_hot.mean(dim=0)
                perplexity = torch.exp(-torch.sum(avg_probs * (avg_probs + 1e-12).log())).item()

        return z_q_st, indices.view(z_e.shape[:-1]), loss_vq, float(perplexity)


class MixedDecoder(nn.Module):
    """
    Decodes latent [B,T,D] into:
      - continuous regression: [B,T,C_cont]
      - categorical logits: dict(name -> [B,T,num_ids])
      - canopy scalar: [B] using final time step only
    """
    def __init__(self, emb_dim: int, cont_dim: int, cat_vocab_sizes: Dict[str, int], hidden: int = 128):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_names = list(cat_vocab_sizes.keys())
        self.cat_vocab_sizes = cat_vocab_sizes

        self.backbone = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.cont_head = nn.Linear(hidden, cont_dim) if cont_dim > 0 else None
        self.cat_heads = nn.ModuleDict({name: nn.Linear(hidden, v) for name, v in cat_vocab_sizes.items()})
        self.canopy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor):
        h = self.backbone(z)  # [B,T,H]
        cont = self.cont_head(h) if self.cont_head is not None else None
        cat_logits = {name: head(h) for name, head in self.cat_heads.items()}

        # canopy from final time step
        h_last = h[:, -1, :]                          # [B,H]
        canopy_pred = self.canopy_head(h_last).squeeze(-1)  # [B]
        return cont, cat_logits, canopy_pred

class VQVAE(nn.Module):
    def __init__(self,
                 cont_dim: int,
                 cat_vocab_sizes: Dict[str, int],
                 naip_bands: int,
                 emb_dim: int,
                 codebook_size: int,
                 beta: float = 0.25,
                 hidden: int = 128,
                 cat_emb_dim: int = 6,
                 quantizer: str = "st",        # {"st", "ema"}
                 ema_decay: float = 0.99,
                 ema_eps: float = 1e-5):
        super().__init__()
        self.encoder = MixedInputEncoder(
            cont_dim=cont_dim,
            cat_dims=list(cat_vocab_sizes.values()),
            naip_bands=naip_bands,
            emb_dim=emb_dim,
            cat_emb_dim=cat_emb_dim,
            hidden=hidden,
        )
        if quantizer.lower() == "st":
          self.quant = VectorQuantizerST(codebook_size=codebook_size, emb_dim=emb_dim, beta=beta)

        elif quantizer.lower() == "ema":
          self.quant = VectorQuantizerEMA(codebook_size=codebook_size, emb_dim=emb_dim,
                                          decay=ema_decay, eps=ema_eps, beta=beta)
        else:
          raise ValueError(f"Unknown quantizer '{quantizer}'. Use 'st' or 'ema'.")
        self.decoder = MixedDecoder(emb_dim=emb_dim, cont_dim=cont_dim,
                                    cat_vocab_sizes=cat_vocab_sizes, hidden=hidden)
        
        # Placeholder: allow trainer to attach manager post-init
        self.codebook_manager: Optional[CodebookManager] = None

    def attach_codebook_manager(self, manager: CodebookManager):
        """Optionally attach a shared CodebookManager (for DDP safety)."""
        self.codebook_manager = manager
        if isinstance(self.quant, (VectorQuantizerST, VectorQuantizerEMA)):
          self.quant.codebook_manager = manager

    def forward(self, batch):
        z_e = self.encoder(batch)
        z_q, _, vq_loss, perplexity = self.quant(z_e)
        cont_pred, cat_logits, canopy_pred = self.decoder(z_q)
        return cont_pred, cat_logits, canopy_pred, vq_loss, perplexity


    # --- Exposed modular API ---
    @torch.inference_mode(False)
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode batch dict -> z_e [B,T,D]."""
        return self.encoder(batch)

    @torch.inference_mode(False)
    def quantize(self, z_e: torch.Tensor):
        """
        Quantize z_e -> (z_q_st, indices, vq_loss, perplexity)
        indices has shape z_e.shape[:-1].
        """
        return self.quant(z_e)

    @torch.inference_mode(False)
    def decode(self, z_q: torch.Tensor):
        """Decode z_q [B,T,D] -> (cont_pred, cat_logits, canopy_pred)."""
        return self.decoder(z_q)
      
      
    # --- Training-friendly combined forward ---
    def forward(self, batch: Dict[str, torch.Tensor]):
        z_e = self.encode(batch)
        z_q, indices, vq_loss, perplexity = self.quantize(z_e)
        cont_pred, cat_logits, canopy_pred = self.decode(z_q)
        # indices are available if the caller needs them, but we keep the
        # forward signature aligned with the trainer’s expectations.
        return cont_pred, cat_logits, canopy_pred, vq_loss, perplexity      
