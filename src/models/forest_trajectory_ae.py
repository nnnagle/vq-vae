import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    """
    Simple 1D residual block with LayerNorm + SiLU.

    Expects input of shape [B, C, T].
    """

    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(dim, dim, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size,
                               padding=padding, dilation=dilation)

        # LayerNorm over channel dimension; we permute to [B, T, C] for LN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = x.permute(0, 2, 1)          # [B, T, C]
        h = self.norm1(h)
        h = self.act(h)
        h = h.permute(0, 2, 1)          # [B, C, T]
        h = self.conv1(h)

        h2 = h.permute(0, 2, 1)
        h2 = self.norm2(h2)
        h2 = self.act(h2)
        h2 = h2.permute(0, 2, 1)
        h2 = self.conv2(h2)

        return x + h2


class ResDilatedBlock2d(nn.Module):
    """
    2D residual block with dilation, GroupNorm and SiLU.

    Used for "spatial context" with increasing dilation to gather
    a larger receptive field without pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 8,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

        # Match channels for residual connection if needed
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return self.skip(x) + h


class JointVectorQuantizerEMA(nn.Module):
    """
    EMA-based vector quantizer over a 2D feature map.

    Expects input x: [B, D, H, W]
    Returns:
      - x_q_st: quantized tensor with straight-through gradients, same shape as x
      - vq_loss: codebook + commitment loss
      - codes: [B, H, W] integer code indices
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost

        # Codebook: [K, D]
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.normal_()

        # EMA buffers
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", torch.zeros(num_codes, code_dim))

    def forward(self, x: torch.Tensor):
        """
        x: [B, D, H, W]
        """
        B, D, H, W = x.shape
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [N, D], N = B*H*W

        emb = self.embedding.weight  # [K, D]

        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            + emb.pow(2).sum(dim=1)
            - 2 * flat_x @ emb.t()
        )  # [N, K]

        codes = torch.argmin(distances, dim=1)  # [N]
        flat_x_q = self.embedding(codes)        # [N, D]

        x_q = flat_x_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        if self.training:
            one_hot = F.one_hot(codes, self.num_codes).type_as(flat_x)  # [N, K]
            cluster_size = one_hot.sum(dim=0)                            # [K]
            embed_sum = one_hot.t() @ flat_x                             # [K, D]

            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace-smoothed cluster sizes
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.num_codes * self.eps)
                * n
            )

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # VQ loss: codebook + commitment
        vq_loss = (
            F.mse_loss(x_q.detach(), x)
            + self.commitment_cost * F.mse_loss(x_q, x.detach())
        )

        # Straight-through estimator
        x_q_st = x + (x_q - x).detach()

        codes_map = codes.view(B, H, W)

        return x_q_st, vq_loss, codes_map


import torch
import torch.nn as nn
import torch.nn.functional as F

class JointVectorQuantizerEMA_LowMem(nn.Module):
    """
    Low-memory VQ layer using EMA codebook updates.

    This does exactly what a normal VQ-VAE quantizer does, but avoids materializing
    the full [N, K] distance matrix or one-hot assignment matrix — both are too
    expensive when:
         - N = batch * spatial positions
         - K = codebook size

    Instead:
        * distances are computed in codebook chunks → reduces peak memory.
        * assignment accumulation is done via bincount + index_add (no one-hot).
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        k_chunk_size: int = 32,   # number of codes processed at a time
    ):
        """
        Args:
            num_codes: number of codebook entries K.
            code_dim: dimensionality D of each embedding and x channel size.
            decay: EMA decay rate for codebook stats.
            eps: numerical stability constant for smoothing.
            commitment_cost: weight for VQ commitment loss term.
            k_chunk_size: chunk size for streaming over codebook to reduce memory.
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.k_chunk_size = k_chunk_size

        # Codebook parameters: [K, D]
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.normal_()

        # EMA tracking buffers — not learnable parameters
        self.register_buffer("cluster_size", torch.zeros(num_codes))      # [K]
        self.register_buffer("embed_avg", torch.zeros(num_codes, code_dim))  # [K, D]


    def forward(self, x: torch.Tensor):
        """
        Input:
            x: latent tensor [B, D, H, W]

        Returns:
            x_q_st: straight-through output, same shape as x
            vq_loss: scalar tensor
            codes_map: discrete code assignments [B, H, W]
        """
        B, D, H, W = x.shape
        device = x.device

        # Flatten spatial dims → one vector per spatial position
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, D)   # [N, D]
        N = flat_x.shape[0]

        emb = self.embedding.weight  # full codebook [K, D]
        K = emb.shape[0]

        # Precompute ||x||^2 term for all x (~free) → reused per chunk
        x2 = flat_x.pow(2).sum(dim=1, keepdim=True)  # [N, 1]

        # Track the running nearest code assignment per input vector
        best_dist = torch.full((N,), float("inf"), device=device)
        best_idx  = torch.zeros(N, dtype=torch.long, device=device)

        # --------------------------------------------------------------------
        # CHUNKED DISTANCE COMPUTATION
        # Stream the codebook in chunks to avoid computing [N x K] distances
        # all at once (which is too memory-hungry for large K or large spatial maps).
        # --------------------------------------------------------------------
        for start in range(0, K, self.k_chunk_size):
            end = min(start + self.k_chunk_size, K)

            # emb_chunk: subset of codebook [k_chunk, D]
            emb_chunk = emb[start:end]

            # Precompute squared norm ||e||^2 per code in chunk
            e2_chunk = emb_chunk.pow(2).sum(dim=1)  # [k_chunk]

            # Compute pairwise distances between x and chunk of codes:
            #   ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
            dist_chunk = x2 + e2_chunk.unsqueeze(0) - 2 * (flat_x @ emb_chunk.t())  # [N, k_chunk]

            # For each x, find the closest code within the chunk
            chunk_min_dist, chunk_min_idx = dist_chunk.min(dim=1)

            # Update global best where this chunk beats previous best
            better = chunk_min_dist < best_dist
            best_dist[better] = chunk_min_dist[better]
            best_idx[better] = chunk_min_idx[better] + start  # offset local index

            del dist_chunk  # release reference to help fragmentation

        codes = best_idx  # Final nearest-code indices [N]

        # Lookup quantized embeddings
        flat_x_q = self.embedding(codes)  # [N, D]

        # Reshape quantized vectors to original layout
        x_q = flat_x_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # --------------------------------------------------------------------
        # EMA CODEBOOK UPDATE (NO ONE-HOT)
        # --------------------------------------------------------------------
        if self.training:
            # Count assignments per code k — no full one-hot matrix needed
            cluster_size = torch.bincount(codes, minlength=K).float()  # [K]

            # embed_sum[k] = sum of vectors assigned to that code
            # Using index_add avoids creating a huge sparse matrix
            embed_sum = torch.zeros(K, D, device=device)
            embed_sum.index_add_(0, codes, flat_x)

            # EMA accumulation
            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Smooth small clusters to avoid dead entries / division by zero
            n = self.cluster_size.sum()
            cluster_size_smoothed = (
                (self.cluster_size + self.eps)
                / (n + K * self.eps)
                * n
            )

            # Normalize average vectors → updated codebook means
            embed_normalized = self.embed_avg / cluster_size_smoothed.unsqueeze(1)

            # Update codebook in-place without letting gradients through
            with torch.no_grad():
                self.embedding.weight.copy_(embed_normalized)

        # --------------------------------------------------------------------
        # LOSS AND STRAIGHT-THROUGH ESTIMATOR
        # --------------------------------------------------------------------
        vq_loss = (
            F.mse_loss(x_q.detach(), x)                     # codebook loss
            + self.commitment_cost * F.mse_loss(x_q, x.detach())  # commitment loss
        )

        # Forward uses quantized signal; backward uses original gradient
        x_q_st = x + (x_q - x).detach()

        # Reshape discrete assignments back to spatial form
        codes_map = codes.view(B, H, W)

        return x_q_st, vq_loss, codes_map

class ForestTrajectoryAE(nn.Module):
    """
    ForestTrajectoryAE + joint VQ
    -----------------------------

    A temporal + spatial autoencoder that:

      - Accepts input [B, C_in, H, W] where C_in = time_steps * C_per_timestep.
      - Reshapes to [B, T, C_per_timestep, H, W].
      - Applies a 1x1 conv "feature embedding" per timestep: C_per_timestep -> feature_channels.
      - Runs a 1D temporal encoder per pixel (sequence over T).
      - Splits temporal representation into:
          * z_traj (trajectory-type code, size = feature_channels)
          * age (scalar "time since event").
      - Applies a 2D dilated residual stack over space on z_traj.
      - Forms a joint spatial map [z_traj, spatial context, rescaled age].
      - Quantizes that joint map with a VQ codebook.
      - Decodes from the quantized joint state.

    The forward still returns:
      - recon:  [B, C_in, H, W]
      - mu:     [B, 1]  (zeros, for compatibility)
      - logvar: [B, 1]  (zeros, for compatibility)
    """

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        feature_channels: int = 48,
        temporal_hidden: int = 64,
        num_codes: int = 512,
        vq_decay: float = 0.99,
        vq_commitment_cost: float = 0.25,
        age_rescale: float = 1.0,  # manual scale factor after tanh
    ):
        super().__init__()

        assert in_channels % time_steps == 0, (
            f"in_channels={in_channels} must be divisible by "
            f"time_steps={time_steps}"
        )

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.c_per_t = in_channels // time_steps
        self.feature_channels = feature_channels
        self.context_channels = feature_channels
        self.temporal_hidden = temporal_hidden
        self.age_rescale = age_rescale

        # ------------------------------------------------------------------
        # Feature embedding: per-timestep 1x1 conv over spatial dims
        # [B, T, C_per_t, H, W] -> [B, T, feature_channels, H, W]
        # ------------------------------------------------------------------
        self.feature_embedding = nn.Conv2d(
            in_channels=self.c_per_t,
            out_channels=self.feature_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # ------------------------------------------------------------------
        # Temporal encoder (per pixel)
        #   - reshape to [B*H*W, feature_channels, T]
        #   - Conv1d proj + a couple of ResBlock1d
        #   - learned aggregator over time -> [B*H*W, temporal_hidden]
        # ------------------------------------------------------------------
        self.temporal_proj = nn.Conv1d(
            in_channels=self.feature_channels,
            out_channels=self.temporal_hidden,
            kernel_size=1,
        )
        self.temporal_blocks = nn.ModuleList(
            [
                ResBlock1d(self.temporal_hidden, kernel_size=3, dilation=1),
                ResBlock1d(self.temporal_hidden, kernel_size=3, dilation=2),
            ]
        )

        # Learned temporal aggregation across the full sequence
        self.temporal_agg = nn.Conv1d(
            in_channels=self.temporal_hidden,
            out_channels=self.temporal_hidden,
            kernel_size=self.time_steps,  # span entire T
            padding=0,
        )

        # Heads: trajectory-type code and age (time-since-event scalar)
        self.traj_head = nn.Linear(self.temporal_hidden, self.feature_channels)
        self.age_head = nn.Linear(self.temporal_hidden, 1)

        # ------------------------------------------------------------------
        # Spatial context: 2D dilated residual stack
        # Input:  [B, feature_channels, H, W]  (z_traj map)
        # Output: [B, feature_channels, H, W]  (via channel projection)
        # ------------------------------------------------------------------
        self.spatial_blocks = nn.Sequential(
            ResDilatedBlock2d(self.feature_channels, 64, kernel_size=3, dilation=1),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=2),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=4),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=8),
            # ResDilatedBlock2d(64, 64, kernel_size=3, dilation=16),
        )
        self.channel_proj = nn.Conv2d(64, self.context_channels, kernel_size=1)

        # ------------------------------------------------------------------
        # Joint VQ over [z_traj, spatial context, age_rescaled]
        # latent_channels = feature_channels (z_traj) + context_channels
        # joint_dim = latent_channels + 1 (age scalar channel)
        # ------------------------------------------------------------------
        self.latent_channels = self.feature_channels + self.context_channels
        joint_dim = self.latent_channels + 1

        self.vq = JointVectorQuantizerEMA_LowMem(
            num_codes=num_codes,
            code_dim=joint_dim,
            decay=vq_decay,
            commitment_cost=vq_commitment_cost,
        )

        # ------------------------------------------------------------------
        # Decoder:
        #   joint_q: [B, joint_dim, H, W] after VQ
        #   code_to_traj:   -> [B, 64, H, W]
        #   temporal_dec:   -> [B, T*feature_channels, H, W]
        #   feature_dec:    per-timestep 1x1 conv to C_per_t
        # ------------------------------------------------------------------
        self.code_to_traj = nn.Sequential(
            nn.Conv2d(
                in_channels=joint_dim,
                out_channels=64,
                kernel_size=1,
            ),
            nn.SiLU(),
        )

        self.temporal_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=self.time_steps * self.feature_channels,
                kernel_size=1,
            ),
            nn.SiLU(),
        )

        # Temporal refinement in the decoder:
        self.dec_temporal_blocks = nn.ModuleList(
            [
                ResBlock1d(self.feature_channels, kernel_size=3, dilation=1),
                # ResBlock1d(self.feature_channels, kernel_size=3, dilation=2),
            ]
        )

        self.feature_decoder = nn.Conv2d(
            in_channels=self.feature_channels,
            out_channels=self.c_per_t,
            kernel_size=1,
        )

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x: [B, C_in, H, W], where C_in = time_steps * C_per_timestep
        returns:
          - recon:  [B, C_in, H, W]
          - mu:     [B, 1]  (zeros, for compatibility)
          - logvar: [B, 1]  (zeros, for compatibility)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, "Unexpected channel count."

        T = self.time_steps

        # --------------------------------------------------------------
        # Reshape to [B, T, C_per_t, H, W]
        # --------------------------------------------------------------
        x_t = x.view(B, T, self.c_per_t, H, W)

        # --------------------------------------------------------------
        # Feature embedding per timestep: [B*T, C_per_t, H, W] -> [B, T, C_feat, H, W]
        # --------------------------------------------------------------
        x_emb = self.feature_embedding(
            x_t.reshape(B * T, self.c_per_t, H, W)
        )  # [B*T, feature_channels, H, W]
        x_emb = x_emb.view(B, T, self.feature_channels, H, W)

        # --------------------------------------------------------------
        # Temporal encoder (per pixel)
        #   - rearrange to [B*H*W, feature_channels, T]
        # --------------------------------------------------------------
        x_seq = (
            x_emb.permute(0, 3, 4, 2, 1)
            .contiguous()
            .view(B * H * W, self.feature_channels, T)
        )

        h = self.temporal_proj(x_seq)  # [B*H*W, temporal_hidden, T]
        for block in self.temporal_blocks:
            h = block(h)

        # Learned aggregation over time: [N, Hid, T] -> [N, Hid]
        h_pool = self.temporal_agg(h).squeeze(-1)

        # Split into trajectory code and age
        z_traj = self.traj_head(h_pool)   # [N, feature_channels]
        age = self.age_head(h_pool)       # [N, 1]

        # Reshape to spatial maps
        z_traj_map = (
            z_traj.view(B, H, W, self.feature_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [B, feature_channels, H, W]

        age_map = (
            age.view(B, H, W, 1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [B, 1, H, W]

        # --------------------------------------------------------------
        # Spatial context on trajectory-type code
        # --------------------------------------------------------------
        h_ctx = self.spatial_blocks(z_traj_map)
        h_ctx = self.channel_proj(h_ctx)  # [B, feature_channels, H, W]

        # Latent combining per-pixel trajectory code and spatial refinement
        latent = torch.cat([z_traj_map, h_ctx], dim=1)   # [B, latent_channels, H, W]

        # --------------------------------------------------------------
        # Joint VQ over [latent, rescaled age]
        # Manual rescale of age: bound via tanh, then scale
        # --------------------------------------------------------------
        age_rescaled = torch.tanh(age_map) * self.age_rescale  # [B, 1, H, W]

        joint = torch.cat([latent, age_rescaled], dim=1)       # [B, joint_dim, H, W]

        joint_q, vq_loss, codes = self.vq(joint)

        # Expose VQ stats for trainer / diagnostics
        self._vq_loss = vq_loss
        self._vq_codes = codes

        # --------------------------------------------------------------
        # Decoder from quantized joint state
        # --------------------------------------------------------------
        d = self.code_to_traj(joint_q)               # [B, 64, H, W]
        d = self.temporal_decoder(d)                 # [B, T*feature_channels, H, W]
        d = d.view(B, T, self.feature_channels, H, W)

        # --------------------------------------------------------------
        # Temporal refinement in decoder: [B*H*W, C_feat, T]
        # --------------------------------------------------------------
        d_seq = (
            d.permute(0, 3, 4, 2, 1)    # [B, H, W, C_feat, T]
            .contiguous()
            .view(B * H * W, self.feature_channels, T)
        )

        for block in self.dec_temporal_blocks:
            d_seq = block(d_seq)

        # reshape back to [B, T, C_feat, H, W]
        d = (
            d_seq.view(B, H, W, self.feature_channels, T)
            .permute(0, 4, 3, 1, 2)     # [B, T, C_feat, H, W]
            .contiguous()
        )

        # Per-timestep 1x1 conv to original per-time channels
        d_flat = d.view(B * T, self.feature_channels, H, W)
        out_per_t = self.feature_decoder(d_flat)     # [B*T, C_per_t, H, W]

        recon = out_per_t.view(B, T * self.c_per_t, H, W)

        # Dummy mu/logvar so Trainer's VAE code doesn't break
        mu = x.new_zeros(B, 1)
        logvar = x.new_zeros(B, 1)

        return recon, mu, logvar
