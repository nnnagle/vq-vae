#src/model/ForestTrajectoryAE.py
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
        # Apply LayerNorm over channels by permuting to [B, T, C]
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


class ForestTrajectoryAE(nn.Module):
    """
    ForestTrajectoryAE
    ------------------

    A temporal + spatial autoencoder that:

      - Accepts input [B, C_in, H, W] where C_in = time_steps * C_per_timestep.
      - Reshapes to [B, T, C_per_timestep, H, W].
      - Applies a 1x1 conv "feature embedding" per timestep: C_per_timestep -> feature_channels.
      - Runs a 1D temporal encoder per pixel (sequence over T).
      - Applies a 2D dilated residual stack over space (spatial context).
      - Decodes back to [B, C_in, H, W] via a simple temporal decoder + 1x1 conv.

    This is a deterministic autoencoder: we return (recon, mu, logvar) only
    so we can drop it into the existing VAE trainer. mu/logvar are zeros, so
    any KL term will be identically zero.
    """

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        feature_channels: int = 32,
        temporal_hidden: int = 64,
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
        self.temporal_hidden = temporal_hidden

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
        #   - mean over time -> [B*H*W, temporal_hidden]
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

        # ------------------------------------------------------------------
        # Spatial context: 2D dilated residual stack
        # Input:  [B, temporal_hidden, H, W]
        # Output: [B, feature_channels, H, W]  (via channel projection)
        # ------------------------------------------------------------------
        self.spatial_blocks = nn.Sequential(
            ResDilatedBlock2d(self.temporal_hidden, 64, kernel_size=3, dilation=1),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=2),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=4),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=8),
            ResDilatedBlock2d(64, 64, kernel_size=3, dilation=16),
        )
        self.channel_proj = nn.Conv2d(64, self.feature_channels, kernel_size=1)

        # ------------------------------------------------------------------
        # Decoder:
        #   code_to_traj:   [B, feature_channels, H, W] -> [B, 64, H, W]
        #   temporal_dec:   [B, 64, H, W] -> [B, T*feature_channels, H, W]
        #   feature_dec:    per-timestep 1x1 conv to C_per_t
        # ------------------------------------------------------------------
        self.code_to_traj = nn.Conv2d(
            in_channels=self.feature_channels,
            out_channels=64,
            kernel_size=1,
        )

        # This is a cheap "resmlp_to_sequence" approximation:
        # we learn a per-pixel linear map 64 -> T*feature_channels.
        self.temporal_decoder = nn.Conv2d(
            in_channels=64,
            out_channels=self.time_steps * self.feature_channels,
            kernel_size=1,
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
        # Feature embedding per timestep
        #   - apply 1x1 conv to each timestep independently
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

        h = self.temporal_proj(x_seq)
        for block in self.temporal_blocks:
            h = block(h)

        # "attention pool": simple mean over time
        h_pool = h.mean(dim=-1)  # [B*H*W, temporal_hidden]

        # reshape back to spatial feature map [B, temporal_hidden, H, W]
        h_spatial = (
            h_pool.view(B, H, W, self.temporal_hidden)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # --------------------------------------------------------------
        # Spatial context with dilated residual blocks
        # --------------------------------------------------------------
        h_ctx = self.spatial_blocks(h_spatial)
        latent = self.channel_proj(h_ctx)  # [B, feature_channels, H, W]

        # --------------------------------------------------------------
        # Decoder
        # --------------------------------------------------------------
        d = self.code_to_traj(latent)              # [B, 64, H, W]
        d = self.temporal_decoder(d)               # [B, T*feature_channels, H, W]
        d = d.view(B, T, self.feature_channels, H, W)

        # Per-timestep 1x1 conv to original per-time channels
        d_flat = d.view(B * T, self.feature_channels, H, W)
        out_per_t = self.feature_decoder(d_flat)   # [B*T, C_per_t, H, W]

        recon = out_per_t.view(B, T * self.c_per_t, H, W)

        # Dummy mu/logvar so Trainer's VAE code doesn't break
        mu = x.new_zeros(B, 1)
        logvar = x.new_zeros(B, 1)

        return recon, mu, logvar
