from __future__ import annotations
"""
Spatial processing modules including gated residual convolutions.

The GatedResidualConv2D applies learnable spatial smoothing while preserving
sharp edges through a gating mechanism.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualConv2D(nn.Module):
    """
    Gated residual 2D convolution for adaptive spatial smoothing.

    This module applies spatial convolutions with a learned gating mechanism
    that controls how much of the smoothed features vs. the original features
    to use at each spatial location. This helps preserve sharp edges while
    smoothing homogeneous regions.

    Architecture:
        input -> conv_layers -> smoothed_features
        input -> gate_network -> gate (per-location weights in [0,1])
        output = gate * smoothed_features + (1 - gate) * input

    Args:
        channels: Number of input and output channels (preserved)
        num_layers: Number of convolutional layers
        kernel_size: Kernel size for spatial convolutions
        padding: Padding size (typically (kernel_size - 1) // 2 for 'same')
        gate_hidden: Hidden dimension for gate network
        gate_kernel_size: Kernel size for gate computation (typically 1)

    Example:
        >>> spatial_conv = GatedResidualConv2D(
        ...     channels=128,
        ...     num_layers=2,
        ...     kernel_size=3,
        ...     padding=1,
        ...     gate_hidden=64,
        ...     gate_kernel_size=1
        ... )
        >>> x = torch.randn(2, 128, 32, 32)
        >>> out = spatial_conv(x)  # [2, 128, 32, 32]
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        gate_hidden: int = 64,
        gate_kernel_size: int = 1,
    ):
        super().__init__()

        self.channels = channels

        # Main convolutional path
        conv_layers = []
        for i in range(num_layers):
            conv_layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            # Add normalization and activation except for last layer
            if i < num_layers - 1:
                conv_layers.append(nn.GroupNorm(min(32, channels), channels))
                conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*conv_layers)

        # Gate network (predicts per-location, per-channel gates)
        gate_pad = (gate_kernel_size - 1) // 2
        self.gate_network = nn.Sequential(
            nn.Conv2d(
                channels,
                gate_hidden,
                kernel_size=gate_kernel_size,
                padding=gate_pad,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                gate_hidden,
                channels,
                kernel_size=gate_kernel_size,
                padding=gate_pad,
            ),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, H, W]
            return_gate: If True, also return the gate tensor

        Returns:
            If return_gate=False: [B, C, H, W] with adaptive spatial smoothing
            If return_gate=True: tuple of (output, gate) where gate is [B, C, H, W]
        """
        smoothed = self.conv_layers(x)
        residual = x - smoothed
        gate = self.gate_network(residual)
        # gate≈1: high-residual (edge) → sharpen; gate≈0: low-residual → smooth
        output = x + gate * residual

        if return_gate:
            return output, gate
        return output


def build_gated_residual_conv2d_from_config(config: dict) -> GatedResidualConv2D:
    """
    Build GatedResidualConv2D from configuration dict.

    Example config:
        {
            'channels': 128,
            'conv': {
                'layers': 2,
                'kernel_size': 3,
                'padding': 1
            },
            'gate': {
                'hidden': 64,
                'kernel_size': 1
            }
        }
    """
    channels = config['channels']

    # Conv parameters
    conv_config = config.get('conv', {})
    num_layers = conv_config.get('layers', 2)
    kernel_size = conv_config.get('kernel_size', 3)
    padding = conv_config.get('padding', 1)

    # Gate parameters
    gate_config = config.get('gate', {})
    gate_hidden = gate_config.get('hidden', 64)
    gate_kernel_size = gate_config.get('kernel_size', 1)

    return GatedResidualConv2D(
        channels=channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        padding=padding,
        gate_hidden=gate_hidden,
        gate_kernel_size=gate_kernel_size,
    )


class EdgeAwareSmoothingConv2D(nn.Module):
    """Edge-aware smoothing via a directional filter bank and residual edge gate.

    Two complementary mechanisms work together:

    1. **Directional filter bank** (K = num_directions × 2 fixed filters, two spatial
       scales per orientation) with **rank-R factored per-channel mixing weights**:
       concentrates smoothing along edges while suppressing cross-edge blurring.
       Mixing weights are factored as W[k,c] = Σ_r A[k,r] · B[c,r], where A holds
       R shared direction-basis patterns and B holds per-channel mixture coefficients.
       This enforces correlation across channels (they share R basis patterns) while
       still allowing each channel its own directional preference — e.g. topographic
       channels can load on coarse-scale bases, spectral channels on fine-scale bases.

    2. **Residual edge gate**: where no safe smooth direction exists (corners, strong
       cross-edge content), gate ≈ 1 preserves the original features.

    Blending formula (Form A)::

        output = smoothed + gate * (x − smoothed)
            gate ≈ 0  →  homogeneous / along-edge  →  output ≈ smoothed
            gate ≈ 1  →  cross-edge / corner        →  output ≈ x (preserved)

    Args:
        channels: Number of input/output channels (preserved).
        num_layers: Accepted for signature compatibility; unused in this design.
        kernel_size: Fine-scale directional filter kernel size (3).
        padding: Padding for fine-scale filters (1).
        gate_hidden: Hidden channels for the mixing backbone and gate network.
        gate_kernel_size: Spatial kernel size for the gate network convolutions.
        num_directions: Number of orientations; total K = num_directions × 2 (fine+coarse).
        coarse_dilation: Dilation for coarse-scale bank filters (default 3 → 7×7 effective).
        rank: Number of shared direction-basis patterns R. Controls the degree of
            cross-channel correlation in the mixing weights. Larger R allows more
            independent channel behaviour; smaller R enforces tighter coupling.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        gate_hidden: int = 64,
        gate_kernel_size: int = 3,
        num_directions: int = 4,
        coarse_dilation: int = 3,
        rank: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.num_directions = num_directions
        self.coarse_dilation = coarse_dilation
        self.K = num_directions * 2
        self.rank = rank

        # --- Fixed directional filter bank ---
        # Four orientations, two scales each (fine dilation=1, coarse dilation=coarse_dilation).
        direction_templates = [
            torch.tensor([[0., 0., 0.], [1 / 3, 1 / 3, 1 / 3], [0., 0., 0.]]),
            torch.tensor([[0., 1 / 3, 0.], [0., 1 / 3, 0.], [0., 1 / 3, 0.]]),
            torch.tensor([[1 / 3, 0., 0.], [0., 1 / 3, 0.], [0., 0., 1 / 3]]),
            torch.tensor([[0., 0., 1 / 3], [0., 1 / 3, 0.], [1 / 3, 0., 0.]]),
        ]
        bank = torch.stack(direction_templates[:num_directions])  # [D, 3, 3]
        # Pre-expand to [D, C, 1, 3, 3] for grouped depthwise conv (groups=C).
        bank_expanded = (
            bank.unsqueeze(1).unsqueeze(1)           # [D, 1, 1, 3, 3]
            .expand(num_directions, channels, 1, 3, 3)
            .contiguous()
        )
        self.register_buffer("bank", bank_expanded)  # [D, C, 1, 3, 3]

        # --- Fixed Sobel filters (per-channel gradient estimation) ---
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        self.register_buffer(
            "sobel_x",
            sobel_x.reshape(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous(),
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.reshape(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous(),
        )

        # --- Factored mixing weight network ---
        # Shared backbone extracts spatial features from per-channel Sobel gradients.
        # head_A: K*R outputs → A[k, r] — K-way softmax direction weights for R slots.
        # head_B: C*R outputs → B[c, r] — R-way softmax channel mixture over slots.
        # Effective per-channel weight: w[k,c] = Σ_r A[k,r] · B[c,r].
        # Largest single output tensor is [B, C*R, H, W] = [B, 256, H, W] vs
        # the unfactored [B, K*C, H, W] = [B, 512, H, W].
        self.mix_backbone = nn.Sequential(
            nn.Conv2d(2 * channels, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mix_head_A = nn.Conv2d(gate_hidden, self.K * rank, kernel_size=1)
        self.mix_head_B = nn.Conv2d(gate_hidden, channels * rank, kernel_size=1)

        # --- Residual edge gate ---
        gate_pad = (gate_kernel_size - 1) // 2
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, gate_hidden, kernel_size=gate_kernel_size, padding=gate_pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, channels, kernel_size=gate_kernel_size, padding=gate_pad),
            nn.Sigmoid(),
        )
        # Curriculum floor: clamped from below during early training to prevent
        # the model using smoothing as a shortcut before spectral structure develops.
        # 1.0 = identity (no smoothing); 0.0 = unconstrained.  Set via set_min_gate().
        self.min_gate: float = 0.0

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, H, W].
            return_gate: If True, also return the gate tensor.

        Returns:
            If return_gate=False: output [B, C, H, W].
            If return_gate=True: (output, gate) where gate is [B, C, H, W] in [0, 1].
        """
        B, C, H, W = x.shape
        K = self.K
        R = self.rank

        # Per-channel Sobel gradients → shared mixing backbone
        dx = F.conv2d(x, self.sobel_x, padding=1, groups=C)  # [B, C, H, W]
        dy = F.conv2d(x, self.sobel_y, padding=1, groups=C)
        feat = self.mix_backbone(torch.cat([dx, dy], dim=1))  # [B, gate_hidden, H, W]

        # A: K-way softmax direction weights for each of R basis slots
        A = torch.softmax(
            self.mix_head_A(feat).reshape(B, K, R, H, W), dim=1
        )  # [B, K, R, H, W]

        # B: R-way softmax per-channel mixture over the R basis slots
        B_w = torch.softmax(
            self.mix_head_B(feat).reshape(B, C, R, H, W), dim=2
        )  # [B, C, R, H, W]

        # Accumulate slot smoothed outputs:
        #   slot[b, c, r, h, w] = Σ_k  A[b, k, r, h, w] · filtered_k[b, c, h, w]
        # A[:, k]: [B, R, H, W] → unsqueeze(1) → [B, 1, R, H, W]
        # filtered:  [B, C, H, W] → unsqueeze(2) → [B, C, 1, H, W]
        # product broadcasts to [B, C, R, H, W]
        slot = torch.zeros(B, C, R, H, W, device=x.device, dtype=x.dtype)
        for i in range(self.num_directions):
            filt = self.bank[i]  # [C, 1, 3, 3]
            fine = F.conv2d(x, filt, padding=1, groups=C)
            coarse = F.conv2d(
                x, filt,
                padding=self.coarse_dilation,
                dilation=self.coarse_dilation,
                groups=C,
            )
            slot = slot + fine.unsqueeze(2) * A[:, 2 * i].unsqueeze(1)
            slot = slot + coarse.unsqueeze(2) * A[:, 2 * i + 1].unsqueeze(1)

        # Per-channel mix over R slots: smoothed[b,c,h,w] = Σ_r B[b,c,r,h,w] · slot[b,c,r,h,w]
        smoothed = (B_w * slot).sum(dim=2)  # [B, C, H, W]

        # Residual edge gate: high residual (cross-edge/corner) → gate ≈ 1 → preserve x
        residual = x - smoothed
        gate = self.gate_net(residual)
        if self.min_gate > 0.0:
            gate = gate.clamp(min=self.min_gate)
        output = smoothed + gate * residual

        if return_gate:
            return output, gate
        return output

    def set_min_gate(self, value: float) -> None:
        """Set the curriculum floor for the residual gate (0.0 = unconstrained, 1.0 = identity)."""
        self.min_gate = float(value)
