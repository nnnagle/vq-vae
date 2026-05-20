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
       scales per orientation) with **per-channel learned mixing weights**: concentrates
       smoothing along edges while suppressing cross-edge blurring.

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
        gate_hidden: Hidden channels for both the mixing network and gate network.
        gate_kernel_size: Spatial kernel size for the gate network convolutions.
        num_directions: Number of orientations; total K = num_directions × 2 (fine+coarse).
        coarse_dilation: Dilation for coarse-scale bank filters (default 3 → 7×7 effective).
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
    ):
        super().__init__()

        self.channels = channels
        self.num_directions = num_directions
        self.coarse_dilation = coarse_dilation
        self.K = num_directions * 2

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

        # --- Mixing weight network ---
        # Maps [B, 2C, H, W] (per-channel Sobel gradients) to K-way softmax
        # weights [B, K, C, H, W] — each channel chooses its own blend direction.
        self.mixing_net = nn.Sequential(
            nn.Conv2d(2 * channels, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, self.K * channels, kernel_size=1),
        )

        # --- Residual edge gate ---
        gate_pad = (gate_kernel_size - 1) // 2
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, gate_hidden, kernel_size=gate_kernel_size, padding=gate_pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, channels, kernel_size=gate_kernel_size, padding=gate_pad),
            nn.Sigmoid(),
        )

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

        # Per-channel Sobel gradients → mixing weight input
        dx = F.conv2d(x, self.sobel_x, padding=1, groups=C)  # [B, C, H, W]
        dy = F.conv2d(x, self.sobel_y, padding=1, groups=C)

        # Per-channel K-way softmax mixing weights
        w_raw = self.mixing_net(torch.cat([dx, dy], dim=1))       # [B, K*C, H, W]
        w = torch.softmax(w_raw.reshape(B, K, C, H, W), dim=1)   # [B, K, C, H, W]

        # Directional weighted smoothing
        smoothed = torch.zeros_like(x)
        for i in range(self.num_directions):
            filt = self.bank[i]  # [C, 1, 3, 3]
            fine = F.conv2d(x, filt, padding=1, groups=C)
            coarse = F.conv2d(
                x, filt,
                padding=self.coarse_dilation,
                dilation=self.coarse_dilation,
                groups=C,
            )
            smoothed = smoothed + w[:, 2 * i] * fine + w[:, 2 * i + 1] * coarse

        # Residual edge gate: high residual (cross-edge/corner) → gate ≈ 1 → preserve x
        residual = x - smoothed
        gate = self.gate_net(residual)
        output = smoothed + gate * residual

        if return_gate:
            return output, gate
        return output
