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
        # OLD #
        # Compute smoothed features
        #smoothed = self.conv_layers(x)

        # Compute per-location gates
        #gate = self.gate_network(x)

        # Gated residual combination
        # gate ≈ 1: use smoothed features (smooth regions)
        # gate ≈ 0: use original input (preserve edges)
        #output = gate * smoothed + (1 - gate) * x
        
        # NEW #
        smoothed = self.conv_layers(x)
        residual = x - smoothed
        gate = self.gate_network(residual)
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
