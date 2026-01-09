"""
Conv2D encoder for static [B, C, H, W] tensors.

Features:
- Multiple convolutional layers with configurable channels
- Dropout (postconv placement)
- GroupNorm (postconv placement)
- Flexible kernel sizes and padding
- ReLU activation
"""

from typing import List, Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DEncoder(nn.Module):
    """
    Stack of 2D convolutional layers for static spatial inputs.

    Architecture per layer:
        input -> conv -> norm -> activation -> dropout -> output

    Args:
        in_channels: Number of input channels
        channels: List of output channels for each layer
        kernel_size: Convolution kernel size (int or list per layer)
        padding: Padding size (int or list per layer, or 'same')
        dropout_rate: Dropout probability (float or list per layer)
        num_groups: Number of groups for GroupNorm (int or list per layer)
        activation: Activation function ('relu', 'none')
        out_channels: If specified, adds final conv to this many channels

    Example:
        >>> encoder = Conv2DEncoder(
        ...     in_channels=47,
        ...     channels=[128, 64],
        ...     kernel_size=1,
        ...     padding=0,
        ...     dropout_rate=0.1,
        ...     num_groups=[16, 8]
        ... )
        >>> x = torch.randn(2, 47, 32, 32)
        >>> out = encoder(x)  # [2, 64, 32, 32]
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: Union[int, List[int]] = 1,
        padding: Union[int, str, List[int]] = 0,
        dropout_rate: Union[float, List[float]] = 0.0,
        num_groups: Union[int, List[int]] = 8,
        activation: Literal['relu', 'none'] = 'relu',
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        assert len(channels) > 0, "Must have at least one layer"

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else channels[-1]

        # Normalize parameters to lists
        num_layers = len(channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        if isinstance(padding, int):
            padding = [padding] * num_layers
        elif padding == 'same':
            # Compute padding to maintain spatial dimensions
            padding = [(k - 1) // 2 for k in kernel_size]
        if isinstance(dropout_rate, (int, float)):
            dropout_rate = [dropout_rate] * num_layers
        if isinstance(num_groups, int):
            num_groups = [num_groups] * num_layers

        assert len(kernel_size) == num_layers
        assert len(padding) == num_layers
        assert len(dropout_rate) == num_layers
        assert len(num_groups) == num_layers

        # Build layers
        layers = []
        prev_channels = in_channels

        for i, (out_ch, ks, pad, drop, groups) in enumerate(
            zip(channels, kernel_size, padding, dropout_rate, num_groups)
        ):
            # Conv layer
            layers.append(
                nn.Conv2d(
                    prev_channels,
                    out_ch,
                    kernel_size=ks,
                    padding=pad,
                    bias=False,  # Bias is redundant with normalization
                )
            )

            # Normalization (postconv)
            layers.append(nn.GroupNorm(groups, out_ch))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            # else: no activation

            # Dropout (postconv)
            if drop > 0:
                layers.append(nn.Dropout2d(drop))

            prev_channels = out_ch

        # Optional final projection
        if out_channels is not None and out_channels != channels[-1]:
            layers.append(
                nn.Conv2d(
                    channels[-1],
                    out_channels,
                    kernel_size=1,
                    padding=0,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            [B, C_out, H, W]
        """
        return self.layers(x)


def build_conv2d_from_config(config: dict) -> Conv2DEncoder:
    """
    Build Conv2D encoder from configuration dict.

    Example config:
        {
            'in_channels': 47,
            'channels': [128, 64],
            'kernel_size': 1,
            'padding': 0,
            'dropout': {
                'rate': 0.10,
                'kind': 'dropout2d',
                'placement': 'postconv'
            },
            'norm': {
                'kind': 'groupnorm',
                'num_groups': [16, 8],
                'placement': 'postconv'
            },
            'activation': 'relu',
            'out_channels': 64  # optional
        }
    """
    # Extract basic parameters
    in_channels = config['in_channels']
    channels = config['channels']
    kernel_size = config.get('kernel_size', 1)
    padding = config.get('padding', 0)

    # Dropout
    dropout_config = config.get('dropout', {})
    if isinstance(dropout_config, dict):
        dropout_rate = dropout_config.get('rate', 0.0)
    else:
        dropout_rate = 0.0

    # Normalization
    norm_config = config.get('norm', {})
    num_groups = norm_config.get('num_groups', 8)

    # Activation
    activation = config.get('activation', 'relu')

    # Optional output channels
    out_channels = config.get('out_channels', None)

    return Conv2DEncoder(
        in_channels=in_channels,
        channels=channels,
        kernel_size=kernel_size,
        padding=padding,
        dropout_rate=dropout_rate,
        num_groups=num_groups,
        activation=activation,
        out_channels=out_channels,
    )
