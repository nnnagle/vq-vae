"""
Prediction heads for various outputs.

Includes:
- MLP heads (per-pixel predictions)
- Linear heads (simple projection)
- Conv2D heads (spatial predictions)
"""

from typing import List, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for per-pixel predictions.

    Applied independently to each spatial location, processes
    [B, in_dim, H, W] -> [B, out_dim, H, W] using 1x1 convolutions.

    Args:
        in_dim: Input feature dimension
        layers: List of hidden layer dimensions
        output_dim: Output dimension
        dropout: Dropout rate between layers
        activation: Activation function ('relu', 'sigmoid', 'none')

    Example:
        >>> head = MLPHead(in_dim=128, layers=[256, 64], output_dim=64, dropout=0.1)
        >>> x = torch.randn(2, 128, 32, 32)
        >>> out = head(x)  # [2, 64, 32, 32]
    """

    def __init__(
        self,
        in_dim: int,
        layers: List[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: Literal['relu', 'sigmoid', 'none'] = 'relu',
    ):
        super().__init__()

        self.in_dim = in_dim
        self.output_dim = output_dim

        # Build hidden layers
        mlp_layers = []
        prev_dim = in_dim

        for hidden_dim in layers:
            mlp_layers.append(nn.Conv2d(prev_dim, hidden_dim, kernel_size=1))
            mlp_layers.append(nn.ReLU(inplace=True))

            if dropout > 0:
                mlp_layers.append(nn.Dropout2d(dropout))

            prev_dim = hidden_dim

        # Output layer
        mlp_layers.append(nn.Conv2d(prev_dim, output_dim, kernel_size=1))

        # Final activation
        if activation == 'relu':
            mlp_layers.append(nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            mlp_layers.append(nn.Sigmoid())
        # else: no activation (linear output)

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim, H, W]

        Returns:
            [B, output_dim, H, W]
        """
        return self.mlp(x)


class LinearHead(nn.Module):
    """
    Simple linear projection head.

    Args:
        in_dim: Input feature dimension
        out_dim: Output dimension
        activation: Optional activation function

    Example:
        >>> head = LinearHead(in_dim=64, out_dim=50)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = head(x)  # [2, 50, 32, 32]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Literal['relu', 'sigmoid', 'none'] = 'none',
    ):
        super().__init__()

        self.projection = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim, H, W]

        Returns:
            [B, out_dim, H, W]
        """
        return self.activation(self.projection(x))


class Conv2DHead(nn.Module):
    """
    Convolutional head with multiple layers for spatial predictions.

    Similar to Conv2DEncoder but designed for output heads with
    optional final activation.

    Args:
        in_channels: Input channels
        channels: List of intermediate channel dimensions
        out_channels: Final output channels
        kernel_size: Convolution kernel size
        padding: Padding size
        activation: Final activation ('relu', 'sigmoid', 'none')

    Example:
        >>> head = Conv2DHead(
        ...     in_channels=64,
        ...     channels=[128, 64, 32],
        ...     out_channels=20,
        ...     kernel_size=1,
        ...     activation='none'
        ... )
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = head(x)  # [2, 20, 32, 32]
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        activation: Literal['relu', 'sigmoid', 'none'] = 'none',
    ):
        super().__init__()

        layers = []
        prev_channels = in_channels

        # Build intermediate layers
        for ch in channels:
            layers.append(
                nn.Conv2d(prev_channels, ch, kernel_size=kernel_size, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))
            prev_channels = ch

        # Final projection
        layers.append(
            nn.Conv2d(prev_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

        # Final activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            [B, out_channels, H, W]
        """
        return self.layers(x)


def build_mlp_from_config(config: dict) -> MLPHead:
    """
    Build MLP head from configuration dict.

    Example config:
        {
            'in_dim': 128,
            'layers': [256, 64],
            'output_dim': 64,
            'dropout': 0.0,
            'activation': 'relu'
        }
    """
    in_dim = config['in_dim']
    layers = config['layers']
    output_dim = config['output_dim']
    dropout = config.get('dropout', 0.0)
    activation = config.get('activation', 'none')

    return MLPHead(
        in_dim=in_dim,
        layers=layers,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
    )


def build_linear_from_config(config: dict) -> LinearHead:
    """
    Build linear head from configuration dict.

    Example config:
        {
            'in_dim': 64,
            'out_dim': 50,
            'activation': 'none'
        }
    """
    in_dim = config['in_dim']
    out_dim = config['out_dim']
    activation = config.get('activation', 'none')

    return LinearHead(in_dim=in_dim, out_dim=out_dim, activation=activation)


def build_conv2d_head_from_config(config: dict) -> Conv2DHead:
    """
    Build Conv2D head from configuration dict.

    Example config:
        {
            'in_channels': 64,
            'channels': [128, 64, 32],
            'out_channels': 20,
            'kernel_size': 1,
            'activation': 'none'
        }
    """
    in_channels = config['in_channels']
    channels = config['channels']
    out_channels = config['out_channels']
    kernel_size = config.get('kernel_size', 1)
    padding = config.get('padding', 0)
    activation = config.get('activation', 'none')

    return Conv2DHead(
        in_channels=in_channels,
        channels=channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
    )
