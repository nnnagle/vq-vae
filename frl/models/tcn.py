"""
TCN (Temporal Convolutional Network) encoder for [B, C, T, H, W] tensors.

Features:
- Dilated causal or non-causal 1D convolutions along time dimension
- Gated residual connections with learnable projection
- Configurable dropout (preconv placement)
- GroupNorm with preact placement (conv -> norm -> activation)
- Optional temporal pooling (mean/std statistics or none)
- Spatial-aware processing (applies same TCN to each spatial location)
"""

from typing import List, Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualBlock(nn.Module):
    """
    Gated residual block for TCN.

    Architecture:
        input -> [dropout] -> conv -> norm -> activation -> gate * output + (1-gate) * residual

    The gate is computed from the pre-activation features to control how much of the
    new features vs. residual to use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_rate: float = 0.0,
        num_groups: int = 8,
        projection_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Padding to maintain temporal dimension (non-causal, centered)
        self.padding = (kernel_size - 1) * dilation // 2

        # Dropout (preconv placement)
        self.dropout = nn.Dropout1d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Main conv path
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        # GroupNorm (preact placement: apply after conv, before activation)
        self.norm = nn.GroupNorm(num_groups, out_channels)

        # Gate path (1x1 conv from pre-activation features)
        self.gate = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        # Residual projection if channels change
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            proj_channels = projection_channels if projection_channels else out_channels
            self.projection = nn.Conv1d(in_channels, proj_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T]

        Returns:
            [B, C_out, T]
        """
        # Compute residual
        residual = self.projection(x)

        # Main path: dropout -> conv -> norm
        out = self.dropout(x)
        out = self.conv(out)
        out = self.norm(out)

        # Compute gate from pre-activation features
        gate = torch.sigmoid(self.gate(out))

        # Apply activation after gating decision point
        out = F.relu(out)

        # Gated residual: gate * new_features + (1 - gate) * residual
        if self.needs_projection:
            # If we projected, we need to match dimensions
            if residual.size(1) != out.size(1):
                # Pad or slice residual to match output channels
                if residual.size(1) < out.size(1):
                    pad_size = out.size(1) - residual.size(1)
                    residual = F.pad(residual, (0, 0, 0, pad_size))
                else:
                    residual = residual[:, :out.size(1), :]

        return gate * out + (1 - gate) * residual


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder for [B, C, T, H, W] tensors.

    Processes each spatial location independently by applying 1D convolutions
    along the time dimension with gated residual connections.

    Args:
        in_channels: Number of input channels
        channels: List of output channels for each layer
        kernel_size: Convolution kernel size (applied to all layers)
        dilations: List of dilation rates for each layer
        dropout_rate: Dropout probability
        num_groups: Number of groups for GroupNorm
        projection_channels: Channels for residual projection (if None, uses output channels)
        pooling: Type of temporal pooling ('stats', 'none')
        post_pool_norm: Apply LayerNorm after pooling
        normalized_shape: Shape for post-pool LayerNorm (e.g., [C, T] or [2*C])

    Example:
        >>> tcn = TCNEncoder(
        ...     in_channels=7,
        ...     channels=[128, 128, 128],
        ...     kernel_size=3,
        ...     dilations=[1, 2, 4],
        ...     dropout_rate=0.1,
        ...     num_groups=16,
        ...     pooling='stats'
        ... )
        >>> x = torch.randn(2, 7, 10, 32, 32)  # [B, C, T, H, W]
        >>> out = tcn(x)  # [B, 256, 32, 32] if pooling='stats' (mean+std)
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout_rate: float = 0.0,
        num_groups: int = 8,
        projection_channels: Optional[int] = None,
        pooling: Literal['stats', 'none'] = 'none',
        post_pool_norm: bool = False,
        normalized_shape: Optional[List[int]] = None,
    ):
        super().__init__()

        assert len(channels) > 0, "Must have at least one layer"

        if dilations is None:
            dilations = [1] * len(channels)
        assert len(dilations) == len(channels), "dilations must match channels"

        self.in_channels = in_channels
        self.out_channels = channels[-1]
        self.pooling = pooling
        self.post_pool_norm = post_pool_norm

        # Build TCN layers
        layers = []
        prev_channels = in_channels

        for i, (out_ch, dilation) in enumerate(zip(channels, dilations)):
            layers.append(
                GatedResidualBlock(
                    in_channels=prev_channels,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout_rate=dropout_rate,
                    num_groups=num_groups,
                    projection_channels=projection_channels,
                )
            )
            prev_channels = out_ch

        self.layers = nn.ModuleList(layers)

        # Post-pooling normalization
        if post_pool_norm:
            if pooling == 'stats':
                # After stats pooling, we have [B, 2*C, H, W]
                if normalized_shape is None:
                    normalized_shape = [2 * self.out_channels]
                # LayerNorm expects [B, ..., normalized_shape]
                # We'll apply it to [B, H, W, C] format
                self.post_norm = nn.LayerNorm(normalized_shape)
            elif pooling == 'none':
                # Output is [B, C, T, H, W]
                if normalized_shape is None:
                    # Default: normalize over [C, T]
                    normalized_shape = [self.out_channels]
                self.post_norm = nn.LayerNorm(normalized_shape)
            else:
                self.post_norm = nn.Identity()
        else:
            self.post_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            If pooling='stats': [B, 2*C_out, H, W] (mean and std concatenated)
            If pooling='none': [B, C_out, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Reshape to [B*H*W, C, T] for 1D convolutions along time
        x = x.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
        x = x.reshape(B * H * W, C, T)  # [B*H*W, C, T]

        # Apply TCN layers
        for layer in self.layers:
            x = layer(x)  # [B*H*W, C_out, T]

        C_out = x.size(1)

        # Apply pooling
        if self.pooling == 'stats':
            # Compute mean and std across time dimension
            mean = x.mean(dim=2)  # [B*H*W, C_out]
            std = x.std(dim=2)    # [B*H*W, C_out]
            x = torch.cat([mean, std], dim=1)  # [B*H*W, 2*C_out]

            # Reshape back to [B, 2*C_out, H, W]
            x = x.reshape(B, H, W, 2 * C_out)
            x = x.permute(0, 3, 1, 2)  # [B, 2*C_out, H, W]

            # Post-pool normalization
            if self.post_pool_norm:
                # LayerNorm over channel dimension
                x = x.permute(0, 2, 3, 1)  # [B, H, W, 2*C_out]
                x = self.post_norm(x)
                x = x.permute(0, 3, 1, 2)  # [B, 2*C_out, H, W]

        elif self.pooling == 'none':
            # Keep temporal dimension: [B*H*W, C_out, T] -> [B, C_out, T, H, W]
            x = x.reshape(B, H, W, C_out, T)
            x = x.permute(0, 3, 4, 1, 2)  # [B, C_out, T, H, W]

            # Post-pool normalization (applied per spatial location, over C dimension)
            if self.post_pool_norm:
                # Normalize over channel dimension at each time/spatial location
                x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C_out]
                x = self.post_norm(x)
                x = x.permute(0, 4, 1, 2, 3)  # [B, C_out, T, H, W]

        return x


def build_tcn_from_config(config: dict) -> TCNEncoder:
    """
    Build TCN encoder from configuration dict.

    Example config:
        {
            'in_channels': 7,
            'channels': [128, 128, 128],
            'kernel_size': 3,
            'dilations': [1, 2, 4],
            'residual': {
                'kind': 'gated',
                'projection_channels': 128
            },
            'dropout': {
                'rate': 0.10,
                'kind': 'dropout1d',
                'placement': 'preconv'
            },
            'norm': {
                'kind': 'groupnorm',
                'num_groups': 16,
                'placement': 'preact'
            },
            'pooling': {
                'kind': 'stats',
                'stats': ['mean', 'std']
            },
            'post_pool_norm': {
                'kind': 'layernorm'
            }
        }
    """
    # Extract basic parameters
    in_channels = config['in_channels']
    channels = config['channels']
    kernel_size = config.get('kernel_size', 3)
    dilations = config.get('dilations', None)

    # Dropout
    dropout_config = config.get('dropout', {})
    dropout_rate = dropout_config.get('rate', 0.0)

    # Normalization
    norm_config = config.get('norm', {})
    num_groups = norm_config.get('num_groups', 8)

    # Residual projection
    residual_config = config.get('residual', {})
    projection_channels = residual_config.get('projection_channels', None)

    # Pooling
    pooling_config = config.get('pooling', {})
    pooling = pooling_config.get('kind', 'none')

    # Post-pool normalization
    post_pool_config = config.get('post_pool_norm', {})
    if isinstance(post_pool_config, dict):
        post_pool_norm = post_pool_config.get('kind', None) is not None
        normalized_shape = post_pool_config.get('normalized_shape', None)
    else:
        post_pool_norm = False
        normalized_shape = None

    return TCNEncoder(
        in_channels=in_channels,
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout_rate=dropout_rate,
        num_groups=num_groups,
        projection_channels=projection_channels,
        pooling=pooling,
        post_pool_norm=post_pool_norm,
        normalized_shape=normalized_shape,
    )
