"""
TCN (Temporal Convolutional Network) encoder.

Accepts either ``[N, C, T]`` (batch of sequences) or ``[B, C, T, H, W]``
(spatial grid of sequences).  The 5-D path reshapes to ``[B*H*W, C, T]``,
runs the same 1-D convolution stack, then reshapes back.

Features:
- Dilated causal or non-causal 1D convolutions along time dimension
- Gated residual connections with learnable projection
- Configurable dropout (preconv placement)
- GroupNorm with preact placement (conv -> norm -> activation)
- Optional temporal pooling (mean/std statistics or none)
- Mask-aware temporal pooling for handling invalid timesteps
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
    Temporal Convolutional Network encoder.

    Accepts **3-D** ``[N, C, T]`` or **5-D** ``[B, C, T, H, W]`` inputs.
    The 5-D path flattens spatial dims to the batch axis, runs the same
    1-D convolution stack, then reshapes back.  The 3-D path skips the
    spatial bookkeeping entirely.

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
        >>> # 5-D spatial input
        >>> x = torch.randn(2, 7, 10, 32, 32)  # [B, C, T, H, W]
        >>> mask = torch.ones(2, 10, 32, 32, dtype=torch.bool)
        >>> out = tcn(x, mask=mask)  # [B, 256, 32, 32] if pooling='stats'
        >>> # 3-D sequence input
        >>> x = torch.randn(500, 7, 10)  # [N, C, T]
        >>> out = tcn(x)  # [N, 256] if pooling='stats', [N, 128, 10] if 'none'
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor, either:
                - ``[N, C, T]``  — batch of sequences (no spatial dims), or
                - ``[B, C, T, H, W]`` — spatial grid of sequences.
            mask: Optional validity mask.
                - For 3-D input: ``[N, T]``
                - For 5-D input: ``[B, T, H, W]``
                True/1.0 = valid, False/0.0 = masked.
                Used during pooling to exclude invalid timesteps.

        Returns:
            For 3-D input:
                - pooling='stats': ``[N, 2*C_out]``
                - pooling='none':  ``[N, C_out, T]``
            For 5-D input:
                - pooling='stats': ``[B, 2*C_out, H, W]``
                - pooling='none':  ``[B, C_out, T, H, W]``
        """
        spatial = x.ndim == 5

        if spatial:
            B, C, T, H, W = x.shape
            # Flatten spatial dims into batch: [B*H*W, C, T]
            x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
            if mask is not None:
                mask = mask.permute(0, 2, 3, 1).reshape(B * H * W, T)
                mask = mask.float()
        else:
            T = x.size(2)
            if mask is not None:
                mask = mask.float()

        # Apply TCN layers
        for layer in self.layers:
            x = layer(x)  # [N, C_out, T]

        C_out = x.size(1)

        # Apply pooling
        if self.pooling == 'stats':
            if mask is not None:
                mask_expanded = mask.unsqueeze(1)  # [N, 1, T]
                valid_count = mask_expanded.sum(dim=2, keepdim=False)  # [N, 1]
                valid_count = torch.clamp(valid_count, min=1.0)
                masked_x = x * mask_expanded
                mean = masked_x.sum(dim=2) / valid_count  # [N, C_out]
                mean_expanded = mean.unsqueeze(2)
                squared_diff = ((x - mean_expanded) ** 2) * mask_expanded
                variance = squared_diff.sum(dim=2) / valid_count
                std = torch.sqrt(variance + 1e-8)
            else:
                mean = x.mean(dim=2)  # [N, C_out]
                std = x.std(dim=2)

            x = torch.cat([mean, std], dim=1)  # [N, 2*C_out]

            if spatial:
                x = x.reshape(B, H, W, 2 * C_out).permute(0, 3, 1, 2)
                if self.post_pool_norm:
                    x = x.permute(0, 2, 3, 1)
                    x = self.post_norm(x)
                    x = x.permute(0, 3, 1, 2)
            else:
                if self.post_pool_norm:
                    x = self.post_norm(x)  # [N, 2*C_out]

        elif self.pooling == 'none':
            if spatial:
                x = x.reshape(B, H, W, C_out, T).permute(0, 3, 4, 1, 2)
                if self.post_pool_norm:
                    x = x.permute(0, 2, 3, 4, 1)
                    x = self.post_norm(x)
                    x = x.permute(0, 4, 1, 2, 3)
            else:
                # x is already [N, C_out, T]
                if self.post_pool_norm:
                    x = x.permute(0, 2, 1)  # [N, T, C_out]
                    x = self.post_norm(x)
                    x = x.permute(0, 2, 1)  # [N, C_out, T]

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
