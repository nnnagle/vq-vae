"""
Neural network modules for forest state representation VQ-VAE.

This package provides encoder modules for processing temporal and static
geospatial data into latent representations.

Modules:
- tcn: Temporal Convolutional Networks for [B, C, T, H, W] tensors
- conv2d_encoder: 2D convolutional encoders for [B, C, H, W] tensors
- spatial: Spatial processing modules including gated residual convolutions
- conditioning: FiLM and other conditioning mechanisms
- heads: Prediction heads (MLP, Linear, Conv2D)
"""

# TCN encoder
from .tcn import (
    TCNEncoder,
    GatedResidualBlock,
    build_tcn_from_config,
)

# Conv2D encoder
from .conv2d_encoder import (
    Conv2DEncoder,
    build_conv2d_from_config,
)

# Spatial modules
from .spatial import (
    GatedResidualConv2D,
    build_gated_residual_conv2d_from_config,
)

# Representation model (unified encoder pipeline)
from .representation import RepresentationModel

# Conditioning
from .conditioning import (
    FiLMLayer,
    FiLMConditionedBlock,
    broadcast_to_time,
    build_film_from_config,
)

# Heads
from .heads import (
    MLPHead,
    LinearHead,
    Conv2DHead,
    build_mlp_from_config,
    build_linear_from_config,
    build_conv2d_head_from_config,
)

__all__ = [
    # TCN
    'TCNEncoder',
    'GatedResidualBlock',
    'build_tcn_from_config',
    # Conv2D
    'Conv2DEncoder',
    'build_conv2d_from_config',
    # Spatial
    'GatedResidualConv2D',
    'build_gated_residual_conv2d_from_config',
    # Representation
    'RepresentationModel',
    # Conditioning
    'FiLMLayer',
    'FiLMConditionedBlock',
    'broadcast_to_time',
    'build_film_from_config',
    # Heads
    'MLPHead',
    'LinearHead',
    'Conv2DHead',
    'build_mlp_from_config',
    'build_linear_from_config',
    'build_conv2d_head_from_config',
]
