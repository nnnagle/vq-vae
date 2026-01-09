"""
Conditioning modules including FiLM (Feature-wise Linear Modulation).

FiLM applies affine transformations to features based on conditioning information:
    output = gamma * input + beta

This is used to condition phase pathway on type latents.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Generates gamma (scale) and beta (shift) parameters from conditioning input
    to modulate target features via affine transformation:
        modulated = gamma * features + beta

    Args:
        cond_dim: Dimension of conditioning input (e.g., z_type_cont dimension)
        target_dim: Dimension of features to modulate (e.g., phase features)
        hidden_dim: Hidden dimension for parameter generation network
        use_bias: Whether to generate beta (shift) parameters

    Example:
        >>> film = FiLMLayer(cond_dim=64, target_dim=128, hidden_dim=64)
        >>> z_type = torch.randn(2, 64, 32, 32)  # conditioning
        >>> h_phase = torch.randn(2, 128, 32, 32)  # features to modulate
        >>> gamma, beta = film(z_type)
        >>> modulated = gamma * h_phase + beta
    """

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        hidden_dim: Optional[int] = None,
        use_bias: bool = True,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.use_bias = use_bias

        if hidden_dim is None:
            hidden_dim = max(cond_dim, target_dim) // 2

        # Network to generate gamma (scale)
        self.gamma_network = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, target_dim, kernel_size=1),
        )

        # Network to generate beta (shift)
        if use_bias:
            self.beta_network = nn.Sequential(
                nn.Conv2d(cond_dim, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, target_dim, kernel_size=1),
            )
        else:
            self.beta_network = None

        # Initialize gamma to output near 1.0 (start close to identity)
        nn.init.zeros_(self.gamma_network[-1].weight)
        nn.init.ones_(self.gamma_network[-1].bias)

        if use_bias:
            nn.init.zeros_(self.beta_network[-1].weight)
            nn.init.zeros_(self.beta_network[-1].bias)

    def forward(
        self, conditioning: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate FiLM parameters from conditioning input.

        Args:
            conditioning: [B, cond_dim, H, W] conditioning tensor

        Returns:
            gamma: [B, target_dim, H, W] scale parameters
            beta: [B, target_dim, H, W] shift parameters (or None if use_bias=False)
        """
        gamma = self.gamma_network(conditioning)

        if self.use_bias:
            beta = self.beta_network(conditioning)
        else:
            beta = None

        return gamma, beta

    def modulate(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to features.

        Args:
            features: [B, target_dim, ...] features to modulate
            gamma: [B, target_dim, ...] scale parameters
            beta: [B, target_dim, ...] shift parameters (optional)

        Returns:
            [B, target_dim, ...] modulated features
        """
        modulated = gamma * features
        if beta is not None:
            modulated = modulated + beta
        return modulated


class FiLMConditionedBlock(nn.Module):
    """
    Block that applies FiLM conditioning followed by processing layers.

    This is useful for building phase pathway that's conditioned on type latents.

    Args:
        cond_dim: Dimension of conditioning (z_type_cont)
        feature_dim: Dimension of features to modulate
        hidden_dim: Hidden dimension for FiLM network

    Example:
        >>> block = FiLMConditionedBlock(cond_dim=64, feature_dim=128)
        >>> z_type = torch.randn(2, 64, 32, 32)
        >>> h_phase = torch.randn(2, 128, 32, 32)
        >>> out = block(h_phase, z_type)
    """

    def __init__(
        self,
        cond_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.film = FiLMLayer(
            cond_dim=cond_dim,
            target_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        features: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, feature_dim, H, W] features to modulate
            conditioning: [B, cond_dim, H, W] conditioning tensor

        Returns:
            [B, feature_dim, H, W] modulated features
        """
        gamma, beta = self.film(conditioning)
        return self.film.modulate(features, gamma, beta)


def broadcast_to_time(
    spatial_tensor: torch.Tensor, T: int
) -> torch.Tensor:
    """
    Broadcast spatial tensor [B, C, H, W] to temporal [B, C, T, H, W].

    Args:
        spatial_tensor: [B, C, H, W]
        T: Number of time steps

    Returns:
        [B, C, T, H, W] with values repeated along time dimension
    """
    B, C, H, W = spatial_tensor.shape
    # Add time dimension and repeat
    return spatial_tensor.unsqueeze(2).expand(B, C, T, H, W)


def build_film_from_config(config: dict) -> FiLMLayer:
    """
    Build FiLM layer from configuration dict.

    Example config:
        {
            'cond_dim': 64,
            'target_dim': 128,
            'hidden_dim': 64,
            'use_bias': True
        }
    """
    cond_dim = config['cond_dim']
    target_dim = config['target_dim']
    hidden_dim = config.get('hidden_dim', None)
    use_bias = config.get('use_bias', True)

    return FiLMLayer(
        cond_dim=cond_dim,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
        use_bias=use_bias,
    )
