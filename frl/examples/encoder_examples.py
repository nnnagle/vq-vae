"""
Example usage of encoder modules.

This script demonstrates how to:
1. Instantiate encoders from YAML-like configurations
2. Process sample data through the encoders
3. Verify output shapes
"""

import torch
from frl.models import (
    build_tcn_from_config,
    build_conv2d_from_config,
    build_gated_residual_conv2d_from_config,
    build_mlp_from_config,
    FiLMLayer,
    broadcast_to_time,
)


def example_tcn_type_encoder():
    """
    Example: TCN encoder for type pathway (ls8_delta with stats pooling).

    Input: [B, 7, T, H, W] - LS8 delta features over time
    Output: [B, 256, H, W] - Pooled statistics (mean + std)
    """
    print("\n" + "="*70)
    print("TCN Type Encoder (ls8_delta with stats pooling)")
    print("="*70)

    config = {
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

    encoder = build_tcn_from_config(config)

    # Sample input
    B, T, H, W = 2, 10, 32, 32
    x = torch.randn(B, 7, T, H, W)

    print(f"Input shape: {x.shape}")

    # Forward pass
    out = encoder(x)

    print(f"Output shape: {out.shape}")
    print(f"Expected: [B={B}, 256 (128*2 from mean+std), H={H}, W={W}]")
    assert out.shape == (B, 256, H, W), f"Unexpected shape: {out.shape}"
    print("✓ Shape correct!")

    return encoder


def example_tcn_phase_encoder():
    """
    Example: TCN encoder for phase pathway (no pooling, keeps temporal dimension).

    Input: [B, 9, T, H, W] - LS8 level features + position
    Output: [B, 64, T, H, W] - Temporal features preserved
    """
    print("\n" + "="*70)
    print("TCN Phase Encoder (ls8_level without pooling)")
    print("="*70)

    config = {
        'in_channels': 9,
        'channels': [64, 64, 64],
        'kernel_size': 3,
        'dilations': [1, 2, 4],
        'residual': {
            'kind': 'gated',
            'projection_channels': 64
        },
        'dropout': {
            'rate': 0.10,
        },
        'norm': {
            'kind': 'groupnorm',
            'num_groups': 8,
        },
        'pooling': {
            'kind': 'none'
        },
        'post_pool_norm': {
            'kind': 'layernorm',
            'normalized_shape': [64]
        }
    }

    encoder = build_tcn_from_config(config)

    # Sample input
    B, T, H, W = 2, 10, 32, 32
    x = torch.randn(B, 9, T, H, W)

    print(f"Input shape: {x.shape}")

    # Forward pass
    out = encoder(x)

    print(f"Output shape: {out.shape}")
    print(f"Expected: [B={B}, C=64, T={T}, H={H}, W={W}]")
    assert out.shape == (B, 64, T, H, W), f"Unexpected shape: {out.shape}"
    print("✓ Shape correct!")

    return encoder


def example_conv2d_encoder():
    """
    Example: Conv2D encoder for static inputs (ccdc_history).

    Input: [B, 47, H, W] - CCDC history features
    Output: [B, 64, H, W] - Encoded features
    """
    print("\n" + "="*70)
    print("Conv2D Encoder (ccdc_history)")
    print("="*70)

    config = {
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
        'activation': 'relu'
    }

    encoder = build_conv2d_from_config(config)

    # Sample input
    B, H, W = 2, 32, 32
    x = torch.randn(B, 47, H, W)

    print(f"Input shape: {x.shape}")

    # Forward pass
    out = encoder(x)

    print(f"Output shape: {out.shape}")
    print(f"Expected: [B={B}, C=64, H={H}, W={W}]")
    assert out.shape == (B, 64, H, W), f"Unexpected shape: {out.shape}"
    print("✓ Shape correct!")

    return encoder


def example_gated_spatial_conv():
    """
    Example: Gated residual Conv2D for adaptive spatial smoothing.

    Input: [B, 128, H, W]
    Output: [B, 128, H, W] - Smoothed features
    """
    print("\n" + "="*70)
    print("Gated Residual Conv2D (adaptive spatial smoothing)")
    print("="*70)

    config = {
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

    module = build_gated_residual_conv2d_from_config(config)

    # Sample input
    B, H, W = 2, 32, 32
    x = torch.randn(B, 128, H, W)

    print(f"Input shape: {x.shape}")

    # Forward pass
    out = module(x)

    print(f"Output shape: {out.shape}")
    print(f"Expected: [B={B}, C=128, H={H}, W={W}]")
    assert out.shape == (B, 128, H, W), f"Unexpected shape: {out.shape}"
    print("✓ Shape correct!")

    return module


def example_mlp_head():
    """
    Example: MLP head for latent space projection.

    Input: [B, 128, H, W] - Trunk features
    Output: [B, 64, H, W] - Latent representation
    """
    print("\n" + "="*70)
    print("MLP Head (z_type projection)")
    print("="*70)

    config = {
        'in_dim': 128,
        'layers': [256, 64],
        'output_dim': 64,
        'dropout': 0.0
    }

    head = build_mlp_from_config(config)

    # Sample input
    B, H, W = 2, 32, 32
    x = torch.randn(B, 128, H, W)

    print(f"Input shape: {x.shape}")

    # Forward pass
    out = head(x)

    print(f"Output shape: {out.shape}")
    print(f"Expected: [B={B}, C=64, H={H}, W={W}]")
    assert out.shape == (B, 64, H, W), f"Unexpected shape: {out.shape}"
    print("✓ Shape correct!")

    return head


def example_film_conditioning():
    """
    Example: FiLM conditioning for phase pathway.

    Conditioning: [B, 64, H, W] - z_type_cont
    Features: [B, 128, H, W] - phase features
    Output: [B, 128, H, W] - Modulated features
    """
    print("\n" + "="*70)
    print("FiLM Conditioning (type -> phase)")
    print("="*70)

    film = FiLMLayer(cond_dim=64, target_dim=128, hidden_dim=64)

    # Sample inputs
    B, H, W = 2, 32, 32
    z_type = torch.randn(B, 64, H, W)
    h_phase = torch.randn(B, 128, H, W)

    print(f"Conditioning shape: {z_type.shape}")
    print(f"Features shape: {h_phase.shape}")

    # Generate FiLM parameters
    gamma, beta = film(z_type)

    print(f"Gamma shape: {gamma.shape}")
    print(f"Beta shape: {beta.shape}")

    # Apply modulation
    modulated = film.modulate(h_phase, gamma, beta)

    print(f"Modulated output shape: {modulated.shape}")
    print(f"Expected: [B={B}, C=128, H={H}, W={W}]")
    assert modulated.shape == (B, 128, H, W), f"Unexpected shape: {modulated.shape}"
    print("✓ Shape correct!")

    # Demonstrate broadcasting to time
    print("\nBroadcasting to temporal dimension...")
    T = 10
    gamma_t = broadcast_to_time(gamma, T)
    print(f"Gamma temporal shape: {gamma_t.shape}")
    print(f"Expected: [B={B}, C=128, T={T}, H={H}, W={W}]")
    assert gamma_t.shape == (B, 128, T, H, W)
    print("✓ Broadcast correct!")

    return film


def example_full_type_pathway():
    """
    Example: Full type pathway from config.

    Simulates the type pathway from the YAML spec:
    ls8_delta -> TCN -> pool -> concat with static -> trunk -> spatial -> head
    """
    print("\n" + "="*70)
    print("Full Type Pathway Example")
    print("="*70)

    B, T, H, W = 2, 10, 32, 32

    # 1. TCN for ls8_delta
    tcn_config = {
        'in_channels': 7,
        'channels': [128, 128, 128],
        'kernel_size': 3,
        'dilations': [1, 2, 4],
        'residual': {'projection_channels': 128},
        'dropout': {'rate': 0.10},
        'norm': {'num_groups': 16},
        'pooling': {'kind': 'stats'},
        'post_pool_norm': {'kind': 'layernorm'}
    }
    tcn_ls8 = build_tcn_from_config(tcn_config)

    # 2. Conv2D for ccdc_history
    conv_ccdc_config = {
        'in_channels': 47,
        'channels': [128, 64],
        'kernel_size': 1,
        'dropout': {'rate': 0.10},
        'norm': {'num_groups': [16, 8]},
        'activation': 'relu'
    }
    conv_ccdc = build_conv2d_from_config(conv_ccdc_config)

    # 3. Conv2D for topo
    conv_topo_config = {
        'in_channels': 8,
        'channels': [8, 6],
        'kernel_size': 1,
        'dropout': {'rate': 0.40},
        'norm': {'num_groups': 2},
        'activation': 'relu'
    }
    conv_topo = build_conv2d_from_config(conv_topo_config)

    # 4. Trunk (fusion)
    # Input: 256 (ls8 pooled) + 64 (ccdc) + 6 (topo) = 326
    trunk_config = {
        'in_channels': 326,
        'channels': [256, 128],
        'kernel_size': 1,
        'dropout': {'rate': 0.10},
        'norm': {'num_groups': [32, 16]},
        'activation': 'relu'
    }
    trunk = build_conv2d_from_config(trunk_config)

    # 5. Spatial smoothing
    spatial_config = {
        'channels': 128,
        'conv': {'layers': 2, 'kernel_size': 3, 'padding': 1},
        'gate': {'hidden': 64, 'kernel_size': 1}
    }
    spatial = build_gated_residual_conv2d_from_config(spatial_config)

    # 6. Head to z_type
    head_config = {
        'in_dim': 128,
        'layers': [256, 64],
        'output_dim': 64,
        'dropout': 0.0
    }
    head = build_mlp_from_config(head_config)

    print("\nProcessing through type pathway...")

    # Forward pass
    ls8_delta = torch.randn(B, 7, T, H, W)
    ccdc_hist = torch.randn(B, 47, H, W)
    topo = torch.randn(B, 8, H, W)

    print(f"1. ls8_delta input: {ls8_delta.shape}")
    feat_ls8 = tcn_ls8(ls8_delta)
    print(f"   -> TCN output: {feat_ls8.shape}")

    print(f"2. ccdc_hist input: {ccdc_hist.shape}")
    feat_ccdc = conv_ccdc(ccdc_hist)
    print(f"   -> Conv output: {feat_ccdc.shape}")

    print(f"3. topo input: {topo.shape}")
    feat_topo = conv_topo(topo)
    print(f"   -> Conv output: {feat_topo.shape}")

    print("4. Concatenating features...")
    h_type = torch.cat([feat_ls8, feat_ccdc, feat_topo], dim=1)
    print(f"   -> Concatenated: {h_type.shape}")

    print("5. Trunk processing...")
    h_type = trunk(h_type)
    print(f"   -> Trunk output: {h_type.shape}")

    print("6. Spatial smoothing...")
    h_type_smooth = spatial(h_type)
    print(f"   -> Smoothed: {h_type_smooth.shape}")

    print("7. Head projection to z_type...")
    z_type_cont = head(h_type_smooth)
    print(f"   -> z_type_cont: {z_type_cont.shape}")

    assert z_type_cont.shape == (B, 64, H, W)
    print("\n✓ Full type pathway successful!")

    return {
        'tcn_ls8': tcn_ls8,
        'conv_ccdc': conv_ccdc,
        'conv_topo': conv_topo,
        'trunk': trunk,
        'spatial': spatial,
        'head': head,
        'z_type_cont': z_type_cont
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ENCODER EXAMPLES - Forest State Representation VQ-VAE")
    print("="*70)

    # Run examples
    example_tcn_type_encoder()
    example_tcn_phase_encoder()
    example_conv2d_encoder()
    example_gated_spatial_conv()
    example_mlp_head()
    example_film_conditioning()
    example_full_type_pathway()

    print("\n" + "="*70)
    print("All examples completed successfully! ✓")
    print("="*70)
