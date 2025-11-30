# vqvae/vae_v0.py
#
# Minimal convolutional VAE for early pipeline testing.
#
# This model:
#   - Accepts a 2D tensor [B, C_in, 256, 256]
#   - Compresses it into a latent vector z
#   - Reconstructs back to the same shape
#
# Design philosophy for v0:
#   - Keep the architecture simple enough that debugging is trivial.
#   - Prioritize end-to-end functionality over optimal modeling choices.
#   - No fancy temporal modeling, conditioning, or skip connections.
#   - Goal is: Does the pipeline run? Does it backprop? Does loss go down?
#
# Once this works, you can replace it with something more appropriate.


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    A minimal convolutional VAE.

    Input:  [B, C_in, 256, 256]
    Output: [B, C_in, 256, 256]

    Notes:
      - "C_in" = T * C_cont from your PatchDataset (time collapsed into channels)
      - The encoder downsamples 256 -> 16 through 4 strided convs.
      - The latent vector is a simple Gaussian with reparameterization.
      - The decoder mirrors the encoder with ConvTranspose2d layers.
    """

    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # ------------------------------------------------------------
        # Encoder: progressively downsample spatial resolution
        # 256 -> 128 -> 64 -> 32 -> 16
        # ------------------------------------------------------------
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # 128 -> 64
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # 64 -> 32
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),         # 32 -> 16
            nn.ReLU(inplace=True),
        )

        # After four downsamples we have [B, 256, 16, 16] = 65,536 dims
        self.enc_out_dim = 256 * 16 * 16

        # ------------------------------------------------------------
        # Latent distribution: parameterize mean and log-variance
        # ------------------------------------------------------------
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # ------------------------------------------------------------
        # Decoder: linear map from latent vector back to a feature map
        # that can be reshaped into [256, 16, 16]
        # ------------------------------------------------------------
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        # ------------------------------------------------------------
        # Decoder: upsample spatially back to [256, 256]
        # 16 -> 32 -> 64 -> 128 -> 256
        # ------------------------------------------------------------
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32 -> 64
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64 -> 128
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # 128 -> 256
        )

    # ------------------------------------------------------------
    # Core VAE operations
    # ------------------------------------------------------------

    def encode(self, x):
        """Encode input image → (mu, logvar)."""
        h = self.enc(x)                # [B, 256, 16, 16]
        h = h.view(x.size(0), -1)      # flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample latent vector z using the reparameterization trick:
            z = mu + sigma * eps
        where eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector z → reconstructed image."""
        h = self.fc_dec(z)             # [B, enc_out_dim]
        h = h.view(z.size(0), 256, 16, 16)
        x_recon = self.dec(h)
        return x_recon

    def forward(self, x):
        """
        Full VAE forward pass:
          - encode input
          - sample latent variable z
          - decode back to reconstruction
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
