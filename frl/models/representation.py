"""
RepresentationModel — unified encoder pipeline for contrastive representation learning.

This module defines the complete forward pipeline (encoder -> spatial conv -> embedding)
as a single nn.Module. All training scripts, probes, and diagnostics should use this
class rather than assembling components individually.

Checkpoints store ``model_version`` so that downstream scripts can detect architecture
mismatches early instead of encountering cryptic shape errors.

Example
-------
Create, train, and checkpoint::

    from models import RepresentationModel

    model = RepresentationModel().to(device)
    # ... training loop ...
    torch.save({
        'model_version': RepresentationModel.VERSION,
        'model_state_dict': model.state_dict(),
        ...
    }, path)

Load from checkpoint::

    model = RepresentationModel.from_checkpoint(path, device=device)
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d_encoder import Conv2DEncoder
from .conditioning import FiLMLayer
from .spatial import GatedResidualConv2D
from .tcn import TCNEncoder

logger = logging.getLogger(__name__)


class RepresentationModel(nn.Module):
    """Full encoder pipeline for type and phase embeddings.

    **Type pathway** (v1):
        ``[B, C_type, H, W]`` → Conv2DEncoder → GatedResidualConv2D → z_type ``[B, 64, H, W]``

    **Phase pathway** (v3):
        ``[B, C_phase, T, H, W]`` → TCN → 1×1 bottleneck (64→12)
        → FiLM(stopgrad z_type) → z_phase ``[B, 12, T, H, W]``

    Args:
        type_in_channels: Number of input channels for the type pathway
            (must match the ``ccdc_history`` feature). Default 16.
        phase_in_channels: Number of input channels for the phase pathway
            (must match the ``phase_ccdc`` feature). Default 8.

    The bottleneck projects TCN features to the 12-dim embedding space,
    then FiLM applies type-conditioned affine modulation directly in
    that space.  FiLM gamma is initialized near 1 and beta near 0,
    giving near-identity behavior at the start of training.

    Attributes:
        VERSION: Architecture version string. Bump this whenever the forward
            pipeline changes in a checkpoint-incompatible way.
    """

    VERSION = "3"

    def __init__(
        self,
        type_in_channels: int = 16,
        phase_in_channels: int = 8,
    ) -> None:
        super().__init__()

        self.type_in_channels = type_in_channels
        self.phase_in_channels = phase_in_channels

        # --- Type pathway ---
        self.encoder = Conv2DEncoder(
            in_channels=type_in_channels,
            channels=[128, 64],
            kernel_size=1,
            padding=0,
            dropout_rate=0.1,
            num_groups=8,
        )
        self.spatial_conv = GatedResidualConv2D(
            channels=64,
            num_layers=2,
            kernel_size=3,
            padding=1,
            gate_hidden=64,
            gate_kernel_size=1,
        )

        # --- Phase pathway ---
        self.phase_tcn = TCNEncoder(
            in_channels=phase_in_channels,
            channels=[64, 64, 64],
            kernel_size=3,
            dilations=[1, 2, 4],
            dropout_rate=0.1,
            num_groups=8,
            pooling='none',
        )
        # Bottleneck: project TCN output to final embedding dimension
        self.phase_head = nn.Conv2d(64, 12, kernel_size=1)
        # FiLM: generates gamma/beta from z_type [B, 64, H, W]
        # to modulate 12-dim bottleneck output, broadcast across T.
        # Gamma initialized near 1, beta near 0 → near-identity at init.
        self.phase_film = FiLMLayer(
            cond_dim=64,
            target_dim=12,
        )

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Type pathway forward.

        Args:
            x: Input tensor ``[B, C_type, H, W]`` (ccdc_history features).
            return_gate: If True, also return the spatial gate tensor.

        Returns:
            If *return_gate* is False: ``z_type [B, 64, H, W]``.
            If *return_gate* is True: ``(z_type, gate)``.
        """
        h = self.encoder(x)
        if return_gate:
            z, gate = self.spatial_conv(h, return_gate=True)
            return z, gate
        return self.spatial_conv(h)

    def forward_phase(
        self,
        x_phase: torch.Tensor,
        z_type: torch.Tensor,
    ) -> torch.Tensor:
        """Phase pathway forward (dense, full spatial grid).

        .. deprecated::
            Prefer :meth:`forward_phase_at_locations` for training, which
            runs the TCN only on sampled anchor pixels.  This dense method
            is retained for inference when embeddings are needed at every pixel.

        Args:
            x_phase: Temporal input ``[B, C_phase, T, H, W]`` (phase_ccdc features).
            z_type: Type embeddings ``[B, 64, H, W]`` (**caller must
                stop-grad** before passing in).

        Returns:
            z_phase: ``[B, 12, T, H, W]`` phase embeddings.
        """
        B, C, T, H, W = x_phase.shape

        # TCN along time: [B, C_phase, T, H, W] -> [B, 64, T, H, W]
        h = self.phase_tcn(x_phase)

        # Bottleneck per-timestep: reshape to [B*T, 64, H, W] for Conv2d
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, 64, H, W)
        h = self.phase_head(h)  # [B*T, 12, H, W]
        h = h.reshape(B, T, 12, H, W).permute(0, 2, 1, 3, 4)  # [B, 12, T, H, W]

        # L2-normalize across (channel, time) jointly so the TCN controls
        # direction & temporal shape, while FiLM gamma owns the per-channel scale.
        # One norm factor per spatial location — temporal variation is preserved.
        h = F.normalize(h.flatten(1, 2), dim=1).unflatten(1, (12, T))

        # FiLM conditioning
        gamma, beta = self.phase_film(z_type)  # each [B, 12, H, W]
        gamma = gamma.unsqueeze(2)  # [B, 12, 1, H, W]
        beta = beta.unsqueeze(2)    # [B, 12, 1, H, W]
        z_phase = gamma * h + beta  # [B, 12, T, H, W]

        return z_phase  # [B, 12, T, H, W]

    def forward_phase_at_locations(
        self,
        x_phase_pixels: torch.Tensor,
        z_type_pixels: torch.Tensor,
        return_film: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase pathway forward at sampled pixel locations only.

        Runs the same TCN → bottleneck → FiLM pipeline but only on the
        supplied pixel time-series, avoiding the cost of processing the
        full spatial grid.  Produces identical results to extracting from
        the dense ``forward_phase`` output at the same locations.

        Args:
            x_phase_pixels: ``[N, C, T]`` temporal features at N pixels.
            z_type_pixels: ``[N, 64]`` type embeddings at the same N pixels
                (**caller must stop-grad**).
            return_film: If True, also return the data-dependent gamma and
                beta tensors (useful for diagnostics).

        Returns:
            If *return_film* is False: ``z_phase_pixels [N, T, 12]``.
            If *return_film* is True: ``(z_phase_pixels, gamma, beta)``
                where gamma and beta are ``[N, 12]``.
        """
        N, C, T = x_phase_pixels.shape

        # TCN accepts [N, C, T] directly (no spatial reshape needed)
        h = self.phase_tcn(x_phase_pixels)  # [N, 64, T]

        # Bottleneck: phase_head is Conv2d(64, 12, 1×1)
        # Reshape [N, 64, T] -> [N*T, 64, 1, 1] for Conv2d
        h = h.permute(0, 2, 1).reshape(N * T, 64, 1, 1)
        h = self.phase_head(h)  # [N*T, 12, 1, 1]
        h = h.reshape(N, T, 12).permute(0, 2, 1)  # [N, 12, T]

        # L2-normalize across (channel, time) jointly — see forward_phase.
        h = F.normalize(h.flatten(1, 2), dim=1).unflatten(1, (12, T))

        # FiLM conditioning
        # FiLMLayer expects [B, cond_dim, H, W]; reshape to [N, 64, 1, 1]
        z_cond = z_type_pixels.unsqueeze(-1).unsqueeze(-1)  # [N, 64, 1, 1]
        gamma, beta = self.phase_film(z_cond)  # each [N, 12, 1, 1]
        gamma = gamma.squeeze(-1)  # [N, 12, 1]
        beta = beta.squeeze(-1)    # [N, 12, 1]
        z = gamma * h + beta  # [N, 12, T]

        z = z.permute(0, 2, 1)  # [N, T, 12]
        if return_film:
            # Return gamma/beta as [N, 12] (squeeze the broadcast time dim)
            return z, gamma.squeeze(-1), beta.squeeze(-1)
        return z


    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        freeze: bool = True,
    ) -> "RepresentationModel":
        """Construct a model and load weights from a checkpoint.

        Args:
            path: Path to a ``.pt`` checkpoint saved by the training script.
            device: Device to map tensors onto.
            freeze: If True, set all parameters to ``requires_grad=False``
                and call ``eval()``.

        Returns:
            Loaded RepresentationModel.

        Raises:
            RuntimeError: If the checkpoint's ``model_version`` does not match
                the current class VERSION.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        ckpt_version = checkpoint.get("model_version")
        if ckpt_version != cls.VERSION:
            raise RuntimeError(
                f"Checkpoint model_version={ckpt_version!r} does not match "
                f"RepresentationModel.VERSION={cls.VERSION!r}. "
                f"The architecture has changed since this checkpoint was saved."
            )

        model_kwargs = checkpoint.get("model_kwargs", {})
        model = cls(**model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if freeze:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

        logger.info(f"Loaded RepresentationModel v{cls.VERSION} from {path}")
        return model

    @staticmethod
    def source_file() -> Path:
        """Return the absolute path to this source file (for archival copy)."""
        return Path(inspect.getfile(RepresentationModel))
