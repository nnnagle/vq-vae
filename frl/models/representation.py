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

from .conv2d_encoder import Conv2DEncoder
from .conditioning import FiLMLayer
from .spatial import GatedResidualConv2D
from .tcn import TCNEncoder

logger = logging.getLogger(__name__)


class RepresentationModel(nn.Module):
    """Full encoder pipeline for type and phase embeddings.

    **Type pathway** (v1):
        ``[B, 16, H, W]`` → Conv2DEncoder → GatedResidualConv2D → z_type ``[B, 64, H, W]``

    **Phase pathway** (v2):
        ``[B, 8, T, H, W]`` → TCN → FiLM(stopgrad z_type) → 1×1 proj → z_phase ``[B, 12, T, H, W]``

    The FiLM generates gamma/beta once from the spatial z_type and
    broadcasts them across all timesteps (the type identity modulates
    temporal features uniformly).

    Attributes:
        VERSION: Architecture version string. Bump this whenever the forward
            pipeline changes in a checkpoint-incompatible way.
    """

    VERSION = "2"

    def __init__(self) -> None:
        super().__init__()

        # --- Type pathway ---
        self.encoder = Conv2DEncoder(
            in_channels=16,
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
            in_channels=8,
            channels=[64, 64, 64],
            kernel_size=3,
            dilations=[1, 2, 4],
            dropout_rate=0.1,
            num_groups=8,
            pooling='none',
        )
        # FiLM: generates gamma/beta from z_type [B, 64, H, W]
        # to modulate TCN output (64-dim), broadcast across T
        self.phase_film = FiLMLayer(
            cond_dim=64,
            target_dim=64,
            hidden_dim=64,
        )
        # Project to final phase embedding dimension
        self.phase_head = nn.Conv2d(64, 12, kernel_size=1)

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Type pathway forward.

        Args:
            x: Input tensor ``[B, 16, H, W]`` (ccdc_history features).
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

        Use for inference when embeddings are needed at every pixel.

        Args:
            x_phase: Temporal input ``[B, 8, T, H, W]`` (phase_ls8 features).
            z_type: Type embeddings ``[B, 64, H, W]`` (**caller must
                stop-grad** before passing in).

        Returns:
            z_phase: ``[B, 12, T, H, W]`` phase embeddings.
        """
        B, C, T, H, W = x_phase.shape

        # TCN along time: [B, 8, T, H, W] -> [B, 64, T, H, W]
        h = self.phase_tcn(x_phase)

        # FiLM: generate gamma/beta from z_type once, broadcast across T
        gamma, beta = self.phase_film(z_type)  # each [B, 64, H, W]
        gamma = gamma.unsqueeze(2)  # [B, 64, 1, H, W]
        beta = beta.unsqueeze(2)    # [B, 64, 1, H, W]
        h = gamma * h + beta        # [B, 64, T, H, W]

        # Project per-timestep: reshape to [B*T, 64, H, W] for Conv2d
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, 64, H, W)
        z_phase = self.phase_head(h)  # [B*T, 12, H, W]
        z_phase = z_phase.reshape(B, T, 12, H, W).permute(0, 2, 1, 3, 4)

        return z_phase  # [B, 12, T, H, W]

    def forward_phase_at_locations(
        self,
        x_phase_pixels: torch.Tensor,
        z_type_pixels: torch.Tensor,
    ) -> torch.Tensor:
        """Phase pathway forward at sampled pixel locations only.

        Runs the same TCN → FiLM → projection pipeline but only on the
        supplied pixel time-series, avoiding the cost of processing the
        full spatial grid.  Produces identical results to extracting from
        the dense ``forward_phase`` output at the same locations.

        Args:
            x_phase_pixels: ``[N, C, T]`` temporal features at N pixels.
            z_type_pixels: ``[N, 64]`` type embeddings at the same N pixels
                (**caller must stop-grad**).

        Returns:
            z_phase_pixels: ``[N, T, 12]`` phase embeddings.
        """
        N, C, T = x_phase_pixels.shape

        # TCN layers operate on [N, C, T] directly (no spatial reshape)
        h = x_phase_pixels
        for layer in self.phase_tcn.layers:
            h = layer(h)  # [N, 64, T]

        # FiLM: z_type_pixels [N, 64] -> gamma/beta via the same FiLM layer
        # FiLMLayer expects [B, cond_dim, H, W]; reshape to [N, 64, 1, 1]
        z_cond = z_type_pixels.unsqueeze(-1).unsqueeze(-1)  # [N, 64, 1, 1]
        gamma, beta = self.phase_film(z_cond)  # each [N, 64, 1, 1]
        gamma = gamma.squeeze(-1)  # [N, 64, 1]
        beta = beta.squeeze(-1)    # [N, 64, 1]
        h = gamma * h + beta       # [N, 64, T]

        # Project: phase_head is Conv2d(64, 12, 1×1)
        # Reshape [N, 64, T] -> [N*T, 64, 1, 1] for Conv2d
        h = h.permute(0, 2, 1).reshape(N * T, 64, 1, 1)
        z = self.phase_head(h)  # [N*T, 12, 1, 1]
        z = z.reshape(N, T, 12)

        return z  # [N, T, 12]

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

        model = cls().to(device)
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
