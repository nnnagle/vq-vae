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
from .spatial import GatedResidualConv2D

logger = logging.getLogger(__name__)


class RepresentationModel(nn.Module):
    """Full encoder pipeline: Conv2DEncoder -> GatedResidualConv2D.

    Attributes:
        VERSION: Architecture version string. Bump this whenever the forward
            pipeline changes in a checkpoint-incompatible way.
    """

    VERSION = "1"

    def __init__(self) -> None:
        super().__init__()
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

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor ``[B, C_in, H, W]`` (e.g. 16-channel ccdc_history).
            return_gate: If True, also return the spatial gate tensor.

        Returns:
            If *return_gate* is False: ``z [B, D, H, W]`` embeddings.
            If *return_gate* is True: ``(z, gate)`` where gate is ``[B, D, H, W]``.
        """
        h = self.encoder(x)
        if return_gate:
            z, gate = self.spatial_conv(h, return_gate=True)
            return z, gate
        return self.spatial_conv(h)

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
