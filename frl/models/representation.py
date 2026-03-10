"""
RepresentationModel — unified encoder pipeline for contrastive representation learning.

This module defines the complete forward pipeline (encoder -> spatial conv -> embedding)
as a single nn.Module. All training scripts, probes, and diagnostics should use this
class rather than assembling components individually.

Checkpoints store ``model_version``, ``model_config``, ``type_in_channels``, and
``phase_in_channels`` so that downstream scripts can reconstruct the exact architecture
without needing the original YAML on disk.

Example
-------
Create from config dict::

    import yaml
    from models import RepresentationModel

    with open('config/frl_repr_model_v1.yaml') as f:
        model_cfg = yaml.safe_load(f)

    model = RepresentationModel.from_config(
        model_cfg,
        type_in_channels=16,
        phase_in_channels=8,
    ).to(device)

    # ... training loop ...
    torch.save({
        'model_version': RepresentationModel.VERSION,
        'model_config': model_cfg,
        'type_in_channels': model.type_in_channels,
        'phase_in_channels': model.phase_in_channels,
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
from typing import List, Optional

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

    **Type pathway**:
        ``[B, C_type, H, W]`` → Conv2DEncoder → GatedResidualConv2D → z_type ``[B, z_type_dim, H, W]``

    **Phase pathway**:
        ``[B, C_phase, T, H, W]`` → TCN → 1×1 bottleneck → FiLM(stopgrad z_type)
        → z_phase ``[B, z_phase_dim, T, H, W]``

    Construct via :meth:`from_config` rather than calling ``__init__`` directly.

    The bottleneck projects TCN features to the z_phase_dim embedding space,
    then FiLM applies type-conditioned affine modulation directly in
    that space.  FiLM gamma is initialized near 1 and beta near 0,
    giving near-identity behavior at the start of training.

    Attributes:
        VERSION: Checkpoint schema version. Bump when the checkpoint dict
            structure changes in a backward-incompatible way. Architecture
            variants (channel widths, dropout, etc.) are encoded in
            ``model_config`` and do not require a VERSION bump.
    """

    VERSION = "4"

    def __init__(
        self,
        type_in_channels: int,
        phase_in_channels: int,
        # latent dims
        z_type_dim: int = 64,
        z_phase_dim: int = 12,
        # type encoder (Conv2DEncoder)
        type_encoder_channels: List[int] = (128, 64),
        type_encoder_kernel_size: int = 1,
        type_encoder_padding: int = 0,
        type_encoder_dropout: float = 0.1,
        type_encoder_num_groups: int = 8,
        type_encoder_input_dropout: float = 0.0,
        # spatial conv (GatedResidualConv2D)
        spatial_conv_num_layers: int = 2,
        spatial_conv_kernel_size: int = 3,
        spatial_conv_padding: int = 1,
        spatial_conv_gate_hidden: int = 64,
        spatial_conv_gate_kernel_size: int = 1,
        # phase TCN (TCNEncoder)
        phase_tcn_channels: List[int] = (64, 64, 64),
        phase_tcn_kernel_size: int = 3,
        phase_tcn_dilations: List[int] = (1, 2, 4),
        phase_tcn_dropout: float = 0.1,
        phase_tcn_num_groups: int = 8,
    ) -> None:
        super().__init__()

        if list(type_encoder_channels)[-1] != z_type_dim:
            raise ValueError(
                f"type_encoder_channels[-1]={type_encoder_channels[-1]} must equal "
                f"z_type_dim={z_type_dim}"
            )

        self.type_in_channels = type_in_channels
        self.phase_in_channels = phase_in_channels
        self.z_type_dim = z_type_dim
        self.z_phase_dim = z_phase_dim

        # --- Type pathway ---
        self.encoder = Conv2DEncoder(
            in_channels=type_in_channels,
            channels=list(type_encoder_channels),
            kernel_size=type_encoder_kernel_size,
            padding=type_encoder_padding,
            dropout_rate=type_encoder_dropout,
            num_groups=type_encoder_num_groups,
            input_dropout_rate=type_encoder_input_dropout,
        )
        self.spatial_conv = GatedResidualConv2D(
            channels=z_type_dim,
            num_layers=spatial_conv_num_layers,
            kernel_size=spatial_conv_kernel_size,
            padding=spatial_conv_padding,
            gate_hidden=spatial_conv_gate_hidden,
            gate_kernel_size=spatial_conv_gate_kernel_size,
        )

        # --- Phase pathway ---
        self.phase_tcn = TCNEncoder(
            in_channels=phase_in_channels,
            channels=list(phase_tcn_channels),
            kernel_size=phase_tcn_kernel_size,
            dilations=list(phase_tcn_dilations),
            dropout_rate=phase_tcn_dropout,
            num_groups=phase_tcn_num_groups,
            pooling='none',
        )
        tcn_out_dim = list(phase_tcn_channels)[-1]
        # Bottleneck: project TCN output to final embedding dimension
        self.phase_head = nn.Conv2d(tcn_out_dim, z_phase_dim, kernel_size=1)
        # FiLM: generates gamma/beta from z_type to modulate z_phase_dim-dim output.
        # Gamma initialized near 1, beta near 0 → near-identity at init.
        self.phase_film = FiLMLayer(
            cond_dim=z_type_dim,
            target_dim=z_phase_dim,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        type_in_channels: int,
        phase_in_channels: int,
    ) -> "RepresentationModel":
        """Construct a RepresentationModel from a config dict.

        The config dict matches the structure of ``frl_repr_model_v1.yaml``.
        ``type_in_channels`` and ``phase_in_channels`` are passed separately
        because they are determined by the data pipeline (bindings config),
        not the model architecture.

        Args:
            cfg: Architecture config dict (contents of the model YAML).
            type_in_channels: Number of input channels for the type pathway.
            phase_in_channels: Number of input channels for the phase pathway.

        Returns:
            Constructed (untrained) RepresentationModel.

        Raises:
            ValueError: If the config version does not match VERSION, or if
                type_encoder.channels[-1] != latents.z_type_dim.
        """
        cfg_version = str(cfg.get("version", ""))
        if cfg_version != cls.VERSION:
            raise ValueError(
                f"Config version={cfg_version!r} does not match "
                f"RepresentationModel.VERSION={cls.VERSION!r}."
            )

        latents = cfg.get("latents", {})
        z_type_dim = latents.get("z_type_dim", 64)
        z_phase_dim = latents.get("z_phase_dim", 12)

        te = cfg.get("type_encoder", {})
        sc = cfg.get("spatial_conv", {})
        pt = cfg.get("phase_tcn", {})

        # input_dropout may be a scalar (constant) or a schedule dict.
        # At construction time we always use the initial rate:
        #   scalar  -> that value
        #   dict    -> "start" (the rate at epoch 0)
        # The training loop is responsible for calling set_input_dropout_rate()
        # each epoch when a schedule is active.
        input_dropout_cfg = te.get("input_dropout", 0.0)
        if isinstance(input_dropout_cfg, dict):
            type_encoder_input_dropout = float(input_dropout_cfg.get("start", 0.0))
        else:
            type_encoder_input_dropout = float(input_dropout_cfg)

        return cls(
            type_in_channels=type_in_channels,
            phase_in_channels=phase_in_channels,
            z_type_dim=z_type_dim,
            z_phase_dim=z_phase_dim,
            # type encoder
            type_encoder_channels=te.get("channels", [128, 64]),
            type_encoder_kernel_size=te.get("kernel_size", 1),
            type_encoder_padding=te.get("padding", 0),
            type_encoder_dropout=te.get("dropout", 0.1),
            type_encoder_num_groups=te.get("num_groups", 8),
            type_encoder_input_dropout=type_encoder_input_dropout,
            # spatial conv
            spatial_conv_num_layers=sc.get("num_layers", 2),
            spatial_conv_kernel_size=sc.get("kernel_size", 3),
            spatial_conv_padding=sc.get("padding", 1),
            spatial_conv_gate_hidden=sc.get("gate_hidden", 64),
            spatial_conv_gate_kernel_size=sc.get("gate_kernel_size", 1),
            # phase TCN
            phase_tcn_channels=pt.get("channels", [64, 64, 64]),
            phase_tcn_kernel_size=pt.get("kernel_size", 3),
            phase_tcn_dilations=pt.get("dilations", [1, 2, 4]),
            phase_tcn_dropout=pt.get("dropout", 0.1),
            phase_tcn_num_groups=pt.get("num_groups", 8),
        )

    def set_input_dropout_rate(self, rate: float) -> None:
        """Update the type encoder's input dropout rate at runtime.

        Intended for scheduled input dropout: call once per epoch with the
        epoch-appropriate rate before processing any batches.

        Args:
            rate: Dropout probability in [0, 1]. 0.0 disables dropout.
        """
        self.encoder.set_input_dropout_rate(rate)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Type pathway forward.

        Args:
            x: Input tensor ``[B, C_type, H, W]`` (ccdc_history features).
            return_gate: If True, also return the spatial gate tensor.

        Returns:
            If *return_gate* is False: ``z_type [B, z_type_dim, H, W]``.
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
            z_type: Type embeddings ``[B, z_type_dim, H, W]`` (**caller must
                stop-grad** before passing in).

        Returns:
            z_phase: ``[B, z_phase_dim, T, H, W]`` phase embeddings.
        """
        B, C, T, H, W = x_phase.shape
        zp = self.z_phase_dim

        # TCN along time: [B, C_phase, T, H, W] -> [B, tcn_out, T, H, W]
        h = self.phase_tcn(x_phase)

        # Bottleneck per-timestep: reshape to [B*T, tcn_out, H, W] for Conv2d
        tcn_out = h.shape[1]
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, tcn_out, H, W)
        h = self.phase_head(h)  # [B*T, zp, H, W]
        h = h.reshape(B, T, zp, H, W).permute(0, 2, 1, 3, 4)  # [B, zp, T, H, W]

        # L2-normalize across (channel, time) jointly so the TCN controls
        # direction & temporal shape, while FiLM gamma owns the per-channel scale.
        # One norm factor per spatial location — temporal variation is preserved.
        h = F.normalize(h.flatten(1, 2), dim=1).unflatten(1, (zp, T))

        # FiLM conditioning
        gamma, beta = self.phase_film(z_type)  # each [B, zp, H, W]
        gamma = gamma.unsqueeze(2)  # [B, zp, 1, H, W]
        beta = beta.unsqueeze(2)    # [B, zp, 1, H, W]
        z_phase = gamma * h + beta  # [B, zp, T, H, W]

        return z_phase

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
            z_type_pixels: ``[N, z_type_dim]`` type embeddings at the same N pixels
                (**caller must stop-grad**).
            return_film: If True, also return the data-dependent gamma and
                beta tensors (useful for diagnostics).

        Returns:
            If *return_film* is False: ``z_phase_pixels [N, T, z_phase_dim]``.
            If *return_film* is True: ``(z_phase_pixels, gamma, beta)``
                where gamma and beta are ``[N, z_phase_dim]``.
        """
        N, C, T = x_phase_pixels.shape
        zp = self.z_phase_dim

        # TCN accepts [N, C, T] directly (no spatial reshape needed)
        h = self.phase_tcn(x_phase_pixels)  # [N, tcn_out, T]

        # Bottleneck: phase_head is Conv2d(tcn_out, zp, 1×1)
        # Reshape [N, tcn_out, T] -> [N*T, tcn_out, 1, 1] for Conv2d
        tcn_out = h.shape[1]
        h = h.permute(0, 2, 1).reshape(N * T, tcn_out, 1, 1)
        h = self.phase_head(h)  # [N*T, zp, 1, 1]
        h = h.reshape(N, T, zp).permute(0, 2, 1)  # [N, zp, T]

        # L2-normalize across (channel, time) jointly — see forward_phase.
        h = F.normalize(h.flatten(1, 2), dim=1).unflatten(1, (zp, T))

        # FiLM conditioning
        # FiLMLayer expects [B, cond_dim, H, W]; reshape to [N, z_type_dim, 1, 1]
        z_cond = z_type_pixels.unsqueeze(-1).unsqueeze(-1)  # [N, z_type_dim, 1, 1]
        gamma, beta = self.phase_film(z_cond)  # each [N, zp, 1, 1]
        gamma = gamma.squeeze(-1)  # [N, zp, 1]
        beta = beta.squeeze(-1)    # [N, zp, 1]
        z = gamma * h + beta  # [N, zp, T]

        z = z.permute(0, 2, 1)  # [N, T, zp]
        if return_film:
            # Return gamma/beta as [N, zp] (squeeze the broadcast time dim)
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
            RuntimeError: If the checkpoint's ``model_version`` is not supported.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        ckpt_version = checkpoint.get("model_version")
        if ckpt_version != cls.VERSION:
            raise RuntimeError(
                f"Checkpoint model_version={ckpt_version!r} is not supported. "
                f"RepresentationModel.VERSION={cls.VERSION!r}. "
                f"The checkpoint was saved with a different schema version."
            )

        model_config = checkpoint["model_config"]
        type_in_channels = checkpoint["type_in_channels"]
        phase_in_channels = checkpoint["phase_in_channels"]

        model = cls.from_config(
            model_config,
            type_in_channels=type_in_channels,
            phase_in_channels=phase_in_channels,
        ).to(device)
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
