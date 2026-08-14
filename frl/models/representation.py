from __future__ import annotations
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

import inspect
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d_encoder import Conv2DEncoder
from .heads import MLPProjectionHead
from .spatial import GatedResidualConv2D, EdgeAwareSmoothingConv2D
from .tcn import TCNEncoder
from .mature_baseline import RFFMatureBaseline
from .anomaly_transform import AnomalyTransform
from losses.ou_dynamics import OUDynamics

logger = logging.getLogger(__name__)


class RepresentationModel(nn.Module):
    """Full encoder pipeline for type and phase embeddings.

    **Type pathway**:
        ``[B, C_type, H, W]`` → Conv2DEncoder → EdgeAwareSmoothingConv2D → z_type ``[B, z_type_dim, H, W]``

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
        # spatial conv (EdgeAwareSmoothingConv2D)
        spatial_conv_num_layers: int = 2,
        spatial_conv_kernel_size: int = 3,
        spatial_conv_padding: int = 1,
        spatial_conv_gate_hidden: int = 64,
        spatial_conv_gate_kernel_size: int = 3,
        spatial_conv_num_directions: int = 4,
        spatial_conv_coarse_dilation: int = 3,
        spatial_conv_rank: int = 4,
        # phase TCN (TCNEncoder)
        phase_tcn_channels: List[int] = (64, 64, 64),
        phase_tcn_kernel_size: int = 3,
        phase_tcn_dilations: List[int] = (1, 2, 4),
        phase_tcn_dropout: float = 0.1,
        phase_tcn_num_groups: int = 8,
        phase_tcn_norm: str = "none",
        # mature-baseline RFF readout (Step 1) + anomaly transform (Step 2)
        rff_features: int = 1024,
        rff_bandwidth: float = 1.0,
        rff_ridge_lambda: float = 1e-3,
        rff_sigma_floor: float = 1e-2,
        rff_ema_decay: float = 0.99,
        anomaly_delta_only: bool = True,
        anomaly_scale_decay: float = 0.99,
        # Step-4 within-pixel OU dynamics
        ou_rho_init: float = 0.9,
        ou_learn_rho: bool = True,
        ou_s0: float = 0.5,
        ou_s1: float = 0.0,
        ou_learn_s: bool = False,
        ou_tau: float = 2.0,
        # type projection head (SimCLR-style; None = disabled)
        type_proj_hidden_dim: Optional[int] = None,
        type_proj_output_dim: Optional[int] = None,
        type_proj_l2_normalize: bool = True,
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
        self.spatial_conv = EdgeAwareSmoothingConv2D(
            channels=z_type_dim,
            num_layers=spatial_conv_num_layers,
            kernel_size=spatial_conv_kernel_size,
            padding=spatial_conv_padding,
            gate_hidden=spatial_conv_gate_hidden,
            gate_kernel_size=spatial_conv_gate_kernel_size,
            num_directions=spatial_conv_num_directions,
            coarse_dilation=spatial_conv_coarse_dilation,
            rank=spatial_conv_rank,
        )

        # --- Phase pathway (FiLM-free; type enters only via the μ/σ input
        #     normalization below — see docs/phase_rethink_design.md Steps 1-3b) ---
        # Step-1 mature-baseline readout μ(z_type), σ(z_type) and Step-2 anomaly
        # transform a=(x-μ)/σ + Δa.  The TCN consumes the 2*C anomaly input.
        self.mature_baseline = RFFMatureBaseline(
            in_dim=z_type_dim,
            n_channels=phase_in_channels,
            n_features=rff_features,
            bandwidth=rff_bandwidth,
            ridge_lambda=rff_ridge_lambda,
            sigma_floor=rff_sigma_floor,
            ema_decay=rff_ema_decay,
        )
        self.anomaly_transform = AnomalyTransform(
            n_channels=phase_in_channels,
            delta_only=anomaly_delta_only,
            scale_decay=anomaly_scale_decay,
        )
        self.phase_tcn = TCNEncoder(
            in_channels=self.anomaly_transform.out_channels,   # 2 * phase_in_channels
            channels=list(phase_tcn_channels),
            kernel_size=phase_tcn_kernel_size,
            dilations=list(phase_tcn_dilations),
            dropout_rate=phase_tcn_dropout,
            num_groups=phase_tcn_num_groups,
            norm=phase_tcn_norm,      # 'none'/'batch': magnitude-preserving (no per-sample GroupNorm-over-T)
            pooling='none',
        )
        tcn_out_dim = list(phase_tcn_channels)[-1]
        # Bottleneck: project TCN output to final embedding dimension (no FiLM,
        # no pre-FiLM L2-norm — magnitude must survive to become z_phase radius).
        self.phase_head = nn.Conv2d(tcn_out_dim, z_phase_dim, kernel_size=1)

        # Step-4 within-pixel OU dynamics (learnable ρ; loss lives in step.py).
        self.ou_dynamics = OUDynamics(
            rho_init=ou_rho_init, learn_rho=ou_learn_rho,
            s0=ou_s0, s1=ou_s1, learn_s=ou_learn_s, tau=ou_tau,
        )

        # --- Type projection head (SimCLR-style) ---
        # Applied between z_type and InfoNCE losses during training.
        # None when disabled (project_type() acts as identity).
        if type_proj_hidden_dim is not None and type_proj_output_dim is not None:
            self.type_projection: Optional[MLPProjectionHead] = MLPProjectionHead(
                in_dim=z_type_dim,
                hidden_dim=type_proj_hidden_dim,
                output_dim=type_proj_output_dim,
                l2_normalize=type_proj_l2_normalize,
            )
        else:
            self.type_projection = None

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
        tp = cfg.get("type_projection", {})
        mb = cfg.get("mature_baseline", {})   # Step-1 RFF readout
        at = cfg.get("anomaly_transform", {})  # Step-2 anomaly transform
        ou = cfg.get("ou_dynamics", {})        # Step-4 within-pixel OU

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
            spatial_conv_gate_kernel_size=sc.get("gate_kernel_size", 3),
            spatial_conv_num_directions=sc.get("num_directions", 4),
            spatial_conv_coarse_dilation=sc.get("coarse_dilation", 3),
            spatial_conv_rank=sc.get("rank", 4),
            # phase TCN
            phase_tcn_channels=pt.get("channels", [64, 64, 64]),
            phase_tcn_kernel_size=pt.get("kernel_size", 3),
            phase_tcn_dilations=pt.get("dilations", [1, 2, 4]),
            phase_tcn_dropout=pt.get("dropout", 0.1),
            phase_tcn_num_groups=pt.get("num_groups", 8),
            phase_tcn_norm=pt.get("norm", "none"),
            # mature-baseline readout + anomaly transform
            rff_features=mb.get("n_features", 1024),
            rff_bandwidth=mb.get("bandwidth", 1.0),
            rff_ridge_lambda=mb.get("ridge_lambda", 1e-3),
            rff_sigma_floor=mb.get("sigma_floor", 1e-2),
            rff_ema_decay=mb.get("ema_decay", 0.99),
            anomaly_delta_only=at.get("delta_only", True),
            anomaly_scale_decay=at.get("scale_decay", 0.99),
            # OU dynamics
            ou_rho_init=ou.get("rho_init", 0.9),
            ou_learn_rho=ou.get("learn_rho", True),
            ou_s0=ou.get("s0", 0.5),
            ou_s1=ou.get("s1", 0.0),
            ou_learn_s=ou.get("learn_s", False),
            ou_tau=ou.get("tau", 2.0),
            # type projection head
            type_proj_hidden_dim=tp.get("hidden_dim") if tp.get("enabled", False) else None,
            type_proj_output_dim=tp.get("output_dim") if tp.get("enabled", False) else None,
            type_proj_l2_normalize=tp.get("l2_normalize", True),
        )

    def set_spatial_min_gate(self, value: float) -> None:
        """Set the curriculum gate floor on the spatial smoothing conv (0.0 = off, 1.0 = identity)."""
        self.spatial_conv.set_min_gate(value)

    def set_input_dropout_rate(self, rate: float) -> None:
        """Update the type encoder's input dropout rate at runtime.

        Intended for scheduled input dropout: call once per epoch with the
        epoch-appropriate rate before processing any batches.

        Args:
            rate: Dropout probability in [0, 1]. 0.0 disables dropout.
        """
        self.encoder.set_input_dropout_rate(rate)

    def project_type(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the SimCLR projection head to z_type embeddings.

        Used before InfoNCE losses during training. At inference time call
        the model forward() directly and use z_type without projection.

        Args:
            z: Type embeddings ``[N, z_type_dim]``.

        Returns:
            Projected embeddings ``[N, output_dim]`` if the projection head is
            enabled, otherwise ``z`` unchanged.
        """
        if self.type_projection is None:
            return z
        return self.type_projection(z)

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
        """Dense full-grid phase forward — **not yet rewired for the FiLM-free
        anomaly pathway** (Step 7 consolidates inference).

        The training path uses :meth:`forward_phase_at_locations`. Dense
        inference needs the μ/σ readout + anomaly transform applied over the full
        grid; that wiring is deferred, so this raises rather than silently
        returning a FiLM-based embedding that no longer exists.
        """
        raise NotImplementedError(
            "forward_phase (dense) is not wired for the FiLM-free anomaly pathway; "
            "use forward_phase_at_locations. Dense inference is rewired in Step 7."
        )

    def forward_phase_at_locations(
        self,
        x_phase_pixels: torch.Tensor,
        z_type_pixels: torch.Tensor,
        valid: torch.Tensor | None = None,
        return_input: bool = False,
    ) -> torch.Tensor | tuple:
        """Phase pathway forward at sampled pixel locations (FiLM-free).

        Builds the type-conditional anomaly input from the raw phase feature and
        the μ/σ readout, then runs the TCN → bottleneck (no FiLM). Type enters
        **only** through the anomaly normalization; ``z_phase = f(a, Δa)``.

        Args:
            x_phase_pixels: ``[N, C, T]`` **raw** phase feature at N pixels (the
                bindings phase feature, e.g. phase_ccdc — *not* pre-normalized).
            z_type_pixels: ``[N, z_type_dim]`` type embeddings at the same pixels
                (**caller must stop-grad**; the readout also detaches internally).
            valid: optional ``[N, T]`` timestep-validity mask.
            return_input: if True, also return the anomaly input ``[N, 2C, T]``.

        Returns:
            ``z_phase_pixels [N, T, z_phase_dim]`` (and the anomaly input if
            *return_input*).
        """
        # μ/σ from the mature-baseline readout (detached z_type → stop-grad), then
        # the anomaly transform [a ; Δa].  Both are grad-free constant inputs.
        mu, sigma = self.mature_baseline.predict(z_type_pixels)          # [N, C]
        feats, _ = self.anomaly_transform(x_phase_pixels, mu, sigma, valid)  # [N, 2C, T]
        z = self.encode_phase(feats)                                     # [N, T, zp]
        if return_input:
            return z, feats
        return z

    def encode_phase(self, feats: torch.Tensor) -> torch.Tensor:
        """Encode a pre-built anomaly input into z_phase (TCN → bottleneck, no FiLM).

        Split out from :meth:`forward_phase_at_locations` so the training loop can
        run the anomaly transform every batch (to settle the readout / observe the
        Δ scale during warmup) while the encoder itself runs only once the phase
        curriculum is active.

        Args:
            feats: ``[N, 2C, T]`` anomaly input ``[a ; Δa]``.
        Returns:
            ``z_phase [N, T, z_phase_dim]``.
        """
        N, _, T = feats.shape
        zp = self.z_phase_dim
        h = self.phase_tcn(feats)                                        # [N, tcn_out, T]
        tcn_out = h.shape[1]
        h = h.permute(0, 2, 1).reshape(N * T, tcn_out, 1, 1)
        h = self.phase_head(h)                                           # [N*T, zp, 1, 1]
        return h.reshape(N, T, zp)                                       # [N, T, zp]

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
