"""
Training Configuration Parser for Forest Representation Learning

Parses and validates training YAML configurations, provides structured access
to settings, and ensures consistency with bindings and model configurations.

Usage:
    parser = TrainingConfigParser('frl_training_v0.yaml')
    config = parser.parse()
    
    # Query settings
    batch_size = config.training.batch_size
    lr = config.optimizer.lr
    debug_mode = config.spatial_domain.debug_mode
    
    # Validate
    parser.validate()
    
    # Get summary
    print(parser.summary())
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================
# Data Classes for Structured Access
# ================================================================

@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration"""
    save_every_n_epochs: int = 5
    save_top_k: int = 3
    monitor: str = "val/loss_total"
    mode: str = "min"  # 'min' or 'max'
    save_last: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CheckpointConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class RunConfig:
    """Run/experiment configuration"""
    experiment_name: str
    run_root: str = "runs"
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunConfig':
        checkpoint = CheckpointConfig.from_dict(d.get('checkpoint', {}))
        return cls(
            experiment_name=d['experiment_name'],
            run_root=d.get('run_root', 'runs'),
            ckpt_dir=d.get('ckpt_dir', 'checkpoints'),
            log_dir=d.get('log_dir', 'logs'),
            checkpoint=checkpoint
        )

@dataclass
class EpochConfig:
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    mode: str = "full"          # full | frac | number
    sample_frac: Optional[float] = None
    sample_number: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpochConfig":
        d = d or {}
        obj = cls(
            num_epochs=d.get("num_epochs"),
            batch_size=d.get("batch_size"),
            mode=d.get("mode", "full"),
            sample_frac=d.get("sample_frac"),
            sample_number=d.get("sample_number"),
        )

        if obj.mode not in {"full", "frac", "number"}:
            raise ValueError(f"Invalid training.epoch.mode: {obj.mode}")

        if obj.mode == "frac" and obj.sample_frac is None:
            raise ValueError("training.epoch.sample_frac required when mode='frac'")

        if obj.mode == "number" and obj.sample_number is None:
            raise ValueError("training.epoch.sample_number required when mode='number'")

        return obj


@dataclass
class MixedPrecisionConfig:
    """Mixed precision training configuration"""
    enabled: bool = True
    dtype: str = "bfloat16"  # 'float16' or 'bfloat16'
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MixedPrecisionConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class HardwareConfig:
    """Hardware and performance configuration"""
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HardwareConfig':
        mixed_precision = MixedPrecisionConfig.from_dict(d.get('mixed_precision', {}))
        return cls(
            device=d.get('device', 'cuda'),
            gpu_ids=d.get('gpu_ids', [0]),
            num_workers=d.get('num_workers', 4),
            pin_memory=d.get('pin_memory', True),
            mixed_precision=mixed_precision
        )


@dataclass
class GradientClipConfig:
    """Gradient clipping configuration"""
    enabled: bool = True
    max_norm: float = 1.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GradientClipConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    enabled: bool = True
    patience: int = 15
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EarlyStoppingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class ValidationConfig:
    """Validation configuration"""
    enabled: bool = True
    val_every_n_epochs: int = 1
    val_fraction: float = 0.15
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ValidationConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    epoch: EpochConfig
    gradient_clip: GradientClipConfig
    early_stopping: EarlyStoppingConfig
    validation: ValidationConfig
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        gradient_clip = GradientClipConfig.from_dict(d.get("gradient_clip", {}))
        early_stopping = EarlyStoppingConfig.from_dict(d.get("early_stopping", {}))
        validation = ValidationConfig.from_dict(d.get("validation", {}))

        epoch = EpochConfig.from_dict(d.get("epoch", {}))

        # Backwards compatibility:
        num_epochs = d.get("num_epochs", epoch.num_epochs)
        batch_size = d.get("batch_size", epoch.batch_size)

        if num_epochs is None or batch_size is None:
            raise KeyError(
                "num_epochs and batch_size must be defined either under "
                "training: or training.epoch:"
            )

        # Canonicalize: keep values consistent
        epoch.num_epochs = num_epochs
        epoch.batch_size = batch_size

        return cls(
            num_epochs=num_epochs,
            batch_size=batch_size,
            epoch=epoch,
            gradient_clip=gradient_clip,
            early_stopping=early_stopping,
            validation=validation,
        )



@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OptimizerConfig':
        return cls(
            name=d.get('name', 'adamw'),
            lr=d.get('lr', 1e-4),
            weight_decay=d.get('weight_decay', 0.01)
        )


@dataclass
class WarmupConfig:
    """Learning rate warmup configuration"""
    enabled: bool = True
    epochs: int = 5
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WarmupConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    name: str = "cosine_warmup"
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    T_max: int = 95
    eta_min: float = 1e-6
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SchedulerConfig':
        warmup = WarmupConfig.from_dict(d.get('warmup', {}))
        return cls(
            name=d.get('name', 'cosine_warmup'),
            warmup=warmup,
            T_max=d.get('T_max', 95),
            eta_min=d.get('eta_min', 1e-6)
        )


@dataclass
class SpatialWindow:
    """Spatial window specification"""
    origin: List[int]  # [row, col]
    size: List[int]    # [height, width]
    block_grid: Optional[List[int]] = None  # [rows, cols]
    
    @property
    def row_start(self) -> int:
        return self.origin[0]
    
    @property
    def col_start(self) -> int:
        return self.origin[1]
    
    @property
    def height(self) -> int:
        return self.size[0]
    
    @property
    def width(self) -> int:
        return self.size[1]
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpatialWindow':
        return cls(
            origin=d['origin'],
            size=d['size'],
            block_grid=d.get('block_grid')
        )


@dataclass
class SpatialDomainConfig:
    """Spatial domain configuration"""
    debug_mode: bool
    debug_window: Optional[SpatialWindow] = None
    full_domain: Optional[SpatialWindow] = None
    
    @property
    def active_window(self) -> SpatialWindow:
        """Get the currently active spatial window based on debug_mode"""
        if self.debug_mode:
            return self.debug_window
        else:
            return self.full_domain
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpatialDomainConfig':
        debug_window = SpatialWindow.from_dict(d['debug_window']) if 'debug_window' in d else None
        full_domain = SpatialWindow.from_dict(d['full_domain']) if 'full_domain' in d else None
        
        return cls(
            debug_mode=d['debug_mode'],
            debug_window=debug_window,
            full_domain=full_domain
        )


@dataclass
class TemporalBundleConfig:
    """Temporal window bundle configuration"""
    enabled: bool = True
    size: int = 3
    offsets: List[int] = field(default_factory=lambda: [0, -2, -4])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TemporalBundleConfig':
        return cls(
            enabled=d.get('enabled', True),
            size=d.get('size', 3),
            offsets=d.get('offsets', [0, -2, -4])
        )


@dataclass
class TemporalSamplingConfig:
    """Temporal sampling strategy"""
    mode: str = "weighted"  # 'uniform', 'weighted', 'balanced'
    weights: Optional[Dict[int, float]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TemporalSamplingConfig':
        return cls(
            mode=d.get('mode', 'weighted'),
            weights=d.get('weights')
        )


@dataclass
class TemporalDomainConfig:
    """Temporal domain configuration"""
    end_years: List[int]
    window_length: int
    bundle: TemporalBundleConfig = field(default_factory=TemporalBundleConfig)
    sampling: TemporalSamplingConfig = field(default_factory=TemporalSamplingConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TemporalDomainConfig':
        bundle = TemporalBundleConfig.from_dict(d.get('bundle', {}))
        sampling = TemporalSamplingConfig.from_dict(d.get('sampling', {}))
        
        return cls(
            end_years=d['end_years'],
            window_length=d['window_length'],
            bundle=bundle,
            sampling=sampling
        )


@dataclass
class GridSubsampleConfig:
    """Grid subsampling for contrastive learning"""
    enabled: bool = True
    grid_size: List[int] = field(default_factory=lambda: [16, 16])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GridSubsampleConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class ForestSamplesConfig:
    """Supplemental forest sampling configuration"""
    enabled: bool = True
    per_patch: int = 64
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ForestSamplesConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    enabled: bool = True
    random_flip: Optional[Dict[str, Any]] = None
    random_rotation: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AugmentationConfig':
        return cls(
            enabled=d.get('enabled', True),
            random_flip=d.get('random_flip'),
            random_rotation=d.get('random_rotation')
        )


@dataclass
class SamplingConfig:
    """Spatial sampling configuration"""
    patch_size: int
    grid_subsample: GridSubsampleConfig = field(default_factory=GridSubsampleConfig)
    forest_samples: ForestSamplesConfig = field(default_factory=ForestSamplesConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SamplingConfig':
        grid_subsample = GridSubsampleConfig.from_dict(d.get('grid_subsample', {}))
        forest_samples = ForestSamplesConfig.from_dict(d.get('forest_samples', {}))
        augmentation = AugmentationConfig.from_dict(d.get('augmentation', {}))
        
        return cls(
            patch_size=d['patch_size'],
            grid_subsample=grid_subsample,
            forest_samples=forest_samples,
            augmentation=augmentation
        )


@dataclass
class LossWeightSchedule:
    """Loss weight schedule entry"""
    epoch: List[Optional[int]]  # [start, end] where None means open-ended
    value: Union[float, str]    # Float value or 'linear' for ramping
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LossWeightSchedule':
        return cls(
            epoch=d['epoch'],
            value=d['value']
        )
    
    def get_weight_at_epoch(self, epoch: int) -> float:
        """Get the weight value for a given epoch"""
        start, end = self.epoch
        
        # Check if epoch is in range
        if start is not None and epoch < start:
            return 0.0
        if end is not None and epoch > end:
            return 0.0
        
        # Handle linear ramping
        if isinstance(self.value, str) and self.value == 'linear':
            if start is None or end is None:
                raise ValueError("Linear schedule requires both start and end epochs")
            # Linear interpolation from 0 to 1
            progress = (epoch - start) / (end - start)
            return max(0.0, min(1.0, progress))
        
        return float(self.value)


@dataclass
class TripletLossConfig:
    """Triplet loss configuration"""
    enabled: bool = True
    weight_schedule: List[LossWeightSchedule] = field(default_factory=list)
    triplet_mining: Optional[Dict[str, Any]] = None
    
    def get_weight_at_epoch(self, epoch: int) -> float:
        """Get the current weight for this loss at given epoch"""
        if not self.enabled:
            return 0.0
        
        for schedule in self.weight_schedule:
            start, end = schedule.epoch
            if start is not None and epoch < start:
                continue
            if end is not None and epoch > end:
                continue
            return schedule.get_weight_at_epoch(epoch)
        
        return 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TripletLossConfig':
        weight_schedule = [
            LossWeightSchedule.from_dict(s) 
            for s in d.get('weight_schedule', [])
        ]
        
        return cls(
            enabled=d.get('enabled', True),
            weight_schedule=weight_schedule,
            triplet_mining=d.get('triplet_mining')
        )


@dataclass
class VQLossConfig:
    """Vector quantization loss configuration"""
    enabled: bool = True
    weight: float = 1.0
    commitment_cost: float = 0.25
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VQLossConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class MonotonicityConstraint:
    """Phase monotonicity constraint"""
    when: Dict[str, Any]  # Condition (e.g., {ysfc_in: [0, 1]})
    order: List[str]      # Expected order (e.g., [t0, t4, t2])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MonotonicityConstraint':
        return cls(
            when=d['when'],
            order=d['order']
        )


@dataclass
class PhaseMonotonicityLossConfig:
    """Phase monotonicity loss configuration"""
    enabled: bool = True
    weight_schedule: List[LossWeightSchedule] = field(default_factory=list)
    constraints: List[MonotonicityConstraint] = field(default_factory=list)
    
    def get_weight_at_epoch(self, epoch: int) -> float:
        """Get the current weight for this loss at given epoch"""
        if not self.enabled:
            return 0.0
        
        for schedule in self.weight_schedule:
            start, end = schedule.epoch
            if start is not None and epoch < start:
                continue
            if end is not None and epoch > end:
                continue
            return schedule.get_weight_at_epoch(epoch)
        
        return 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PhaseMonotonicityLossConfig':
        weight_schedule = [
            LossWeightSchedule.from_dict(s) 
            for s in d.get('weight_schedule', [])
        ]
        constraints = [
            MonotonicityConstraint.from_dict(c)
            for c in d.get('constraints', [])
        ]
        
        return cls(
            enabled=d.get('enabled', True),
            weight_schedule=weight_schedule,
            constraints=constraints
        )


@dataclass
class LossesConfig:
    """All loss configurations"""
    z_type_triplet: TripletLossConfig = field(default_factory=TripletLossConfig)
    vq_loss: VQLossConfig = field(default_factory=VQLossConfig)
    phase_monotonicity: PhaseMonotonicityLossConfig = field(default_factory=PhaseMonotonicityLossConfig)
    
    def get_total_weight_at_epoch(self, epoch: int) -> Dict[str, float]:
        """Get all loss weights for a given epoch"""
        return {
            'z_type_triplet': self.z_type_triplet.get_weight_at_epoch(epoch),
            'vq_loss': self.vq_loss.weight if self.vq_loss.enabled else 0.0,
            'phase_monotonicity': self.phase_monotonicity.get_weight_at_epoch(epoch)
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LossesConfig':
        z_type_triplet = TripletLossConfig.from_dict(d.get('z_type_triplet', {}))
        vq_loss = VQLossConfig.from_dict(d.get('vq_loss', {}))
        phase_monotonicity = PhaseMonotonicityLossConfig.from_dict(d.get('phase_monotonicity', {}))
        
        return cls(
            z_type_triplet=z_type_triplet,
            vq_loss=vq_loss,
            phase_monotonicity=phase_monotonicity
        )


@dataclass
class MaskingConfig:
    """Masking and weighting configuration"""
    global_mask: Union[str, List[str]]
    per_loss_masks: Optional[Dict[str, List[str]]] = None
    global_weight: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MaskingConfig':
        return cls(
            global_mask=d.get('global_mask', []),
            per_loss_masks=d.get('per_loss_masks'),
            global_weight=d.get('global_weight')
        )


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    train: List[str] = field(default_factory=list)
    validation: List[str] = field(default_factory=list)
    latent_analysis: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MetricsConfig':
        return cls(
            train=d.get('train', []),
            validation=d.get('validation', []),
            latent_analysis=d.get('latent_analysis')
        )


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    enabled: bool = True
    tensorboard: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VisualizationConfig':
        return cls(
            enabled=d.get('enabled', True),
            tensorboard=d.get('tensorboard')
        )


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration"""
    seed: int = 42
    benchmark: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ReproducibilityConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class TrainingConfiguration:
    """Complete training configuration"""
    version: str
    name: str
    config_paths: Dict[str, str]  # bindings_path, model_path
    run: RunConfig
    hardware: HardwareConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    spatial_domain: SpatialDomainConfig
    temporal_domain: TemporalDomainConfig
    sampling: SamplingConfig
    losses: LossesConfig
    masking: MaskingConfig
    metrics: MetricsConfig
    visualization: VisualizationConfig
    reproducibility: ReproducibilityConfig
    
    @property
    def is_debug_mode(self) -> bool:
        """Check if running in debug mode"""
        return self.spatial_domain.debug_mode
    
    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size (accounting for gradient accumulation)"""
        return self.training.batch_size
    
    @property
    def total_training_steps(self) -> int:
        """Estimate total training steps (rough estimate)"""
        # This is approximate - actual depends on dataset size
        window = self.spatial_domain.active_window
        num_patches = (window.height // self.sampling.patch_size) * \
                      (window.width // self.sampling.patch_size)
        steps_per_epoch = num_patches // self.training.batch_size
        return steps_per_epoch * self.training.num_epochs


# ================================================================
# Parser
# ================================================================

class TrainingConfigParser:
    """
    Parser for training configuration YAML files.
    
    Validates configuration, provides structured access, and ensures
    consistency with bindings and model configurations.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize parser.
        
        Args:
            config_path: Path to training YAML configuration file
        """
        self.config_path = Path(config_path)
        self.raw_config: Optional[Dict[str, Any]] = None
        self.config: Optional[TrainingConfiguration] = None
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def parse(self) -> TrainingConfiguration:
        """
        Parse the training configuration file.
        
        Returns:
            TrainingConfiguration object with structured access to all settings
        """
        # Load YAML
        with open(self.config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)
        
        logger.info(f"Loaded training config: {self.config_path}")
        
        # Parse into structured format
        self.config = TrainingConfiguration(
            version=self.raw_config.get('version', '1.0'),
            name=self.raw_config.get('name', 'unnamed'),
            config_paths=self.raw_config.get('config', {}),
            run=RunConfig.from_dict(self.raw_config.get('run', {})),
            hardware=HardwareConfig.from_dict(self.raw_config.get('hardware', {})),
            training=TrainingConfig.from_dict(self.raw_config.get('training', {})),
            optimizer=OptimizerConfig.from_dict(self.raw_config.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(self.raw_config.get('scheduler', {})),
            spatial_domain=SpatialDomainConfig.from_dict(self.raw_config.get('spatial_domain', {})),
            temporal_domain=TemporalDomainConfig.from_dict(self.raw_config.get('temporal_domain', {})),
            sampling=SamplingConfig.from_dict(self.raw_config.get('sampling', {})),
            losses=LossesConfig.from_dict(self.raw_config.get('losses', {})),
            masking=MaskingConfig.from_dict(self.raw_config.get('masking', {})),
            metrics=MetricsConfig.from_dict(self.raw_config.get('metrics', {})),
            visualization=VisualizationConfig.from_dict(self.raw_config.get('visualization', {})),
            reproducibility=ReproducibilityConfig.from_dict(self.raw_config.get('reproducibility', {}))
        )
        
        return self.config
    
    def validate(self) -> bool:
        """
        Validate configuration for common issues.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if self.config is None:
            raise ValueError("Must call parse() before validate()")
        
        errors = []
        warnings = []
        
        # Check config paths exist
        bindings_path = Path(self.config.config_paths.get('bindings_path', ''))
        model_path = Path(self.config.config_paths.get('model_path', ''))
        
        if not bindings_path.exists():
            errors.append(f"Bindings config not found: {bindings_path}")
        
        if not model_path.exists():
            errors.append(f"Model config not found: {model_path}")
        
        # Validate spatial domain
        window = self.config.spatial_domain.active_window
        if window.height % self.config.sampling.patch_size != 0:
            warnings.append(
                f"Window height ({window.height}) not divisible by patch_size "
                f"({self.config.sampling.patch_size})"
            )
        
        if window.width % self.config.sampling.patch_size != 0:
            warnings.append(
                f"Window width ({window.width}) not divisible by patch_size "
                f"({self.config.sampling.patch_size})"
            )
        
        # Validate temporal domain
        if self.config.temporal_domain.bundle.enabled:
            if self.config.temporal_domain.bundle.size != len(self.config.temporal_domain.bundle.offsets):
                errors.append(
                    f"Bundle size ({self.config.temporal_domain.bundle.size}) "
                    f"doesn't match number of offsets ({len(self.config.temporal_domain.bundle.offsets)})"
                )
        
        # Validate loss schedules
        if self.config.losses.z_type_triplet.enabled and not self.config.losses.z_type_triplet.weight_schedule:
            warnings.append("Triplet loss enabled but no weight schedule defined")
        
        # Check hardware settings
        if self.config.hardware.device == 'cuda' and not self.config.hardware.gpu_ids:
            errors.append("Device is 'cuda' but no GPU IDs specified")
        
        # Validate batch size
        if self.config.training.batch_size < 1:
            errors.append(f"Batch size must be >= 1, got {self.config.training.batch_size}")
        
        # Check learning rate
        if self.config.optimizer.lr <= 0:
            errors.append(f"Learning rate must be > 0, got {self.config.optimizer.lr}")
        
        # Validate scheduler
        if self.config.scheduler.warmup.enabled:
            if self.config.scheduler.warmup.epochs >= self.config.training.num_epochs:
                warnings.append(
                    f"Warmup epochs ({self.config.scheduler.warmup.epochs}) >= "
                    f"total epochs ({self.config.training.num_epochs})"
                )
        
        # Print warnings
        if warnings:
            logger.warning("Validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        # Raise errors
        if errors:
            error_msg = "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        logger.info("✓ Configuration validated successfully")
        return True
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the configuration.
        
        Returns:
            Formatted summary string
        """
        if self.config is None:
            raise ValueError("Must call parse() before summary()")
        
        lines = [
            "=" * 70,
            f"Training Configuration: {self.config.name}",
            "=" * 70,
            "",
            "EXPERIMENT",
            f"  Name: {self.config.run.experiment_name}",
            f"  Mode: {'DEBUG' if self.config.is_debug_mode else 'PRODUCTION'}",
            "",
            "TRAINING",
            f"  Epochs: {self.config.training.num_epochs}",
            f"  Batch size: {self.config.training.batch_size}",
            f"  Learning rate: {self.config.optimizer.lr:.2e}",
            f"  Optimizer: {self.config.optimizer.name}",
            f"  Scheduler: {self.config.scheduler.name}",
            "",
            "HARDWARE",
            f"  Device: {self.config.hardware.device}",
            f"  GPUs: {self.config.hardware.gpu_ids}",
            f"  Workers: {self.config.hardware.num_workers}",
            f"  Mixed precision: {self.config.hardware.mixed_precision.enabled} "
            f"({self.config.hardware.mixed_precision.dtype})",
            "",
            "SPATIAL DOMAIN",
        ]
        
        window = self.config.spatial_domain.active_window
        lines.extend([
            f"  Origin: {window.origin} (row, col)",
            f"  Size: {window.size} (height, width)",
            f"  Patches: ~{(window.height // self.config.sampling.patch_size) * (window.width // self.config.sampling.patch_size):,}",
            f"  Patch size: {self.config.sampling.patch_size}",
        ])
        
        lines.extend([
            "",
            "TEMPORAL DOMAIN",
            f"  End years: {self.config.temporal_domain.end_years}",
            f"  Window length: {self.config.temporal_domain.window_length} years",
            f"  Bundle: {self.config.temporal_domain.bundle.enabled} "
            f"(offsets: {self.config.temporal_domain.bundle.offsets})",
            "",
            "LOSSES",
            f"  VQ loss: {self.config.losses.vq_loss.enabled}",
            f"  Triplet loss: {self.config.losses.z_type_triplet.enabled}",
            f"  Phase monotonicity: {self.config.losses.phase_monotonicity.enabled}",
            "",
            "SAMPLING",
            f"  Grid subsample: {self.config.sampling.grid_subsample.enabled}",
            f"  Forest samples: {self.config.sampling.forest_samples.enabled}",
            f"  Augmentation: {self.config.sampling.augmentation.enabled}",
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def get_loss_schedule(self) -> str:
        """
        Generate a summary of the loss schedule across epochs.
        
        Returns:
            Formatted schedule string
        """
        if self.config is None:
            raise ValueError("Must call parse() before get_loss_schedule()")
        
        lines = ["Loss Weight Schedule", "=" * 50]
        
        # Sample epochs
        sample_epochs = [0, 10, 20, 30, 50, self.config.training.num_epochs - 1]
        
        lines.append(f"{'Epoch':<10} {'VQ':<10} {'Triplet':<10} {'Monoton.':<10}")
        lines.append("-" * 50)
        
        for epoch in sample_epochs:
            weights = self.config.losses.get_total_weight_at_epoch(epoch)
            lines.append(
                f"{epoch:<10} "
                f"{weights['vq_loss']:<10.2f} "
                f"{weights['z_type_triplet']:<10.2f} "
                f"{weights['phase_monotonicity']:<10.2f}"
            )
        
        return "\n".join(lines)


# ================================================================
# Convenience Functions
# ================================================================

def load_training_config(config_path: Union[str, Path]) -> TrainingConfiguration:
    """
    Convenience function to load and parse training configuration.
    
    Args:
        config_path: Path to training YAML file
        
    Returns:
        Parsed TrainingConfiguration object
    """
    parser = TrainingConfigParser(config_path)
    config = parser.parse()
    parser.validate()
    return config


if __name__ == '__main__':
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config/frl_training_v1.yaml'
    
    print(f"\nParsing: {config_path}\n")
    
    # Parse and validate
    parser = TrainingConfigParser(config_path)
    config = parser.parse()
    parser.validate()
    
    # Print summary
    print(parser.summary())
    
    # Print loss schedule
    print("\n")
    print(parser.get_loss_schedule())
    
    # Example queries
    print("\n" + "=" * 70)
    print("Example Queries")
    print("=" * 70)
    print(f"Is debug mode? {config.is_debug_mode}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Patch size: {config.sampling.patch_size}")
    print(f"Learning rate: {config.optimizer.lr}")
    print(f"Active window size: {config.spatial_domain.active_window.size}")
    print(f"Temporal end years: {config.temporal_domain.end_years}")
    print(f"VQ loss enabled? {config.losses.vq_loss.enabled}")
