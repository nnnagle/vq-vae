# src/training/loss_config.py
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ForestTrajectoryLossConfig:
    # KL / VAE strength
    beta: float = 1.0

    # Weights on auxiliary losses
    lambda_cat: float = 1.0
    lambda_delta: float = 10.0
    lambda_deriv: float = 10.0
    lambda_spatial_grad: float = 0.0

    # Continuous reconstruction loss
    w_final: float = 2.0   # extra weight on last timestep

    # Which continuous channels to use for temporal/texture losses
    # None -> use [0] (preserves your current behavior)
    channel_indices: Optional[Tuple[int, ...]] = None

    # Threshold (in normalized units) below which temporal changes are ignored
    change_thresh: float = 0.05

    # Spatial gradient loss options
    spatial_grad_mode: str = "huber"  # "l2" | "l1" | "huber"
    spatial_grad_beta: float = 0.05   # Huber beta
    
    lambda_vq: float = 0.0  # weight on VQ loss; keep 0.0 to disable by default
