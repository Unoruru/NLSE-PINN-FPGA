"""
Central configuration for PINN NLSE training.
All hyperparameters and physical parameters in one place.
"""

from dataclasses import dataclass, field


@dataclass
class PINNConfig:
    # --- Physical parameters ---
    beta2: float = -1.0           # second-order dispersion
    gamma: float = 1.0            # nonlinear coefficient (Kerr)
    L: float = 1.0                # propagation length
    n_steps: int = 100            # SSFM steps

    # --- Time axis ---
    T_max: float = 10.0
    N_t: int = 1024

    # --- Pulse ---
    P0: float = 1.0               # peak power
    T0: float = 1.0               # pulse width

    # --- Network ---
    hidden_dim: int = 50
    layers: int = 4

    # --- Training (FP32) ---
    n_epochs: int = 10000
    lr: float = 1e-3
    lr_min: float = 1e-6
    N_res: int = 10000            # collocation points
    resample_every: int = 100     # collocation resampling interval
    grad_clip_norm: float = 1.0
    lambda_ic_update_every: int = 100  # adaptive loss weight update interval

    # --- Training (QAT) ---
    qat_epochs: int = 3000
    qat_lr: float = 5e-4
    qat_lr_min: float = 1e-6
    qat_warmup_epochs: int = 500
    qat_N_res: int = 15000
    qat_ic_boost_factor: float = 2.0
    weight_bit_width: int = 8
    act_bit_width: int = 8

    # --- Validation & Checkpointing ---
    validate_every: int = 1000
    print_every: int = 500

    # --- Reproducibility ---
    seed: int = 42
