"""
FP32 PINN training for NLSE with second-order dispersion and Kerr nonlinearity.

    A_z = i * (beta2/2) * A_tt + i * gamma * |A|^2 * A

Improvements over PINN_2disp_kerr.py:
  - Reproducibility seeds
  - Input normalization (z, t -> [-1, 1])
  - CosineAnnealing LR scheduler
  - Adaptive loss weighting (PDE vs IC)
  - Collocation point resampling
  - Gradient clipping
  - Loss curve tracking (every epoch)
  - Validation against SSFM during training
  - Best model checkpointing

Saves: fp32_pinn_best.pth for downstream QAT fine-tuning.
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import PINNConfig
from shared.ssfm import ssfm_nlse
from shared.pinn_base import PINN_NLSE
from shared.training import set_seed, train_pinn
from shared.evaluation import (
    evaluate_full_field, plot_z0_check, plot_zL_comparison,
    plot_error_heatmap, plot_error_density, plot_loss_curves, print_summary
)


def main():
    cfg = PINNConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Reproducibility ---
    set_seed(cfg.seed)

    # --- Time axis & initial pulse ---
    t = np.linspace(-cfg.T_max, cfg.T_max, cfg.N_t)
    A0 = np.sqrt(cfg.P0) * np.exp(-t**2 / (2 * cfg.T0**2))

    # --- SSFM reference ---
    print("Running SSFM reference...")
    A_L, z_points, A_snaps = ssfm_nlse(A0, t, cfg.beta2, cfg.gamma, cfg.L, cfg.n_steps)
    A_true = np.array(A_snaps)
    A_true_at_L = A_true[-1]
    print(f"SSFM done, {len(z_points)} z-snapshots.")

    # --- Model + Optimizer + Scheduler ---
    model = PINN_NLSE(hidden_dim=cfg.hidden_dim, layers=cfg.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr_min
    )

    # --- Training ---
    save_dir = os.path.dirname(__file__)
    save_path = os.path.join(save_dir, "fp32_pinn_best.pth")

    history = train_pinn(
        model, optimizer, scheduler, cfg, t, A0, A_true_at_L, device,
        save_path=save_path
    )

    # --- Load best model for evaluation ---
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"\nLoaded best model from {save_path}")

    # --- Evaluation ---
    A_pred = evaluate_full_field(model, t, z_points, cfg.L, cfg.T_max, cfg.N_t, device)
    print_summary(t, A_true, A_pred)

    # --- Plots ---
    plot_loss_curves(history, save_path=os.path.join(save_dir, "fp32_loss_curves.png"))
    plot_z0_check(t, A0, A_pred, save_path=os.path.join(save_dir, "fp32_z0_check.png"))
    plot_zL_comparison(t, A_true, A_pred, save_path=os.path.join(save_dir, "fp32_zL_comparison.png"), label="FP32 PINN")
    plot_error_heatmap(t, z_points, cfg.L, A_true, A_pred, save_path=os.path.join(save_dir, "fp32_abs_error.png"))
    plot_error_density(t, z_points, cfg.L, A_true, A_pred, save_path=os.path.join(save_dir, "fp32_error_density.png"))


if __name__ == "__main__":
    main()
