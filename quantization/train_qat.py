"""
Quantization-Aware Training (QAT) for PINN NLSE using Brevitas.

Usage:
    python train_qat.py --fp32-checkpoint ../software/fp32_pinn_best.pth

Improvements over qat_pinn.py:
  - Loads pretrained FP32 weights (standard QAT pipeline)
  - Input normalization (z, t -> [-1, 1])
  - CosineAnnealing LR scheduler (lower LR for fine-tuning)
  - Adaptive loss weighting, collocation resampling, gradient clipping
  - No Unwrap modules (cleaner quantization graph)
  - Loss curve tracking (every epoch)
  - Validation + best model checkpointing
  - Proper exception handling (no bare except)
  - Exports QONNX for FPGA/FINN
"""

import sys
import os
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import PINNConfig
from shared.ssfm import ssfm_nlse
from shared.training import set_seed, train_pinn
from shared.evaluation import (
    evaluate_full_field, plot_z0_check, plot_zL_comparison,
    plot_error_heatmap, plot_error_density, plot_loss_curves, print_summary
)
from quantization.qat_model import QuantPINN_NLSE, load_fp32_into_qat

try:
    from brevitas.export import StdQCDQONNXManager, export_qonnx
except ImportError:
    print("Error: Brevitas not found. Please install it using 'pip install brevitas'.")
    sys.exit(1)


def export_model(model, device, export_dir):
    """Export quantized model to QONNX and fallback to ONNX."""
    dummy_input = torch.randn(1, 2, device=device)

    # Try QONNX first (optimized for FINN)
    try:
        print("Attempting QONNX export (optimized for FINN)...")
        qonnx_path = os.path.join(export_dir, "qat_pinn_model.qonnx")
        try:
            export_qonnx(model.net, args=dummy_input, export_path=qonnx_path, dynamo=False)
        except TypeError:
            export_qonnx(model.net, args=dummy_input, export_path=qonnx_path)
        print(f"QONNX exported to {qonnx_path}")
        return
    except Exception as e:
        print(f"QONNX export failed: {e}")

    # Fallback: Standard QCDQ ONNX
    try:
        print("Attempting Standard QCDQ ONNX export...")
        onnx_path = os.path.join(export_dir, "qat_pinn_model.onnx")
        try:
            StdQCDQONNXManager.export(model.net, args=dummy_input, export_path=onnx_path, dynamo=False)
        except TypeError:
            StdQCDQONNXManager.export(model.net, args=dummy_input, export_path=onnx_path)
        print(f"ONNX exported to {onnx_path}")
    except Exception as e2:
        print(f"All exports failed: {e2}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="QAT fine-tuning for PINN NLSE")
    parser.add_argument("--fp32-checkpoint", type=str, default=None,
                        help="Path to pretrained FP32 model (fp32_pinn_best.pth)")
    args = parser.parse_args()

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

    # --- QAT Model ---
    model = QuantPINN_NLSE(
        hidden_dim=cfg.hidden_dim,
        layers=cfg.layers,
        weight_bit_width=cfg.weight_bit_width,
        act_bit_width=cfg.act_bit_width
    ).to(device)

    # --- Load pretrained FP32 weights ---
    if args.fp32_checkpoint and os.path.exists(args.fp32_checkpoint):
        print(f"Loading FP32 weights from {args.fp32_checkpoint}...")
        load_fp32_into_qat(model, args.fp32_checkpoint, device=device)
    else:
        if args.fp32_checkpoint:
            print(f"Warning: FP32 checkpoint not found at {args.fp32_checkpoint}")
        print("Training QAT from scratch (not recommended).")

    # --- Override config for QAT fine-tuning ---
    qat_cfg = PINNConfig(
        n_epochs=cfg.qat_epochs,
        lr=cfg.qat_lr,
        lr_min=cfg.qat_lr_min,
        print_every=100,
        validate_every=200,
        seed=cfg.seed,
    )

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=qat_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=qat_cfg.n_epochs, eta_min=qat_cfg.lr_min
    )

    # --- Training ---
    export_dir = os.path.dirname(__file__)
    save_path = os.path.join(export_dir, "qat_pinn_best.pth")

    history = train_pinn(
        model, optimizer, scheduler, qat_cfg, t, A0, A_true_at_L, device,
        save_path=save_path
    )

    # --- Load best model ---
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"\nLoaded best QAT model from {save_path}")

    # --- Evaluation ---
    A_pred = evaluate_full_field(model, t, z_points, cfg.L, cfg.T_max, cfg.N_t, device)
    print_summary(t, A_true, A_pred)

    # --- Plots ---
    plot_loss_curves(history, save_path=os.path.join(export_dir, "qat_loss_curves.png"))
    plot_z0_check(t, A0, A_pred, save_path=os.path.join(export_dir, "qat_z0_check.png"))
    plot_zL_comparison(t, A_true, A_pred, save_path=os.path.join(export_dir, "qat_zL_comparison.png"), label="QAT PINN")
    plot_error_heatmap(t, z_points, cfg.L, A_true, A_pred, save_path=os.path.join(export_dir, "qat_abs_error.png"))
    plot_error_density(t, z_points, cfg.L, A_true, A_pred, save_path=os.path.join(export_dir, "qat_error_density.png"))

    # --- Export ---
    export_model(model, device, export_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
