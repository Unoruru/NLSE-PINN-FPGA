# Script to visually compare PC PINN predictions against FPGA-accelerated predictions
# Loads pre-trained checkpoint + deterministic inputs, runs PC inference,
# then overlays PC and FPGA output points on a single constellation diagram.
# Last Updated: 24 Mar 2026

import os
import sys
import pickle
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="brevitas")

# Check working directory is project root
check_path = os.path.join(os.getcwd(), "cwd.check")
if not os.path.isfile(check_path):
    print(f"ERROR: cwd.check not found at {check_path}. Run this script from the project root directory.")
    sys.exit()

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import matplotlib.pyplot as plt

from consolidate.config import complexPINN
from consolidate.helper import synchronize_signals, align_signal, evm, calculate_ser
from consolidate.sigClassify import classify

FINAL_SCALE = 0.00345  # FPGA dequantisation scale (from evalComplexTemp.py)
SLD_WIN = 25           # Must match the window size used during training


def main():
    parser = argparse.ArgumentParser(description="Overlay PC vs FPGA PINN constellation comparison.")
    parser.add_argument("--sig_type", type=str, default="16qam",
                        choices=["16qam", "16apsk", "16psk", "star"],
                        help="Signal type to evaluate. Default '16qam'.")
    parser.add_argument("--dir_inputs", type=str, default="sample_results",
                        help="Directory (relative to project root) containing per-signal checkpoint and inputs. Default 'sample_results'.")
    parser.add_argument("--dir_fpga", type=str,
                        default="pynq-zu/_deployment/20260322_complex_v3/results",
                        help="Directory (relative to project root) containing FPGA output folders. Default 'pynq-zu/_deployment/20260322_complex_v3/results'.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for PC inference. Default 'cpu'.")
    args = parser.parse_args()

    sig_dir = os.path.join(os.getcwd(), args.dir_inputs, args.sig_type)
    checkpoint_path = os.path.join(sig_dir, "complex_pinn_checkpoint.pth")
    inputs_path = os.path.join(sig_dir, "generated_inputs.pklv2")
    fpga_output_path = os.path.join(os.getcwd(), args.dir_fpga,
                                    f"{args.sig_type}_results_fpga", "output_0.npy")
    out_dir = os.path.join(os.getcwd(), "pc2fpga_eval")
    plot_save_path = os.path.join(out_dir, f"{args.sig_type}_pc_vs_fpga_overlay.png")

    # --- Load inputs ---
    if not os.path.isfile(inputs_path):
        print(f"ERROR: inputs file not found at {inputs_path}")
        sys.exit()
    with open(inputs_path, "rb") as f:
        sig_type, clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
    if sig_type != args.sig_type:
        print(f"ERROR: loaded signal type '{sig_type}' does not match requested '{args.sig_type}'.")
        sys.exit()
    print(f"Loaded inputs for {args.sig_type} from {inputs_path}.")

    # --- Load checkpoint and run PC inference ---
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}")
        sys.exit()
    device = torch.device(args.device)
    model = complexPINN(window_size=SLD_WIN, hlayers=3, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        p_out = model(X_train.to(device)).cpu().numpy()
    pc_complex = p_out[:, 0] + 1j * p_out[:, 1]
    print(f"PC inference complete. {len(pc_complex)} output points.")

    # --- Load and dequantise FPGA output ---
    if not os.path.isfile(fpga_output_path):
        print(f"ERROR: FPGA output not found at {fpga_output_path}")
        sys.exit()
    fpga_raw = np.load(fpga_output_path)
    fpga_float = fpga_raw * FINAL_SCALE
    fpga_complex = fpga_float[:, 0] + 1j * fpga_float[:, 1]
    print(f"FPGA output loaded and dequantised. {len(fpga_complex)} output points.")

    # --- Synchronise and align both outputs to the clean reference ---
    clean_sync_pc, pc_sync = synchronize_signals(clean, pc_complex)
    clean_align_pc, pc_aligned = align_signal(clean_sync_pc, pc_sync)

    clean_sync_fpga, fpga_sync = synchronize_signals(clean, fpga_complex)
    clean_align_fpga, fpga_aligned = align_signal(clean_sync_fpga, fpga_sync)

    # --- Compute metrics ---
    pc_evm = evm(clean_align_pc, pc_aligned)
    pc_ser = calculate_ser(*[classify(x, args.sig_type)[1] for x in (clean_align_pc, pc_aligned)])

    fpga_evm = evm(clean_align_fpga, fpga_aligned)
    fpga_ser = calculate_ser(*[classify(x, args.sig_type)[1] for x in (clean_align_fpga, fpga_aligned)])

    print(f"PC   — EVM: {pc_evm:.2f}%  SER: {pc_ser:.2f}%")
    print(f"FPGA — EVM: {fpga_evm:.2f}%  SER: {fpga_ser:.2f}%")
    print(f"Delta — EVM: {fpga_evm - pc_evm:+.2f}%  SER: {fpga_ser - pc_ser:+.2f}%")

    # --- Plot overlay ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fpga_aligned.real, fpga_aligned.imag, s=1, alpha=0.4, color="red", label=f"FPGA (EVM {fpga_evm:.2f}%)")
    ax.scatter(pc_aligned.real, pc_aligned.imag, s=1, alpha=0.4, color="tab:blue", label=f"PC (EVM {pc_evm:.2f}%)")
    ax.set_title(f"{args.sig_type.upper()} — PC vs FPGA PINN Output Overlay")
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", markerscale=6)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    print(f"Overlay plot saved to {plot_save_path}.")


if __name__ == "__main__":
    main()
