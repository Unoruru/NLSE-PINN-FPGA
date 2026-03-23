# (Temp) Script for evaluating the outputs from the accelerator for complex PINN
# Last Updated: 23 Mar 2026

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
# check at project root
check_path = os.path.join(os.getcwd(), "cwd.check")
if not os.path.isfile(check_path):
    print(f"Current working directory is not project root. Expected to find 'cwd.check' at {check_path}. Please change to project root and rerun.")
    sys.exit()
else:
    print(f"Current working directory verified as project root: {os.getcwd()}.")

from consolidate.helper import synchronize_signals, align_signal, evm, calculate_ser
from consolidate.sigClassify import classify

def main():
    parser = argparse.ArgumentParser(description="Evaluate FPGA hardware outputs for complex PINN.")
    parser.add_argument("--dir_inputs", type=str, default="sample_results", help="Path to the directory (relative to cwd) containing the generated inputs.")
    parser.add_argument("--dir_fpga", type=str, default="pynq-zu/_deployment/20260322_complex_v3/results/", help="Path to the directory (relative to cwd) containing the fpga outputs are saved.")
    parser.add_argument("--sig_type", type=str, default="16qam", choices=["16qam", "16apsk", "16psk", "star"], help="Type of modulation scheme for classification.")

    args = parser.parse_args()
    inputs_save_path = os.path.join(os.getcwd(), args.dir_inputs, args.sig_type, "generated_inputs.pklv2")

    try:
        with open(inputs_save_path, "rb") as f:
            sig_type, clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
            print(f"Loaded training data and inputs from {inputs_save_path} for evaluation.")
    except Exception as e:
        print(f"Failed to load inputs: {e}. Ensure the file exists and is a valid .pkl file. Exiting...")
        sys.exit()

    assert(sig_type == args.sig_type), f"Signal type mismatch: expected {args.sig_type}, but got {sig_type} from loaded data. Please check the input file and modulation type." 

    outputs_path = os.path.join(os.getcwd(), args.dir_fpga, args.sig_type+"_results_fpga", "output_0.npy")
    try:
        outputs = np.load(outputs_path)
        print("Loaded FPGA hardware outputs from file.")
    except Exception as e:
        print(f"Load Error: {e}")
        sys.exit()
    
    # dequantization
    FINAL_SCALE = 0.00345 # where required, extract value from brevitas
    outputs_float = outputs * FINAL_SCALE
    outputs_complex = outputs_float[:, 0] + 1j * outputs_float[:, 1]

    # evm
    clean_sync, out_sync = synchronize_signals(clean, outputs_complex)
    clean_align, out_align = align_signal(clean_sync, out_sync)
    out_evm = evm(clean_align, out_align)
    
    # ser
    _, clean_classify = classify(clean_align, args.sig_type)
    _, out_classify = classify(out_align, args.sig_type)
    out_ser = calculate_ser(clean_classify, out_classify)
    
    print(f"EVM = {out_evm:.2f}% | SER = {out_ser:.2f}%") 
    
    metrics_save_path = os.path.join(os.getcwd(), args.dir_fpga, f"{args.sig_type}_fpga_evaluation_results.txt")
    with open(metrics_save_path, "w") as f:
        f.write(f"Evaluation Results (FPGA-Accelerated) for {args.sig_type.upper()} Modulation:\n")
        f.write(f"EVM: {out_evm:.2f}%\n")
        f.write(f"SER: {out_ser:.2f}%\n")
    print(f"Saved evaluation metrics to {metrics_save_path}")

    diagram_save_path = os.path.join(os.getcwd(), args.dir_fpga, f"{args.sig_type}_fpga_constellation.png")
    # draw constellation diagram
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.scatter(out_align.real, out_align.imag, s=1, alpha=0.5)
    ax.set_title(f"{args.sig_type} 8-bit PINN on FPGA Accelerator")
    ax.set_aspect('equal')
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    plt.savefig(diagram_save_path)
    print(f"Saved constellation diagram to {diagram_save_path}")
        
if __name__ == "__main__":
    main()
    