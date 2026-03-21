# Script for training complex PINN for multi-signal-type prediction
# Supports 4 signal types: 16-QAM, 16-APSK, 16-PSK, and Star-QAM
# Last Updated: 21 Mar 2026

# See readme.md for detailed instructions on running this script, including environment setup and dependencies.

import os
import sys
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()

log_file_format = logging.Formatter('%(asctime)s, %(msecs)03d %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('training_pinn.log', mode='a') # appends to log file each run (a append w overwrite)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_file_format)

console_file_format = logging.Formatter('%(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_file_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.log(logging.INFO, "Log saved as training_pinn.log in current working directory.")

# Check if current working directory is correct (should be project root)
check_path = os.path.join(os.getcwd(), "cwd.check")
if not os.path.isfile(check_path):
    logger.log(logging.ERROR, f"Current working directory is not project root. Expected to find 'cwd.check' at {check_path}. Please change to project root and rerun.")
    sys.exit()
else:
    logger.log(logging.INFO, f"Current working directory verified as project root: {os.getcwd()}.")
    # check results directory exists
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(results_dir):
        logger.log(logging.WARNING, f"Results directory does not exist. Creating: {results_dir}.")
        os.makedirs(results_dir)
    else:
        logger.log(logging.INFO, f"Results directory verified: {results_dir}.")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import argparse
import warnings

# Supress UserWarnings for following torch, qonnx and brevitas for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="qonnx")
warnings.filterwarnings("ignore", category=UserWarning, module="brevitas")

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

from brevitas.export import export_qonnx

# Custom imports
from consolidate.config import complexPINN, ssfm_nlse, train
from consolidate.helper import align_signal, evm, calculate_ser, synchronize_signals, str2bool
from consolidate.sigGen import genSignals
from consolidate.sigClassify import classify

# Custom assertion function that logs errors before exiting
def assertlog(condition, message):
    try:
        assert condition, message
    except AssertionError as e:
        logger.log(logging.ERROR, f"Assertion Error: {e}")
        sys.exit()

def main():
    parser = argparse.ArgumentParser(description="Training a Complex PINN for Optical Fibre Communication.")
    parser.add_argument("--sig_type", type=str, default="16qam", help="String. Type of signal to generate/load for training/evaluation. Default '16qam'.")
    parser.add_argument("--load", type=str2bool, default=False, help="Boolean. Load existing model checkpoint if available. Default False.")
    parser.add_argument("--load_path", type=str, default="complex_pinn_checkpoint.pth", help="String. Path to model checkpoint for loading. Must be within the results directory. Default 'complex_pinn_checkpoint.pth'.")
    parser.add_argument("--reinforce", type=str2bool, default=False, help="Boolean. Reinforce physics during training with additional loss term. Requires load. Default False.")
    parser.add_argument("--metrics", type=str2bool, default=True, help="Boolean.Calculate and log EVM and SER metrics after training. Default True.")
    parser.add_argument("--visual", type=str2bool, default=True, help="Boolean.Generate and save constellation comparison figure after evaluation. Requires metrics. Default True.")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="Boolean. Save model checkpoint after training. Default True.")
    parser.add_argument("--checkpoint_path", type=str, default="complex_pinn_checkpoint.pth", help="String. Path to save model checkpoint. Must be within the results directory. Default 'complex_pinn_checkpoint.pth'.")
    parser.add_argument("--epochs", type=int, default=3000, help="Integer. Number of training epochs. Default 3000.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Float. Learning rate for training. Default 5e-4.")
    parser.add_argument("--save_inputs", type=str2bool, default=False, help="Boolean. Save generated training data and inputs as .pkl/.npy files for accelerator use. Default False.")
    parser.add_argument("--load_inputs", type=str2bool, default=False, help="Boolean. Load generated training data and inputs from .pkl/.npy files for accelerator use. Requires checkpoint and .pklv2 input file. Default False.")
    parser.add_argument("--onnx_export", type=str2bool, default=True, help="Boolean. Export trained model to ONNX format for FPGA deployment. Default True.")
    parser.add_argument("--onnx_path", type=str, default="complex_pinn.onnx", help="String. Path to save ONNX model. Must be within the results directory. Default 'complex_pinn.onnx'.")
    parser.add_argument("--finn_convert", type=str2bool, default=True, help="Boolean. Convert exported ONNX model to FINN format using qonnx2finn. Requires onnx_export. Default True.")

    args = parser.parse_args()

    assertlog(args.sig_type in ["16qam", "16apsk", "16psk", "star"], "Unsupported signal type specified. Supported types: '16qam', '16apsk', '16psk', 'star'.")
    logger.log(logging.INFO, f"Script started for training complex PINN with signal type: {args.sig_type}.")

    # Fiber Params
    beta2   = -21e-27  # second-order dispersion param (ps^2/m)
    gamma   = 0.015    # Kerr nonlinear coefficient (1/W/km)
    dt      = 1e-12    # 1ps
    n_steps = 50

    # Signal generation for training and evaluation
    sld_win = 25
    L = 40 # km

    # Paths for saving results
    model_save_path = os.path.join(results_dir, args.checkpoint_path)
    model_load_path = os.path.join(results_dir, args.load_path)
    comparison_fig_path = os.path.join(results_dir, f"{args.sig_type}_constellation_comparison.png")

    inputs_save_path = os.path.join(results_dir, "generated_inputs.pklv2")
    accelerator_inputs_path = os.path.join(results_dir, "accelerator_inputs.npy")

    # check onnx export path validity
    assertlog(args.onnx_path.endswith(".onnx"), "ONNX export path must end with .onnx extension.")
    assertlog(args.onnx_path != "model.onnx", "ONNX export path cannot be 'model.onnx' to avoid overwriting FINN conversion output. Please specify a different name.")
    
    # sanity checks for arguments
    assertlog(args.epochs > 0, "Epoch count must be a positive integer.")
    assertlog(args.lr > 0, "Learning rate must be a positive float.")
    assertlog(not (args.save_inputs and args.load_inputs), "Cannot both save and load inputs in the same run. Please choose one.")
    if args.load_inputs:
        assertlog(os.path.isfile(inputs_save_path), f"Input loading specified but file not found at {inputs_save_path}. Please ensure the file exists or disable input loading.")
        assertlog(args.load, "Input loading specified without model loading. Please enable model loading to use loaded inputs.")

    onnx_export_path = os.path.join(results_dir, args.onnx_path)

    # Set up model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = complexPINN(window_size=sld_win, hlayers=3, hidden_dim=64).to(device)
    logger.log(logging.INFO, f"Torch using {device} for training.")

    # Loading model checkpoint if specified, configuring training skip and reinforcement based on arguments
    skip_training = False
    if os.path.exists(model_load_path) and args.load:
        try:
            model.load_state_dict(torch.load(model_load_path, map_location=device))
            model.to(device)
            logger.log(logging.INFO, f"Loaded model checkpoint from {model_load_path}.")
            if args.reinforce:
                logger.log(logging.INFO, "Reinforce enabled. Continuing reinforcement training.")
                skip_training = False
            else:
                logger.log(logging.INFO, "Reinforce not enabled. Skipping reinforcement training and proceeding to evaluation.")
                skip_training = True
        except Exception as e:
            logger.log(logging.WARNING, f"Failed to load checkpoint: {e}. Starting training from scratch.")
            skip_training = False

    if args.reinforce and not args.load:
        logger.log(logging.WARNING, "Reinforce option enabled without loading a model. Please enable loading to use reinforce. Starting training from scratch without reinforcement.")

    # Load Inputs
    if args.load_inputs:
        try:
            with open(inputs_save_path, "rb") as f:
                sig_type, clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
                assertlog(sig_type == args.sig_type, f"Loaded signal type {sig_type} does not match specified signal type {args.sig_type}. Please check the input file or disable input loading.")
                scale_X, scale_Y = 1.0, 1.0
            logger.log(logging.INFO, f"Loaded training data and inputs from {inputs_save_path} for evaluation.")
        except Exception as e:
            logger.log(logging.ERROR, f"Failed to load inputs: {e}. Ensure the file exists and is a valid .pklv2 file. Exiting...")
            sys.exit()
    else:
        logger.log(logging.INFO, "No input loading specified. Will generate training data and inputs during execution for training and evaluation.")
        clean, distorted, baseline, X_train, Y_train, scale_X, scale_Y, clean_scaled, distorted_scaled = genSignals(args.sig_type, beta2, gamma, dt, n_steps, sld_win, L)

    # save inputs for accelerator use if specified
    if args.save_inputs and not args.load_inputs:
        pack = (args.sig_type, clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled)
        with open(inputs_save_path, "wb") as f:
            pickle.dump(pack, f)
        logger.log(logging.INFO, f"Generated training data and inputs saved to {inputs_save_path} for later evaluation.")

    # Training
    if not skip_training:
        # Train the complex PINN
        logger.log(logging.INFO, "Starting PINN training...")
        model, losses, train_time = train(model, device, X_train, Y_train, epochs=args.epochs, lr=args.lr, beta2=beta2, gamma=gamma, scale_factor=scale_X)
        logger.log(logging.INFO, f"PINN training completed in {train_time:.2f} seconds.")
        logger.log(logging.INFO, f"Final Training Loss: {losses[-1]:.6f} at {args.epochs} epochs.")
        
        # Save Checkpoint
        if args.checkpoint:
            torch.save(model.state_dict(), model_save_path)
            logger.log(logging.INFO, f"Model checkpoint saved at {model_save_path}.")

    # Evaluate and compare with baseline
    logger.log(logging.INFO, "Evaluating PINN performance against baseline...")
    eval_start = time.time()
    model.eval()
    with torch.no_grad():
        p_out = model(X_train.to(device)).cpu().numpy()
        pinn_recovered = (p_out[:, 0] + 1j * p_out[:, 1]) * scale_Y
    eval_end = time.time()
    logger.log(logging.INFO, f"Evaluation completed in {eval_end - eval_start:.6f} seconds.")

    # create inputs for accelerator
    if args.save_inputs and not args.load_inputs:
        with torch.no_grad():
            # Pass the float data ONLY through the first QuantIdentity layer
            quant_input_tensor = model.model[0](X_train.to(device))
            # Extract the raw integer values from the Brevitas QuantTensor and cast them to standard NumPy int8
            fpga_input_data = quant_input_tensor.int().cpu().numpy().astype(np.int8)
        logger.log(logging.INFO, "Hardware input shape is {} and type is {}.".format(fpga_input_data.shape, fpga_input_data.dtype))
        np.save(accelerator_inputs_path, fpga_input_data)
        logger.log(logging.INFO, "Generated quantized inputs for accelerator from the first QuantIdentity layer and saved to {}.".format(accelerator_inputs_path))

    if args.visual and not args.metrics:
        logger.log(logging.WARNING, "Visualization enabled without metrics calculation. Enable metrics to generate constellation comparison figure. Skipping...")

    if args.metrics:
        # Calculate EVM
        clean_sync_dist, dist_sync = synchronize_signals(clean, distorted)
        clean_sync_dbp, dbp_sync   = synchronize_signals(clean, baseline)
        clean_sync_pinn, pinn_sync = synchronize_signals(clean, pinn_recovered)

        clean_dist_n, dist_f = align_signal(clean_sync_dist, dist_sync)
        clean_dbp_n,  dbp_f  = align_signal(clean_sync_dbp, dbp_sync)
        clean_pinn_n, pinn_f = align_signal(clean_sync_pinn, pinn_sync)

        evm_dist = evm(clean_dist_n, dist_f)
        evm_ssfm = evm(clean_dbp_n, dbp_f)
        evm_pinn = evm(clean_pinn_n, pinn_f)
        logger.log(logging.INFO, f"EVM Summary - Distorted: {evm_dist:.2f}%, SSFM: {evm_ssfm:.2f}%, PINN: {evm_pinn:.2f}%")

        # Calculate SER
        _, clean_idx_dist = classify(clean_dist_n, args.sig_type)
        _, clean_idx_dbp  = classify(clean_dbp_n, args.sig_type)
        _, clean_idx_pinn = classify(clean_pinn_n, args.sig_type)

        _, dist_idx  = classify(dist_f, args.sig_type)
        _, ssfm_idx  = classify(dbp_f, args.sig_type)
        _, pinn_idx  = classify(pinn_f, args.sig_type)

        ser_dist = calculate_ser(clean_idx_dist, dist_idx)
        ser_ssfm = calculate_ser(clean_idx_dbp, ssfm_idx)
        ser_pinn = calculate_ser(clean_idx_pinn, pinn_idx)
        logger.log(logging.INFO, f"SER Summary - Distorted: {ser_dist:.2f}%, SSFM: {ser_ssfm:.2f}%, PINN: {ser_pinn:.2f}%")

        if args.visual:
            # visualize constellation diagrams
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sigs = [dist_f, dbp_f, pinn_f]
            titles = ["Distorted", "SSFM Baseline", "8-bit PINN"]
            for i in range(3):
                ax[i].scatter(sigs[i].real, sigs[i].imag, s=1, alpha=0.5)
                ax[i].set_title(titles[i]); ax[i].set_aspect('equal')
                ax[i].grid(visible=True, which='both', linestyle='--', linewidth=0.5)
                ax[i].set_xlim(-1.5, 1.5); ax[i].set_ylim(-1.5, 1.5)
            fig.suptitle(f"Constellation Diagrams for {args.sig_type} Signals", fontsize=16)
            plt.savefig(comparison_fig_path)
            logger.log(logging.INFO, f"Constellation comparison figure saved at {comparison_fig_path}.")

    # ONNX Export for FINN Deployment
    if args.onnx_export:
        logger.log(logging.INFO, f"Exporting QAT model to QONNX format at '{onnx_export_path}'...")
        try:
            model.eval()
            model_cpu = model.cpu()
            
            dummy_input = torch.randn(1, sld_win * 2, dtype=torch.float32)
            
            export_qonnx(model_cpu, dummy_input, onnx_export_path, dynamo=False, opset_version=11)
            logger.log(logging.INFO, f"Successfully exported QONNX model to {onnx_export_path}.")
            
        except Exception as e:
            logger.log(logging.ERROR, f"Failed to export QONNX model: {e}")
            sys.exit()
    else:
        logger.log(logging.INFO, "ONNX export not enabled. Skipping...")

    # ONNX Conversion to FINN Format
    if args.finn_convert and args.onnx_export:
        logger.log(logging.INFO, "Starting conversion of QONNX model to FINN format...")
        try:
            from qonnx2finn.qonnx2finn import conv2finn
            conv2finn(dir_path=results_dir, qonnx_name=args.onnx_path, output_name="model.onnx")
            logger.log(logging.INFO, "Successfully converted QONNX model to FINN format as model.onnx.")
        except Exception as e:
            logger.log(logging.ERROR, f"Failed to convert to FINN format: {e}")
            sys.exit()
    elif args.finn_convert and not args.onnx_export:
        logger.log(logging.WARNING, "FINN conversion requested without ONNX export. Please enable ONNX export to convert to FINN format. Skipping...")
    else:
        logger.log(logging.INFO, "FINN conversion not enabled. Skipping...")

    logger.log(logging.INFO, "Script execution completed successfully.")

if __name__ == "__main__":
    main()