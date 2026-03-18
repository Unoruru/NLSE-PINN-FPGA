# Script for training 16-PSK PINN for multi-signal-type prediction
# Last Updated: 16 Mar 2026

# See readme.md for detailed instructions on running this script, including environment setup and dependencies.

import os
import sys
import logging

# Clear console for cleaner output
# os.system('cls' if os.name == 'nt' else 'clear')

# Set up logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()

log_file_format = logging.Formatter('%(asctime)s, %(msecs)03d %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('pinn_psk.log', mode='a') # appends to log file each run (a append w overwrite)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_file_format)

console_file_format = logging.Formatter('%(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_file_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.log(logging.INFO, "Log saved as pinn_psk.log in current working directory.")

# Check if current working directory is correct (should be project root)
check_path = os.path.join(os.getcwd(), "PSK", "pinn_psk.py")
if not os.path.isfile(check_path):
    logger.log(logging.ERROR, f"Current working directory is not project root. Expected to find 'PSK/pinn_psk.py' at {check_path}. Please change to project root and rerun.")
    sys.exit()
else:
    logger.log(logging.INFO, f"Current working directory verified as project root: {os.getcwd()}.")

# check results directory exists
results_dir = os.path.join(os.getcwd(), "PSK", "results")
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
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
import pickle

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias
from brevitas.export import export_qonnx

logger.log(logging.INFO, "Script started for training 16-PSK PINN.")

# Custom assertion function that logs errors before exiting
def assertlog(condition, message):
    try:
        assert condition, message
    except AssertionError as e:
        logger.log(logging.ERROR, f"Assertion Error: {e}")
        sys.exit()

# Define complex PINN architecture
class complexPINN(nn.Module):
    def __init__(self, window_size=25, hlayers=3, hidden_dim=64, bit_width=8, act_bit_width=8):
        super().__init__()

        layers = []

        # Input Layer (Window -> Hidden)
        layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
        layers.append(qnn.QuantLinear(window_size * 2, hidden_dim, bias=True, weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias))
        layers.append(qnn.QuantHardTanh(bit_width=act_bit_width, max_val=1.0, min_val=-1.0, act_quant=Int8ActPerTensorFloat))

        # Hidden Layers (default 3)
        for _ in range(hlayers):
            layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
            layers.append(qnn.QuantLinear(hidden_dim, hidden_dim, bias=True, weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias))
            layers.append(qnn.QuantHardTanh(bit_width=act_bit_width, max_val=1.0, min_val=-1.0, act_quant=Int8ActPerTensorFloat))

        # Output Layer (Hidden -> 2 for Re/Im)
        layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
        layers.append(qnn.QuantLinear(hidden_dim, 2, bias=True, weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# SSFM for generating training data and reverse physics for baseline comparison
def ssfm_nlse(A_in, L, direction="forward", beta2=-21e-27, gamma=0.015, dt=1e-12, n_steps=50):
    # check direction validity
    assertlog(direction in ["forward", "reverse"], "Direction must be 'forward' or 'reverse'.")

    # reverse physics for baseline comparison
    if direction == "reverse":
        gamma, beta2 = -gamma, -beta2

    omega = 2 * np.pi * np.fft.fftfreq(len(A_in), d=dt)
    dz = L / n_steps
    H = np.exp(-1j * (beta2 / 2) * (omega**2) * dz)

    A = A_in.copy()
    for _ in range(n_steps):
        A *= np.exp(1j * gamma * np.abs(A)**2 * dz)  # Non-linear
        A = np.fft.ifft(np.fft.fft(A) * H)           # Linear

    return A

# Windowing for training and dataset preparation
def windowing(sig, win_size):
    half = win_size // 2
    p = np.pad(sig, (half, half), mode='reflect')
    X = []
    for i in range(len(sig)):
        w = p[i : i + win_size]
        X.append(np.concatenate([np.real(w), np.imag(w)]))
    return torch.tensor(np.array(X), dtype=torch.float32)

# Physics loss computation using the NLSE residuals
def compute_physics_loss(model, x_window, beta2, gamma, scale_factor):
    x_window.requires_grad_(True)

    # 1. Surrogate Forward Pass (Smooth Tanh)
    current_val = x_window

    # We iterate directly over the Sequential blocks in model.model
    for layer in model.model:
        # Pass through Identity (gives input scale) and Linear (does the math)
        if isinstance(layer, (qnn.QuantIdentity, qnn.QuantLinear)):
            current_val = layer(current_val)

        elif isinstance(layer, qnn.QuantHardTanh):
            # Unwrap the Brevitas QuantTensor to a raw PyTorch tensor
            if hasattr(current_val, 'value'):
                raw_tensor = current_val.value
            else:
                raw_tensor = current_val

            # Use Tanh for smooth second derivatives during physics check
            current_val = torch.tanh(raw_tensor)

    # Final unwrap just in case the last layer returned a QuantTensor
    if hasattr(current_val, 'value'):
        current_val = current_val.value

    u = current_val[:, 0]
    v = current_val[:, 1]

    # 2. Gradients for NLSE
    u_t = torch.autograd.grad(u, x_window, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, x_window, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    u_tt = torch.autograd.grad(u_t, x_window, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    v_tt = torch.autograd.grad(v_t, x_window, grad_outputs=torch.ones_like(v_t), create_graph=True)[0]

    # 3. Residuals
    power = (u**2 + v**2) * (scale_factor ** 2)
    u_tt_sum = u_tt.sum(dim=1)
    v_tt_sum = v_tt.sum(dim=1)

    res_real = 0.5 * beta2 * v_tt_sum + gamma * v * power
    res_imag = -0.5 * beta2 * u_tt_sum - gamma * u * power

    return torch.mean(res_real**2 + res_imag**2)

# Training loop for the 16-PSK PINN
def train(model, device, X_train, Y_train, epochs, lr, beta2, gamma, scale_factor):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_start = time.time()

    with trange(epochs, desc="Training 16-PSK PINN") as t:
        for e in t:
            model.train()
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_train.to(device)), Y_train.to(device))
            physics_loss = compute_physics_loss(model, X_train.to(device), beta2, gamma, scale_factor=scale_factor)
            loss = loss + 0.01 * physics_loss
            loss.backward()
            optimizer.step()
            t.set_postfix({"Loss": f"{loss.item():.6f}"})

    train_end = time.time()
    logger.log(logging.INFO, f"PINN training completed in {train_end - train_start:.2f} seconds.")
    logger.log(logging.INFO, f"Final Training Loss: {loss.item():.6f} at {epochs} epochs.")

    return model

# Helper functions
def to_unit_power(sig):
    rms = np.sqrt(np.mean(np.abs(sig)**2))
    return sig / rms if rms > 0 else sig

def align_signal(ref, target):
    """Normalizes power and rotates the grid to align with reference."""
    ref_n = to_unit_power(ref)
    target_n = to_unit_power(target)
    # Find optimal rotation angle
    angle = np.angle(np.mean(target_n * np.conj(ref_n)))
    target_aligned = target_n * np.exp(-1j * angle)
    return ref_n, target_aligned

def evm(ref, target):
    return np.sqrt(np.mean(np.abs(ref - target)**2) / np.mean(np.abs(ref)**2)) * 100

def classify_16psk(signal):
    """Maps continuous complex signals to the nearest discrete 16-PSK points."""
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    ideal_points = np.exp(1j * angles)
    ideal_points /= np.sqrt(np.mean(np.abs(ideal_points)**2))
    distances = np.abs(signal[:, np.newaxis] - ideal_points)
    closest_indices = np.argmin(distances, axis=1)
    return ideal_points[closest_indices], closest_indices

def calculate_ser(clean_indices, recovered_indices):
    """Calculates the percentage of incorrectly identified symbols."""
    errors = np.sum(clean_indices != recovered_indices)
    return (errors / len(clean_indices)) * 100

def synchronize_signals(ref, target):
    """
    Finds the time delay between two signals and aligns them.
    This is critical when comparing windowed neural network outputs.
    """
    # 1. Cross-correlation to find the delay
    correlation = np.correlate(np.abs(target), np.abs(ref), mode='full')
    delay = np.argmax(correlation) - (len(ref) - 1)

    # 2. Shift the signals to match
    if delay > 0:
        # Target is delayed
        ref_sync = ref[:-delay]
        target_sync = target[delay:]
    elif delay < 0:
        # Target is advanced (rare, but possible with padding)
        ref_sync = ref[-delay:]
        target_sync = target[:delay]
    else:
        ref_sync, target_sync = ref, target

    return ref_sync, target_sync

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="16-PSK PINN for Optical Fibre Communication.")
    parser.add_argument("--load", type=str2bool, default=False, help="Boolean. Load existing model checkpoint if available. Default False.")
    parser.add_argument("--load_path", type=str, default="psk_pinn_checkpoint.pth", help="String. Path to model checkpoint for loading. Must be within the results directory. Default 'complex_pinn_checkpoint.pth'.")
    parser.add_argument("--reinforce", type=str2bool, default=False, help="Boolean. Reinforce physics during training with additional loss term. Requires load. Default False.")
    parser.add_argument("--metrics", type=str2bool, default=True, help="Boolean.Calculate and log EVM and SER metrics after training. Default True.")
    parser.add_argument("--visual", type=str2bool, default=True, help="Boolean.Generate and save constellation comparison figure after evaluation. Requires metrics. Default True.")
    parser.add_argument("--checkpoint", type=str2bool, default=True, help="Boolean. Save model checkpoint after training. Default True.")
    parser.add_argument("--checkpoint_path", type=str, default="psk_pinn_checkpoint.pth", help="String. Path to save model checkpoint. Must be within the results directory. Default 'complex_pinn_checkpoint.pth'.")
    parser.add_argument("--epochs", type=int, default=3000, help="Integer. Number of training epochs. Default 3000.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Float. Learning rate for training. Default 5e-4.")
    parser.add_argument("--save_inputs", type=str2bool, default=False, help="Boolean. Save generated training data and inputs as .pkl/.npy files for accelerator use. Default False.")
    parser.add_argument("--load_inputs", type=str2bool, default=False, help="Boolean. Load generated training data and inputs from .pkl/.npy files for accelerator use. Requires checkpoint and .pkl input file. Default False.")
    parser.add_argument("--onnx_export", type=str2bool, default=True, help="Boolean. Export trained model to ONNX format for FPGA deployment. Default True.")
    parser.add_argument("--onnx_path", type=str, default="psk_raw.onnx", help="String. Path to save ONNX model. Must be within the results directory. Default 'psk_raw.onnx'.")
    parser.add_argument("--finn_convert", type=str2bool, default=True, help="Boolean. Convert exported ONNX model to FINN format using qonnx2finn. Requires onnx_export. Default True.")

    args = parser.parse_args()

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
    comparison_fig_path = os.path.join(results_dir, "constellation_comparison.png")

    inputs_save_path = os.path.join(results_dir, "generated_inputs.pkl")
    accelerator_inputs_path = os.path.join(results_dir, "accelerator_inputs.npy")

    # check onnx export path validity
    assertlog(args.onnx_path.endswith(".onnx"), "ONNX export path must end with .onnx extension.")
    assertlog(args.onnx_path != "psk_FINN_ready.onnx", "ONNX export path cannot be 'psk_FINN_ready.onnx' to avoid overwriting FINN conversion output. Please specify a different name.")

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

    if args.load_inputs:
        try:
            with open(inputs_save_path, "rb") as f:
                clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
            logger.log(logging.INFO, f"Loaded training data and inputs from {inputs_save_path} for evaluation.")
        except Exception as e:
            logger.log(logging.ERROR, f"Failed to load inputs: {e}. Ensure the file exists and is a valid .pkl file. Exiting...")
            sys.exit()
    else:
        logger.log(logging.INFO, "No input loading specified. Will generate training data and inputs during execution for training and evaluation.")
        # Generate 16-PSK (uniform ring)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        const = np.exp(1j * angles)
        const /= np.sqrt(np.mean(np.abs(const)**2))  # already unit power

        clean = const[np.random.randint(0, len(const), 6000)]
        clean = clean / np.sqrt(np.mean(np.abs(clean)**2))

        distorted = ssfm_nlse(clean, L, "forward", beta2, gamma, dt, n_steps)
        logger.log(logging.INFO, "Generated training data using SSFM with forward physics.")
        baseline = ssfm_nlse(distorted, L, "reverse", beta2, gamma, dt, n_steps) # ssfm baseline for comparison
        logger.log(logging.INFO, "Generated baseline recovery using SSFM with reverse physics.")

        scale_X = 1.0
        scale_Y = 1.0

        distorted_scaled = distorted / scale_X
        clean_scaled = clean / scale_Y

        X_train = windowing(distorted_scaled, sld_win)
        Y_train = torch.tensor(np.stack([np.real(clean_scaled), np.imag(clean_scaled)], axis=1), dtype=torch.float32)

    # save inputs for accelerator use if specified
    if args.save_inputs and not args.load_inputs:
        pack = (clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled)
        with open(inputs_save_path, "wb") as f:
            pickle.dump(pack, f)
        logger.log(logging.INFO, f"Generated training data and inputs saved to {inputs_save_path} for later evaluation.")

    if not skip_training:
        # Train the 16-PSK PINN
        logger.log(logging.INFO, "Starting PINN training...")
        model = train(model, device, X_train, Y_train, epochs=args.epochs, lr=args.lr, beta2=beta2, gamma=gamma, scale_factor=scale_X)
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
        _, clean_idx_dist = classify_16psk(clean_dist_n)
        _, clean_idx_dbp  = classify_16psk(clean_dbp_n)
        _, clean_idx_pinn = classify_16psk(clean_pinn_n)

        _, dist_idx  = classify_16psk(dist_f)
        _, ssfm_idx  = classify_16psk(dbp_f)
        _, pinn_idx  = classify_16psk(pinn_f)

        ser_dist = calculate_ser(clean_idx_dist, dist_idx)
        ser_ssfm = calculate_ser(clean_idx_dbp, ssfm_idx)
        ser_pinn = calculate_ser(clean_idx_pinn, pinn_idx)
        logger.log(logging.INFO, f"SER Summary - Distorted: {ser_dist:.2f}%, SSFM: {ser_ssfm:.2f}%, PINN: {ser_pinn:.2f}%")

        if args.visual:
            # visualize constellation diagrams
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sigs = [dist_f, dbp_f, pinn_f]
            titles = ["Distorted (16-PSK)", "SSFM Baseline", "8-bit PINN"]
            for i in range(3):
                ax[i].scatter(sigs[i].real, sigs[i].imag, s=1, alpha=0.5)
                ax[i].set_title(titles[i]); ax[i].set_aspect('equal')
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
            conv2finn(dir_path=results_dir, qonnx_name=args.onnx_path, output_name="psk_FINN_ready.onnx")
            logger.log(logging.INFO, "Successfully converted QONNX model to FINN format as psk_FINN_ready.onnx.")
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
