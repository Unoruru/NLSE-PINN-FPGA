# Script to create input .npy files for the complex PINN example
# Last updated: 2026-03-14

# Note: Assumes training script in the same directory, and that the checkpoint file is present.

import os
import argparse
import numpy as np
import pickle
import torch

from pinn_complex import complexPINN, windowing, ssfm_nlse

def main():
    parser = argparse.ArgumentParser(description="Create input .npy files for complex PINN example.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="String. Path to model checkpoint. Default checkpoint.pth.")
    parser.add_argument("--sld_win", type=int, default=25, help="Integer. Sliding window size. Default 25.")
    parser.add_argument("--hlayers", type=int, default=3, help="Integer. Number of hidden layers. Default 3.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Integer. Hidden dimension size. Default 64.")

    args = parser.parse_args()

    # checks
    assert os.path.isfile(args.checkpoint), f"Checkpoint file {args.checkpoint} not found."
    assert os.path.isfile("pinn_complex.py"), "Training script (pinn_complex.py) not found in current directory."
    device = "cpu"

    # fixed params
    beta2   = -21e-27  # second-order dispersion param (ps^2/m)
    gamma   = 0.015    # Kerr nonlinear coefficient (1/W/km)
    dt      = 1e-12    # 1ps
    n_steps = 50
    L = 40             # fiber length in km

    # load model checkpoint
    model = complexPINN(window_size=args.sld_win, hlayers=args.hlayers, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # create input data for model and accelerator
    # Generate 16-QAM
    points = np.array([-3, -1, 1, 3])
    re, im = np.meshgrid(points, points)
    const = (re + 1j*im).flatten()
    const /= np.sqrt(np.mean(np.abs(const)**2))

    clean = const[np.random.randint(0, 16, 6000)]
    distorted = ssfm_nlse(clean, L, "forward", beta2, gamma, dt, n_steps) 
    baseline = ssfm_nlse(distorted, L, "reverse", beta2, gamma, dt, n_steps) # ssfm baseline for comparison
    scale_X = np.max(np.abs(distorted))
    distorted_scaled = distorted / scale_X
    scale_Y = np.max(np.abs(clean)) 
    clean_scaled = clean / scale_Y 
    X_train = windowing(distorted_scaled, args.sld_win)
    Y_train = torch.tensor(np.stack([np.real(clean_scaled), np.imag(clean_scaled)], axis=1), dtype=torch.float32)

    # save inputs as .npy files
    pack = (clean, distorted, baseline, X_train, Y_train, scale_X, scale_Y, clean_scaled, distorted_scaled)
    with open("generated_inputs.pkl", "wb") as f:
        pickle.dump(pack, f)
    print("Generated and saved input data to generated_inputs.pkl")

    # create inputs for accelerator
    with torch.no_grad():
        # Pass the float data ONLY through the first QuantIdentity layer
        quant_input_tensor = model.model[0](X_train.to(device))
        
        # Extract the raw integer values from the Brevitas QuantTensor
        # and cast them to standard NumPy int8
        fpga_input_data = quant_input_tensor.int().cpu().numpy().astype(np.int8)
    
    # 2. Verify the shape and type match the io_shape_dict
    print(f"Hardware Input Shape: {fpga_input_data.shape}") # Should be (N, 50)
    print(f"Hardware Input Type: {fpga_input_data.dtype}")  # Should be int8

    # 3. Save it to a file you can move to the PYNQ board
    np.save("fpga_test_input.npy", fpga_input_data)
    print("Saved fpga_test_input.npy for accelerator input.")

if __name__ == "__main__":
    main()