# Script to benchmark ComplexPINN when evaluating on CPU/GPU as compared to FPGA accelerator
# Last Updated: 22 Mar 2026

import os
import sys
import argparse
import logging

# Set up logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()

log_file_format = logging.Formatter('%(asctime)s, %(msecs)03d %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('benchmark.log', mode='a') # appends to log file each run (a append w overwrite)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_file_format)

console_file_format = logging.Formatter('%(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_file_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.log(logging.INFO, "Log saved as benchmark.log in current working directory.")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias
from pytorch_benchmark import benchmark

from createPINN import complexPINN, assertlog

def main():
    parser = argparse.ArgumentParser(description='Benchmark ComplexPINN on CPU/GPU vs FPGA accelerator')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to benchmark on (default: cpu)')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of runs for benchmarking (default: 100)')
    parser.add_argument('--dir', type=str, default='results', help='Directory to save benchmark results (default: results)')
    parser.add_argument('--load_file', type=str, default='complex_pinn_checkpoint.pth', help='Path to load pre-trained model weights (default: complex_pinn_checkpoint.pth)')
    parser.add_argument('--inputs_file', type=str, default='generated_inputs.pklv2', help='Path to load sample input data (default: generated_inputs.pklv2)')

    args = parser.parse_args()

    assertlog(args.num_runs > 0, "Number of runs must be a positive integer.")
    assertlog(os.path.isfile("cwd.check"), "cwd.check file not found. Ensure you are running this script from the project root directory.")

    # Sanity check for arguments
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.log(logging.WARNING, "CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    model_load_path = os.path.join(os.getcwd(), args.dir, args.load_file)
    inputs_save_path = os.path.join(os.getcwd(), args.dir, args.inputs_file)

    assertlog(os.path.isfile(model_load_path), f"Model checkpoint file not found at {model_load_path}. Ensure the file exists and is a valid .pth file.")
    assertlog(os.path.isfile(inputs_save_path), f"Inputs file not found at {inputs_save_path}. Ensure the file exists and is a valid .pkl file.")

    # Set up model and device
    model = complexPINN(window_size=25, hlayers=3, hidden_dim=64).to(device)
    logger.log(logging.INFO, f"Torch using {device} for benchmarking.")

    # Load pre-trained model weights
    if os.path.exists(model_load_path):
        try:
            model.load_state_dict(torch.load(model_load_path, map_location=device))
            model.to(device)
            logger.log(logging.INFO, f"Loaded model checkpoint from {model_load_path}.")
        except Exception as e:
            logger.log(logging.WARNING, f"Failed to load checkpoint: {e}. Exiting...")
            sys.exit()
    else:
        logger.log(logging.WARNING, f"Checkpoint file not found at {model_load_path}. Exiting...")
        sys.exit()
    
    # Load sample input data for benchmarking
    try:
        with open(inputs_save_path, "rb") as f:
            if inputs_save_path.endswith('.pklv2'): # new version includes signal type for validation
                sig_type, clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
                logger.log(logging.INFO, f"Loaded signal type {sig_type} from {inputs_save_path} for evaluation.")
            else:
                clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
                logger.log(logging.INFO, f"Loaded training data and inputs from {inputs_save_path} for evaluation.")
    except Exception as e:
        logger.log(logging.ERROR, f"Failed to load inputs: {e}. Ensure the file exists and is a valid .pkl file. Exiting...")
        sys.exit()

    # Benchmark model inference on specified device
    model.eval()
    results = benchmark(model, (X_train.to(device)), num_runs=args.num_runs)
    logger.log(logging.INFO, f"Benchmark results on {device}: {results}")

    logger.log(logging.INFO, "Benchmarking completed. Results saved.")

if __name__ == "__main__":
    main()