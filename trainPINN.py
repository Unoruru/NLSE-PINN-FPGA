# Script to run the complex PINN training and evaluation process for reinforcement learning.
# Supports 4 signal types: 16-QAM, 16-APSK, 16-PSK, and Star-QAM.
# Last Updated: 20 Mar 2026

import os
import argparse
import logging
import sys
import time

# Set up logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()

log_file_format = logging.Formatter('%(asctime)s, %(msecs)03d %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('pinn_complex.log', mode='a') # appends to log file each run (a append w overwrite)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_file_format)

console_file_format = logging.Formatter('%(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_file_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

with open("training_pinn.log", "w") as log_file: # Clear log file at the start of each run
    log_file.write("")

logger.log(logging.INFO, "Log saved as training_pinn.log in current working directory.")

from createPINN import assertlog
from consolidate.helper import str2bool
from consolidate.trainEval import load_get, write_save, plot_perf

def main():
    parser = argparse.ArgumentParser(description="Run Complex PINN Training and Evaluation with Reinforcement Learning.")
    parser.add_argument("--loop", type=int, default=15, help="Integer. Number of reinforcement learning iterations to perform. Default 15.")
    parser.add_argument("--epochs", type=int, default=350, help="Integer. Number of epochs for each training iteration (post-first loop). Default 350.")
    parser.add_argument("--script_path", type=str, default="createPINN.py", help="String. Path to the complex PINN training script. Default 'createPINN.py'.")
    parser.add_argument("--no_initial", type=str2bool, default=False, help="Boolean. Skip initial training and start directly with reinforcement learning iterations. Expects load checkpoint at results/complex_pinn_checkpoint.pth. Default False.")
    parser.add_argument("--sig_type", type=str, default="16qam", help="String. Type of signal to train on. Supported types: '16qam', '16apsk', '16psk', 'star'. Default '16qam'.")
    args = parser.parse_args()

    # sanity checks for arguments
    assertlog(args.loop > 0, "Loop count must be a positive integer.")
    assertlog(args.epochs > 0, "Epoch count must be a positive integer.")
    assertlog(os.path.isfile(args.script_path), f"Script path '{args.script_path}' does not exist or is not a file.")
    assertlog(args.sig_type in ["16qam", "16apsk", "16psk", "star"], "Unsupported signal type specified. Supported types: '16qam', '16apsk', '16psk', 'star'.")

    metrics_path = os.path.join("results", "training_perf_metrics.pklv2")
    
    startTime = time.time()
    # run initial training
    if args.no_initial:
        logger.log(logging.INFO, "Skipping initial training as per argument. Starting directly with reinforcement learning iterations. Ensure checkpoints are available for loading.")
    else:
        logger.log(logging.INFO, "Starting initial training to generate PINN.")
        os.system("python " + args.script_path + " --sig_type " + args.sig_type + " --onnx_export False --finn_convert False --visual False")
    
    sig_type, losses, accuracies = load_get(metrics_path)

    # run reinforcement training and evaluation using checkpoint from training
    logger.log(logging.INFO, f"Starting Reinforcement training and evaluation with reinforcement learning for {args.loop} iterations and {args.epochs} epochs each.")
    for i in range(args.loop - 1):
        logger.log(logging.INFO, f"Reinforcement Learning Iteration {i+1}/{args.loop}")
        os.system("python " + args.script_path + " --sig_type " + args.sig_type + " --load True --reinforce True --onnx_export False --finn_convert False --visual False --epochs {}".format(args.epochs))
        cur_sig_type, cur_losses, cur_accuracies = load_get(metrics_path)
        losses += cur_losses
        accuracies += cur_accuracies

    logger.log(logging.INFO, f"Final Reinforcement Learning Iteration {args.loop}/{args.loop}")
    os.system("python " + args.script_path + " --sig_type " + args.sig_type + " --load True --reinforce True --visual False --epochs {}".format(args.epochs))
    cur_sig_type, cur_losses, cur_accuracies = load_get(metrics_path)
    losses += cur_losses
    accuracies += cur_accuracies
    logger.log(logging.INFO, "Reinforcement learning iterations completed.")

    # Final evaluation with metrics and visualization for new random input
    logger.log(logging.INFO, "Final evaluation with new random input for metrics and visualization. Saving generated inputs for accelerator testing.")
    os.system("python " + args.script_path + " --sig_type " + args.sig_type + " --load True --save_inputs True --onnx_export False --finn_convert False")

    # Create training performance plots and save metrics
    write_save(metrics_path, args.sig_type, losses, accuracies)
    plot_perf(metrics_path)
    logger.log(logging.INFO, f"Training performance metrics saved and plots generated for signal type {args.sig_type}.")

    endTime = time.time()
    logger.log(logging.INFO, f"Total execution time: {endTime - startTime:.2f} seconds.")

if __name__ == "__main__":
    main()