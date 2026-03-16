# Script to run the 16-APSK PINN training and evaluation process for reinforcement learning.
# Last Updated: 16 Mar 2026

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
file_handler = logging.FileHandler('pinn_apsk.log', mode='a') # appends to log file each run (a append w overwrite)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_file_format)

console_file_format = logging.Formatter('%(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_file_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

with open("pinn_apsk.log", "w") as log_file: # Clear log file at the start of each run
    log_file.write("")

logger.log(logging.INFO, "Log saved as pinn_apsk.log in current working directory.")

from pinn_apsk import assertlog, str2bool

def main():
    parser = argparse.ArgumentParser(description="Run 16-APSK PINN Training and Evaluation with Reinforcement Learning.")
    parser.add_argument("--loop", type=int, default=15, help="Integer. Number of reinforcement learning iterations to perform. Default 15.")
    parser.add_argument("--epochs", type=int, default=250, help="Integer. Number of epochs for each training iteration (post-first loop). Default 250.")
    parser.add_argument("--script_path", type=str, default="APSK/pinn_apsk.py", help="String. Path to the 16-APSK PINN training script. Default 'APSK/pinn_apsk.py'.")
    parser.add_argument("--no_initial", type=str2bool, default=False, help="Boolean. Skip initial training and start directly with reinforcement learning iterations. Expects load checkpoint at results/complex_pinn_checkpoint.pth. Default False.")
    args = parser.parse_args()

    # sanity checks for arguments
    assertlog(args.loop > 0, "Loop count must be a positive integer.")
    assertlog(args.epochs > 0, "Epoch count must be a positive integer.")
    assertlog(os.path.isfile(args.script_path), f"Script path '{args.script_path}' does not exist or is not a file.")

    startTime = time.time()
    # run initial training
    if args.no_initial:
        logger.log(logging.INFO, "Skipping initial training as per argument. Starting directly with reinforcement learning iterations. Ensure checkpoints are available for loading.")
    else:
        logger.log(logging.INFO, "Starting initial training to generate PINN.")
        os.system("python " + args.script_path + " --onnx_export False --finn_convert False --visual False")

    # run reinforcement training and evaluation using checkpoint from training
    logger.log(logging.INFO, f"Starting Reinforcement training and evaluation with reinforcement learning for {args.loop} iterations and {args.epochs} epochs each.")
    for i in range(args.loop - 1):
        logger.log(logging.INFO, f"Reinforcement Learning Iteration {i+1}/{args.loop}")
        os.system("python " + args.script_path + " --load True --reinforce True --onnx_export False --finn_convert False --visual False --epochs {}".format(args.epochs))

    logger.log(logging.INFO, f"Final Reinforcement Learning Iteration {args.loop}/{args.loop}")
    os.system("python " + args.script_path + " --load True --reinforce True --finn_convert False --visual False --epochs {}".format(args.epochs))
    logger.log(logging.INFO, "Reinforcement learning iterations completed.")

    # Final evaluation with metrics and visualization for new random input
    logger.log(logging.INFO, "Final evaluation with new random input for metrics and visualization.")
    os.system("python " + args.script_path + " --load True")

    endTime = time.time()
    logger.log(logging.INFO, f"Total execution time: {endTime - startTime:.2f} seconds.")

if __name__ == "__main__":
    main()
