# Script to run complex hardware accelerated PINN on PYNQ-ZU
# Last updated: 2026-03-14

# Note: This script is intended to be run on the PYNQ-ZU board itself, and assumes that
# the driver files (driver.py, driver_base.py) are present in the same directory. The 
# bitfiles should also be present at the expected location (../bitfile/finn-accel.bit) 
# or specified via --bitfile.

import argparse
import csv
import logging
import sys
from pathlib import Path
import os
import time

import numpy as np
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent

def save_results_csv(data, filename, output_dir=SCRIPT_DIR):
    """Save inference results as CSV next to this script."""
    filepath = output_dir / filename
    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        if isinstance(data, dict):
            writer.writerow(["Metric", "Value"])
            for key, value in data.items():
                writer.writerow([key, value])
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                writer.writerow([f"col_{j}" for j in range(data.shape[0])])
                writer.writerow(data.tolist())
            else:
                writer.writerow([f"col_{j}" for j in range(data.shape[-1])])
                for row in data.reshape(-1, data.shape[-1]):
                    writer.writerow(row.tolist())
    logging.info("Saved CSV to %s", filepath)

# attempt to import driver artifacts (they should be present next to this script)
try:
    from driver import io_shape_dict
    from driver_base import FINNExampleOverlay
except Exception as e:
    io_shape_dict = None
    FINNExampleOverlay = None
    _import_error = e

def main():
    parser = argparse.ArgumentParser(description="Run FINN overlay on PYNQ-ZU (or summarize inputs).")
    default_bit = (
        Path(__file__).resolve().parents[1]
        / "bitfile"
        / "finn-accel.bit"
    )
    parser.add_argument("--bitfile", default=str(default_bit), help='path to bitfile (default: generated finn-accel.bit)')
    parser.add_argument("--runtime-weight-dir", default="runtime_weights/", help="runtime weights folder")
    parser.add_argument("--test", action="store_true", default=False, help="run throughput_test() instead of executing inputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # check bitfile exists
    bitfile_path = Path(args.bitfile)
    if not bitfile_path.exists():
        logging.error("Bitfile not found at %s.", bitfile_path)
        sys.exit()
    else:
        logging.info("Using bitfile: %s", bitfile_path)

    # check drivers exist and imported
    if FINNExampleOverlay is None:
        logging.error("Failed to import driver modules. Ensure `driver.py` and `driver_base.py` are next to this script.")
        logging.debug("Import error: %s", _import_error)
        sys.exit()

    # initialize driver
    if args.test:
        batch_size_driver = 1
    else:
        ibufs = np.load("accelerator_inputs.npy")
        batch_size_driver = len(ibufs) # placeholder, implement logic here

    driver = FINNExampleOverlay(
            bitfile_name=str(bitfile_path),
            platform="zynq-iodma",
            io_shape_dict=io_shape_dict,
            batch_size = batch_size_driver,
            runtime_weight_dir=args.runtime_weight_dir,
        )
        
    # run throughput test if requested
    if args.test:
        logging.info("Running throughput test...")
        res = driver.throughput_test()
        logging.info("Throughput results: %s", res)
        if isinstance(res, dict):
            save_results_csv(res, "throughput_results.csv")
        return

    # create inputs for execution on board
    # ibufs = []
    # d_inp = np.array(ins, dtype=np.int8)
    # ibufs.append(d_inp)

    # execute on board
    exe_start = time.time()
    obuf = driver.execute(ibufs)
    exe_end = time.time()
    logging.info("Execution time: %.6f seconds", exe_end - exe_start)

    # save outputs to files next to inputs
    if not isinstance(obuf, list):
        obuf = [obuf]
    for i, out in enumerate(obuf):
        outp = f"output_{i}.npy"
        np.save(outp, out)
        logging.info("Saved output to %s", outp)
        save_results_csv(out, f"output_{i}.csv")


if __name__ == "__main__":
    main()


