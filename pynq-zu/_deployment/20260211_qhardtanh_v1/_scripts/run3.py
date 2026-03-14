# Script to run hardware accelerated network on PYNQ-ZU
# Last updated: 2026-03-04

# Note: This script is intended to be run on the PYNQ-ZU board itself, and assumes that the driver files (driver.py, driver_base.py) are present in the same directory.

import argparse
import csv
import logging
import sys
from pathlib import Path
import os

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
    parser.add_argument("--batchsize", type=int, default=1, help="batch size for inference")
    parser.add_argument("--runtime-weight-dir", default="runtime_weights/", help="runtime weights folder")
    parser.add_argument("--inputs", nargs="*", help="optional input .npy files to run on accelerator")
    parser.add_argument("--test", action="store_true", default=False, help="run throughput_test() instead of executing inputs")
    parser.add_argument("--defs", type=str, default="z=0", help="z=0 or z=L only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # check defs:
    defs_permitted = {"z=0", "z=L"}
    assert args.defs in defs_permitted, "--defs arg value not accepted."

    bitfile_path = Path(args.bitfile)
    if not bitfile_path.exists():
        logging.warning("Bitfile not found at %s (placeholder 'resizer.bit' may have been used).", bitfile_path)
    else:
        logging.info("Using bitfile: %s", bitfile_path)

    if FINNExampleOverlay is None:
        logging.error("Failed to import driver modules. Ensure `driver.py` and `driver_base.py` are next to this script.")
        logging.debug("Import error: %s", _import_error)
        sys.exit(2)

    # instantiate overlay
    if args.inputs or args.test:
        batch_size_driver = args.batchsize
    elif args.defs == "z=0":
        batch_size_driver = int(255)
    elif args.defs == "z=L":
        batch_size_driver = int(101)
        assert os.path.isfile("ssfm_results.pkl"), "ssfm_results.pkl is missing, cannot continue with instantiation."
    else:
        logging.error("Failed to define batch size for overlay instantiation.")
        sys.exit()
    
    driver = FINNExampleOverlay(
            bitfile_name=str(bitfile_path),
            platform="zynq-iodma",
            io_shape_dict=io_shape_dict,
            # batch_size=args.batchsize,
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

    # execute inputs on board
    ibufs = []
    # if inputs provided
    if args.inputs:
        for fn in args.input_npy:
            p = Path(fn)
            if not p.exists():
                logging.error("Input file not found: %s", p)
                sys.exit(3)
            ibufs.append(np.load(p))
        logging.info("Executing overlay on %d input files...", len(ibufs))
    # run defaults if inputs not provided
    else:
        # logging.info("No .npy inputs provided and --throughput-test not set. Exiting after initialization.")
        logging.info("No inputs provided, executing with default inputs.")
        
        ins = []
    
        # z = 0
        if args.defs == "z=0":
            logging.info("Executing defaults at z=0.")
            t = np.linspace(-127, 127, 255, dtype=np.int8)
            for num in t:
                ins.append([0, num])
            
        # z = L
        elif args.defs == "z=L":
            logging.info("Executing defaults at z=L.")
            t = np.linspace(-127, 127, 101, dtype=np.int8)
            with open("ssfm_results.pkl", "rb") as f:
                sd = pickle.load(f)
            (time, A_L, z_points, A_true) = sd
            z_points = np.round(z_points*127).astype(np.int8)
            for i in range(len(t)):
                ins.append([z_points[i], t[i]])
        
        d_inp = np.array(ins, dtype=np.int8)
        ibufs.append(d_inp)

    # execute on board
    obuf = driver.execute(ibufs)

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


