# Script to run hardware accelerated network on PYNQ-ZU
# Last updated: 2026-02-27

# Note: This script is intended to be run on the PYNQ-ZU board itself, and assumes that the driver files (driver.py, driver_base.py) are present in the same directory.

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pypickle

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


def safe_load_pickle(pkl_path: Path):
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    try:
        with pkl_path.open("rb") as f:
            return pypickle.load(f)
    except Exception:
        # some pypickle versions accept a filename string directly
        try:
            return pypickle.load(str(pkl_path))
        except Exception as e:
            raise


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
    parser.add_argument("--pickle", default="ssfm_results.pkl", help="optional pickle file to load and inspect")
    parser.add_argument("--inputs", nargs="*", help="optional input .npy files to run on accelerator")
    parser.add_argument("--test", action="store_true", default=True, help="run throughput_test() instead of executing inputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    driver = FINNExampleOverlay(
        bitfile_name=str(bitfile_path),
        platform="zynq-iodma",
        io_shape_dict=io_shape_dict,
        batch_size=args.batchsize,
        runtime_weight_dir=args.runtime_weight_dir,
    )

    # optionally load and inspect pickle
    pkl_path = Path(args.pickle)
    if pkl_path.exists():
        try:
            data = safe_load_pickle(pkl_path)
            logging.info("Loaded pickle '%s' — type: %s", pkl_path, type(data))
            # if it's a tuple/list, print shapes of contained arrays
            if isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    try:
                        logging.info(" item %d: type=%s, shape=%s", i, type(item), getattr(item, "shape", None))
                    except Exception:
                        logging.info(" item %d: type=%s", i, type(item))
            elif isinstance(data, dict):
                for k, v in data.items():
                    logging.info(" key=%s: type=%s, shape=%s", k, type(v), getattr(v, "shape", None))
        except Exception as e:
            logging.error("Failed to load pickle %s: %s", pkl_path, e)
    else:
        logging.info("No pickle found at %s — skipping pickle load.", pkl_path)

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
        t = np.linspace(-T_max, T_max, N_t)
        z = np.zeros(t)
        ibufs.append([t, z])

    obuf = driver.execute(ibufs)
    # save outputs to files next to inputs
    if not isinstance(obuf, list):
        obuf = [obuf]
    for i, out in enumerate(obuf):
        outp = Path(args.input_npy[0]).parent / f"output_{i}.npy"
        np.save(outp, out)
        logging.info("Saved output to %s", outp)
        save_results_csv(out, f"output_{i}.csv")


if __name__ == "__main__":
    main()


