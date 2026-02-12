# Script to run hardware accelerated network on PYNQ-ZU
# Last updated: 2026-02-12

# Note: This script is intended to be run on the PYNQ-ZU board itself, and assumes that the driver files (driver.py, driver_base.py) are present in the same directory.

import numpy as np
import pypickle
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

bitfile = "resizer.bit" # birfile directory
bsize = 1 # batch size for inference (int)

driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform="zynq-iodma",
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
        runtime_weight_dir="runtime_weights/",
    )

t, A_L, z_points, A_true = pypickle.load("ssfm_results.pkl")

