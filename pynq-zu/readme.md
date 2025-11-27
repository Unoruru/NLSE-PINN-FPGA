# Setup FINN Framework for Pynq-ZU

> [!IMPORTANT]
> This guide assumes the use of Ubuntu 24.04.3 LTS and Vivado 2022.2. Please note that other versions of the operating system/software has not been tested/verified. Linux is a requirement to run the FINN framework.

 1. Follow setup instruction (1) in the quickstart [here](https://finn.readthedocs.io/en/latest/getting_started.html) to setup docker. Ensure that docker is working and can run without root. 
 2. Download the `v1: FINN Framework for Pynq-ZU Board Latest` release from the repo. The release can be found [here](https://github.com/Anthony062966/PINN_FPGA_Project/releases/tag/Pynq-ZU). This will be the modified FINN framework, used in lieu of the original FINN framework. Please note that there is **no need** to clone the original repo. Extract the framework into a folder. The folder containing the framework should be named `finn`. This folder will henceforth be referred to as `$finn_path`.
 3. Download the `sample` folder in this repo. This will be the folder containing the quantised neural network for later processing. This folder will henceforth be referred to as `$model_path`. **Do not** place this folder within `$finn_path`.
 4. In `$finn_path`, open `en_var.sh` and alter the path and version number of vivado to target the locally installed path and version. The default board is the Pynq-ZU development board.
 5. Launch the locally installed copy of Vivado. Ensure that the Pynq-ZU board files are downloaded. Once confirmed, Vivado can be safely closed.
 6. Open a terminal window within `$finn_path`. Run `source en_var.sh` to setup the required environmental variables in the local terminal window.
 7. Run `./run-docker.sh quicktest` to verify that both docker and the FINN framework is working properly.

 > [!NOTE]
 > The `en_var.sh` folder only sets the required environmental variables for the current local session terminal window. Once the terminal window is killed, or when operating within a different terminal window, you must run the script again to setup the enviornmental variables. Running the framework without the correct environmental variables can lead to irreparable damage to the modifications on the original FINN framework, which will break compatibility with the Pynq-ZU FPGA development board.

# Running FINN Framework for Pynq-ZU

> [!NOTE]
> This part assumes that a ONNX file has been created for the quantised neural network.

 1. [Required] Place the quantised neural network file (ONNX format) within `$model_path`. Rename the file as `model.onnx`.
 2. [Required] Adjust the parameters within the `$model_path/dataflow_build_config.json` file to suit your build needs. Remove references where applicable.
 3. [Optional] Adjust the parameters within `$model_path/folding_config.json` file to suit your build needs. For alternatives, see the wiki for the FINN framework [here](https://finn.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode). You must remove references to this file within `$model_path/dataflow_build_config.json` if unused.
 4. With the setup terminal window from the previous section, ensure that the terminal window is operating at `$finn_path`. Run `./run-docker.sh build_dataflow $model_path` to start the build.

# Running FINN_EXAMPLES on the Pynq-ZU
A bitfile containing the `tfc-w1a1` neural network compiled for the Pynq-ZU is available in `/bitfiles`. This can be used with the example found in the FINN_EXAMPLES repo [here](https://github.com/Xilinx/finn-examples/blob/main/finn_examples/notebooks/0_mnist_with_fc_networks.ipynb). Please note that both the bitfile (`.bit`) and the metadata file (`.hwh`) must be placed in the folder specified by the python script in FINN_EXAMPLES.