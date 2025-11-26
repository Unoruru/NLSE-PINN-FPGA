# Setup FINN Framework for PYNQ-ZU
 1. Follow setup instructions in the "quick start" page on FINN github.
 2. Before running the docker quick start test, you must setup the enviornmental variables in the local terminal window. Place `en_var.sh` into the main folder. Alter the path and version number based on the installed version of Xilinx Vivado. Alter the board part to suit needs. `cd` into the folder containing the cloned FINN repo and run `source en_var.sh`.
 3. Replace the following file within the cloned FINN framework for support of PYNQ-ZU:
    # replace the following:
    $finn_repo/src/finn/transformation/fpgadataflow/templates.py
    # with the following:
    $this_repo/replacements/templates.py
 4. Ensure that the PYNQ-ZU board is downloaded within Vivado Store.

# Running FINN Framework for PYNQ-ZU
This part assumes that a ONNX file has been created for the quantised neural network.
 1. [Required] Create a new folder outside of the cloned FINN repo. Place the quantised neural network file (ONNX format) within the folder. Rename it as `model.onnx`.
 2. [Required] Copy the `sample/dataflow_build_config.json` file from this repo and place it into the newly created folder. Adjust the parameters within the file to suit your build needs.
 3. [Optional] Place the `sample/folding_config.json` file from this repo and place it into the newly created folder. Adjust the parameters within the file to suit your build needs. For alternatives, see the wiki for the FINN framework.
 4. `cd` back into the folder containing the cloned FINN framework. Run `./run-docker.sh build_dataflow $path_to_folder`, where `$path_to_folder` is the path to the newly created folder containing the above.

 A sample bitfile for the is available in ``/bitfile`. This contains the `tfc-w1a1` neural network from the FINN_EXAMPLES repo, for juypter notebook example 0.