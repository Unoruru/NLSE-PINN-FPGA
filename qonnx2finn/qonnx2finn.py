# Script to Convert QONNX/QNNX/ONNX Exports to FINN-ONNX for Hardware Deployment
# Last Updated: 12 Feb 2026

# Install prerequisites using "pip install -r req.txt"

# Usage:
# Call "python qonnx2finn.py" with arguments (--i is required).

# Arguments:
# --dir:    directory of the folder containing .qonnx/.qnnx/.onnx model export, relative to current working directory. override with full path.
# (--d)     defaults to current working directory. provided directory must exist.
# 
# --input:  file name of the .qonnx/.qnnx/.onnx model export, ready for conversion. should contain only file name with extension.
# (--i)     default empty. provided file extension must be one of: .onnx/.qnnx/.qonnx.
#
# --output: file name for the output FINN-format .onnx file. should contain only file name with extension
# (--o)     defaults to "model.onnx". provided file extension must be .onnx.

import os
import argparse
from qonnx.util.cleanup import cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


def conv2finn(dir_path, qonnx_name, output_name):
    qonnx_path = os.path.join(dir_path, qonnx_name)
    qonnx_clean_path = os.path.join(dir_path, "model_clean.onnx")
    output_path = os.path.join(dir_path, output_name)

    cleanup(qonnx_path, out_file=qonnx_clean_path)

    model = ModelWrapper(qonnx_clean_path)
    model = model.transform(ConvertQONNXtoFINN())
    model.save(output_path)

    os.remove(qonnx_clean_path)

    print(f"Model converted to FINN format and exported as {output_path}")


def main():
    parse = argparse.ArgumentParser(description="QONNX to FINN-ONNX Converter")
    parse.add_argument("--dir", "--d", type=str, default=os.getcwd(), help="Path to Folder Containing QONNX Model")
    parse.add_argument("--input", "--i", type=str, default="", help="File Name of QONNX Model for Conversion (include .qonnx/.qnnx/.onnx)")
    parse.add_argument("--output", "--o", type=str, default="model.onnx", help="File Name of FINN-ONNX File Output (include .onnx)")
    
    args = parse.parse_args()

    assert os.path.isdir(args.dir), "Provided directory is invalid"
    assert args.input != "", "Specify model file name for conversion."

    input_path = os.path.join(os.getcwd(), args.dir)
    assert os.path.isfile(os.path.join(input_path, args.input)), "Input file does not exist."

    assert (args.input[-5:] == ".onnx" or args.input[-5:] == ".qnnx" or args.input[-6:] == ".qonnx"), "Input path extension must be one of: .onnx/.qnnx/.qonnx."

    assert args.output[-5:] == ".onnx", "Output path extension must be .onnx."


    # print(input_path, args.input, args.output) # debug only

    try:
        conv2finn(input_path, args.input, args.output)
    except Exception as e:
        print(f"Error during conversion: {e}")
        
if __name__ == "__main__":
    main()