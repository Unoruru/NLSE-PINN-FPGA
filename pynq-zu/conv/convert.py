# Script to convert qonnx/onnx file format to finn-onnx file format
# Last updated: 05 Feb 2026

# requires brevitas, qonnx and FINN
# pip3 install brevitas qonnx
# install FINN by cloning git repository and running "pip install -e ." in the root directory of the repository

from qonnx.util.cleanup import cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

qonnx_path = "qat_pinn_model.qnnx"
qonnx_clean_path = qonnx_path.replace(".qnnx", "_cleaned.qnnx")

cleanup(qonnx_path, out_file=qonnx_clean_path)

model = ModelWrapper(qonnx_clean_path)
model = model.transform(ConvertQONNXtoFINN())
finn_onnx_path = qonnx_path.replace(".qnnx", ".onnx")
finn_onnx_path = "finn_" + finn_onnx_path
model.save(finn_onnx_path)