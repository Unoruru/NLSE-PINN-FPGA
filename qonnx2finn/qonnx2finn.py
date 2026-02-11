# ======================
# Convert to FINN format
# ======================

import os
from qonnx.util.cleanup import cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


qonnx_path = "qat_4bit_pinn_model.qonnx"

os.makedirs("qonnx2finn", exist_ok=True)
qonnx_clean_path = "qonnx2finn/model_clean.onnx"

cleanup("qonnx2finn/"+qonnx_path, out_file=qonnx_clean_path)

model = ModelWrapper(qonnx_clean_path)
model = model.transform(ConvertQONNXtoFINN())
finn_onnx_path = "qonnx2finn/model.onnx"
model.save(finn_onnx_path)
print(f"Model converted to FINN format and exported as {finn_onnx_path}")