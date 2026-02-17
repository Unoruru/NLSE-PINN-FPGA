"""
Quantized PINN model using Brevitas for NLSE.

Key changes from original qat_pinn.py:
  - Removed Unwrap modules: use return_quant_tensor=False on QuantLinear instead
  - QuantLinear(return_quant_tensor=False) -> QuantHardTanh(return_quant_tensor=True)
  - QuantHardTanh fuses activation + requantization into one FINN-synthesizable node
  - Cleaner graph, fully quantized, FINN-compatible
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8Bias, Int32Bias

BIAS_QUANT_MAP = {
    4: Int8Bias,
    8: Int8Bias,
    16: Int32Bias,
}


class QuantPINN_NLSE(nn.Module):
    """
    Quantized PINN for NLSE.
    Input: (z, t) — normalized to [-1, 1]
    Output: (u, v) = (Re(A), Im(A))
    """
    def __init__(self, hidden_dim=50, layers=4, weight_bit_width=8, act_bit_width=8):
        super().__init__()

        bias_quant = BIAS_QUANT_MAP.get(weight_bit_width, Int8Bias)

        self.net = nn.Sequential()

        # Input quantization
        self.net.add_module("input_quant", qnn.QuantIdentity(
            bit_width=act_bit_width, return_quant_tensor=True))

        # Layer 1: QuantLinear -> QuantHardTanh (fused activation + requantization)
        self.net.add_module("layer1", qnn.QuantLinear(
            2, hidden_dim, bias=True,
            weight_bit_width=weight_bit_width,
            bias_quant=bias_quant,
            return_quant_tensor=False))
        self.net.add_module("act1", qnn.QuantHardTanh(
            bit_width=act_bit_width, min_val=-1.0, max_val=1.0,
            return_quant_tensor=True))

        # Hidden layers
        for i in range(layers - 1):
            self.net.add_module(f"layer{i+2}", qnn.QuantLinear(
                hidden_dim, hidden_dim, bias=True,
                weight_bit_width=weight_bit_width,
                bias_quant=bias_quant,
                return_quant_tensor=False))
            self.net.add_module(f"act{i+2}", qnn.QuantHardTanh(
                bit_width=act_bit_width, min_val=-1.0, max_val=1.0,
                return_quant_tensor=True))

        # Output layer (no activation — raw linear output)
        self.net.add_module("layer_out", qnn.QuantLinear(
            hidden_dim, 2, bias=True,
            weight_bit_width=weight_bit_width,
            bias_quant=bias_quant,
            return_quant_tensor=False))

    def forward(self, z, t):
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([z, t], dim=1)
        return self.net(x)


def load_fp32_into_qat(qat_model, fp32_path, device="cpu"):
    """
    Load FP32 PINN weights into the QAT model.

    FP32 model uses nn.Sequential with keys like:
        net.0.weight, net.0.bias   (Linear 2->hidden)
        net.2.weight, net.2.bias   (Linear hidden->hidden)
        net.4.weight, net.4.bias   (Linear hidden->hidden)
        ...
        net.{2*layers}.weight/bias (Linear hidden->2)

    QAT model uses named layers:
        net.layer1.weight/bias
        net.layer2.weight/bias
        ...
        net.layer_out.weight/bias
    """
    fp32_state = torch.load(fp32_path, map_location=device, weights_only=True)

    # Build mapping: extract Linear layer indices from FP32 sequential
    # FP32: [Linear, Tanh, Linear, Tanh, ..., Linear]
    # Indices of Linear layers: 0, 2, 4, ..., 2*layers
    linear_keys = []
    for key in sorted(fp32_state.keys()):
        if key.startswith("net.") and ".weight" in key:
            idx = key.split(".")[1]
            linear_keys.append(int(idx))
    linear_keys = sorted(set(linear_keys))

    # QAT layer names: layer1, layer2, ..., layer_out
    n_layers = len(linear_keys)
    qat_names = [f"layer{i+1}" for i in range(n_layers - 1)] + ["layer_out"]

    # Build new state dict mapping
    mapping = {}
    for fp32_idx, qat_name in zip(linear_keys, qat_names):
        for suffix in ["weight", "bias"]:
            fp32_key = f"net.{fp32_idx}.{suffix}"
            qat_key = f"net.{qat_name}.{suffix}"
            if fp32_key in fp32_state:
                mapping[qat_key] = fp32_state[fp32_key]

    # Directly copy weight/bias data into the QAT model's parameters.
    # We can't use load_state_dict because QuantIdentity scaling params
    # are lazily initialized and would cause missing-key errors.
    loaded = 0
    for qat_key, fp32_tensor in mapping.items():
        # Navigate to the parameter: "net.layer1.weight" -> model.net.layer1.weight
        parts = qat_key.split(".")
        obj = qat_model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        param_name = parts[-1]
        param = getattr(obj, param_name)
        if param.shape == fp32_tensor.shape:
            param.data.copy_(fp32_tensor)
            loaded += 1
        else:
            print(f"Warning: shape mismatch for {qat_key}: "
                  f"QAT={param.shape} vs FP32={fp32_tensor.shape}, skipping")

    print(f"Loaded {loaded}/{len(mapping)} FP32 parameter tensors into QAT model")
    return qat_model
