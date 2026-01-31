# PINNs QAT Project

This repository implements Physics-Informed Neural Networks (PINNs) for solving the Non-Linear Schrödinger Equation (NLSE). It includes both a standard PINN implementation and a Quantization-Aware Training (QAT) version using Brevitas to explore model quantization for efficient hardware deployment.

## Project Structure

The project is organized into two main components:

- **`software/pinn_2disp_kerr/`**: Standard PINN implementation (Float32).
- **`quantization/`**: Quantization-Aware Training (QAT) implementation (Int8).

## Environment Setup

The project uses Python. Each component has its own dependencies.

### Standard PINN
Requires `torch`, `numpy`, `matplotlib`, and `scipy`.
```bash
pip install -r software/pinn_2disp_kerr/requirements.txt
```

### Quantization-Aware Training (QAT)
Requires `brevitas` in addition to the standard libraries.
```bash
pip install -r quantization/requirements.txt
```

---

## 1. Standard PINN (Float32)

This module solves the NLSE considering second-order dispersion and Kerr non-linearity.

### Run the Simulation
```bash
python software/pinn_2disp_kerr/PINN_2disp_kerr.py
```

### Workflow
1.  **SSFM Reference**: Generates ground-truth data using the Split-Step Fourier Method (SSFM).
2.  **PINN Training**: Optimizes a standard neural network using physics-based loss (NLSE residuals).
3.  **Evaluation**: Compares the PINN predictions against the SSFM baseline.

### Outputs
- `zt_cmpr_0_L.png`: Comparison at $z=0$ and $z=L$.
- `abs_Err.png`: Absolute error heatmap.
- `ErrDens.png`: Normalized error density.

---

## 2. Quantization-Aware Training (QAT)

This module implements a quantized PINN to simulate hardware constraints (e.g., FPGA deployment). It uses **Brevitas** to simulate 8-bit integer weights and activations during training.

### Run the QAT Simulation
```bash
python quantization/qat_pinn.py
```

### Process
1.  **SSFM Generation**: Generates ground-truth data for validation.
2.  **Model Initialization**: Initializes a `QuantPINN_NLSE` model using Brevitas `QuantLinear` layers (8-bit weights and activations).
3.  **Training**: Optimizes the model using physics-based loss and IC loss.
4.  **Export**: Exports the trained model to ONNX / QONNX formats.

### Model Details
The model `QuantPINN_NLSE` replaces standard PyTorch layers with:
- **`qnn.QuantLinear`**: For 8-bit weight quantization.
- **`qnn.QuantIdentity`**: For 8-bit activation quantization.

### Outputs (in `quantization/` folder)
- **`qat_pinn_model.pth`**: Trained model weights.
- **`qat_pinn_model.qnnx`**: Exported quantized model (for FINN/hardware).
- **`qat_z0_check.png`**: Initial pulse check ($z=0$).
- **`qat_zL_comparison.png`**: Intensity waveform comparison at $z=L$.
- **`qat_error_density.png`**: Error density heatmap.