## Branch Structure

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

### Products & Performance Evaluation

The execution of `qat_pinn.py` generates several key artifacts and visualizations located in the `quantization/` directory. These products serve to validate the Quantized PINN against the Split-Step Fourier Method (SSFM) benchmark—the same ground truth used for the standard Float32 PINN.

#### Generated Files
- **`qat_pinn_model.qnnx`**: The final Quantized Neural Network exported in QONNX format. This model is ready for deployment on FPGA accelerators (e.g., via AMD/Xilinx FINN), representing the core deliverable of the QAT process.
- **`qat_pinn_model.pth`**: The PyTorch state dictionary containing the trained 8-bit weights and quantization thresholds.

#### Performance Visualization (Plots)
These plots illustrate the comparative performance of the QAT PINN against the SSFM ground truth, demonstrating that 8-bit quantization maintains solution fidelity.

- **`qat_zL_comparison.png`**: A direct comparison of the pulse intensity at the final propagation distance ($z=L$). It plots the **QAT PINN prediction** (dashed line) against the **SSFM reference** (solid line) and effectively contrasts the quantized model's accuracy with the theoretical limit.
- **`qat_error_density.png`**: A heatmap showing the normalized error density distribution over the entire spatiotemporal domain ($z$ vs. $t$). This highlights specific regions where the quantized model might deviate from the SSFM solution.
- **`qat_z0_check.png`**: Verification of the initial condition ($z=0$), confirming that the quantized network correctly learns the starting Gaussian pulse profile.
