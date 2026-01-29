# Quantization-Aware Training (QAT) for PINNs

This directory contains the implementation of Quantization-Aware Training (QAT) for the Physics-Informed Neural Network (PINN) modeling the Nonlinear Schrödinger Equation (NLSE).

## Overview

As part of **Step 4.2** of the project proposal, this module adapts the baseline floating-point PINN to a quantized model using the **Brevitas** library. This prepares the model for efficient deployment on FPGA hardware (via FINN) by simulating low-precision arithmetic (8-bit integers) during the training phase.

## Files

- `qat_pinn.py`: The main script that defines the quantized model architecture, trains it using the physics-informed loss function, and evaluates the results against the SSFM baseline.
- `requirements.txt`: List of Python dependencies required to run the QAT script.

## Setup & Installation

Ensure you have a Python environment (3.8+) ready.

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This requires `torch` and `brevitas`.*

## Workflow

### 1. Run the Training Script
Execute the Python script to start the Quantization-Aware Training:
```bash
python qat_pinn.py
```

### 2. Process
The script performs the following steps:
1.  **SSFM Generation**: Generates ground-truth data using the Split-Step Fourier Method (SSFM) for validation.
2.  **Model Initialization**: Initializes a `QuantPINN_NLSE` model using Brevitas `QuantLinear` layers (8-bit weights and activations).
3.  **Training**: Optimizes the model using the physics-based loss (NLSE residuals) and Initial Condition (IC) loss.
4.  **Validation**: Compares the quantized PINN predictions against the SSFM baseline.

### 3. Outputs
After execution, the following artifacts are generated in this directory:

- **`qat_pinn_model.pth`**: The saved state dictionary of the trained quantized model.
- **`qat_z0_check.png`**: Plot comparing the initial pulse prediction at $z=0$ with the ground truth.
- **`qat_zL_comparison.png`**: Comparison of the intensity waveform at propagation distance $z=L$ between QAT-PINN and SSFM.
- **`qat_error_density.png`**: Heatmap showing the error density over the entire time-space domain.

## Model Details

The model `QuantPINN_NLSE` replaces standard PyTorch layers with:
- **`qnn.QuantLinear`**: For 8-bit weight quantization.
- **`qnn.QuantIdentity`**: For 8-bit activation quantization.

This ensures that the model learns weights that are robust to quantization noise, minimizing accuracy loss when deployed to hardware.
