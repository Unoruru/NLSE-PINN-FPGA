# Quantization-Aware Training (QAT) for PINN

This directory contains the implementation of a Quantized Physics-Informed Neural Network (PINN) for solving the Non-Linear Schrödinger Equation (NLSE). It utilizes **Brevitas** to simulate hardware-aware quantization (specifically 8-bit integer weights and activations) during the training process.

## Overview

The goal of this module is to train a PINN that can be deployed on integer-only hardware accelerators (like FPGAs) while maintaining the accuracy of the standard floating-point model.

## Requirements

Install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`
- `brevitas` (for quantization)
- `numpy`, `scipy`, `matplotlib` (for physics simulation and plotting)

## Usage

Run the main QAT training and evaluation script:

```bash
python qat_pinn.py
```

## Workflow

1.  **SSFM Reference**: The script first calculates the exact solution using the Split-Step Fourier Method (SSFM) to serve as the ground truth.
2.  **Quantized Model Setup**: Initializes a `QuantPINN_NLSE` using Brevitas layers (`QuantLinear`, `QuantIdentity`).
3.  **Training**: Trains the model using the physics loss (NLSE residual) and initial condition loss.
4.  **Export**: Saves the trained model weights and exports the network to QONNX format for hardware compilation.

## Products & Outputs

After execution, the following files are generated in this directory:

### Model Files
- **`qat_pinn_model.qnnx`**: The exported quantized model in QONNX format, compatible with the FINN compiler for FPGA deployment.
- **`qat_pinn_model.pth`**: PyTorch checkpoint containing the trained parameters (weights, biases, and quantization scales).

### Performance Plots
These visualizations compare the Quantized PINN's predictions against the SSFM ground truth:

- **`qat_zL_comparison.png`**: Compares the intensity profile at $z=L$ between the QAT PINN (dashed) and SSFM (solid). Overlap indicates high accuracy despite quantization.
- **`qat_error_density.png`**: A heatmap of the normalized error distribution across the $z$-$t$ domain.
- **`qat_z0_check.png`**: Verifies that the model correctly learns the initial Gaussian pulse at $z=0$.
