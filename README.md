
Physics-Informed Neural Network (PINN) that solves the Nonlinear Schrodinger Equation (NLSE) with second-order dispersion and Kerr nonlinearity, then quantizes the trained model to configurable 4-bit, 8-bit, or 16-bit precision via Quantization-Aware Training (QAT) for FPGA deployment.

## Physics Background

The NLSE describes pulse propagation in optical fiber:

```
A_z = i * (beta2/2) * A_tt + i * gamma * |A|^2 * A
```

where `A(z,t)` is the complex envelope, `beta2` is the group-velocity dispersion parameter, and `gamma` is the Kerr nonlinear coefficient. Ground-truth solutions are generated via the symmetric Split-Step Fourier Method (SSFM).

## Project Structure

```
PINNs QAT/
├── shared/                          # Shared modules used by both FP32 and QAT
│   ├── __init__.py
│   ├── config.py                    # PINNConfig dataclass — all hyperparameters
│   ├── ssfm.py                      # Split-Step Fourier Method solver
│   ├── pinn_base.py                 # PINN_NLSE model + PDE residual function
│   ├── training.py                  # Training loop, normalization, collocation, seeding
│   └── evaluation.py               # Full-field evaluation, plotting, error metrics
│
├── software/
│   ├── train_fp32.py                # Step 1: Train FP32 PINN → fp32_pinn_best.pth
│   └── pinn_2disp_kerr/
│       ├── PINN_2disp_kerr.py       # [DEPRECATED] Original monolithic FP32 script
│       └── requirements.txt
│
├── quantization/
│   ├── train_qat.py                 # Step 2: QAT fine-tuning → .qonnx export
│   ├── qat_model.py                 # QuantPINN_NLSE model + FP32→QAT weight loader
│   ├── qat_pinn.py                  # [DEPRECATED] Original monolithic QAT script
│   └── requirements.txt             # All dependencies (covers both FP32 and QAT)
│
└── README.md
```

## Environment Setup

A single requirements file covers all dependencies:

```bash
pip install -r quantization/requirements.txt
```

Key packages: `torch`, `brevitas`, `numpy`, `scipy`, `matplotlib`, `onnx`.

## Quick Start

The workflow has two steps. Both scripts should be run from the project root directory.

### Step 1 — Train the FP32 PINN

```bash
python software/train_fp32.py
```

Trains a standard float32 PINN from scratch using SSFM ground truth. Saves the best checkpoint to `software/fp32_pinn_best.pth`.

### Step 2 — QAT Fine-Tuning

```bash
# 8-bit (default)
python quantization/train_qat.py --fp32-checkpoint software/fp32_pinn_best.pth

# 4-bit
python quantization/train_qat.py --fp32-checkpoint software/fp32_pinn_best.pth --bit-width 4

# 16-bit
python quantization/train_qat.py --fp32-checkpoint software/fp32_pinn_best.pth --bit-width 16
```

Loads the pretrained FP32 weights into a quantized model (Brevitas) at the specified bit width (4, 8, or 16), fine-tunes with the same physics-informed loss, and exports a `.qonnx` file for FPGA synthesis (e.g., AMD/Xilinx FINN). The `--bit-width` flag controls both weight and activation quantization precision (default: 8).

## Architecture

### FP32 Model (`PINN_NLSE` in `shared/pinn_base.py`)

```
Input (z, t) → [Linear 2→50] → Tanh → [Linear 50→50] → Tanh
             → [Linear 50→50] → Tanh → [Linear 50→50] → Tanh
             → [Linear 50→2] → Output (u, v)
```

- **`qat_zL_comparison.png`**: A direct comparison of the pulse intensity at the final propagation distance ($z=L$). It plots the **QAT PINN prediction** (dashed line) against the **SSFM reference** (solid line) and effectively contrasts the quantized model's accuracy with the theoretical limit.
- **`qat_error_density.png`**: A heatmap showing the normalized error density distribution over the entire spatiotemporal domain ($z$ vs. $t$). This highlights specific regions where the quantized model might deviate from the SSFM solution.
- **`qat_z0_check.png`**: Verification of the initial condition ($z=0$), confirming that the quantized network correctly learns the starting Gaussian pulse profile.
