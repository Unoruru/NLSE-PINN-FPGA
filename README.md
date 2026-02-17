# PINN-QAT

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
│   ├── qat_model.py                 # QuantPINN_NLSE (QuantHardTanh) + FP32→QAT loader
│   ├── qat_pinn.py                  # [DEPRECATED] Original monolithic QAT script
│   ├── qat_pinn2.py                 # [DEPRECATED] Intermediate QAT revision
│   └── requirements.txt             # All dependencies (covers both FP32 and QAT)
│
├── qonnx2finn/
│   ├── qonnx2finn.py                # Step 3: Convert QONNX export → FINN-ONNX
│   └── req.txt                      # FINN/QONNX dependencies
│
└── README.md
```

## Environment Setup

A single requirements file covers FP32 and QAT dependencies:

```bash
pip install -r quantization/requirements.txt
```

For the FINN conversion step, install the FINN-specific dependencies separately:

```bash
pip install -r qonnx2finn/req.txt
```

Key packages: `torch`, `brevitas`, `numpy`, `scipy`, `matplotlib`, `onnx`, `qonnx`, `finn`.

## Quick Start

The workflow has three steps. Steps 1 and 2 should be run from the project root directory.

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

Additional CLI options for tuning QAT training:

| Flag | Description | Default |
|------|-------------|---------|
| `--qat-epochs` | Total QAT training epochs | 3000 |
| `--qat-warmup` | Warmup scheduler epochs | 500 |
| `--qat-nres` | Number of collocation points | 15000 |
| `--output-dir` | Directory for output files | script directory |
| `--export-only` | Skip plots/evaluation, only train and export | off |

Example with custom QAT parameters:

```bash
python quantization/train_qat.py \
  --fp32-checkpoint software/fp32_pinn_best.pth \
  --bit-width 4 \
  --qat-epochs 5000 \
  --qat-warmup 800 \
  --qat-nres 20000
```

### Step 3 — FINN Conversion

```bash
python qonnx2finn/qonnx2finn.py \
  --dir quantization \
  --input qat_8bit_pinn_model.qonnx \
  --output finn_8bit_model.onnx
```

Converts the QONNX export from Step 2 into a FINN-compatible ONNX format for hardware synthesis. The `--dir` flag specifies the directory containing the `.qonnx` file, `--input` is the exported model filename, and `--output` is the desired FINN-ONNX filename.

## Architecture

### FP32 Model (`PINN_NLSE` in `shared/pinn_base.py`)

```
Input (z, t) → Linear(2→50) → Tanh
             → Linear(50→50) → Tanh  (×3)
             → Linear(50→2) → Output (u, v)
```

### QAT Model (`QuantPINN_NLSE` in `quantization/qat_model.py`)

```
Input (z, t) → QuantIdentity → QuantLinear(2→50) → QuantHardTanh
             → QuantLinear(50→50) → QuantHardTanh  (×3)
             → QuantLinear(50→2) → Output (u, v)
```

The QAT model uses `QuantHardTanh` instead of `Tanh` because FINN cannot synthesize `nn.Tanh` (or `qnn.QuantTanh`) into hardware logic. `QuantHardTanh` clamps outputs to [-1, 1] and fuses the activation with requantization into a single FINN-synthesizable node. A `QuantIdentity` layer at the input quantizes the incoming (z, t) values before the first linear layer.

## QAT Training Details

QAT fine-tuning uses several mechanisms to recover accuracy lost from quantization:

- **Warmup-cosine LR schedule**: Learning rate ramps linearly from 0 to `qat_lr` over 500 warmup epochs, then decays via cosine annealing. This prevents early gradient instability in the quantized graph.
- **IC boost during warmup**: The initial-condition loss weight is multiplied by 2× during the warmup phase, anchoring the quantized model to the correct pulse shape before the PDE loss takes over.
- **Increased collocation points**: QAT uses 15,000 collocation points (vs 10,000 for FP32) to compensate for reduced representational capacity at lower bit widths.
- **Extended training**: 3,000 epochs by default (vs 10,000 for FP32, but starting from pretrained weights).

## Output Plots

Both FP32 (`software/train_fp32.py`) and QAT (`quantization/train_qat.py`) produce the same set of diagnostic plots. QAT plot filenames are prefixed with `qat_{bit}bit_`.

- **`loss_curves.png`**: Training loss components (PDE residual, initial condition, boundary) over epochs.
- **`zL_comparison.png`**: Pulse intensity at the final propagation distance (z=L) — PINN prediction (dashed) vs SSFM reference (solid).
- **`error_density.png`**: Normalized error density heatmap over the full spatiotemporal domain (z vs t).
- **`z0_check.png`**: Initial condition verification at z=0, confirming the network learns the starting Gaussian pulse profile.
- **`abs_error.png`**: Absolute error heatmap across the full domain.

## Default Hyperparameters

All defaults are defined in `shared/config.py` (`PINNConfig` dataclass).

### Physics

| Parameter | Value | Description |
|-----------|-------|-------------|
| `beta2` | -1.0 | Second-order dispersion |
| `gamma` | 1.0 | Kerr nonlinear coefficient |
| `L` | 1.0 | Propagation length |
| `P0` | 1.0 | Peak power |
| `T0` | 1.0 | Pulse width |

### FP32 Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_epochs` | 10,000 | Training epochs |
| `lr` | 1e-3 | Learning rate |
| `N_res` | 10,000 | Collocation points |
| `hidden_dim` | 50 | Hidden layer width |
| `layers` | 4 | Hidden layers |

### QAT Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `qat_epochs` | 3,000 | Training epochs |
| `qat_lr` | 5e-4 | Learning rate |
| `qat_warmup_epochs` | 500 | Warmup epochs |
| `qat_N_res` | 15,000 | Collocation points |
| `qat_ic_boost_factor` | 2.0 | IC loss multiplier during warmup |
| `weight_bit_width` | 8 | Weight quantization bits |
| `act_bit_width` | 8 | Activation quantization bits |
