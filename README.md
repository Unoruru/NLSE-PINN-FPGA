# PINN for Nonlinear Schrodinger Equation with Configurable Quantization-Aware Training (4/8/16-bit)

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

- 4 hidden layers, 50 neurons each, Tanh activation
- Inputs normalized to `[-1, 1]`; PDE residual applies chain-rule correction for physical coordinates
- Output `(u, v)` represents `Re(A)` and `Im(A)`

### QAT Model (`QuantPINN_NLSE` in `quantization/qat_model.py`)

Same architecture with Brevitas quantization layers at configurable bit width:
- `QuantIdentity` (N-bit activations) before each `QuantLinear`
- `QuantLinear` (N-bit weights, bias quantized via `BIAS_QUANT_MAP`: `Int8Bias` for 4/8-bit, `Int32Bias` for 16-bit)
- Standard `nn.Tanh()` between layers (operates on float tensors)
- Bit width is set by the `--bit-width` CLI flag (4, 8, or 16)

FP32 weights are transferred via `load_fp32_into_qat()`, which maps the sequential index-based keys to named layer keys.

## Training Features

Both scripts share the training loop from `shared/training.py`:

- **Input normalization** — `z` and `t` mapped to `[-1, 1]` with chain-rule correction in the PDE residual
- **CosineAnnealing LR scheduler** — decays from `lr` to `lr_min` over the full training run
- **Adaptive loss weighting** — `lambda_ic` auto-balances PDE loss vs. IC loss (updated every `lambda_ic_update_every` epochs)
- **Collocation point resampling** — fresh random `(z, t)` points every `resample_every` epochs
- **Gradient clipping** — `clip_grad_norm` at `1.0`
- **Validation against SSFM** — mean absolute error at `z=L` computed every `validate_every` epochs
- **Best model checkpointing** — saves only when validation error improves
- **Reproducibility** — `torch.manual_seed`, `np.random.seed`, and CUDA seeds set from `cfg.seed`
- **Loss curve tracking** — total, PDE, and IC loss recorded every epoch

## Configuration

All hyperparameters are centralized in `shared/config.py` as a `PINNConfig` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `beta2` | `-1.0` | Second-order dispersion |
| `gamma` | `1.0` | Kerr nonlinear coefficient |
| `L` | `1.0` | Propagation length |
| `n_steps` | `100` | SSFM steps |
| `T_max` | `10.0` | Time window half-width |
| `N_t` | `1024` | Time-axis grid points |
| `P0` | `1.0` | Peak power |
| `T0` | `1.0` | Pulse width |
| `hidden_dim` | `50` | Neurons per hidden layer |
| `layers` | `4` | Number of hidden layers |
| `n_epochs` | `10000` | FP32 training epochs |
| `lr` | `1e-3` | FP32 initial learning rate |
| `N_res` | `10000` | Collocation points per batch |
| `resample_every` | `100` | Collocation resampling interval |
| `grad_clip_norm` | `1.0` | Max gradient norm |
| `qat_epochs` | `1000` | QAT fine-tuning epochs |
| `qat_lr` | `5e-4` | QAT initial learning rate |
| `weight_bit_width` | `8` | Quantized weight bit-width (set via `--bit-width`: 4, 8, or 16) |
| `act_bit_width` | `8` | Quantized activation bit-width (set via `--bit-width`: 4, 8, or 16) |
| `seed` | `42` | Random seed |

## Outputs

### `software/train_fp32.py`

| File | Description |
|---|---|
| `fp32_pinn_best.pth` | Best FP32 model checkpoint |
| `fp32_loss_curves.png` | Training loss (total/PDE/IC) and LR schedule |
| `fp32_z0_check.png` | Initial condition comparison (SSFM vs PINN at `z=0`) |
| `fp32_zL_comparison.png` | Pulse intensity comparison at `z=0` and `z=L` |
| `fp32_abs_error.png` | Absolute error heatmap over full `(z, t)` domain |
| `fp32_error_density.png` | Normalized & smoothed error density heatmap |

### `quantization/train_qat.py`

All QAT output filenames include the bit-width prefix `qat_{N}bit_*`, where `{N}` is the value of `--bit-width` (default 8).

| File | Description |
|---|---|
| `qat_{N}bit_pinn_best.pth` | Best QAT model checkpoint |
| `qat_{N}bit_pinn_model.qonnx` | Exported QONNX model for FPGA (FINN-compatible) |
| `qat_{N}bit_loss_curves.png` | QAT training loss and LR schedule |
| `qat_{N}bit_z0_check.png` | Initial condition check |
| `qat_{N}bit_zL_comparison.png` | QAT PINN vs SSFM at `z=L` |
| `qat_{N}bit_abs_error.png` | Absolute error heatmap |
| `qat_{N}bit_error_density.png` | Normalized error density heatmap |

## Legacy Scripts

The following scripts are **deprecated** and kept for reference only:

- **`software/pinn_2disp_kerr/PINN_2disp_kerr.py`** — Original monolithic FP32 PINN. Superseded by `software/train_fp32.py`.
- **`quantization/qat_pinn.py`** — Original monolithic QAT script. Superseded by `quantization/train_qat.py`.

Both legacy scripts lack input normalization, LR scheduling, adaptive loss weighting, validation, and checkpointing.
