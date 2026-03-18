# FPGA-Accelerated Physics Informed Neural Network for Optical Fibre Communications (16-APSK Network)

This script trains a Physics Informed Neural Network (PINN) to recover 16-APSK signals received at the endpoint through optical fibre. Noise is introduced due to physical constraints described by the Non-Linear Schrodingers Equation (NLSE).

## Physics Background

The NLSE describes pulse propagation in optical fiber:

```
A_z = i * (beta2/2) * A_tt + i * gamma * |A|^2 * A
```

where `A(z,t)` is the complex envelope, `beta2` is the group-velocity dispersion parameter, and `gamma` is the Kerr nonlinear coefficient. Ground-truth solutions are generated via the symmetric Split-Step Fourier Method (SSFM).

## Project Structure

```
PINNs QAT/
├── APSK/
│   ├── pinn_apsk.py                 # 16-APSK Quantization Aware PINN Training and Conversion Script
│   ├── run_apsk.py                  # Reinforcement training runner for 16-APSK PINN
│   └── readme.md                    # This file
│
├── shared/                          # See outer README.md for description
│   ├── __init__.py
│   ├── config.py
│   ├── ssfm.py
│   ├── pinn_base.py
│   ├── training.py
│   └── evaluation.py
│
├── software/                        # See outer README.md for description
│   ├── train_fp32.py
│   └── pinn_2disp_kerr/
│       ├── PINN_2disp_kerr.py
│       └── requirements.txt
│
├── quantization/                    # See outer README.md for description
│   ├── train_qat.py
│   ├── qat_model.py
│   ├── qat_pinn.py
│   ├── qat_pinn2.py
│   └── requirements.txt
│
├── qonnx2finn/
│   ├── qonnx2finn.py                # Function for FINN-ONNX export conversion
│   └── req.txt                      # FINN/QONNX dependencies [Not Required Here]
│
└── README.md                        # Outer README.md file
```

## Environment Setup

To run this 16-APSK quantization aware (QA) PINN training script, a Python virtual environment is required. **The expected Python version is 3.12.**
It is expected that all commands should run from the project root directory. To begin, create a virtual environment by the following:

```bash
python -m venv env
```

Activate the environment by the following:

```bash
./env/scripts/activate  # Windows
./env/bin/activate      # macOS/Linux
```

Install the required dependencies by the following:

```bash
pip install --no-deps --ignore-requires-python -r req.txt  # Designed for CUDA-accelerated workflows
```

Flags are required as there are dependency and python version conflicts between packages. This has been tested to be functional for the script.

## Quick Start

To begin, simply run the script by calling:

```bash
python APSK/pinn_apsk.py
```

This will invoke training from scratch and will generate all available outputs (including metrics, visuals, checkpoints and exports).

Additional CLI options are available. Please run:

```bash
python APSK/pinn_apsk.py --help
```

for more information.

## Reinforcement Training Quick Start

In order for the network to be able to predict the recovery for different input 16-APSK signals, reinforcement training is required. This can be done by calling:

```bash
python APSK/run_apsk.py
```

This will invoke an initial 3000 epoch training, then reinforcement training of ``x`` iterations at ``y`` epochs each, based on either default or CLI inputs.

Additional CLI options are available. Please run:

```bash
python APSK/run_apsk.py --help
```

for more information.

## Architecture

The architecture of the model is as follows:

```
Input (W×2) → QuantIdentity → QuantLinear(W×2 → H) → QuantHardTanh
            → QuantIdentity → QuantLinear(H → H)   → QuantHardTanh  (×L)
            → QuantIdentity → QuantLinear(H → 2)   → Output (Re, Im)

W = window_size, H = hidden_dim, L = hlayers
```

The input takes a flattened sliding window of complex symbols, doubled to account for both ``Re`` and ``Im`` components. The model uses `QuantHardTanh` instead of `Tanh` as FINN is unable to synthesize `nn.Tanh` (or `qnn.QuantTanh`) into hardware logic. `QuantHardTanh` clamps outputs to [-1, 1] and fuses the activation with requantization into a single FINN-synthesizable node. A `QuantIdentity` layer at the input quantizes the incoming (z, t) values before the first linear layer. The final linear layer funnels the 64-wide hidden dimension down to an output size of 2, representing the single corrected real and imaginary values of the target symbol.

## Output Metrics/Visuals

Running the model provides 2 sets of metrics and 1 set of visualisation. The metrics include EVM (Error Vector Magnitude) and SER (Symbol Error Rate). The visualisation shows the distorted, SSFM-recovered and PINN-recovered constellation diagram of the 16-APSK signal, with symbols normalised.

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 3,000 | Training epochs |
| `lr` | 5e-4 | Learning rate |
| `bit_width` | 8 | Weight quantization bits |
| `act_bit_width` | 8 | Activation quantization bits |

## Results

Using the default hyperparameters and running the script to train from scratch, the following visual was generated, which illustrates the recovery of 16-APSK.

![results](sample_results/constellation_comparison.png)
