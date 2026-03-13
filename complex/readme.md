# FPGA-Accelerated Physics Informed Neural Network for Optical Fibre Communications (Complex Network)

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
├── complex/
│   ├── pinn_complex.py              # Complex Quantization Aware PINN Training and Conversion Script
│   ├── req.txt                      # Requirements to run this complex pinn script
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
├── software/
│   ├── train_fp32.py                # See outer README.md for description
│   └── pinn_2disp_kerr/
│       ├── PINN_2disp_kerr.py       
│       └── requirements.txt
│
├── quantization/
│   ├── train_qat.py                 # See outer README.md for description
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
To run this complex quantization aware (QA) PINN training script, a Python virtual enviornment is required. **The expected Python version is 3.12.1.**
It is expected that all commands should run from the project root directory. To begin, create a virtual enviornment by the following:
```bash
python -m venv env
```
Activate the enviornment by the following:
```bash
./env/scripts/activate/ # Windows
./env/bin/activate/     # macOS/Linux
```
Install the required dependencies by the following:
```bash
pip install -r complex/req.txt
```

## Quick Start
To begin, simply run the script by calling:
```bash
python complex/pinn_complex.py
```
This will invoke training from scratch and will generate all available outputs (including metrics, visuals, checkpoints and exports).

Additional CLI options are available. Please run:
```bash
python complex/pinn_complex.py --help
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

Running the model provides 2 sets of metrics and 1 set of visualisation. The metrics include EVM (Error Vector Magnitude) and SER (Symbol Error Rate). The visualisation shows the distorted, SSFM-recovered and PINN-recovered constellation diagram of the 16-QAM signal, with symbols normalised.

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 3,000 | Training epochs |
| `lr` | 5e-4 | Learning rate |
| `bit_width` | 8 | Weight quantization bits |
| `act_bit_width` | 8 | Activation quantization bits |
