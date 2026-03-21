# Script
# Last Updated: 20 Mar 2026

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import trange

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias

# Define complex PINN architecture
class complexPINN(nn.Module):
    def __init__(self, window_size=21, hlayers=3, hidden_dim=64, bit_width=8, act_bit_width=8):
        super().__init__()
        
        layers = []

        # Input Layer (Window -> Hidden)
        layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
        layers.append(qnn.QuantLinear(window_size * 2, hidden_dim, bias=True, weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias))
        layers.append(qnn.QuantHardTanh(bit_width=act_bit_width, max_val=1.0, min_val=-1.0, act_quant=Int8ActPerTensorFloat))

        # Hidden Layers (default 3)
        for _ in range(hlayers):
            layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
            layers.append(qnn.QuantLinear(hidden_dim, hidden_dim, bias=True, weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias))
            layers.append(qnn.QuantHardTanh(bit_width=act_bit_width, max_val=1.0, min_val=-1.0, act_quant=Int8ActPerTensorFloat))

        # Output Layer (Hidden -> 2 for Re/Im)
        layers.append(qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True))
        layers.append(qnn.QuantLinear(hidden_dim, 2, bias=True, weight_bit_width=bit_width, weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# SSFM for generating training data and reverse physics for baseline comparison
def ssfm_nlse(A_in, L, direction="forward", beta2=-21e-27, gamma=1.3, dt=1e-12, n_steps=50):
    # check direction validity
    # assertlog(direction in ["forward", "reverse"], "Direction must be 'forward' or 'reverse'.")
    assert direction in ["forward", "reverse"], "Direction must be 'forward' or 'reverse'."
    
    # reverse physics for baseline comparison
    if direction == "reverse":
        gamma, beta2 = -gamma, -beta2

    omega = 2 * np.pi * np.fft.fftfreq(len(A_in), d=dt)
    dz = L / n_steps
    H = np.exp(-1j * (beta2 / 2) * (omega**2) * dz)
    
    A = A_in.copy()
    for _ in range(n_steps):
        A *= np.exp(1j * gamma * np.abs(A)**2 * dz)  # Non-linear
        A = np.fft.ifft(np.fft.fft(A) * H)           # Linear

    return A

# Windowing for training and dataset preparation
def windowing(sig, win_size):
    half = win_size // 2
    p = np.pad(sig, (half, half), mode='reflect')
    X = []
    for i in range(len(sig)):
        w = p[i : i + win_size]
        X.append(np.concatenate([np.real(w), np.imag(w)]))
    return torch.tensor(np.array(X), dtype=torch.float32)

# Physics loss computation using the NLSE residuals
def compute_physics_loss(model, x_window, beta2, gamma, scale_factor):
    x_window.requires_grad_(True)

    # 1. Surrogate Forward Pass (Smooth Tanh)
    current_val = x_window
    
    # We iterate directly over the Sequential blocks in model.model
    for layer in model.model:
        # Pass through Identity (gives input scale) and Linear (does the math)
        if isinstance(layer, (qnn.QuantIdentity, qnn.QuantLinear)):
            current_val = layer(current_val)
            
        elif isinstance(layer, qnn.QuantHardTanh):
            # Unwrap the Brevitas QuantTensor to a raw PyTorch tensor
            if hasattr(current_val, 'value'):
                raw_tensor = current_val.value
            else:
                raw_tensor = current_val
                
            # Use Tanh for smooth second derivatives during physics check
            current_val = torch.tanh(raw_tensor)
    
    # Final unwrap just in case the last layer returned a QuantTensor
    if hasattr(current_val, 'value'):
        current_val = current_val.value

    u = current_val[:, 0]
    v = current_val[:, 1]

    # 2. Gradients for NLSE
    u_t = torch.autograd.grad(u, x_window, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, x_window, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    u_tt = torch.autograd.grad(u_t, x_window, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    v_tt = torch.autograd.grad(v_t, x_window, grad_outputs=torch.ones_like(v_t), create_graph=True)[0]

    # 3. Residuals
    power = (u**2 + v**2) * (scale_factor ** 2)
    u_tt_sum = u_tt.sum(dim=1)
    v_tt_sum = v_tt.sum(dim=1)

    res_real = 0.5 * beta2 * v_tt_sum + gamma * v * power
    res_imag = -0.5 * beta2 * u_tt_sum - gamma * u * power

    return torch.mean(res_real**2 + res_imag**2)

# Training loop for the complex PINN
def train(model, device, X_train, Y_train, epochs, lr, beta2, gamma, scale_factor):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    train_start = time.time()

    with trange(epochs, desc="Training Complex PINN") as t:
        for e in t:
            model.train()
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_train.to(device)), Y_train.to(device))
            physics_loss = compute_physics_loss(model, X_train.to(device), beta2, gamma, scale_factor=scale_factor)
            loss = loss + 0.01 * physics_loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            t.set_postfix({"Loss": f"{loss.item():.6f}"})

    train_end = time.time()

    return model, losses, train_end - train_start
