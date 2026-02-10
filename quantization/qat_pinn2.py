"""
Quantization-Aware Training (QAT) for PINN NLSE using Brevitas.
Based on software/pinn_2disp_kerr/PINN_2disp_kerr.py

Install required packages:
- pip install -r requirements.txt
- pip install qonnx

git clone FINN repository and cd into the root directory:
- pip install -e .  # Install FINN
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
import sys

from qonnx.util.cleanup import cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from finn.transformation.fpgadataflow.convert_to_hw_layers import InferQuantizedMatrixVectorActivation
from finn.transformation.streamline import Streamline
from qonnx.transformation.general import GiveUniqueNodeNames

# Try importing Brevitas
try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias
    from brevitas.export import StdQCDQONNXManager, export_qonnx
except ImportError:
    print("Error: Brevitas not found. Please install it using 'pip install brevitas'.")
    exit(1)

# Ensure quantization directory exists
os.makedirs("quantization", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Parameters
# ======================
beta2 = -1.0          # second-order dispersion param
gamma = 1             # nonlinear coefficient (Kerr)
L = 1.0               # propagation length
n_steps = 100         # SSFM steps

# time axis
T_max = 10          
N_t = 1024            # time-sample count
t = np.linspace(-T_max, T_max, N_t)

# Gaussian pulse
P0 = 1.0              
T0 = 1.0             
A0 = np.sqrt(P0) * np.exp(-t**2 / (2 * T0**2))  

# ======================
# SSFM (Reference)
# ======================
def ssfm_nlse(A0, t, beta2, gamma, L, n_steps):
    dt = t[1] - t[0]
    omega = 2 * np.pi * np.fft.fftfreq(len(t), d=dt)

    dz = L / n_steps
    H_half = np.exp(-1j * beta2 / 2 * omega**2 * (dz / 2.0))

    A = A0.astype(np.complex128).copy()
    snapshots = [A.copy()]
    z_points = [0.0]

    for k in range(n_steps):
        # 1/2 linear
        Af = np.fft.fft(A)
        Af *= H_half
        A = np.fft.ifft(Af)

        # non-linear
        nl_phase = np.exp(1j * gamma * np.abs(A)**2 * dz)
        A = A * nl_phase

        # 1/2 linear
        Af = np.fft.fft(A)
        Af *= H_half
        A = np.fft.ifft(Af)

        snapshots.append(A.copy())
        z_points.append((k + 1) * dz)

    return A, np.array(z_points), snapshots

print("Running SSFM (Reference)...")
A_L, z_points, A_snaps = ssfm_nlse(A0, t, beta2, gamma, L, n_steps)
A_true = np.array(A_snaps)
print("SSFM done.")

# ======================
# Quantized PINN Model
# ======================

class QuantPINN_NLSE(nn.Module):
    """
    Quantized version of the PINN using Brevitas.
    Input: (z, t)
    Output: (u, v) = (Re(A), Im(A))
    """
    def __init__(self, hidden_dim=50, layers=4, weight_bit_width=8, act_bit_width=8):
        super().__init__()
        
        # We construct the network as a list of modules
        # Note: We use QuantIdentity for activation quantization before linear layers
        # where appropriate. 
        
        self.net = nn.Sequential(

            # Input Quantization
            qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True),

            # Input Layer
            qnn.QuantLinear(2, hidden_dim, bias=True, 
                            weight_bit_width=weight_bit_width,
                            act_quant=Int8ActPerTensorFloat,
                            bias_quant=Int8Bias,
                            return_quant_tensor=True),
            # nn.Tanh(),
            qnn.QuantTanh(bit_width=act_bit_width, return_quant_tensor=True),

            # Hidden Layers
            *[nn.Sequential(
                qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True),
                qnn.QuantLinear(hidden_dim, hidden_dim, bias=True, 
                                weight_bit_width=weight_bit_width,
                                act_quant=Int8ActPerTensorFloat,
                                bias_quant=Int8Bias,
                                return_quant_tensor=True),
                # nn.Tanh()
                qnn.QuantTanh(bit_width=act_bit_width, return_quant_tensor=True)
            ) for _ in range(layers - 1)],

            # Output Layer
            qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True),
            qnn.QuantLinear(hidden_dim, 2, bias=True, 
                             weight_bit_width=weight_bit_width,
                             act_quant=Int8ActPerTensorFloat,
                             bias_quant=Int8Bias,
                             return_quant_tensor=False)

        )

    def forward(self, z, t):
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([z, t], dim=1)  # [N,2]
        # Run through network and extract the float value from the final QuantTensor
        return self.net(x)

# ======================
# Physics Loss
# ======================
def gradients(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

def residual_pde(model, z, t, beta2, gamma):
    z = z.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    out = model(z, t)      # [N,2]
    u = out[:, 0]
    v = out[:, 1]

    u_z = gradients(u, z)
    v_z = gradients(v, z)

    u_t = gradients(u, t)
    v_t = gradients(v, t)

    u_tt = gradients(u_t, t)
    v_tt = gradients(v_t, t)

    absA2 = u**2 + v**2

    res_real = u_z + 0.5 * beta2 * v_tt + gamma * v * absA2
    res_imag = v_z - 0.5 * beta2 * u_tt - gamma * u * absA2

    return res_real, res_imag

# ======================
# Setup Training
# ======================
t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
A0_real_t = torch.tensor(np.real(A0), dtype=torch.float32, device=device)
A0_imag_t = torch.tensor(np.imag(A0), dtype=torch.float32, device=device)

# IC: z = 0
z0 = torch.zeros_like(t_tensor, device=device)

# Collocation points
N_res = 10000
z_res = torch.rand(N_res, device=device) * L            
t_res = (torch.rand(N_res, device=device) * 2 - 1) * T_max 

# Initialize QAT Model
model = QuantPINN_NLSE(hidden_dim=50, layers=4).to(device)
model_path = os.path.join("quantization", "qat_pinn_model.pth")

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded. Skipping training.")
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 2000 

    print(f"Start QAT training for {n_epochs} epochs...")
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # PDE residual loss
        res_r, res_i = residual_pde(model, z_res, t_res, beta2, gamma)
        loss_pde = torch.mean(res_r**2 + res_i**2)

        # IC constraint loss
        A_pred0 = model(z0, t_tensor)  
        loss_ic = torch.mean((A_pred0[:, 0] - A0_real_t)**2 +
                             (A_pred0[:, 1] - A0_imag_t)**2)

        loss = loss_pde + loss_ic
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4e}, PDE: {loss_pde.item():.4e}, IC: {loss_ic.item():.4e}")
            history.append(loss.item())

    print("QAT Finished.")
    # ======================
    # Save Model
    # ======================
    torch.save(model.state_dict(), model_path)
    print(f"Quantized model saved to {model_path}")

# ======================
# Evaluation & Plotting
# ======================
model.eval()

# Check z=0
with torch.no_grad():
    A_pred0 = model(z0, t_tensor)          
    u0 = A_pred0[:, 0].cpu().numpy()
    v0 = A_pred0[:, 1].cpu().numpy()
    I0_pred = u0**2 + v0**2                

I0_true = np.abs(A0)**2

plt.figure(figsize=(6,4))
plt.plot(t, I0_true, label="True")
plt.plot(t, I0_pred, "--", label="QAT PINN")
plt.xlabel("t")
plt.ylabel("Intensity")
plt.title("z=0 Check (QAT)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("quantization", "qat_z0_check.png"))
plt.close()

# Check full field
z_test = torch.tensor(z_points, dtype=torch.float32, device=device)
t_test = torch.tensor(t, dtype=torch.float32, device=device)
Z, T = torch.meshgrid(z_test, t_test, indexing='ij')
Z_flat = Z.reshape(-1)
T_flat = T.reshape(-1)

with torch.no_grad():
    pred_flat = model(Z_flat, T_flat)    
pred_flat = pred_flat.cpu().numpy().reshape(len(z_points), N_t, 2)
A_pred_complex = pred_flat[..., 0] + 1j * pred_flat[..., 1]

err_last = np.abs(A_pred_complex[-1] - A_true[-1])
print("Mean abs error at z=L (QAT):", np.mean(err_last))

# Plot z=L comparison
plt.figure(figsize=(6, 4))
plt.plot(t, np.abs(A_true[-1])**2, label="SSFM")
plt.plot(t, np.abs(A_pred_complex[-1])**2, '--', label="QAT PINN")
plt.title("z=L Intensity (QAT vs SSFM)")
plt.legend()
plt.savefig(os.path.join("quantization", "qat_zL_comparison.png"))
plt.close()

# Error Density
err = np.abs(A_pred_complex - A_true)
err_norm = err / np.max(err)
try:
    err_norm_smooth = ndimage.gaussian_filter(err_norm, sigma=1.0)
    plt.figure(figsize=(6, 10))
    plt.imshow(err_norm_smooth,
               extent=[t[0], t[-1], 0, L],
               aspect='auto',
               origin='lower',
               cmap='turbo')
    plt.colorbar(label="Normalized error density")
    plt.title("QAT Error Density")
    plt.savefig(os.path.join("quantization", "qat_error_density.png"))
    plt.close()
except:
    print("Skipping smoothed error plot (scipy issue?)")

# ======================
# Export to QONNX / ONNX
# ======================
print("Exporting...")
model.eval()
opset_version = 11

dummy_input = torch.randn(1, 2, device=device)
try:
    print("Attempting QONNX export (optimized for FINN)...")
    export_path_qonnx = os.path.join("quantization", "qat_pinn_model.qnnx")
    # Try passing dynamo=False to avoid tracing issues
    try:
        export_qonnx(model.net, args=dummy_input, export_path=export_path_qonnx, dynamo=False, opset_version=opset_version)
        print(f"Model exported to {export_path_qonnx}")
    except TypeError:
         # If dynamo arg not supported, try without
         export_qonnx(model.net, args=dummy_input, export_path=export_path_qonnx, opset_version=opset_version)
         print(f"Model exported to {export_path_qonnx}")

except Exception as e:
    print(f"QONNX export failed: {e}")
    exit()

# ======================
# Convert to FINN format
# ======================

qonnx_path = "qat_pinn_model.qnnx"
qonnx_clean_path = "quantization/" + qonnx_path.replace(".qnnx", "_cleaned.qnnx")

cleanup("quantization/"+qonnx_path, out_file=qonnx_clean_path)

model = ModelWrapper(qonnx_clean_path)
model = model.transform(ConvertQONNXtoFINN())
finn_onnx_path = "quantization/model.onxx"
model.save(finn_onnx_path)
print(f"Model converted to FINN format and exported as {finn_onnx_path}")
