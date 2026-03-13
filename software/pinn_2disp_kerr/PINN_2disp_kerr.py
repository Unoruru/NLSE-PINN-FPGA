"""
PINN for NLSE with second-order dispersion and kerr non-linear

- By employing the normalized equation: A_z = i * (beta2/2) * A_tt + i * gamma * abs(A)**2 * A

DEPRECATED: This script is superseded by software/train_fp32.py which adds:
  - Reproducibility seeds
  - Input normalization (z, t -> [-1, 1])
  - CosineAnnealing LR scheduler
  - Adaptive loss weighting, collocation resampling, gradient clipping
  - Validation against SSFM + best model checkpointing
  - Modular code in shared/ for reuse by QAT pipeline

Usage:  python software/train_fp32.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# param

# A_z = i * (beta2/2) * A_tt
beta2 = -1.0          # second-order dispersion param
gamma = 1             # nonlinear coefficient (Kerr)
L = 1.0               # propagation length: 0~5
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
# SSFM (full NLSE)
# ======================
def ssfm_nlse(A0, t, beta2, gamma, L, n_steps):
    """
    NLSE:
        A_z = i * (beta2/2) * A_tt + i * gamma * |A|^2 * A
    (using symmetric split-step)
    """
    dt = t[1] - t[0]
    omega = 2 * np.pi * np.fft.fftfreq(len(t), d=dt)

    dz = L / n_steps
    # linear half step
    H_half = np.exp(-1j * beta2 / 2 * omega**2 * (dz / 2.0))

    A = A0.astype(np.complex128).copy()
    snapshots = [A.copy()]
    z_points = [0.0]

    for k in range(n_steps):
        # --- 1/2 linear ---
        Af = np.fft.fft(A)
        Af *= H_half
        A = np.fft.ifft(Af)

        # --- non-linear ---
        # A_z = i * gamma * |A|^2 * A  ⇒  A(z+dz) = A(z) * exp(i*gamma*|A|^2*dz)
        nl_phase = np.exp(1j * gamma * np.abs(A)**2 * dz)
        A = A * nl_phase

        # --- 1/2 linear ---
        Af = np.fft.fft(A)
        Af *= H_half
        A = np.fft.ifft(Af)

        snapshots.append(A.copy())
        z_points.append((k + 1) * dz)

    return A, np.array(z_points), snapshots

print("Running SSFM (full NLSE) to generate reference data.")
A_L, z_points, A_snaps = ssfm_nlse(A0, t, beta2, gamma, L, n_steps)
print("SSFM done, got", len(z_points), "z-snapshots.")


# ======================
# PINN
# ======================
class PINN_NLSE(nn.Module):
    """
    Input: (z, t)
    Output: (u, v) = (Re(A), Im(A))
    """
    def __init__(self, hidden_dim=100, layers=6):
        super().__init__()
        net = []
        net.append(nn.Linear(2, hidden_dim))
        net.append(nn.Tanh())
        for _ in range(layers - 1):
            net.append(nn.Linear(hidden_dim, hidden_dim))
            net.append(nn.Tanh())
        net.append(nn.Linear(hidden_dim, 2))  # real and im outputs
        self.net = nn.Sequential(*net)

    def forward(self, z, t):
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([z, t], dim=1)  # [N,2]
        return self.net(x)  # [N,2]

def gradients(y, x):

    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

def residual_pde(model, z, t, beta2, gamma):
    """
    compute PDE residual for full NLSE:
        A_z = i * (beta2/2) * A_tt + i * gamma * |A|^2 * A

    In terms of u = Re(A), v = Im(A):

        u_z = -(beta2/2)*v_tt - gamma * v * (u^2 + v^2)
        v_z =  (beta2/2)*u_tt + gamma * u * (u^2 + v^2)

    So residuals (set to 0):

        res_real = u_z + (beta2/2)*v_tt + gamma*v*(u^2+v^2)
        res_imag = v_z - (beta2/2)*u_tt - gamma*u*(u^2+v^2)
    """
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
# IC + collocation
# ======================

t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
A0_real_t = torch.tensor(np.real(A0), dtype=torch.float32, device=device)
A0_imag_t = torch.tensor(np.imag(A0), dtype=torch.float32, device=device)

# IC: z = 0
z0 = torch.zeros_like(t_tensor, device=device)

# collocation
N_res = 10000
z_res = torch.rand(N_res, device=device) * L            # [0, L]
t_res = (torch.rand(N_res, device=device) * 2 - 1) * T_max  # [-T_max, T_max]


model = PINN_NLSE(hidden_dim=50, layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


n_epochs = 10000

print("Start training PINN...")
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
        print(f"Epoch {epoch}/{n_epochs}, "
              f"Loss: {loss.item():.4e}, "
              f"PDE: {loss_pde.item():.4e}, "
              f"IC: {loss_ic.item():.4e}")

print("Training finished.")


model.eval()
##
# ===== check z = 0  PINN vs SSFM=====
with torch.no_grad():

    # t_tensor = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
    # z0 = torch.zeros_like(t_tensor)

    A_pred0 = model(z0, t_tensor)          # [N, 2]
    u0 = A_pred0[:, 0].cpu().numpy()
    v0 = A_pred0[:, 1].cpu().numpy()
    I0_pred = u0**2 + v0**2                # PINN: |A|^2

# SSFM Initial pulse
I0_true = np.abs(A0)**2

plt.figure(figsize=(6,4))
plt.plot(t, I0_true, label="Initial |A|^2 (true)")
plt.plot(t, I0_pred, "--", label="Initial |A|^2 (PINN)")
plt.xlabel("t")
plt.ylabel("Intensity")
plt.title("z = 0: Initial pulse, SSFM vs PINN")
plt.legend()
plt.tight_layout()
plt.show()
# ======================================

##
z_test = torch.tensor(z_points, dtype=torch.float32, device=device)   # [Nz]
t_test = torch.tensor(t, dtype=torch.float32, device=device)          # [Nt]

Z, T = torch.meshgrid(z_test, t_test, indexing='ij')  # [Nz, Nt]

Z_flat = Z.reshape(-1)
T_flat = T.reshape(-1)

with torch.no_grad():
    pred_flat = model(Z_flat, T_flat)    # [Nz*Nt, 2]
pred_flat = pred_flat.cpu().numpy().reshape(len(z_points), N_t, 2)

A_pred_complex = pred_flat[..., 0] + 1j * pred_flat[..., 1]  # [Nz, Nt]

A_true = np.array(A_snaps)  

err_last = np.abs(A_pred_complex[-1] - A_true[-1])

print("Mean abs error at z = L:", np.mean(err_last))


# comparison of the time-domain intensity waveforms
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("z=0, Initial pulse")
plt.plot(t, np.abs(A0)**2, label="Initial |A|^2")
plt.xlabel("t (normalized)")
plt.ylabel("Intensity")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("z=L, SSFM vs PINN")
plt.plot(t, np.abs(A_true[-1])**2, label="SSFM |A|^2")
plt.plot(t, np.abs(A_pred_complex[-1])**2, '--', label="PINN |A|^2")
plt.xlabel("t (normalized)")
plt.legend()

plt.tight_layout()
plt.savefig("zt_cmpr_0_L.png", dpi=300)
plt.show()

# abs error 
plt.figure(figsize=(6, 4))
plt.imshow(np.abs(A_pred_complex - A_true),
           aspect="auto",
           # extent=[t[0], t[-1], z_points[0], z_points[-1]],
           extent=[t[0], t[-1], 0, L], 
           origin="lower")
plt.colorbar(label="|A_pred - A_true|")
plt.xlabel("t (normalized)")
plt.ylabel("z (normalized)")
plt.title("PINN vs SSFM absolute error")
plt.savefig("abs_Err.png", dpi=300)
plt.show()

# error density

err = np.abs(A_pred_complex - A_true)

# normalization
err_norm = err / np.max(err)

# smooth
err_norm_smooth = ndimage.gaussian_filter(err_norm, sigma=1.0)

plt.figure(figsize=(6, 10))
plt.imshow(err_norm_smooth,
           extent=[t[0], t[-1], z_points[0], z_points[-1]],
           aspect='auto',
           origin='lower',
           cmap='turbo')

plt.colorbar(label="Normalized error density")
plt.xlabel("t (normalized)")
plt.ylabel("z (normalized)")
plt.title("Normalized & smoothed error density")
plt.savefig("ErrDens.png", dpi=300)
plt.show()