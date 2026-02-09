"""
Base PINN model and physics-informed loss functions for NLSE.
Supports input normalization with correct chain-rule scaling.
"""

import torch
import torch.nn as nn


class PINN_NLSE(nn.Module):
    """
    Input: (z, t) — normalized to [-1, 1]
    Output: (u, v) = (Re(A), Im(A))
    """
    def __init__(self, hidden_dim=50, layers=4):
        super().__init__()
        net = []
        net.append(nn.Linear(2, hidden_dim))
        net.append(nn.Tanh())
        for _ in range(layers - 1):
            net.append(nn.Linear(hidden_dim, hidden_dim))
            net.append(nn.Tanh())
        net.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*net)

    def forward(self, z, t):
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([z, t], dim=1)
        return self.net(x)


def gradients(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]


def residual_pde(model, z, t, beta2, gamma, z_std, t_std):
    """
    PDE residual for NLSE with chain-rule correction for normalized inputs.

    z, t are in normalized coordinates.
    z_std, t_std are the normalization scale factors so that:
        z_physical = z_normalized * z_std + z_mean
        t_physical = t_normalized * t_std

    Chain rule:
        du/dz_physical = du/dz_norm / z_std
        d^2u/dt_physical^2 = d^2u/dt_norm^2 / t_std^2
    """
    z = z.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    out = model(z, t)
    u = out[:, 0]
    v = out[:, 1]

    # Gradients in normalized coordinates
    u_z = gradients(u, z)
    v_z = gradients(v, z)

    u_t = gradients(u, t)
    v_t = gradients(v, t)

    u_tt = gradients(u_t, t)
    v_tt = gradients(v_t, t)

    # Convert to physical coordinates via chain rule
    u_z_phys = u_z / z_std
    v_z_phys = v_z / z_std
    u_tt_phys = u_tt / (t_std ** 2)
    v_tt_phys = v_tt / (t_std ** 2)

    absA2 = u**2 + v**2

    res_real = u_z_phys + 0.5 * beta2 * v_tt_phys + gamma * v * absA2
    res_imag = v_z_phys - 0.5 * beta2 * u_tt_phys - gamma * u * absA2

    return res_real, res_imag
