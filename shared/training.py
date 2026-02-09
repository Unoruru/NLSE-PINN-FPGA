"""
Training loop utilities for PINN NLSE.
Handles adaptive loss weighting, collocation resampling,
gradient clipping, LR scheduling, validation, and checkpointing.
"""

import numpy as np
import torch


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_z(z_raw, L):
    """Normalize z from [0, L] to [-1, 1]."""
    return 2.0 * z_raw / L - 1.0


def normalize_t(t_raw, T_max):
    """Normalize t from [-T_max, T_max] to [-1, 1]."""
    return t_raw / T_max


def sample_collocation(N_res, L, T_max, device):
    """Sample collocation points and return them in normalized coordinates."""
    z_raw = torch.rand(N_res, device=device) * L
    t_raw = (torch.rand(N_res, device=device) * 2 - 1) * T_max
    z_norm = normalize_z(z_raw, L)
    t_norm = normalize_t(t_raw, T_max)
    return z_norm, t_norm


def compute_losses(model, z_res, t_res, z0_norm, t_norm, A0_real, A0_imag,
                   beta2, gamma, z_std, t_std):
    """Compute PDE residual loss and initial condition loss."""
    from shared.pinn_base import residual_pde

    res_r, res_i = residual_pde(model, z_res, t_res, beta2, gamma, z_std, t_std)
    loss_pde = torch.mean(res_r**2 + res_i**2)

    A_pred0 = model(z0_norm, t_norm)
    loss_ic = torch.mean((A_pred0[:, 0] - A0_real)**2 +
                         (A_pred0[:, 1] - A0_imag)**2)

    return loss_pde, loss_ic


def validate_against_ssfm(model, t, A_true_at_L, L, T_max, device):
    """
    Evaluate model at z=L against SSFM ground truth.
    Returns mean absolute error.
    """
    model.eval()
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    t_norm = normalize_t(t_tensor, T_max)
    z_L_norm = torch.full_like(t_norm, 1.0)  # z=L normalizes to 1.0

    with torch.no_grad():
        pred = model(z_L_norm, t_norm)
        u = pred[:, 0].cpu().numpy()
        v = pred[:, 1].cpu().numpy()

    A_pred = u + 1j * v
    err = np.mean(np.abs(A_pred - A_true_at_L))
    model.train()
    return err


def train_pinn(model, optimizer, scheduler, cfg, t, A0, A_true_at_L, device,
               save_path="fp32_pinn_best.pth"):
    """
    Full training loop with all improvements:
    - Adaptive loss weighting
    - Collocation resampling
    - Gradient clipping
    - LR scheduling
    - Validation against SSFM
    - Best model checkpointing
    - Loss curve tracking
    """
    L, T_max = cfg.L, cfg.T_max
    z_std = L / 2.0       # normalization scale for z
    t_std = T_max          # normalization scale for t

    # Prepare IC data (normalized)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    t_norm = normalize_t(t_tensor, T_max)
    z0_norm = torch.full_like(t_norm, -1.0)  # z=0 normalizes to -1.0

    A0_real = torch.tensor(np.real(A0), dtype=torch.float32, device=device)
    A0_imag = torch.tensor(np.imag(A0), dtype=torch.float32, device=device)

    # Initial collocation points
    z_res, t_res = sample_collocation(cfg.N_res, L, T_max, device)

    # Adaptive loss weight
    lambda_ic = 1.0

    # Tracking
    history = {"loss": [], "loss_pde": [], "loss_ic": [], "lr": []}
    best_val_error = float("inf")

    print(f"Starting training for {cfg.n_epochs} epochs...")
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss_pde, loss_ic = compute_losses(
            model, z_res, t_res, z0_norm, t_norm, A0_real, A0_imag,
            cfg.beta2, cfg.gamma, z_std, t_std
        )

        loss = loss_pde + lambda_ic * loss_ic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        # Record every epoch
        history["loss"].append(loss.item())
        history["loss_pde"].append(loss_pde.item())
        history["loss_ic"].append(loss_ic.item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Adaptive loss weight update
        if epoch % cfg.lambda_ic_update_every == 0:
            with torch.no_grad():
                if loss_ic.item() > 1e-12:
                    lambda_ic = loss_pde.item() / loss_ic.item()

        # Collocation resampling
        if epoch % cfg.resample_every == 0:
            z_res, t_res = sample_collocation(cfg.N_res, L, T_max, device)

        # Printing
        if epoch % cfg.print_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{cfg.n_epochs}  "
                  f"Loss: {loss.item():.4e}  "
                  f"PDE: {loss_pde.item():.4e}  "
                  f"IC: {loss_ic.item():.4e}  "
                  f"lambda_ic: {lambda_ic:.2f}  "
                  f"LR: {current_lr:.2e}")

        # Validation + checkpointing
        if epoch % cfg.validate_every == 0:
            val_err = validate_against_ssfm(model, t, A_true_at_L, L, T_max, device)
            print(f"  -> Validation error at z=L: {val_err:.6e}")
            if val_err < best_val_error:
                best_val_error = val_err
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved (val_err={val_err:.6e})")

    # Final save if never validated better
    if best_val_error == float("inf"):
        torch.save(model.state_dict(), save_path)

    print(f"Training finished. Best validation error: {best_val_error:.6e}")
    return history
