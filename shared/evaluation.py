"""
Evaluation and plotting utilities for PINN NLSE.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from shared.training import normalize_z, normalize_t


def evaluate_full_field(model, t, z_points, L, T_max, N_t, device):
    """
    Evaluate the PINN over the full (z, t) domain.
    Returns complex field A_pred[Nz, Nt].
    """
    model.eval()
    z_test = torch.tensor(z_points, dtype=torch.float32, device=device)
    t_test = torch.tensor(t, dtype=torch.float32, device=device)

    z_test_norm = normalize_z(z_test, L)
    t_test_norm = normalize_t(t_test, T_max)

    Z, T = torch.meshgrid(z_test_norm, t_test_norm, indexing='ij')
    Z_flat = Z.reshape(-1)
    T_flat = T.reshape(-1)

    with torch.no_grad():
        pred_flat = model(Z_flat, T_flat)
    pred_flat = pred_flat.cpu().numpy().reshape(len(z_points), N_t, 2)

    A_pred_complex = pred_flat[..., 0] + 1j * pred_flat[..., 1]
    return A_pred_complex


def plot_z0_check(t, A0, A_pred_complex, save_path=None):
    """Plot initial condition comparison."""
    I0_true = np.abs(A0)**2
    I0_pred = np.abs(A_pred_complex[0])**2

    plt.figure(figsize=(6, 4))
    plt.plot(t, I0_true, label="True |A|^2")
    plt.plot(t, I0_pred, "--", label="PINN |A|^2")
    plt.xlabel("t")
    plt.ylabel("Intensity")
    plt.title("z=0: Initial pulse, SSFM vs PINN")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_zL_comparison(t, A_true, A_pred_complex, save_path=None, label="PINN"):
    """Plot intensity comparison at z=L."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("z=0, Initial pulse")
    plt.plot(t, np.abs(A_true[0])**2, label="SSFM |A|^2")
    plt.plot(t, np.abs(A_pred_complex[0])**2, '--', label=f"{label} |A|^2")
    plt.xlabel("t")
    plt.ylabel("Intensity")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("z=L, SSFM vs " + label)
    plt.plot(t, np.abs(A_true[-1])**2, label="SSFM |A|^2")
    plt.plot(t, np.abs(A_pred_complex[-1])**2, '--', label=f"{label} |A|^2")
    plt.xlabel("t")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_error_heatmap(t, z_points, L, A_true, A_pred_complex, save_path=None):
    """Plot absolute error heatmap over full domain."""
    err = np.abs(A_pred_complex - A_true)
    plt.figure(figsize=(6, 4))
    plt.imshow(err,
               aspect="auto",
               extent=[t[0], t[-1], 0, L],
               origin="lower")
    plt.colorbar(label="|A_pred - A_true|")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title("PINN vs SSFM absolute error")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_error_density(t, z_points, L, A_true, A_pred_complex, save_path=None):
    """Plot normalized and smoothed error density."""
    err = np.abs(A_pred_complex - A_true)
    err_norm = err / (np.max(err) + 1e-10)

    err_norm_smooth = ndimage.gaussian_filter(err_norm, sigma=1.0)

    plt.figure(figsize=(6, 10))
    plt.imshow(err_norm_smooth,
               extent=[t[0], t[-1], z_points[0], z_points[-1]],
               aspect='auto',
               origin='lower',
               cmap='turbo')
    plt.colorbar(label="Normalized error density")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title("Normalized & smoothed error density")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_loss_curves(history, save_path=None):
    """Plot training loss curves (total, PDE, IC) and learning rate."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    epochs = range(1, len(history["loss"]) + 1)

    ax1.semilogy(epochs, history["loss"], label="Total Loss", alpha=0.7)
    ax1.semilogy(epochs, history["loss_pde"], label="PDE Loss", alpha=0.7)
    ax1.semilogy(epochs, history["loss_ic"], label="IC Loss", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["lr"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def print_summary(t, A_true, A_pred_complex):
    """Print evaluation summary metrics."""
    err_last = np.abs(A_pred_complex[-1] - A_true[-1])
    err_all = np.abs(A_pred_complex - A_true)
    print(f"Mean abs error at z=L:       {np.mean(err_last):.6e}")
    print(f"Max  abs error at z=L:       {np.max(err_last):.6e}")
    print(f"Mean abs error (full field): {np.mean(err_all):.6e}")
