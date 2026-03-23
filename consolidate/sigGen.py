# Script to generate training/evaluation signals for ComplexPINN benchmarking, including 16-QAM, 16-APSK, 16-PSK, and Star-QAM modulation schemes
# Last Updated: 20 Mar 2026

import numpy as np
import torch

from consolidate.config import ssfm_nlse, windowing

def gen_16qam(beta2, gamma, dt, n_steps, sld_win, L):
    points = np.array([-3, -1, 1, 3])
    re, im = np.meshgrid(points, points)
    const = (re + 1j*im).flatten()
    const /= np.sqrt(np.mean(np.abs(const)**2))
    clean = const[np.random.randint(0, 16, 6000)]
    clean = clean / np.sqrt(np.mean(np.abs(clean)**2))

    distorted = ssfm_nlse(clean, L, "forward", beta2, gamma, dt, n_steps) 
    baseline = ssfm_nlse(distorted, L, "reverse", beta2, gamma, dt, n_steps) # ssfm baseline for comparison

    scale_X = 1.0
    scale_Y = 1.0

    distorted_scaled = distorted / scale_X
    clean_scaled = clean / scale_Y 

    X_train = windowing(distorted_scaled, sld_win)
    Y_train = torch.tensor(np.stack([np.real(clean_scaled), np.imag(clean_scaled)], axis=1), dtype=torch.float32)

    return clean, distorted, baseline, X_train, Y_train, scale_X, scale_Y, clean_scaled, distorted_scaled

def gen_16apsk(beta2, gamma, dt, n_steps, sld_win, L):
    r1, r2 = 1.0, 2.53
    inner = r1 * np.exp(1j * np.linspace(0, 2*np.pi, 4, endpoint=False))
    outer = r2 * np.exp(1j * (np.linspace(0, 2*np.pi, 12, endpoint=False) + np.pi/12))
    const = np.concatenate([inner, outer])
    const /= np.sqrt(np.mean(np.abs(const)**2))

    clean = const[np.random.randint(0, len(const), 6000)]
    clean = clean / np.sqrt(np.mean(np.abs(clean)**2))

    distorted = ssfm_nlse(clean, L, "forward", beta2, gamma, dt, n_steps)
    baseline = ssfm_nlse(distorted, L, "reverse", beta2, gamma, dt, n_steps) # ssfm baseline for comparison

    scale_X = 1.0
    scale_Y = 1.0

    distorted_scaled = distorted / scale_X
    clean_scaled = clean / scale_Y

    X_train = windowing(distorted_scaled, sld_win)
    Y_train = torch.tensor(np.stack([np.real(clean_scaled), np.imag(clean_scaled)], axis=1), dtype=torch.float32)

    return clean, distorted, baseline, X_train, Y_train, scale_X, scale_Y, clean_scaled, distorted_scaled

def gen_16psk(beta2, gamma, dt, n_steps, sld_win, L):
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    const = np.exp(1j * angles)
    const /= np.sqrt(np.mean(np.abs(const)**2))  # already unit power

    clean = const[np.random.randint(0, len(const), 6000)]
    clean = clean / np.sqrt(np.mean(np.abs(clean)**2))

    distorted = ssfm_nlse(clean, L, "forward", beta2, gamma, dt, n_steps)
    baseline = ssfm_nlse(distorted, L, "reverse", beta2, gamma, dt, n_steps) # ssfm baseline for comparison

    scale_X = 1.0
    scale_Y = 1.0

    distorted_scaled = distorted / scale_X
    clean_scaled = clean / scale_Y

    X_train = windowing(distorted_scaled, sld_win)
    Y_train = torch.tensor(np.stack([np.real(clean_scaled), np.imag(clean_scaled)], axis=1), dtype=torch.float32)

    return clean, distorted, baseline, X_train, Y_train, scale_X, scale_Y, clean_scaled, distorted_scaled

def gen_star(beta2, gamma, dt, n_steps, sld_win, L):
    r1, r2 = 1.0, 2.0
    inner = r1 * np.exp(1j * np.linspace(0, 2*np.pi, 8, endpoint=False))
    outer = r2 * np.exp(1j * (np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8))
    const = np.concatenate([inner, outer])
    const /= np.sqrt(np.mean(np.abs(const)**2))

    clean = const[np.random.randint(0, len(const), 6000)]
    clean = clean / np.sqrt(np.mean(np.abs(clean)**2))

    distorted = ssfm_nlse(clean, L, "forward", beta2, gamma, dt, n_steps)
    baseline = ssfm_nlse(distorted, L, "reverse", beta2, gamma, dt, n_steps) # ssfm baseline for comparison

    scale_X = 1.0
    scale_Y = 1.0

    distorted_scaled = distorted / scale_X
    clean_scaled = clean / scale_Y

    X_train = windowing(distorted_scaled, sld_win)
    Y_train = torch.tensor(np.stack([np.real(clean_scaled), np.imag(clean_scaled)], axis=1), dtype=torch.float32)

    return clean, distorted, baseline, X_train, Y_train, scale_X, scale_Y, clean_scaled, distorted_scaled


def genSignals(type, beta2, gamma, dt, n_steps, sld_win, L):
    if type == "16qam":
        return gen_16qam(beta2, gamma, dt, n_steps, sld_win, L)
    elif type == "16apsk":
        return gen_16apsk(beta2, gamma, dt, n_steps, sld_win, L)
    elif type == "16psk":
        return gen_16psk(beta2, gamma, dt, n_steps, sld_win, L)
    elif type == "star":
        return gen_star(beta2, gamma, dt, n_steps, sld_win, L)
    else:
        raise ValueError("Unsupported signal type specified.")