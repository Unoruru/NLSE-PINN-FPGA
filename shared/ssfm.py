"""
Split-Step Fourier Method (SSFM) for the Nonlinear Schrodinger Equation.
"""

import numpy as np


def ssfm_nlse(A0, t, beta2, gamma, L, n_steps):
    """
    Solve NLSE via symmetric split-step Fourier method:
        A_z = i * (beta2/2) * A_tt + i * gamma * |A|^2 * A

    Returns:
        A_final: complex field at z=L
        z_points: array of z positions
        snapshots: list of complex field at each z step
    """
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
