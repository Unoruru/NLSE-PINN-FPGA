# Script to generate and save SSFM reference solution for hardware verification
# Last updated: 2026-02-28

import numpy as np
import pickle

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
# Generate SSFM reference solution
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

sd = (t, A_L, z_points, A_true)
with open("ssfm_results.pkl", "wb") as f:
    pickle.dump(sd, f)
print("Saved SSFM results to ssfm_results.pkl")