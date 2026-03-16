# Script for evaluating the outputs from the accelerator for complex PINN
# Last Updated: 16 Mar 2026

import os
import sys # <-- ADDED MISSING IMPORT
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# Helper functions
def to_unit_power(sig):
    rms = np.sqrt(np.mean(np.abs(sig)**2))
    return sig / rms if rms > 0 else sig

def align_signal(ref, target):
    """Normalizes power and rotates the grid to align with reference."""
    ref_n = to_unit_power(ref)
    target_n = to_unit_power(target)
    # Find optimal rotation angle
    angle = np.angle(np.mean(target_n * np.conj(ref_n)))
    target_aligned = target_n * np.exp(-1j * angle)
    return ref_n, target_aligned

def evm(ref, target):
    return np.sqrt(np.mean(np.abs(ref - target)**2) / np.mean(np.abs(ref)**2)) * 100

def classify_16qam(signal):
    """Maps continuous complex signals to the nearest discrete 16-QAM points."""
    levels = np.array([-3, -1, 1, 3])
    ideal_points = np.array([re + 1j*im for re in levels for im in levels])
    ideal_points /= np.sqrt(np.mean(np.abs(ideal_points)**2))
    
    distances = np.abs(signal[:, np.newaxis] - ideal_points)
    closest_indices = np.argmin(distances, axis=1)
    
    return ideal_points[closest_indices], closest_indices

def calculate_ser(clean_indices, recovered_indices):
    """Calculates the percentage of incorrectly identified symbols."""
    errors = np.sum(clean_indices != recovered_indices)
    return (errors / len(clean_indices)) * 100

def synchronize_signals(ref, target):
    """Finds the time delay between two signals and aligns them."""
    correlation = np.correlate(np.abs(target), np.abs(ref), mode='full')
    delay = np.argmax(correlation) - (len(ref) - 1)
    
    if delay > 0:
        ref_sync = ref[:-delay]
        target_sync = target[delay:]
    elif delay < 0:
        ref_sync = ref[-delay:]
        target_sync = target[:delay]
    else:
        ref_sync, target_sync = ref, target
        
    return ref_sync, target_sync

def main():

    inputs_save_path = "generated_inputs.pkl"

    try:
        with open(inputs_save_path, "rb") as f:
            clean, distorted, baseline, X_train, Y_train, clean_scaled, distorted_scaled = pickle.load(f)
            print(f"Loaded training data and inputs from {inputs_save_path} for evaluation.")
    except Exception as e:
        print(f"Failed to load inputs: {e}. Ensure the file exists and is a valid .pkl file. Exiting...")
        sys.exit()

    outputs_path = "output_0.npy"
    try:
        outputs = np.load(outputs_path)
        print("Loaded FPGA hardware outputs from file.")
    except Exception as e:
        print(f"Error: {e}") # <-- FIXED EXCEPTION PRINTING
        sys.exit()
    
    # dequantization
    # 1. Replace this with the final_scale you extracted from Brevitas!
    FINAL_SCALE = 0.00345  
    # 2. Un-quantize integers to floats
    outputs_float = outputs * FINAL_SCALE
    # 3. Combine Real (col 0) and Imag (col 1) into a 1D complex array
    outputs_complex = outputs_float[:, 0] + 1j * outputs_float[:, 1]

    # evm
    clean_sync, out_sync = synchronize_signals(clean, outputs_complex) # <-- PASSED COMPLEX ARRAY
    clean_align, out_align = align_signal(clean_sync, out_sync)
    out_evm = evm(clean_align, out_align)
    
    # ser
    _, clean_classify = classify_16qam(clean_align)
    _, out_classify = classify_16qam(out_align)
    out_ser = calculate_ser(clean_classify, out_classify)
    
    # <-- FIXED STRING FORMATTING
    print(f"EVM = {out_evm:.2f}% | SER = {out_ser:.2f}%") 
    
    # draw constellation diagram
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.scatter(out_align.real, out_align.imag, s=1, alpha=0.5) # <-- FIXED AXES INDEXING
    ax.set_title("8-bit PINN on Accelerator")
    ax.set_aspect('equal')
    plt.savefig("out.png")
    print("Saved constellation diagram to out.png")
        
if __name__ == "__main__":
    main()
    