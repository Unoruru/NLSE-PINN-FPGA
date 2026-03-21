# Script containing helper functions for signal processing and evaluation in ComplexPINN benchmarking
# Last Updated: 20 Mar 2026

import numpy as np
import argparse

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

def calculate_ser(clean_indices, recovered_indices):
    """Calculates the percentage of incorrectly identified symbols."""
    errors = np.sum(clean_indices != recovered_indices)
    return (errors / len(clean_indices)) * 100

def synchronize_signals(ref, target):
    """
    Finds the time delay between two signals and aligns them.
    This is critical when comparing windowed neural network outputs.
    """
    # 1. Cross-correlation to find the delay
    correlation = np.correlate(np.abs(target), np.abs(ref), mode='full')
    delay = np.argmax(correlation) - (len(ref) - 1)
    
    # 2. Shift the signals to match
    if delay > 0:
        # Target is delayed
        ref_sync = ref[:-delay]
        target_sync = target[delay:]
    elif delay < 0:
        # Target is advanced (rare, but possible with padding)
        ref_sync = ref[-delay:]
        target_sync = target[:delay]
    else:
        ref_sync, target_sync = ref, target
        
    return ref_sync, target_sync

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')