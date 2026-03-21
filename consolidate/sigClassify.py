# Script to classify continuous complex signals to nearest discrete constellation points for 16-QAM, 16-APSK, 16-PSK, and Star-QAM modulation schemes
# Last Updated: 20 Mar 2026

import numpy as np

def classify_16qam(signal):
    """Maps continuous complex signals to the nearest discrete 16-QAM points."""
    levels = np.array([-3, -1, 1, 3])
    ideal_points = np.array([re + 1j*im for re in levels for im in levels])
    ideal_points /= np.sqrt(np.mean(np.abs(ideal_points)**2))
    distances = np.abs(signal[:, np.newaxis] - ideal_points)
    closest_indices = np.argmin(distances, axis=1)
    
    return ideal_points[closest_indices], closest_indices

def classify_16apsk(signal):
    """Maps continuous complex signals to the nearest discrete 16-APSK points."""
    r1, r2 = 1.0, 2.53
    inner = r1 * np.exp(1j * np.linspace(0, 2*np.pi, 4, endpoint=False))
    outer = r2 * np.exp(1j * (np.linspace(0, 2*np.pi, 12, endpoint=False) + np.pi/12))
    ideal_points = np.concatenate([inner, outer])
    ideal_points /= np.sqrt(np.mean(np.abs(ideal_points)**2))
    distances = np.abs(signal[:, np.newaxis] - ideal_points)
    closest_indices = np.argmin(distances, axis=1)

    return ideal_points[closest_indices], closest_indices

def classify_16psk(signal):
    """Maps continuous complex signals to the nearest discrete 16-PSK points."""
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    ideal_points = np.exp(1j * angles)
    ideal_points /= np.sqrt(np.mean(np.abs(ideal_points)**2))
    distances = np.abs(signal[:, np.newaxis] - ideal_points)
    closest_indices = np.argmin(distances, axis=1)

    return ideal_points[closest_indices], closest_indices

def classify_star_qam(signal):
    """Maps continuous complex signals to the nearest discrete Star-QAM points."""
    r1, r2 = 1.0, 2.0
    inner = r1 * np.exp(1j * np.linspace(0, 2*np.pi, 8, endpoint=False))
    outer = r2 * np.exp(1j * (np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8))
    ideal_points = np.concatenate([inner, outer])
    ideal_points /= np.sqrt(np.mean(np.abs(ideal_points)**2))
    distances = np.abs(signal[:, np.newaxis] - ideal_points)
    closest_indices = np.argmin(distances, axis=1)

    return ideal_points[closest_indices], closest_indices

def classify(signal, modulation_type):
    """Classifies continuous complex signals to the nearest discrete constellation points based on modulation type."""
    if modulation_type == '16qam':
        return classify_16qam(signal)
    elif modulation_type == '16apsk':
        return classify_16apsk(signal)
    elif modulation_type == '16psk':
        return classify_16psk(signal)
    elif modulation_type == 'star':
        return classify_star_qam(signal)
    else:
        raise ValueError(f"Unsupported modulation type: {modulation_type}")