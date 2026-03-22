# Script to evaluate the training loss and accuracy of the complex PINN model
# Last Updated: 22 Mar 2026

import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_perf(file_path="results/training_perf_metrics.pklv2"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading performance data from {file_path}: {e}")

    sig_type, losses, accuracies = data

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(losses, label="Training Loss")
    plt.semilogy(losses, label="Training Loss (Log Scale)")
    plt.title(f"{sig_type} Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.grid()
    plt.xlim(-100, len(losses))  # Set x-axis limits to the number of epochs
    plt.ylim(bottom=min(losses)*0.9)  # Set y-axis lower limit for log scale

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Training Accuracy (%)")
    plt.title(f"{sig_type} Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.xlim(-100, len(accuracies))  # Set x-axis limits to the number of epochs
    plt.ylim(0, 100)  # Set y-axis limits to 0-100% for accuracy

    plt.tight_layout()
    plt.savefig(f"results/{sig_type}_training_performance.png")

    return

def load_get(file_path="results/training_perf_metrics.pklv2"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading performance data from {file_path}: {e}")

    sig_type, losses, accuracies = data
    return sig_type, losses, accuracies

def write_save(file_path="results/training_perf_metrics.pklv2", sig_type=None, losses=None, accuracies=None):
    if sig_type is None or losses is None or accuracies is None:
        raise ValueError("Error: sig_type, losses, and accuracies must all be provided to save performance data.")

    data = (sig_type, losses, accuracies)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        raise ValueError(f"Error saving performance data to {file_path}: {e}")
    
def main():
    plot_perf()
    print("Performance metrics loaded and plots generated successfully.")

if __name__ == "__main__":
    main()