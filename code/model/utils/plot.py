# Utilities for plotting statistics

###### import libraries ######
# Standard libraries
# import os

# Data Handling
import numpy as np

# Visualization
import matplotlib.pyplot as plt


def save_temperature_error_histogram(
    errors: np.ndarray,
    save_path: str,
    temp_max: float = 80.0,
    title: str = 'Prediction Error Distribution'
):
    """
    Save histogram of prediction errors.
    
    Args:
        errors: Error values in normalized [0, 1] range
        save_path: Path to save plot
        temp_max: Max LST for Celsius conversion
        title: Plot title
    """
    # Convert to Celsius
    errors_c = errors * temp_max
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors_c, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    
    # Add statistics lines
    mean_err = errors_c.mean()
    median_err = np.median(errors_c)
    p95_err = np.percentile(errors_c, 95)
    
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}°C')
    ax.axvline(median_err, color='green', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}°C')
    ax.axvline(p95_err, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95_err:.2f}°C')
    
    ax.set_xlabel('Absolute Error (°C)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    

def save_temperature_prediction_scatter(
    targets: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    temp_max: float = 80.0,
    title: str = 'Latent Temperature predictor: Target vs Prediction'
):
    """
    Save scatter plot of target vs predicted LST values.
    
    Args:
        targets: Target values in normalized [0, 1] range
        predictions: Predicted values in normalized [0, 1] range
        save_path: Path to save plot
        temp_max: Max LST for Celsius conversion
        title: Plot title
    """
    # Convert to Celsius
    targets_c = targets * temp_max
    predictions_c = predictions * temp_max
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(targets_c, predictions_c, alpha=0.5, s=20, c='steelblue')
    
    # Perfect prediction line
    min_val = min(targets_c.min(), predictions_c.min())
    max_val = max(targets_c.max(), predictions_c.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Compute metrics
    mae = np.abs(targets_c - predictions_c).mean()
    rmse = np.sqrt(((targets_c - predictions_c) ** 2).mean())
    r2 = 1 - np.sum((targets_c - predictions_c) ** 2) / np.sum((targets_c - targets_c.mean()) ** 2)
    
    # Add metrics text
    metrics_text = f'MAE: {mae:.2f}°C\nRMSE: {rmse:.2f}°C\nR²: {r2:.3f}\nn={len(targets)}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Target Temperature p95 (°C)', fontsize=12)
    ax.set_ylabel('Predicted LST p95 (°C)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()