import numpy as np
from sklearn.metrics import r2_score

def plot_tracking(y, r, time=None):
    """
    Plots the reference signal r and the tracked output y, and computes tracking metrics.

    Parameters:
    - y: np.ndarray, tracked signal (output)
    - r: np.ndarray, reference signal (desired trajectory)
    - time: Optional time vector
    - label_r: Label for reference signal
    - label_y: Label for tracked signal
    - title: Title for the plot

    Returns:
    - metrics: dict containing relative_mae, rmse, r2
    """
    if time is None:
        time = np.arange(len(y))

    # --- Compute metrics ---
    abs_error = np.abs(y - r)
    # total_error = np.sum(abs_error)
    # total_reference = np.sum(np.abs(r)) + 1e-8  # Avoid division by zero

    # relative_mae = 1 - total_error / total_reference

    mrae = np.mean(np.abs((y - r) / r))

    rmse = np.sqrt(np.mean((y - r) ** 2))
    r2 = r2_score(r, y)

    return {
        'mrae': mrae,
        'rmse': rmse,
        'r2': r2
    }