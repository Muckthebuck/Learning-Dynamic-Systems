import numpy as np
from scipy.linalg import solve_discrete_lyapunov, eigvals

def compute_worst_case_J(K, A_in_set, B_in_set, Q, R, x_0=None):
    max_J = -np.inf
    for A, B in zip(A_in_set, B_in_set):
        max_J = max(calc_J(A, B, K, Q, R, x_0), max_J)
    return max_J

def calc_J(A, B, K, Q, R, x_0=None):
    stability_radius = 1 - 1e-6  # Slightly stricter stability criterion
    Phi = A - B @ K
    Q_tilde = Q + K.T @ (R * K) if np.isscalar(R) else Q + K.T @ R @ K

    if np.all(np.abs(eigvals(Phi)) < stability_radius):
        S = solve_discrete_lyapunov(Phi.T, Q_tilde)
        J = np.trace(S) if x_0 is None else x_0.T @ S @ x_0 
        return J.item()
    else:
        return np.inf