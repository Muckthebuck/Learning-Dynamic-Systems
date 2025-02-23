import os
from dtfs import d_tfs
from armax import ARMAX

os.environ['CUPY_ACCELERATORS'] = 'cub'

import os
import torch
try:
    torch.cuda.current_device()
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    device = 'cuda'
except:
    # Fall back to unoptimised versions
    import numpy as cp
    from scipy.linalg import solve_triangular
    from scipy import signal
    device = None

class SPS_direct_model:
    def __init__(self, m, q, N=50):
        self.N = N
        self.m = m
        self.q = q
        self.alpha = cp.random.randn(m,N)
        self.alpha = cp.sign(self.alpha)
        self.alpha[0, :] = 1
        self.pi_order = cp.random.permutation(cp.arange(m))

    def create_phi_optimized(self, Y, U, n_a, n_b):
        m, t = Y.shape
        
        # Initialize phi with zeros
        phi = cp.zeros((m, t, n_a + n_b), dtype=Y.dtype)
        
        # Handle Y lags
        for i in range(1, n_a + 1):
            phi[:, i:, i-1] = -1*Y[:, :-i]
        
        # Handle U lags
        for i in range(1, n_b + 1):
            phi[:, i:, n_a+i-1] = U[:-i]
        
        return phi
    
    # Returns (u_bar, Y_tilde) tuple
    def direct_sps(self, G, H, F, L, Y_t, R_t, n_a, n_b):
        G = d_tfs(G)
        H = d_tfs(H)
        F = d_tfs(F)
        L = d_tfs(L)
        
        try:
            Y_t = cp.asarray(Y_t)
            R_t = cp.asarray(R_t)

            # Regenerate the inputs
            N_hat = self.calculate_Nt(G, H, F, L, Y_t, R_t)
            N_hat_par = N_hat[-self.N:]

            # Perturb the inputs
            N_hat_bar = cp.multiply(self.alpha, N_hat_par)

            U_bar = self.calculate_U_bar(F, L, Y_t, R_t)
            U_bar_par = U_bar[-self.N-1:-1]
            
            Y_tilde = self.calculate_Y_bar(G, H, F, L, U_bar_par, N_hat_bar)
            Y_tilde = cp.asarray(Y_tilde)
            phi_tilde = self.create_phi_optimized(Y_tilde, U_bar_par, n_a, n_b)
            
            # Calculate SPS with new system inputs

            S = cp.zeros(self.m)
            # This loop could be optimised using method in indirect.open_loop_sps_2
            for i in range(self.m):
                phi_col = phi_tilde[i]
                R_i = cp.matmul(phi_col.T, phi_col) / Y_tilde.shape[1]
                L = cp.linalg.cholesky(R_i)  # Cholesky decomposition to get lower-triangular L
                R_root_inv = solve_triangular(L, cp.eye(L.shape[0]), lower=True)  # Compute L_inv
                weighted_sum = cp.matmul(phi_tilde[i].T,N_hat_bar[i].reshape(-1,1))
                # norm squared
                S[i] = cp.sum(cp.square(cp.matmul(R_root_inv, weighted_sum)))

            combined = cp.array(list(zip(self.pi_order, S)))
            order= cp.lexsort(combined.T)
            rank_R = cp.where(order == 0)[0][0] + 1
            return rank_R <= self.m - self.q , S

        except Exception as e:
            print(e)
            pass

    def calculate_Nt(self, G: d_tfs, H: d_tfs, F: d_tfs, L: d_tfs, Y_t: cp.array, R_t: cp.array):
        # N_hat = H^-1*[(1+G*F)*Y_t - G*L*R_t]
        try:

            GLR = G.apply_shift_operator(L.apply_shift_operator(R_t))
            
            YGU = Y_t + F.apply_shift_operator(G.apply_shift_operator(Y_t)) - GLR
            N_hat = (~H).apply_shift_operator(YGU)

            return N_hat
        except Exception as e:
            raise ValueError(f"Error in direct SPS: {e}")

    def calculate_U_bar(self, F: d_tfs, L: d_tfs, Y_t: cp.array, R_t: cp.array):
        # U_bar = L*R_t - FY_bar
        try:

            U_bar = L.apply_shift_operator(R_t) - F.apply_shift_operator(Y_t)

            return U_bar
        except Exception as e:
            raise ValueError(f"Error in direct SPS: {e}")

    def calculate_Y_bar(self, G: d_tfs, H: d_tfs, F: d_tfs, L: d_tfs, U_bar: cp.array, N_t: cp.array):
        try:            
            Y_bar = [
                G.apply_shift_operator(U_bar) + (~H).apply_shift_operator(N_t[i])
                for i in range(self.m)
            ]

            return Y_bar
        except Exception as e:
            raise ValueError(f"Error in direct SPS: {e}")
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # Example usage
    A = [1, -0.33]  # A(z^-1) = 1 - 0.33z^-1
    B = [0.22]      # B(z^-1) = 0.22z^-1
    C = [1, 0.15]   # C(z^-1) = 1 + 0.15z^-1
    F = [0.31, 0.23] # F(z^-1) = 0.31 + 0.23z^-1
    L = [1]        # L(z^-1) = 1

    N_SAMPLES = 100
    armax_model = ARMAX(A, B, C, F, L)

    # square wave reference signal
    R = signal.square(cp.linspace(0, 10*cp.pi, N_SAMPLES))

    Y, U, N, R = armax_model.simulate(N_SAMPLES, R, noise_std=0.2)

    

    G = (B, A)
    H = (C, A)
    F = (F, [1])
    L = (L, [1])

    m = 100
    q = 5
    T = N_SAMPLES  # Example sample size

    import cProfile, pstats
    profiler = cProfile.Profile()

    direct_model = SPS_direct_model(m, q)

    C = cp.array([1, 0.15])         # C(z^-1) = 1 + 0.15z^-1
    F = (cp.array([0.31, 0.23]), cp.array([1]))  # F(z^-1) = 0.31 + 0.23z^-1
    L = (cp.array([1]), cp.array([1]))         # L(z^-1) = 1

    direct_model = SPS_direct_model(m, q)

    # Create arrays for a and b values
    a_values = cp.arange(0, 0.6, 0.02)
    b_values = cp.arange(0, 0.6, 0.02)

    # Create a meshgrid for a and b values
    a_grid, b_grid = cp.meshgrid(a_values, b_values, indexing='ij')

    # Flatten the grids for iteration
    a_flat = a_grid.ravel()
    b_flat = b_grid.ravel()

    # Store results in a list
    results = []

    # Create torch tensors directly on GPU
    a_torch = torch.tensor(a_flat, dtype=torch.float32, device=device)
    b_torch = torch.tensor(b_flat, dtype=torch.float32, device=device)

    # Vectorized operations to minimize Python loops
    A_torch = torch.stack([torch.ones_like(a_torch), -a_torch], dim=-1)
    B_torch = torch.stack([torch.zeros_like(b_torch), b_torch], dim=-1)

    # Perform the operations in batch
    A = cp.from_dlpack(A_torch)
    B = cp.from_dlpack(B_torch)

    # Assuming 'model' is predefined and contains the necessary methods
    for i in range(len(a_flat)):
        G = (B[i], A[i])  # G should be a tuple of arrays
        H = (C, A[i])  # H should be a tuple of arrays

        # Check the condition and store the result if true
        in_sps, S1 = direct_model.direct_sps(G, H, F, L, Y, U, 1, 1)
        if in_sps:
            results.append((a_flat[i].item(), b_flat[i].item()))

    # Convert the results to a NumPy array
    results = cp.array(results)
    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(results[:, 0], results[:, 1], 'bo')  # Plot the points as blue dots

    # Labeling the plot
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    # axis limits
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([-0.1, 1])
    ax.set_title('Points for which closed_loop_sps returns True')

    # Plot the true values
    ax.plot(0.33, 0.22, 'ro')  # Plot the true values as a red dot

    # Show the plot
    plt.show()
