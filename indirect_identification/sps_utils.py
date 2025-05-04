from numba import njit, float64, prange
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import lfilter_numba
from dB.sim_db import SPSType
import numpy as np
from typing import Optional, Union, Tuple, Callable

all = [
    'get_U_perturbed_nhat',
    'compute_S',
    'get_construct_ss_from_params_method',
    'get_phi_method',
]
#---------------------------------------------------------------------------
# SPS mimo methods
#---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def get_U_perturbed_nhat(N_hat: np.ndarray, 
                         U_t: np.ndarray, 
                         alpha: np.ndarray, 
                         N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N_hat_par = N_hat[:, -N:]
    U_t_par = U_t[:, -N:]
    perturbed_N_hat = np.multiply(alpha, N_hat_par)
    return perturbed_N_hat, U_t_par, N_hat_par

@njit(cache=True, fastmath=True, parallel=True)
def compute_S(N_hat_, N_hat_perturbed_, phi_tilde_, N, Lambda=None):
    """
    Compute ranking matrix S using manual loops for matrix multiplications.
    Keep Cholesky decomposition with numpy.
    """
    #trim all the matrices to the last N elements
    N_hat = N_hat_[:, -N:]
    N_hat_perturbed = N_hat_perturbed_[:, :, -N:]
    phi_tilde = phi_tilde_[:, -N:, :, :]

    Lambda_inv, Lambda_n = compute_cov_matrices(N_hat, Lambda)
    Delta_lambda, Delta_lambda_n, Delta_lambda_Y = compute_phi_lambda_phiT_and_phi_lambda_Y(
                                                phi_tilde, Lambda_inv, Lambda_n, N_hat_perturbed)
    
    m, r, _ = Delta_lambda.shape
    S = np.zeros((m, ))

    for i in prange(m):
        # Step 1: Invert Delta_lambda[i] (still using numpy)
        Delta_inv = np.linalg.inv(Delta_lambda[i])
        Delta_lambda_n_i = Delta_lambda_n[i]

        # Step 2: temp = Delta_inv @ Delta_lambda_n[i]
        temp = np.zeros((r, r))
        for a in range(r):
            for b in range(r):
                for c in range(r):
                    temp[a, b] += Delta_inv[a, c] * Delta_lambda_n_i[c, b]

        # Step 3: R_ = temp @ Delta_inv
        R_ = np.zeros((r, r))
        for a in range(r):
            for b in range(r):
                for c in range(r):
                    R_[a, b] += temp[a, c] * Delta_inv[c, b]

        # Step 4: Cholesky decomposition using numpy
        W = np.linalg.cholesky(R_)

        # Step 5: S_i = W @ Delta_lambda_Y[i]
        S_i = np.zeros((r, 1))
        for a in range(r):
            for b in range(r):
                S_i[a, 0] += W[a, b] * Delta_lambda_Y[i, b, 0]

        # Step 6: Compute norm of S_i
        norm = 0.0
        for a in range(r):
            norm += S_i[a, 0] ** 2
        S[i] = norm

    return S


@njit(cache=True, fastmath=True)
def compute_cov_matrices(epsilon: np.ndarray, Lambda: Optional[np.ndarray] = None, shrink_factor=0.1, threshold=1e-5):
    d, N = epsilon.shape

    # Step 1: Compute Lambda (covariance matrix)
    # Lambda = 1 / (N-1) * epsilon_centered @ epsilon_centered.T
    if Lambda is None:
        # Step 1: Compute mean
        mean_epsilon = np.zeros(d)
        for i in range(d):
            for n in range(N):
                mean_epsilon[i] += epsilon[i, n]
            mean_epsilon[i] /= N

        # Step 2: Compute covariance and shrinkage together
        Lambda = np.zeros((d, d))
        for n in range(N):
            for i in range(d):
                centered_i = epsilon[i, n] - mean_epsilon[i]
                for j in range(d):
                    centered_j = epsilon[j, n] - mean_epsilon[j]
                    Lambda[i, j] += centered_i * centered_j
        Lambda /= (N - 1)
        for i in range(d):
            for j in range(d):
                if i == j:
                    # Keep diagonal as is
                    continue
                else:
                    # Shrink and threshold off-diagonals
                    Lambda[i, j] = (1.0 - shrink_factor) * Lambda[i, j]
                    if np.abs(Lambda[i, j]) < threshold:
                        Lambda[i, j] = 0.0

    # Step 2: Compute Lambda_n = 1/N * epsilon @ epsilon.T
    Lambda_n = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            sum_ = 0.0
            for n in range(N):
                sum_ += epsilon[i, n] * epsilon[j, n]
            Lambda_n[i, j] = sum_ / N

    # Step 3: Compute Lambda_inv
    Lambda_inv = np.linalg.inv(Lambda)

    return Lambda_inv, Lambda_n


@njit(cache=True, fastmath=True, parallel=True)
def compute_phi_lambda_phiT_and_phi_lambda_Y(phi_tilde, Lambda_inv, Lambda_n, N_hat_perturbed):
    """
    Computes:
      - Delta_lambda = phi @ Lambda_inv @ phi^T (averaged over T)
      - Delta_lambda_n = phi @ Lambda_inv @ Lambda_n @ Lambda_inv @ phi^T (averaged over T)
      - Delta_lambda_Y = phi @ Lambda_inv @ N_hat_perturbed (summed over T)
    """
    m, T, r, c = phi_tilde.shape
    Delta_lambda = np.zeros((m, r, r))  
    Delta_lambda_n = np.zeros((m, r, r)) 
    Delta_lambda_Y = np.zeros((m, r, 1)) 

    for k in prange(m):
        for t in range(T):
            phi = phi_tilde[k, t]  # (r, c)

            # Precompute phi_Lambda_inv
            phi_Lambda_inv = np.zeros((r, c))
            for i in range(r):
                for l in range(c):
                    for q in range(c):
                        phi_Lambda_inv[i, l] += phi[i, q] * Lambda_inv[q, l]

            # Compute Delta_lambda and Delta_lambda_n
            for i in range(r):
                for j in range(r):
                    sum1 = 0.0
                    sum2 = 0.0
                    for l in range(c):
                        sum1 += phi_Lambda_inv[i, l] * phi[j, l]
                        for p in range(c):
                            sum2 += phi_Lambda_inv[i, l] * Lambda_n[l, p] * phi_Lambda_inv[j, p]
                    Delta_lambda[k, i, j] += sum1
                    Delta_lambda_n[k, i, j] += sum2

            # Compute Delta_lambda_Y = phi_Lambda_inv @ N_hat_perturbed
            for i in range(r):
                acc = 0.0
                for l in range(c):
                    acc += phi_Lambda_inv[i, l] * N_hat_perturbed[k, l, t]
                Delta_lambda_Y[k, i, 0] += acc

    # Average over time T
    Delta_lambda /= T
    Delta_lambda_n /= T
    Delta_lambda_Y /= T

    return Delta_lambda, Delta_lambda_n, Delta_lambda_Y



def get_construct_ss_from_params_method(n_states: int, n_inputs: int, n_outputs: int, C: np.ndarray):
    """
    Returns the function to construct state space matrices from parameters.
    """

    @njit(cache=True)
    def _construct_ss_from_params(params: np.ndarray):
        """
        Returns state space matrices A_obs, B_obs, C_obs, D_obs and the A, B polynomials.
        """
        # Extract A parameters
        A = params[:n_states]
        # A_obs: Observable canonical form of A
        top_block = np.zeros((1, n_states - 1))
        bottom_block = np.eye(n_states - 1)
        A_obs_left = np.vstack((top_block, bottom_block))
        A_obs_right = -np.flipud(A.reshape(-1, 1))
        A_obs = np.hstack((A_obs_left, A_obs_right))

        # Extract B parameters and build B_obs
        B = params[n_states:n_states + n_states * n_inputs].reshape(n_inputs, n_states)
        B_obs = np.flipud(B.T)

        # C and D matrices
        C_obs = C
        D_obs = np.zeros((n_outputs, n_inputs))

        # Polynomial form
        A_poly = np.hstack((np.array([1.0]), A))  # Ensure array concatenation
        B_poly = np.hstack((np.zeros((n_inputs, 1)), B))

        if n_inputs == 1 and n_outputs == 1:
            A_poly = A_poly.flatten()
            B_poly = B_poly.flatten()

        return A_obs, B_obs, C_obs, D_obs, A_poly, B_poly

    return _construct_ss_from_params



#---------------------------------------------------------------------------
# phi methods
#---------------------------------------------------------------------------

def get_phi_method(n_inputs: int, n_outputs: int, n_noise: int):
    if n_inputs == 1 and n_outputs==1:
        if n_noise==-1:
            return create_phi_optimized_siso
        else:
            return create_phi_optimized_general_siso
    elif n_inputs==1 and n_outputs == 2 and n_noise==-1:
        # full state observation of 2 state system, 1 input
        return create_phi_optimized_2states_1input
    else:
        return NotImplementedError


@njit(cache=True)
def create_phi_optimized_siso(Y: np.ndarray, U: np.ndarray, A: np.ndarray, B: np.ndarray, C=None) -> np.ndarray:
    """
    Create the phi matrix for perturbed output (Y: m x t) and single input (U: t).
    
    Parameters:
    Y (array): Output matrix of shape (m, t).
    U (array): Input vector of shape (t,).
    len_a (int): len of A polynomial
    len_b (int): len of B polynomial
    c (float): c0 of the C polynomial.
    
    Returns:
    array: Phi matrix of shape (m, t, n_a + n_b).
    """
    n_a=len(A)-1
    n_b=len(B)-1
    m, t = Y.shape
    phi = np.zeros((m, t, n_a + n_b), dtype=Y.dtype)
    cl = U.ndim == 2
    for j in range(m):  # for each output dimension
        for lag in range(1, n_a + 1):
            for i in range(lag, t):
                phi[j, i, lag - 1] = Y[j, i - lag] 

    for lag in range(1, n_b + 1):
        for i in range(lag, t):
            for j in range(m):  # input is shared across outputs
                if cl:
                    _U = U[j, i-lag]
                else:
                    _U = U[i - lag]
                phi[j, i, n_a + lag - 1] = -_U

    return phi





@njit(cache=True, fastmath=True)
def create_phi_optimized_general_siso(Y, U, A, B, C):
    m, t = Y.shape  # m perturbation, t time steps
    n_a = A.size - 1
    n_b = B.size - 1
    n_c = C.size

    idx_b = n_a
    idx_c = n_a + n_b
    total_phi_len = n_a + n_b + n_c

    phi = np.zeros((m, t, total_phi_len), dtype=np.float64)

    ones = np.empty(1, dtype=np.float64)
    ones[0] = 1.0


    for j in range(m):  # loop over each output dimension

        # Input filtered once (shared across outputs)
        u = U[j]
        filtered_U = lfilter_numba(ones, C, u)
        B_U = lfilter_numba(B, C, filtered_U)
        
        y = Y[j]

        filtered_Y = lfilter_numba(ones, C, y)
        A_Y = lfilter_numba(A, C, filtered_Y)
        eps_t = A_Y - B_U

        for i in range(t):
            if i > 0:
                for lag in range(1, n_a + 1):
                    if i - lag >= 0:
                        phi[j, i, lag - 1] = filtered_Y[i - lag]
                for lag in range(1, n_b + 1):
                    if i - lag >= 0:
                        phi[j, i, idx_b + lag - 1] = -filtered_U[i - lag]
            for lag in range(n_c):
                if i - lag >= 0:
                    phi[j, i, idx_c + lag] = -eps_t[i - lag]

    return phi

@njit(cache=True, fastmath=True)
def create_phi_optimized_2states_1input(Y, U, A, B, C):
    m, _, t = Y.shape
    U_eff  = U[:,0,:]
    
    
    # Preallocate J once with the correct size
    J = np.zeros((m, t, 2, 4), dtype=np.float64)
    
    # Cache commonly used values from A and B
    a1, a2 = A[1], A[2]
    b1, b2 = B[0, 1], B[0, 2]

    for k in range(m):
        Y1 = Y[k, 0]
        Y2 = Y[k, 1]
        Uk = U_eff[k]  

        # Handle i = 1 separately (no im2 index)
        if t > 1:
            Jk0 = J[k, 1, 0]
            Jk1 = J[k, 1, 1]

            Jk0[0] = Y1[0]
            Jk0[2] = -Uk[0]
            Jk1[0] = Y2[0]
            Jk1[3] = -Uk[0]

        # Loop for i = 2 to t
        for i in range(2, t):
            im1, im2 = i - 1, i - 2
            Jk0 = J[k, i, 0]
            Jk1 = J[k, i, 1]

            # Reusing values instead of multiple calls to Y and U
            y1_im1, y1_im2 = Y1[im1], Y1[im2]
            y2_im1, y2_im2 = Y2[im1], Y2[im2]
            u_im1, u_im2 = Uk[im1], Uk[im2]

            # Assign values to Jk0 and Jk1 directly without intermediate variables
            Jk0[0] = y1_im1
            Jk0[1] = y1_im2
            Jk0[2] = -u_im1
            Jk0[3] = -u_im2

            Jk1[0] = y2_im1 - b2 * u_im2
            Jk1[1] = y2_im2 + b1 * u_im2
            Jk1[2] = a2 * u_im2
            Jk1[3] = -(u_im1 + a1 * u_im2)

    return J.transpose(0, 1, 3, 2)





