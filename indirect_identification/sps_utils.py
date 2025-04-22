from numba import njit, float64
import numpy as np


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
def _construct_ss_from_params(params: np.array, n_states: int, n_inputs: int, n_outputs: int, C: np.array):
    """
    Returns state space matrices A_obs,B_obs,C_obs,D_obs and the A,B polynomials
    """
    # A: n_state x n_state matrix
    A =  params[:n_states]
    A_obs = np.hstack([np.vstack([np.zeros(n_states-1), np.eye(n_states-1)]), -np.flipud(A.reshape(A.size,-1))])
    # B: n_state x n_input matrix
    B = params[n_states:n_states+n_states*n_inputs].reshape(n_inputs,n_states)
    B_obs = np.flipud(B.T)
    # C: n_output x n_state matrix
    C_obs = C
    # D: n_output x n_input matrix: zero matrix for now
    D_obs = np.zeros((n_outputs,n_inputs))

    A = np.hstack([1, A])
    B = np.hstack([np.zeros((n_inputs,1)), B])

    return A_obs, B_obs, C_obs, D_obs, A, B


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

    for j in range(m):  # for each output dimension
        for lag in range(1, n_a + 1):
            for i in range(lag, t):
                phi[j, i, lag - 1] = -Y[j, i - lag] 

    for lag in range(1, n_b + 1):
        for i in range(lag, t):
            for j in range(m):  # input is shared across outputs
                phi[j, i, n_a + lag - 1] = U[i - lag] 

    return phi

@njit(cache=True)
def lfilter_numba(b, a, x):
    N = len(a)
    M = len(b)
    n = len(x)

    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    y = np.zeros(n)

    for i in range(n):
        for j in range(M):
            if i - j >= 0:
                y[i] += b[j] * x[i - j]
        for j in range(1, N):
            if i - j >= 0:
                y[i] -= a[j] * y[i - j]

    return y



@njit(cache=True, fastmath=True)
def create_phi_optimized_general_siso(Y, U, A, B, C):
    m, t = Y.shape  # m perturbation, t time steps
    n_a = len(A) - 1
    n_b = len(B) - 1
    n_c = len(C)

    idx_b = n_a
    idx_c = n_a + n_b
    total_phi_len = n_a + n_b + n_c

    phi = np.zeros((m, t, total_phi_len), dtype=np.float64)

    ones = np.empty(1, dtype=np.float64)
    ones[0] = 1.0

    # Input filtered once (shared across outputs)
    filtered_U = lfilter_numba(ones, C, U)
    B_U = lfilter_numba(B, C, filtered_U)

    for j in range(m):  # loop over each output dimension
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
    U = U[0,:]
    B = B[0, :]
    J = np.zeros((m, t, 2, 4), dtype=np.float64)

    a1= A[1]
    a2= A[2]
    b1= B[1]
    b2= B[2]

    for k in range(m):
        Y1 = Y[k, 0]
        Y2 = Y[k, 1]

        # i = 1
        if t > 1:
            i = 1
            im1 = i - 1
            Jk0 = J[k, i, 0]
            Jk1 = J[k, i, 1]

            y1_im1 = Y1[im1]
            y2_im1 = Y2[im1]
            u_im1 = U[im1]

            Jk0[0] = y1_im1
            Jk0[2] = -u_im1
            Jk1[0] = y2_im1
            Jk1[3] = -u_im1

        # i = 2 to t
        for i in range(2, t):
            im1 = i - 1
            im2 = i - 2
            Jk0 = J[k, i, 0]
            Jk1 = J[k, i, 1]

            y1_im1 = Y1[im1]
            y1_im2 = Y1[im2]
            y2_im1 = Y2[im1]
            y2_im2 = Y2[im2]
            u_im1 = U[im1]
            u_im2 = U[im2]

            # Row 0
            Jk0[0] = y1_im1
            Jk0[1] = y1_im2
            Jk0[2] = -u_im1
            Jk0[3] = -u_im2

            # Row 1
            Jk1[0] = y2_im1 - b2 * u_im2
            Jk1[1] = y2_im2 + b1 * u_im2
            Jk1[2]= a2 * u_im2
            Jk1[3]= -(u_im1 + a1 * u_im2)

    return J.transpose(0,1,3,2)