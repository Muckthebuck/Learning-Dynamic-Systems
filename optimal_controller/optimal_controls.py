import numpy as np
import cupy as cp
import scipy
from numpy.linalg import LinAlgError

def dlqr(A, B, Q, R):
    """Solves the infinte horizon problem for discrete state-space system (A,B), with costs defined by (Q,R). Returns the optimal full-state feedback controller K and Riccati solution P."""
    # ... ensure numpy arrays, required for scipy
    try:
        A = cp.asnumpy(A)
        B = cp.asnumpy(B)
        Q = cp.asnumpy(Q)
        R = cp.asnumpy(R)
    except AttributeError:
        pass  # already an np.ndarray
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    if np.isscalar(A):
        K = (R + B * P * B)**-1 * (B * P * A)
        K = K.squeeze()
        P = P.squeeze()
    else:
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def dlqr_custom(A, B, Q, R, tol=1e-6, max_iters=1000):
    """
    Solves the discrete-time algebraic Riccati equation iteratively and computes the optimal state feedback gain.
    See https://au.mathworks.com/help/control/ref/lti.dlqr.html#mw_6e129893-5cab-4a33-a5b8-d8b145402851 for details.
    """

    if np.ndim(A) == 0:
        # scalar eqns
        p = Q
        a = A
        b = B
        q = Q
        r = R
        
        for i in range(max_iters):
            p_next = a * p * a - (a * p * b) * (r + b * p * b)**-1 * (b * p * a) + q
            
            if np.abs(p_next - p) < tol:
                break
            
            p = p_next
        else:
            raise LinAlgError('Unable to iteratively solve the Riccati equation.')

        # 
        k = (r + b * p * b)**-1 * b * p * a
        return k, p
    else:
        # matrix eqns
        P = Q.copy()
        for i in range(max_iters):
            P_next = A.T @ P @ A - (A.T @ P @ B) @ np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
            
            # if np.linalg.norm(P_next - P, 'fro') < tol:
            if np.max(np.abs(P_next - P)) < tol:
                break
            
            P = P_next
        else:
            raise LinAlgError('Unable to iteratively solve the Riccati equation.')

        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K, P

def get_optimal_controller_s(a_in_set,b_in_set,q,r):
    """
    Computes the min-max optimal controller, for a first order (scalar) system.
    """
    
    k_set = []
    J_set = []

    for i in range(len(a_in_set)):

        # load candidate system params (a,b)
        a = a_in_set[i]
        b = b_in_set[i]

        k, _ = dlqr_custom(a,b,q,r)
        J = (q + r*k**2) / (1 - (a-b*k)**2)

        k_set.append(k)
        J_set.append(J.item())

    # evaluate the optimal K
    max_J_idx = np.argmax(np.array(J_set))
    max_J_idx = int(max_J_idx.item())
    optimal_k = k_set[max_J_idx]

    return optimal_k

def get_optimal_controller_v_version1(A_in_set,B_in_set,Q,R,x_0):
    """
    Finds optimal controller, cost calculations based on solving a Lyapunov equation.
    """

    K_set = []
    J_set = []

    for i in range(len(A_in_set)):

        # load candidate system matrices (A,B)
        A = A_in_set[i,:]
        B = B_in_set[i,:]

        # assuming this (A,B), what is the optimal controller K?
        K, _ = dlqr_custom(A, B, Q, R)
        K_set.append(K)

        # what is the associated cost J?
        Phi = A - B @ K
        Q_tilde = Q + K.T @ R @ K
        # ... ensure they're numpy arrays, required for scipy's solve_discrete_lyapunov function
        try:
            Phi = cp.asnumpy(Phi)  
            Q_tilde = cp.asnumpy(Q_tilde)
            x_0 = cp.asnumpy(x_0)
        except AttributeError:
            pass  # already an np.ndarray
        P = scipy.linalg.solve_discrete_lyapunov(Phi.T, Q_tilde)
        J = (x_0.T @ P @ x_0).item()
        J_set.append(J)
    
    # evaluate the optimal K
    max_J_idx = np.argmax(np.array(J_set)) # this will be a scalar, but still stored as 0-dimensional ndarray
    max_J_idx = int(max_J_idx.item()) # ensure it is an int so to index with it
    optimal_K = K_set[max_J_idx]
    return optimal_K

def get_optimal_controller_v_version2(A_in_set,B_in_set,Q,R,x_0):
    """
    Finds optimal controller by evaluating cost using Riccati solution.
    """
    K_set = []
    J_set = []

    for i in range(len(A_in_set)):

        # load candidate system matrices (A,B)
        A = A_in_set[i,:]
        B = B_in_set[i,:]

        # assuming this (A,B), what is the optimal controller K, and associated cost J?
        K, P = dlqr_custom(A, B, Q, R)
        K_set.append(K)
        J = (x_0.T @ P @ x_0).item()
        J_set.append(J)
    
    # evaluate the optimal K
    max_J_idx = np.argmax(np.array(J_set)) # this will be a scalar, but still stored as 0-dimensional ndarray
    max_J_idx = int(max_J_idx.item()) # convert 0-d array to int
    optimal_K = K_set[max_J_idx]
    return optimal_K

def get_optimal_controller(A_in_set,B_in_set,Q,R,x_0=None):
    """
    Returns the controller K that is min-max optimal for the set of plants. NB: (A,B) are state-space matrices, not transfer function coefficients. 
    """

    if np.ndim(A_in_set[0]) == 0:
        # scalar eqns
        return get_optimal_controller_s(A_in_set,B_in_set,Q,R)

    else:
        # matrix eqns
        assert x_0 is not None
        n = len(A_in_set[0])
        assert len(B_in_set[0]) == n
        assert len(Q) == n
        assert len(x_0) == n
        return get_optimal_controller_v_version2(A_in_set,B_in_set,Q,R,x_0)

def tf_to_ccf(num,den):
    B = np.array(num, copy=True)
    A = np.array(den, copy=True)
    assert B[0] == 0 # must be strictly proper transfer function

    # Ensure both polynomials have the same length by padding with zeros at the back
    if len(B) > len(A):
        A = np.pad(A, (0, len(B) - len(A)))
    elif len(B) < len(A):
        B = np.pad(B, (0, len(A) - len(B)))

    # Ensure monic denominator (coefficient associated with z^0 should be 1)
    B = np.divide(B, A[0])
    A = np.divide(A, A[0])

    # Now create the state space matrices
    n = len(A) - 1 # order of the system i.e. number of states
    ss_A = np.vstack([np.hstack([np.zeros((n-1, 1)), np.eye(n-1)]), -np.flip(A[1:])])
    ss_B = np.zeros((n, 1))
    ss_B[n-1] = 1  # Last element is 1
    ss_C = np.flip(B)[0:n]
    ss_C = ss_C.reshape(1,n)

    return ss_A, ss_B, ss_C

def tf_to_ocf(num,den):
    """
    Returns the A, B, C matrices for the Observer Canonical Form (OCF) of a strictly proper transfer function.
    num: numerator coefficients, in descending powers of z e.g. [0 1 0.5] = 0 + z^-1 + 0.5z^-2
    den: denominator coefficients, in descending powers of z e.g. [0 1 0.5] = 0 + z^-1 + 0.5z^-2
    """
    poly_orders = [len(num), len(den)]
    poly_orders = np.array(poly_orders)
    tf_order = np.max(poly_orders).item() - 1
    n = tf_order
    A_ccf, B_ccf, C_ccf = tf_to_ccf(num, den)
    A_ocf = A_ccf.T
    B_ocf = C_ccf.T
    C_ocf = B_ccf.T.reshape(1,n)
    return A_ocf, B_ocf, C_ocf

def tf_list_to_ocf_list(identified_coeffs):
    """
    Given a set of strictly proper transfer functions, returns the A, B, C matrices corresponding to their Observer Canonical Form (OCF).
    num: numerator coefficients, in descending powers of z e.g. [0 1 0.5] = 0 + z^-1 + 0.5z^-2
    den: denominator coefficients, in descending powers of z e.g. [0 1 0.5] = 0 + z^-1 + 0.5z^-2
    """
    poly_orders = np.array([len(identified_coeffs[0][0]), len(identified_coeffs[0][1])])
    tf_order = np.max(poly_orders).item() - 1

    n_states = tf_order
    n_plants = len(identified_coeffs)
    A_in_set = np.zeros([n_plants, n_states, n_states])
    B_in_set = np.zeros([n_plants, n_states, 1])
    for i in range(n_plants):
        num = identified_coeffs[i][0]
        den = identified_coeffs[i][1]
        ss_A_ocf, ss_B_ocf, _ = tf_to_ocf(num, den)
        A_in_set[i,:] = ss_A_ocf
        B_in_set[i,:] = ss_B_ocf

    return A_in_set, B_in_set