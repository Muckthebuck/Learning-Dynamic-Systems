import numpy as np

def generate_perturbed(matrix, n_samples, pct):
    """
    Generates a list of perturbed versions of a given matrix (A or B).
    The original matrix is always included, stored as the first element of the list.
    Perturbations are applied assuming OCF state space matrices, i.e. if an A matrix, only the final column is perturbed.

    Parameters:
    ----------
    matrix : np.ndarray
        The original matrix to perturb. Should be either square (interpreted as an A matrix) or Nx1 rectangular (interpreted as a B matrix).
    n_samples : int
        Number of perturbed matrices to generate, including the original matrix.
    pct : float
        Maximum percentage perturbation (relative to the absolute value of each element) applied to the matrix.
        For example, pct = 0.2 allows ±20% perturbations.

    Returns:
    -------
    perturbed_matrices : list of np.ndarray
        List containing the original matrix and (n_samples - 1) perturbed versions. Perturbations are random but bounded as follows:
        - For A matrices (square), only the last column is perturbed.
        - For B matrices (non-square), all elements are perturbed independently.
        This corresponds to perturbing the transfer function coefficients.
    """    
    display_generated_matrices = False

    if matrix.shape[0] == matrix.shape[1]:
        # Square matrix (A matrix)
        n_states = matrix.shape[0]
        delta_A_min = -pct * np.hstack((np.zeros((n_states, n_states - 1)), np.abs(matrix[:, n_states - 1:n_states])))
        delta_A_max = pct * np.hstack((np.zeros((n_states, n_states - 1)), np.abs(matrix[:, n_states - 1:n_states])))
        perturbed_matrices = [matrix]
        for _ in range(1, n_samples):
            delta_A = delta_A_min + (delta_A_max - delta_A_min) * np.random.randn(n_states, n_states)
            perturbed_matrices.append(matrix + delta_A)
    else:
        # Non-square matrix (B matrix)
        n_states, n_inputs = matrix.shape
        delta_B_min = -pct * np.abs(matrix)
        delta_B_max = pct * np.abs(matrix)
        perturbed_matrices = [matrix]
        for _ in range(1, n_samples):
            delta_B = delta_B_min + (delta_B_max - delta_B_min) * np.random.randn(n_states, n_inputs)
            perturbed_matrices.append(matrix + delta_B)
    
    if display_generated_matrices:
        for i, mat in enumerate(perturbed_matrices):
            print(f'perturbed_matrices[{i}] =\n', mat)
    
    return perturbed_matrices

def tf_to_ccf(num,den):
    """
    Convert a strictly proper SISO discrete-time transfer function to its
    Controller Canonical Form (CCF) state-space representation.

    Parameters
    ----------
    num : array_like
        Numerator coefficients of the transfer function in descending powers of z^-1.
        Example: [0, 1, 0.5] represents z^-1 + 0.5·z^-2.

    den : array_like
        Denominator coefficients of the transfer function in descending powers of z^-1.
        Example: [1, 0.5] represents 1 + 0.5·z^-1.

    Returns
    -------
    A : ndarray, shape (n, n)
        State matrix in controller canonical form.

    B : ndarray, shape (n, 1)
        Input matrix in controller canonical form.

    C : ndarray, shape (1, n)
        Output matrix in controller canonical form.

    Raises
    ------
    ValueError
        If the transfer function is not strictly proper (i.e., if num[0] != 0).
    """

    # Ensure coefficients are stored as numpy arrays, and enforce that transfer function is strictly proper
    num = np.array(num, copy=True) # numerator coefficients, G = B/A
    den = np.array(den, copy=True) # denominator coefficients, G = B/A
    if num[0] == 0:
        pass # we have a strictly proper transfer function
    else:
        raise ValueError("Transfer function must be strictly proper (numerator order < denominator order)")

    # Ensure both polynomials have the same length by padding with zeros at the back
    if len(num) > len(den):
        den = np.pad(den, (0, len(num) - len(den)))
    elif len(num) < len(den):
        num = np.pad(num, (0, len(den) - len(num)))

    # Ensure monic denominator (coefficient associated with z^0 should be 1)
    num = np.divide(num, den[0])
    den = np.divide(den, den[0])

    # Now create the state space matrices
    n = len(den) - 1 # order of the system i.e. number of states
    ss_A = np.vstack([np.hstack([np.zeros((n-1, 1)), np.eye(n-1)]), -np.flip(den[1:])])
    ss_B = np.zeros((n, 1))
    ss_B[n-1] = 1  # Last element is 1
    ss_C = np.flip(num)[0:n]
    ss_C = ss_C.reshape(1,n)

    return ss_A, ss_B, ss_C


def tf_to_ocf(num,den):
    """
    Convert a strictly proper SISO discrete-time transfer function to its
    Observer Canonical Form (OCF) state-space representation.

    Parameters
    ----------
    num : array_like
        Numerator coefficients of the transfer function in descending powers of z^-1.
        Example: [0, 1, 0.5] represents z^-1 + 0.5·z^-2.

    den : array_like
        Denominator coefficients of the transfer function in descending powers of z^-1.
        Example: [1, 0.5] represents 1 + 0.5·z^-1.
    
    Raises
    ------
    ValueError
        If the transfer function is not strictly proper (i.e., if num[0] != 0).
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
    Convert a list of strictly proper SISO discrete-time transfer functions
    into their Observer Canonical Form (OCF) state-space representations.

    Parameters
    ----------
    identified_coeffs : list of plant transfer functions.
        identified_coeffs[i] should be the (num, den) coefficients of the i'th plant.
        The coefficient pairs may be stored as tuples or arrays: (num,den) or [num,den].
        Example:
        [
            ([0, 0.2], [1, 0.5]), # G = 0.2 z^-1 / (1 + 0.5z^-1)
            ([0, 0.4], [1, 0.6])  # G = 0.4 z^-1 / (1 + 0.6z^-1)
        ]

    Returns
    -------
    A_set : ndarray, shape (N, n, n)
        Array of N A matrices, one for each plant, in observer canonical form.

    B_set : ndarray, shape (N, n, 1)
        Array of N B matrices, one for each plant, in observer canonical form.

    Notes
    -----
    - Assumes all transfer functions are of the same order.
    - Only A and B matrices are returned; C is not relevant since it's constant across all plants.

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
        A_in_set[i] = ss_A_ocf
        B_in_set[i] = ss_B_ocf

    return A_in_set, B_in_set