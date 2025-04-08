import numpy as np
from scipy import optimize
from indirect_identification.d_tfs import d_tfs, invert_matrix
from indirect_identification.armax import ARMAX

from indirect_identification.sps_indirect import SPS_indirect_model

sps = SPS_indirect_model(100, 5, 50)


def test_ls(params, K: float, U, Y):
    """
        params: vector of parameters to test
        K:      gain of the controller for this test run. Currently assumed to be constant
        U: U_t signal
        Y: Y_t signal
    """
    print(f"Calculating loss with params {params}")
    print(f"K={K}")
    [print(f"Number of inputs: {len(U)}")]
    [print(f"Number of outputs: {len(Y)}")]
    a, b, c = params


    A = [1, a]
    B = [0, b]
    C = [1, c]
    F = [1] # F(z^-1) = 0.31 + 0.23z^-1
    L = [1]        # L(z^-1) = 1

    G = d_tfs((B, A))
    H = d_tfs((C,A))
    F = d_tfs((F, [1]))
    L = d_tfs((L, [1]))

    G_0, H_0 = sps.transform_to_open_loop(G, H, F, L, check_stability=False)

    YGU = Y - G_0 * U
    N_hat = (1/H_0) * YGU

    return np.sum([N_hat**2]) # Return SSE

if __name__ == "__main__":
    # Example usage
    A = [1, -0.33]  # A(z^-1) = 1 - 0.33z^-1
    B = [0.22]      # B(z^-1) = 0.22z^-1
    C = [1, 0.15]   # C(z^-1) = 1 + 0.15z^-1
    F = [1] # F(z^-1) = 0.31 + 0.23z^-1
    L = [1]        # L(z^-1) = 1


    print("Generating data")
    armax_model = ARMAX(A, B, C, F, L)

    n_samples = 100
    R = 3*np.ones(n_samples)
    Y, U, N, R = armax_model.simulate(n_samples, R, noise_std=0.1)
    K = 1

    # sse = test_ls( (0.2, 0.2, 0.1), 1, U=U, Y=Y )

    x0 = np.array([1, 0.22, 0.15])
    res = optimize.least_squares(test_ls, x0, args=(K, U, Y))
    print(res)
