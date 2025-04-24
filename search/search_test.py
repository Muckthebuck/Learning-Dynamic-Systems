from search.search import SPSSearch
from indirect_identification.sps_indirect import SPS_indirect_model
from indirect_identification.armax import ARMAX

import numpy as np
from scipy import signal


# Initialise SPS model
A = [1, -0.33]  # A(z^-1) = 1 - 0.33z^-1
B = [0.22]      # B(z^-1) = 0.22z^-1
C = [1, 0.15]   # C(z^-1) = 1 + 0.15z^-1
F = [0.31, 0.23] # F(z^-1) = 0.31 + 0.23z^-1
L = [1]        # L(z^-1) = 1


n_samples = 50

m = 100
q = 5

armax_model = ARMAX(A, B, C, F, L)

n_samples = 100
# square wave reference signal
# R = signal.square(np.linspace(0, 10*np.pi, n_samples))
R = 3*np.ones(n_samples)
Y, U, N, R = armax_model.simulate(n_samples, R, noise_std=0.2)


def sps_test_function(params):
    print("Testing", params)
    a, b = params
    a = -a
    A = [1, a]
    B = [0, b]

    G = (B, A)  # G should be a tuple of arrays
    H = (C, A)  # H should be a tuple of arrays

    F = (np.array([0.31, 0.23]), np.array([1]))  # F(z^-1) = 0.31 + 0.23z^-1
    L = ([1], [1])

    # Transform to open loop
    try:
        G_0, H_0 = model.transform_to_open_loop(G, H, F, L)  # Assuming F and L are defined
    except ValueError:
        print("Unstable system... skipping")
        return False
    # Check the condition and store the result if true
    in_sps, S1 = model.open_loop_sps(G_0, H_0, Y, R, 1, 1)  # Assuming Y and U are defined
    return in_sps



model = SPS_indirect_model(m, q)
search = SPSSearch([0, 0], [0.4, 0.4], n_dimensions=2, n_points = [11, 51], test_cb=sps_test_function)
search.go()

results = search.get_results()
print(results.shape)

# search.plot_results_2d()
