import numpy as np

def generate_perturbed(param, n_samples, pct):
    display_generated_matrices = False

    if param.shape[0] == param.shape[1]:
        # Square matrix (A matrix)
        n_states = param.shape[0]
        delta_A_min = -pct * np.hstack((np.zeros((n_states, n_states - 1)), np.abs(param[:, n_states - 1:n_states])))
        delta_A_max = pct * np.hstack((np.zeros((n_states, n_states - 1)), np.abs(param[:, n_states - 1:n_states])))
        perturbed_matrices = [param]
        for _ in range(1, n_samples):
            delta_A = delta_A_min + (delta_A_max - delta_A_min) * np.random.randn(n_states, n_states)
            perturbed_matrices.append(param + delta_A)
    else:
        # Non-square matrix (B matrix)
        n_states, n_inputs = param.shape
        assert n_inputs == 1
        delta_B_min = -pct * np.abs(param)
        delta_B_max = pct * np.abs(param)
        perturbed_matrices = [param]
        for _ in range(1, n_samples):
            delta_B = delta_B_min + (delta_B_max - delta_B_min) * np.random.randn(n_states, n_inputs)
            perturbed_matrices.append(param + delta_B)
    
    if display_generated_matrices:
        for i, mat in enumerate(perturbed_matrices):
            print(f'perturbed_matrices[{i}] =\n', mat)
    
    return perturbed_matrices