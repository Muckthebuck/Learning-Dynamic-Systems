import numpy as np
import matplotlib.pyplot as plt

class RadialSearch:
    def __init__(self,
                 n_dim: int = 2,
                 n_vectors: int = 30,
                 starting_radius = 0.1,
                 center: np.ndarray = None,     # Center of the radial search. Should be least-squares estimate
                 sps_test_function: callable = None, # Test function which should accept only an n_dim length coordinate
                 max_iter = 100,                 # Maximum iterations per direction
                 epsilon = 0.01,                 # Convergence value
                 max_radius = 10,                # Maximum radius to include in the confidence region
                 ):
        self.n_dim = n_dim
        self.n_vectors = n_vectors
        self.vectors = []

        # Parameters
        self.search_radii = np.ones( (n_vectors) ) * starting_radius
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.max_radius = max_radius

        self.center = np.zeros(n_dim) if center is None else center

        self.sps_test_function = sps_test_function

    def _generate_unit_vectors(self):
        """Generates N unit vectors of length n_dim"""
        basis_vectors = np.eye(self.n_dim)

        vectors = []
        vectors.extend(basis_vectors)
        
        for _ in range(self.n_vectors):
            rand_vec = np.random.uniform(-1, 1, self.n_dim) * basis_vectors
            rand_vec = np.sum(rand_vec, axis=1)
            rand_vec /= np.linalg.norm(rand_vec)    # Normalise to create unit vector
            vectors.extend(np.array([rand_vec]))

        self.vectors = np.array(self.vectors)

    def _test_one_direction(self, vector_index):
        vector = self.vectors[vector_index]
        radius = self.search_radii[vector_index]

        attempt_no = 0
        
        highest_true = None
        lowest_false = None

        current_error = 1

        ins = []
        outs = []
        boundary = None

        while current_error > self.epsilon and attempt_no < self.max_iter and radius < self.max_radius:
            attempt_no += 1

            # Convert coordinates to cartesian
            coords = radius * vector + self.center

            in_sps = self.sps_test_function( tuple(coords) )

            # Update search params
            if in_sps:
                ins.append(np.array(coords))
                highest_true = radius

                if lowest_false:
                    radius = highest_true + (lowest_false - highest_true) / 2
                else:
                    radius *= 2

            else:
                outs.append(np.array(coords))
                lowest_false = radius

                if highest_true:
                    radius = highest_true + (lowest_false - highest_true) / 2
                else:
                    radius /= 2

            if highest_true and lowest_false:
                current_error = lowest_false - highest_true

            # TODO: Calculate n-dimensional boundary point
            boundary = None
            return (ins, outs, boundary)


    def plot_unit_vectors_2d(self):
        """Plot the generated unit vectors. Only available for 2d system"""
        if self.n_dim == 2:
            fig, ax = plt.subplots()
            ax.set_title("Generated Unit Vectors")
            ax.set_xlabel("a")
            ax.set_ylabel("b")

            for i in range(len(self.vectors)):
                ax.plot([0, self.vectors[i, 0]], [0, self.vectors[i, 1]])


if __name__ == "__main__":
    search = RadialSearch()
    search._generate_unit_vectors()
    search.plot_unit_vectors_2d()
    plt.waitforbuttonpress()
