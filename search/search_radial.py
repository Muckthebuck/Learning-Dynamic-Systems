import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from search.hull_helpers import expand_convex_hull


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

        # Outputs
        self.ins = []
        self.outs = []
        self.boundaries = []
        self.hull = []
        self.expanded_hull = []

        # Test function
        self.sps_test_function = sps_test_function


    def _generate_unit_vectors(self):
        """Generates N unit vectors of length n_dim."""
        basis_vectors = np.eye(self.n_dim)

        vectors = []
        vectors.extend(basis_vectors)
        
        for _ in range(self.n_vectors):
            rand_vec = np.random.uniform(-1, 1, self.n_dim) * basis_vectors
            rand_vec = np.sum(rand_vec, axis=1)
            rand_vec /= np.linalg.norm(rand_vec)    # Normalise to create unit vector
            vectors.extend(np.array([rand_vec]))

        self.vectors = np.array(self.vectors)


    def search(self):
        """Perform an epoch of the search."""
        ins = []
        outs = []
        boundaries = []

        # Test each direction
        for vector in self.vectors:
            new_ins, new_outs, boundary= self._test_one_direction(vector)
            ins.extend(new_ins)
            outs.extend(new_outs)
            boundaries.append(boundary)

        # Get results
        ins = np.array(ins)
        outs = np.array(outs)
        boundaries.append(boundaries[0])
        boundaries = np.array(boundaries)

        hull = []
        expanded_hull = []

        try:
            hull = ConvexHull(ins)
            expanded_hull = expand_convex_hull(ins[hull.vertices], expansion_factor=0.01)
        except:
            pass
        
        return (ins, outs, boundaries, hull, expanded_hull)


    def _test_one_direction(self, vector_index):
        """Find the boundary in one direction from the search center."""
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
            boundary = radius
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

    # Plot the results
    def plot_2d_results(self, a_true, b_true):
        if self.n_dim == 2:
            a0 = self.center[0]
            b0 = self.center[1]

            fig, ax = plt.subplots()
            ax.plot(a0, b0, 'o', label="Least Squares Estimate")

            if len(self.ins) > 0:
                ax.scatter(self.ins[:, 0], self.ins[:, 1], marker='.', c='red', label="Tested in SPS")
            if len(self.outs) > 0:
                ax.scatter(self.outs[:, 0], self.outs[:, 1], marker='.', c='blue', label="Tested not in SPS")
            # if len(boundaries) > 0:
            #     ax.plot(boundaries[:, 0], boundaries[:, 1], marker='.', c='green', label="linear boundary")
            if len(self.boundaries) > 0:
                ax.plot(self.expanded_hull[:, 0], self.expanded_hull[:, 1], marker='.', c='orange', label="Convex Hull (expanded)")
            ax.plot(a_true, b_true, '*', c='black', label="True Parameter", markersize=10)
            ax.legend()

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("a")
            ax.set_ylabel("b")
            ax.set_title("Radial search output")
            plt.savefig("figures/radial_output.png")

if __name__ == "__main__":
    search = RadialSearch()
    search._generate_unit_vectors()
    search.plot_unit_vectors_2d()
    plt.waitforbuttonpress()
