import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from search.hull_helpers import expand_convex_hull
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadialSearch:
    def __init__(self,
                 n_dimensions: int = 2,
                 n_vectors: int = 30,
                 starting_radius = 0.1,
                 center_options: np.ndarray = None, # Options to test for to set the search center.
                                                # If multiple points are passed in, they will be tested until one is tested True. 
                                                # The first True value will then be used as the center of the search.
                 test_cb: callable = None, # Test function which should accept only an n_dimensions length coordinate
                 max_iter = 100,                 # Maximum iterations per direction
                 epsilon = 0.01,                 # Convergence value
                 max_radius = 2,                # Maximum radius to include in the confidence region
                 next_starting_radius_multiplier = 0.75,    # Value to multiply the boundary of one search epoch by to set the start of the next epoch.
                 ):
        self.n_dimensions = n_dimensions
        self.n_vectors = n_vectors
        self.vectors = []

        # Parameters
        self.search_radii = np.ones( (n_vectors) ) * starting_radius
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.max_radius = max_radius
        self.next_starting_radius_multiplier = next_starting_radius_multiplier

        self.center_options = np.zeros(n_dimensions) if center_options is None else center_options

        # Outputs
        self.ins = []
        self.outs = []
        self.boundaries = []
        self.hull = []
        self.expanded_hull = []

        # Test function
        self.test_cb = test_cb

        self._generate_unit_vectors()


    def _generate_unit_vectors(self):
        """Generates N unit vectors of length n_dimensions."""
        basis_vectors = np.eye(self.n_dimensions)

        vectors = []
        vectors.extend(basis_vectors)

        for _ in range(self.n_vectors - self.n_dimensions):
            rand_vec = np.random.uniform(-1, 1, self.n_dimensions) * basis_vectors
            rand_vec = np.sum(rand_vec, axis=1)
            rand_vec /= np.linalg.norm(rand_vec)    # Normalise to create unit vector
            vectors.extend(np.array([rand_vec]))

        self.vectors = np.array(vectors)


    def search(self):
        """Perform an epoch of the search."""
        ins = []
        outs = []
        boundaries = []

        # Validate that the center is in the confidence region
        self._test_center()
        logger.info("[Radial Search] Search Center Found")
        # Test each direction
        for idx in range(self.n_vectors):
            new_ins, new_outs, boundary = self._test_one_direction(idx)
            ins.extend(new_ins)
            outs.extend(new_outs)
            boundaries.append(boundary)
            # logger.info(f"tested {idx} directions")

        # Get results
        ins = np.array(ins)
        outs = np.array(outs)
        boundaries = np.array(boundaries)

        hull = []
        expanded_hull = []

        try:
            hull = ConvexHull(ins)
            expanded_hull = expand_convex_hull(ins[hull.vertices], expansion_factor=0.01)
        except:
            pass

        return (ins, outs, boundaries, hull, expanded_hull)

    def _test_center(self):
        """Test whether the search center(s) is in the SPS region.
        The first successful point will be used as the search center.
        If no points are successful, an exception is raised."""
        self.center_options = np.array(self.center_options)

        if len(self.center_options.shape) > 1:    # Check if 2d or greater array is passed in 
            for point in self.center_options:
                if self.test_cb(point):
                    self.center = point
                    return
                
        else:
            if self.test_cb(self.center_options):
                self.center = self.center_options
                return
            
        raise Exception("[RADIAL] No provided center values found in confidence region.")


    def _test_one_direction(self, vector_index):
        """Find the boundary in one direction from the search center."""
        vector = self.vectors[vector_index]
        radius = self.search_radii[vector_index]

        # Search state variables
        attempt_no = 0

        highest_true = None
        lowest_false = None

        current_error = 1
        ins = []
        outs = []
        boundary = None

        # Loop until gap between ins and outs is small
        while current_error > self.epsilon and attempt_no < self.max_iter and radius < self.max_radius:
            attempt_no += 1

            # Convert coordinates to cartesian
            coords = radius * vector + self.center
            in_sps = self.test_cb( np.array   (coords) )

            # Check if the coordinate is inside the confidence region
            # If it is, expand the search radius
            if in_sps:
                ins.append(np.array(coords))
                highest_true = radius

                if lowest_false:
                    radius = highest_true + (lowest_false - highest_true) / 2
                else:
                    radius *= 2

            # Otherwise append the coordinate to the outs and reduce the search radius.
            else:
                outs.append(np.array(coords))
                lowest_false = radius

                if highest_true:
                    radius = highest_true + (lowest_false - highest_true) / 2
                else:
                    radius /= 2

            # Update stopping condition
            if highest_true and lowest_false:
                current_error = lowest_false - highest_true

        # Set the boundary to be the current radius (which will be halfway between max_in and min_out after update)
        boundary = radius * vector
        self.search_radii[vector_index] = self.next_starting_radius_multiplier * radius    # Set the next starting radius for this direction
        return (ins, outs, boundary)


    def plot_unit_vectors_2d(self):
        """Plot the generated unit vectors. Only available for 2d system"""
        if self.n_dimensions == 2:
            fig, ax = plt.subplots()
            ax.set_title("Generated Unit Vectors")
            ax.set_xlabel("a")
            ax.set_ylabel("b")

            for i in range(len(self.vectors)):
                ax.plot([0, self.vectors[i, 0]], [0, self.vectors[i, 1]])

    # Plot the results
    def plot_2d_results(self, a_true, b_true):
        if self.n_dimensions == 2:
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
