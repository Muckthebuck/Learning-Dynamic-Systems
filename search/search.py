import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, KMeans
from sklearn.neighbors import KNeighborsClassifier


# TODO: Use meshgrid instead?
def generate_mesh_coordinates(min, max, n_dimensions, num_points):
    """Generates an n-dimensional meshgrid of coordinates"""
    output = []

    for _ in range(n_dimensions):
        output.append(np.linspace(min, max, num_points))

    return tuple(output)


class SPSSearch:
    def __init__(
        self,
        min: float = -1,
        max: float = 1,
        n_dimensions: int = 2,
        test_cb: callable = None,
    ):
        self.NUM_POINTS: int = 31  # Number of values tested for each parameter
        self.INITIALISATION_SIZE: float = (
            0.1  # Proportion of meshgrid which should be randomly searched at start
        )
        self.SEARCH_SIZE: float = (
            0.8  # Proportion of the meshgrid which should be searched after initialisation, inclusive of initialisation.
        )
        self.NUM_EPOCHS: int = (
            25  # How many epochs to break the search into, e.g. how many times should the results be predicted. Inclusive of initialisation.
        )

        self.min = min
        self.max = max
        self.n_dimensions = n_dimensions

        self.test_coordinate = lambda _: False if test_cb is None else test_cb

        # Generate an n-dimensional meshgrid
        coords = []
        for _ in range(self.n_dimensions):
            coords.append(np.arange(self.NUM_POINTS))
        self.search_grid = np.array(np.meshgrid(*coords))

        self.remaining_coords = set(
            self.search_grid.copy()
            .tranpose()
            .reshape(self.NUM_POINTS**self.n_dimensions, self.n_dimensions)
        )
        print(self.remaining_coords)

        self.results = -1 * np.ones(self.n_dimensions * [self.NUM_POINTS])


if __name__ == "__main__":
    search = SPSSearch(-1, 1, 3)
