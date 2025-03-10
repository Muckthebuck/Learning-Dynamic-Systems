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

        self.CHANCE_TEST_EXPECTED_POSITIVE = 0.8  # The chance we choose a point from the predicted confidence region to test

        self.min = min
        self.max = max
        self.n_dimensions = n_dimensions

        self.test_coordinate: callable = lambda _: False if test_cb is None else test_cb

        self.results = -1 * np.ones(self.n_dimensions * [self.NUM_POINTS])
        """
        Results matrix for storing outcome\n
        - (-1)  : Coordinate has **not been tested**.
        - ( 0)  : Coordinate is tested **OUT** of confidence region.
        - ( 1)  : Coordinate is tested **IN** confidence region.
        """

        self.output = []
        """
        List of [coordinate tuple, result value] lists.
        """

        # Generate an n-dimensional meshgrid
        coords = []
        for _ in range(self.n_dimensions):
            coords.append(np.arange(self.NUM_POINTS))
        self.search_grid = np.array(np.meshgrid(*coords))

        self.remaining_coords = (
            self.search_grid.copy()
            .transpose()
            .reshape(self.NUM_POINTS**self.n_dimensions, self.n_dimensions)
            .tolist()
        )

        self.remaining_coords = set(tuple([tuple(x) for x in self.remaining_coords]))

        # Lists of predicted results
        self.predict_in_confidence_region = []
        self.predict_out_confidence_region = []

        self.n_random = int(self.INITIALISATION_SIZE * len(self.remaining_coords))

        # Print some stats
        print("Number of datapoints:", len(self.remaining_coords))
        print("Number of random points:", self.n_random)
        print("Number of points which will be tested:", int(self.SEARCH_SIZE * len(self.remaining_coords)))

    def get_random_coordinate(self) -> tuple:
        """Returns an n-tuple random coordinate."""
        output = []
        for _ in range(self.n_dimensions):
            output.append(np.random.randint(0, self.NUM_POINTS))

        return tuple(output)

    def random_search(self):
        """Test self.n_random points as an initialisation of the search"""
        for _ in range(self.n_random):
            random_coord = self.get_random_coordinate()
            while self.results[random_coord] != -1:
                # Get a new random coordinate if we've already tested this one.
                # Assuming clashes will be minimal here
                random_coord = self.get_random_coordinate()

            # TODO: Convert back to actual meshgrid values from integer coordinates
            self.results[random_coord] = self.test_coordinate(random_coord)
            self.remaining_coords.remove(random_coord)
            self.output.append(random_coord, self.results[random_coord])    # Append to secondary structure for convenience.


    def knn_search(self):
        # Generate random number to determine whether we want to test inside or outside the confidence region
        is_test_confidence_region = np.random.random() < self.CHANCE_TEST_EXPECTED_POSITIVE

        pass

    def calculate_knn(self):
        """Recalculate the KNN predictors"""
        pass


if __name__ == "__main__":
    search = SPSSearch(-1, 1, 2)
