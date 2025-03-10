import numpy as np
import matplotlib.pyplot as plt
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
        self.predict_in = []
        self.predict_out = []

        self.n_random = int(self.INITIALISATION_SIZE * len(self.remaining_coords))
        self.n_per_epoch = int(self.SEARCH_SIZE * len(self.remaining_coords) / (self.NUM_EPOCHS-1))

        # Print some stats
        print("Number of datapoints:", len(self.remaining_coords))
        print("Number of random points:", self.n_random)
        print("Number of points which will be tested:", int(self.SEARCH_SIZE * len(self.remaining_coords)))
        print("Number of points per epoch:", self.n_per_epoch)

        # TODO: Allow a model with fit() and predict() functions to be passed in?
        # Otherwise, expose hyperparameters
        self.knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

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
            self.output.append([*random_coord, self.results[random_coord]])    # Append to secondary structure for convenience.

    def calculate_knn_predictions(self):
        """Recalculate the KNN predictors"""
        X = np.array(self.output)[:, 0:self.n_dimensions]
        y = np.array(self.output)[:, self.n_dimensions]

        self.knn.fit(X, y)

        X_test = np.array([list(x) for x in list(self.remaining_coords)])
        y_pred = self.knn.predict(X_test)

        # Concatenate the results
        concat = np.concatenate([X_test, np.array([y_pred]).transpose()], axis=1)
        # TODO: Check this with tuple coords
        self.pred_in = concat[np.where(concat[:,2] == 1)]
        self.pred_out = concat[np.where(concat[:,2] == 0)]

    def knn_search(self):
        for _ in range(self.n_per_epoch):
        # Generate random number to determine whether we want to test inside or outside the confidence region
            is_test_confidence_region = np.random.random() < self.CHANCE_TEST_EXPECTED_POSITIVE
            is_coord_untested = False
            while not is_coord_untested:
                if is_test_confidence_region and len(self.pred_in) > 0:
                    # Test random point from expected confidence region

                    index = np.random.randint(0, len(self.pred_in) - 1) if len(self.pred_in) > 1 else 0
                    coord = self.pred_in[index][:self.n_dimensions]
                    if index == 0:
                        self.pred_in = self.pred_in[1:]
                    else:
                        self.pred_in = np.concatenate([self.pred_in[:index-1], self.pred_in[index:]])
                elif len(self.pred_out) > 0:
                    # Test random point from outside expected confidence region
                    index = np.random.randint(0, len(self.pred_out) - 1) if len(self.pred_out) > 1 else 0
                    coord = self.pred_out[index][:self.n_dimensions]
                    if index == 0:
                        self.pred_out = self.pred_out[1:]
                    else:
                        self.pred_out = np.concatenate([self.pred_out[:index-1], self.pred_out[index:]])

                else:
                    print("Run out of points to test! Stopping")
                    break

                coord = tuple(coord.astype(int))
                x = coord[0]
                y = coord[1]
                is_coord_untested = coord in self.remaining_coords

            self.results[coord] = self.test_coordinate(coord)
            self.remaining_coords.remove(coord)
            self.output.append([*coord, self.results[coord]])

    def go(self):
        # Perform random search
        self.random_search()

        for i in range(1, self.NUM_EPOCHS):
            print("Epoch", i)
            # Perform KNN
            self.calculate_knn_predictions()

            # Search KNN
            self.knn_search()

        # TODO: return ??


if __name__ == "__main__":
    search = SPSSearch(-1, 1, 2)
    search.go()
