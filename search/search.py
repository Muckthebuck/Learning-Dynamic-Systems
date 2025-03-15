import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score


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
        n_points: int = 31, # Number of values tested for each parameter
        initialisation_size: float = 0.1, # Ratio of total points to use for random initialisation
        search_size: float = 0.8, # Proportion of the meshgrid which should be searched after initialisation, inclusive of initialisation.
        n_epochs: int = 25, # How many epochs to break the search into, e.g. how many times should the results be predicted. Inclusive of initialisation.
        chance_test_in_region: float = 0.8
    ):
        self.NUM_POINTS = n_points
        self.INITIALISATION_SIZE = initialisation_size
        self.SEARCH_SIZE = search_size
        self.NUM_EPOCHS = n_epochs
        self.CHANCE_TEST_IN_REGION = chance_test_in_region

        self.min = min
        self.max = max
        self.n_dimensions = n_dimensions

        if test_cb is None:
            self.test_coordinate = lambda x : False
        else:
            self.test_coordinate = staticmethod(test_cb)

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

        # Normalise to be span the coordinate mapping
        params = np.linspace(self.min, self.max, self.NUM_POINTS)
        params = np.array(self.n_dimensions * list(params)).reshape(self.n_dimensions, self.NUM_POINTS)
        self.parameter_map = np.array(np.meshgrid(*params))

        self.remaining_coords = (
            self.search_grid.copy()
            .transpose()
            .reshape(self.NUM_POINTS**self.n_dimensions, self.n_dimensions)
            .tolist()
        )

        self.remaining_coords = set(tuple([tuple(x) for x in self.remaining_coords]))

        # Lists of predicted results
        self.pred_in = []
        self.pred_out = []

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

        self.is_store_results = True
        if self.is_store_results:
            self.plot_data = {}
            self.confusion_coords = set()
            self.confusion_data = []

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

            mapped_coord = self.parameter_map[:, *random_coord]
            self.results[random_coord] = self.test_coordinate(mapped_coord)
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
        self.pred_in = concat[np.where(concat[:,self.n_dimensions] == 1)]
        self.pred_out = concat[np.where(concat[:,self.n_dimensions] == 0)]

    def knn_search(self):
        """Logic to search through and test the KNN predictions."""
        confusion_true = []
        confusion_pred = []

        for _ in range(self.n_per_epoch):
        # Generate random number to determine whether we want to test inside or outside the confidence region
            is_test_confidence_region = np.random.random() < self.CHANCE_TEST_IN_REGION
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
                is_coord_untested = coord in self.remaining_coords

            mapped_coord = self.parameter_map[:, *coord]
            self.results[coord] = self.test_coordinate(mapped_coord)    # Use the callback function to get a True/False value
            if self.is_store_results:
                # Calculate confusion matrix data for this epoch
                confusion_true.append(int(self.results[coord]))
                confusion_pred.append(1 if is_test_confidence_region else 0)

            self.remaining_coords.remove(coord)
            self.output.append([*coord, self.results[coord]])
        self.confusion_data.append( (confusion_true, confusion_pred) )
            


    def store_plot_data_2d(self, epoch_number):
        """Preprocess and store data for plots."""
        if self.is_store_results:
            correct = np.where(self.results == 1)
            mapped_correct_x = self.parameter_map[0,0,correct[0]]
            mapped_correct_y = self.parameter_map[1,correct[1],0]

            incorrect = np.where(self.results == 0)
            mapped_incorrect_x = self.parameter_map[0,0,incorrect[0]]
            mapped_incorrect_y = self.parameter_map[1,incorrect[1],0]

            mapped_pred_in_x = self.parameter_map[0,0, self.pred_in[:,0].astype('int')]
            mapped_pred_in_y = self.parameter_map[0,0, self.pred_in[:,1].astype('int')]
            mapped_pred_out_x = self.parameter_map[0,0, self.pred_out[:,0].astype('int')]
            mapped_pred_out_y = self.parameter_map[0,0, self.pred_out[:,1].astype('int')]
            
            self.plot_data["correct_x_%d" % epoch_number]   = mapped_correct_x
            self.plot_data["correct_y_%d" % epoch_number]   = mapped_correct_y
            self.plot_data["incorrect_x_%d" % epoch_number] = mapped_incorrect_x
            self.plot_data["incorrect_y_%d" % epoch_number] = mapped_incorrect_y
            self.plot_data["pred_in_x_%d" % epoch_number]   = mapped_pred_in_x
            self.plot_data["pred_in_y_%d" % epoch_number]   = mapped_pred_in_y
            self.plot_data["pred_out_x_%d" % epoch_number]  = mapped_pred_out_x
            self.plot_data["pred_out_y_%d" % epoch_number]  = mapped_pred_out_y


    def plot_results_2d(self):
        """Plot the data points in a 2-dimensional search."""
        if self.n_dimensions != 2:
            raise "Cannot plot non-2d data"
        
        plt.figure()
        for n_epoch in range(1, self.NUM_EPOCHS):
        # Plot the initialised grid
            plt.subplot(5, 5, n_epoch+1)

            plt.scatter(self.plot_data["pred_in_x_%d" % n_epoch],   self.plot_data["pred_in_y_%d" % n_epoch], color='y')
            plt.scatter(self.plot_data["pred_out_x_%d" % n_epoch],      self.plot_data["pred_out_y_%d" % n_epoch], color='b')
            plt.scatter(self.plot_data["incorrect_x_%d" % n_epoch],      self.plot_data["incorrect_y_%d" % n_epoch], color='g')
            plt.scatter(self.plot_data["correct_x_%d" % n_epoch],        self.plot_data["correct_y_%d" % n_epoch], color='r')
        plt.legend(['Predicted correct', 'Predicted incorrect', 'Tested correct', 'Tested incorrect'])

        # TODO: Plot confusion matrix
        # Currently plotting f score
        plt.figure()
        # plt.subplot(5, 5, n_epoch+1)
        x_vals = np.arange(1, self.NUM_EPOCHS)
        f_scores = []

        for n_epoch in range(self.NUM_EPOCHS-1):
            tn, fp, fn, tp = confusion_matrix(*self.confusion_data[n_epoch]).ravel()
            f_scores.append(f1_score(*self.confusion_data[n_epoch]))


        plt.scatter(x_vals, f_scores)
        plt.title("F Score over epochs of KNN search")
        plt.xlabel("Epoch Number")
        plt.ylabel("F Score")
        plt.show()
        plt.waitforbuttonpress()

    def go(self):
        # Perform random search
        self.random_search()

        for i in range(1, self.NUM_EPOCHS):
            print("Epoch", i)
            # Perform KNN
            self.calculate_knn_predictions()

            # Search KNN
            self.knn_search()
            self.store_plot_data_2d(epoch_number=i)

        return self.parameter_map, self.results


if __name__ == "__main__":
    def test(random_coord):
        p_success = 0.6
        return np.random.random() > 1 - p_success

    search = SPSSearch(-1, 1, 3, test_cb=test)
    search.go()
