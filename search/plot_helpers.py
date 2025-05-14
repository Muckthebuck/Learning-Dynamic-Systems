import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def plot_nd_data(points: np.ndarray, fig=None, ax=None, is_scatter=True, color='red'):
    n_dims = points.shape[1]

    indices = np.arange(n_dims)
    options = list(combinations(indices, 2))

    if fig is None or ax is None:
        fig, ax = plt.subplots(n_dims, 1)
    for i in range(len(options)):
        if is_scatter:
            ax[i].scatter(points[:, options[i][0]], points[:, options[i][1]], color=color)
        else:
            ax[i].plot(points[:, options[i][0]], points[:, options[i][1]], color=color)
        ax[i].set_title("Dimensions %d-%d" % (options[i][0], options[i][1]))

    return fig, ax
