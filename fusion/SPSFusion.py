from typing import Union
import numpy as np
import numba
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
EPS = 1e-12


class Fusion:
    def __init__(self, 
                 bounds: np.ndarray, 
                 num_points: np.ndarray, 
                 dim: int, 
                 p: float = 0.95,
                 forget: float = 0.0,
                 random_centers: int = 50):
        self.bounds = bounds
        self.X: np.ndarray = self.sample_uniform_combinations_meshgrid(bounds, num_points,dim)
        self.dim: float = dim
        self.hull: ConvexHull = None
        self.curr_p_map: np.ndarray = None
        self.p: float = p
        self.forget: float = forget
        self.new_update: bool = False
        self.step: int = 0
        self.n_centers: int = random_centers
        self.center_pts: np.ndarray = None
        if self.dim > 3:
            self.pca = PCA(n_components=2)
            self.proj = self.pca.fit_transform(self.X)
        self.initialise_plot()
    
    def sample_uniform_combinations_meshgrid(self, bounds, num_points,d):
        """
        Generate uniform grid points using np.meshgrid (NumPy only).
        
        Parameters:
            bounds: (d, 2) array-like of [min, max] per dimension
            num_points: (d,) array-like, number of samples per dimension

        Returns:
            points: (N, d) array of uniformly spaced points,
                    where N = product of num_points
        """
        axes = []
        for i in range(d):
            axes.append(np.linspace(bounds[i, 0], bounds[i, 1], num_points[i]))

        # Create meshgrid in indexing='ij' to match nested loops
        mesh = np.meshgrid(*axes, indexing='ij')

        # Stack and reshape to (N, d)
        points = np.stack(mesh, axis=-1).reshape(-1, d)
        return points

    def fuse(self, new_hull: ConvexHull):
        if self.hull is not None:
            p_B, _ = point_in_hull_prob(self.X, new_hull.equations, p=self.p)
            self.curr_p_map = fuse_numba(new_info=p_B, prior=self.curr_p_map, forget=self.forget)
            selected_pts = sample_conf_region_from_points(self.X, self.curr_p_map, cumprob=self.p)
            self.hull = ConvexHull(selected_pts)
        else:
            self.hull = new_hull
            self.curr_p_map, insides = point_in_hull_prob(self.X, new_hull.equations, p=self.p)
            selected_pts = self.X[insides]
        self.choose_random_centers(selected_pts)
        self.step+=1
        self.new_update = True

    
    def choose_random_centers(self, selected_pts):
        print(selected_pts.shape)
        n = selected_pts.shape[0]
        if n>self.n_centers:
            idx = np.random.choice(n, size=self.n_centers, replace=False)
            self.center_pts = selected_pts[idx].reshape(-1, self.dim)
        else:
            self.center_pts=selected_pts

    def initialise_plot(self):
        """
        Initialise interactive plot and setup figure, including PCA projection for dim > 3.
        """
        self.new_update = False  # reset update flag
        plt.ion()  # interactive mode on
        self.fig = plt.figure(figsize=(8, 6))

        if self.dim == 2:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("SPS Region (2D)")
            self.ax.set_xlim(self.bounds[0])
            self.ax.set_ylim(self.bounds[1])
            self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=10)
            self.line_collection = LineCollection([], colors='r', linewidths=2)
            self.ax.add_collection(self.line_collection)

        elif self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_title("SPS Region (3D)")
            self.ax.set_xlim(self.bounds[0])
            self.ax.set_ylim(self.bounds[1])
            self.ax.set_zlim(self.bounds[2])
            self.scatter = self.ax.scatter([], [], [], c=[], cmap='viridis', s=10)
            self.lines3d = []  # Will store line objects for hull edges
            # For 3D colorbar, must be added after first update (optional)

        elif self.dim > 3:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("SPS Region (PCA Projection)")
            self.ax.set_xlabel("PC1")
            self.ax.set_ylabel("PC2")

            # Initialize PCA scatter with projected coordinates
            self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=10)
            self.line_collection = LineCollection([], colors='r', linewidths=2)
            self.ax.add_collection(self.line_collection)

            # Optional: set axis limits based on PCA-projected X
            if hasattr(self, "proj"):
                xlim = [np.min(self.proj[:, 0]), np.max(self.proj[:, 0])]
                ylim = [np.min(self.proj[:, 1]), np.max(self.proj[:, 1])]
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

        self.colorbar = self.fig.colorbar(self.scatter, ax=self.ax, label='Fused Posterior Probability')

        self.fig.canvas.draw()
        plt.show(block=False)

    def plot_curr_region(self):
        if not self.new_update or self.curr_p_map is None:
            plt.pause(1.0)  # keep GUI responsive
            return  # No update needed or nothing to plot

        if self.dim == 2:
            self.scatter.set_offsets(self.X)
            self.scatter.set_array(self.curr_p_map)

            # Update hull edges
            segments = [
                [self.hull.points[simplex[0]], self.hull.points[simplex[1]]]
                for simplex in self.hull.simplices
            ]
            self.line_collection.set_segments(segments)

        elif self.dim == 3:
            self.ax.clear()
            self.ax.set_title("SPS Region (3D)")
            self.ax.set_xlim(self.bounds[0])
            self.ax.set_ylim(self.bounds[1])
            self.ax.set_zlim(self.bounds[2])

            self.scatter = self.ax.scatter(
                self.X[:, 0], self.X[:, 1], self.X[:, 2],
                c=self.curr_p_map, cmap='viridis', s=10
            )

            for simplex in self.hull.simplices:
                pts = self.hull.points[simplex]
                self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', linewidth=1)

        elif self.dim > 3:
            self.ax.clear()
            self.ax.set_title("SPS Region (PCA Projection)")
            self.ax.set_xlabel("PC1")
            self.ax.set_ylabel("PC2")

            proj_posterior = self.curr_p_map
            self.scatter = self.ax.scatter(
                self.proj[:, 0], self.proj[:, 1],
                c=proj_posterior, cmap='viridis', s=10
            )
            hull_points = self.pca.transform(self.hull.points)
            for simplex in self.hull.simplices:
                pts = hull_points[simplex]
                self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1)
        
        # Rescale the color limits based on new data
        vmin = np.min(self.curr_p_map)
        vmax = np.max(self.curr_p_map)
        self.scatter.set_clim(vmin, vmax)
        self.colorbar.update_normal(self.scatter)
        # Refresh the plot
        self.ax.figure.canvas.draw_idle()
        self.new_update = False  # Reset update flag
        plt.pause(0.1)  # keep GUI responsive






@numba.njit
def sample_conf_region_from_points(points: np.ndarray, probs: np.ndarray, cumprob: float = 0.95):
    """
    Select the most probable subset of points until the cumulative probability reaches `cumprob`.

    Args:
        points: ndarray of shape (N, D) — raw coordinates.
        probs: ndarray of shape (N,) — probabilities associated with each point.
        cumprob: float — desired cumulative probability (e.g., 0.95).

    Returns:
        selected_points: ndarray of shape (M, D) where M <= N
        actual_cumprob: float — cumulative probability of selected points
    """
    assert points.shape[0] == probs.shape[0], "Mismatch between number of points and probabilities"
    probs = probs / probs.sum()  # Ensure normalization

    # Sort probabilities descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_points = points[sorted_indices]

    # Cumulative sum to determine cutoff
    cum_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cum_probs, cumprob, side='right')

    selected_points = sorted_points[:cutoff_idx + 1]
    actual_cumprob = cum_probs[cutoff_idx]

    return selected_points

@numba.njit
def point_in_hull_prob(points, equations, p):
    """
    points: (N, d)
    equations: (M, d+1), rows are [normal, offset]
    p: probability assigned to points inside the hull

    Returns:
        normalized_probs: (N,) np.ndarray of probabilities normalized to sum to 1
        inside: (N,) np.ndarray of bools for inside hull
    """
    N, d = points.shape
    M = equations.shape[0]
    inside = np.ones(N, dtype=numba.boolean)
    probs = np.empty(N, dtype=np.float64)

    for i in range(N):
        is_inside = True
        for j in range(M):
            val = 0.0
            for k in range(d):
                val += equations[j, k] * points[i, k]
            val += equations[j, -1]
            if val > EPS:
                is_inside = False
                break
        inside[i] = is_inside
        probs[i] = p if is_inside else (1.0 - p)

    prob_sum = np.sum(probs)
    if prob_sum > 0:
        probs /= prob_sum
    return probs, inside

@numba.njit
def fuse_numba(new_info: np.ndarray, prior: np.ndarray, forget = 0.0):
    # https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation#Model
    #  - not sure how to theoretically justify the forgetting factor part

    # forget: forgetting factor such that
    #   1 = disregard all past data
    #   0 = assume past data is always relevant (no change in plant over time)

    if not 0.0 <= forget <= 1.0:
        forget = max(min(forget, 1), 0)
        
    modified_prior = np.mean(prior) * np.ones(prior.shape) * forget + prior * (1-forget)
    posterior = np.multiply(modified_prior, new_info) # Hadamard product
    posterior /= np.sum(np.abs(posterior)) # normalisation
    posterior = np.maximum(EPS, posterior) # to avoid floating point error, this is the new "zero" value
    return posterior

