from typing import Union
import numpy as np
import numba
from numba import prange
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from optimal_controller.lowres_MVEE import LowResMVEE
import logging

EPS = 1e-12
MAX_LOG = 700.0
MIN_LOG = -700.0
MAX_POINTS_TO_PLOT = 10000  # adjustable upper limit

# error for no centers in hull
class NoCentersError(Exception):
    """Exception raised when no centers are found."""
    pass
class Fusion:
    def __init__(self, 
                 bounds: np.ndarray, 
                 num_points: np.ndarray, 
                 dim: int, 
                 fn_stability_check: callable,
                 p: float = 0.80,
                 forget: float = 0.0,
                 random_centers: int = 50, 
                 fusion_prob: float = 0.95,
                 steepness: float = 100.0,
                 K: int = 10
                 ):
        
        self.bounds = bounds
        self.X: np.ndarray = self.sample_uniform_combinations_meshgrid(bounds, num_points,dim)
        self.dim: float = dim
        self.hull: ConvexHull = None
        self.curr_p_map: np.ndarray = None
        self.hull_pmap: np.ndarray = None
        self.p: float = p
        self.fusion_prob: float = fusion_prob
        self.forget: float = forget
        self.new_update: bool = False
        self.step: int = 0
        self.n_centers: int = random_centers
        self.steepness: float = steepness
        self.center_pts: np.ndarray = None
        self.n_vertices = min(10 * self.dim, self.dim ** 2)
        self.best_points_reached = False
        self.selected_pts = None
        self.fn_stability_check = fn_stability_check
        self.plot_points = None
        self.plot_probs = None
        self.k_probable = None
        self.used_idx_mask  = np.zeros(self.X.shape[0], dtype=bool)
        self.last_succesfull_centers = None
        self.last_succesfull_centers_idx = None
        self.curr_p_map_sorted_idx = None

        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None

        self.K = K
        if self.dim >= 3:
            self.pca = PCA(n_components=2)
            self.proj = self.pca.fit_transform(self.X)
        self.initialise_plot()

    def approximate_hull(self):
        """
        Approximate convex hull with MVEE and sample points on the ellipsoid surface.
        Auto-picks number of samples based on dimensionality.
        """
        if self.dim == 2:
            if self.best_points_reached:
                logging.info("[Fusion] Best points reached, returning selected points.")
                # check if the selected points are less than dim, if so keep the best points reached
                if self.selected_pts is None or self.selected_pts.shape[0] > self.dim:
                    self.best_points_reached = False    
             
                return self.selected_pts.reshape(-1, self.dim)
            return self.hull.points[self.hull.vertices]

        # points = self.hull.points[self.hull.vertices]
        # if points.shape[0] < self.n_vertices:
        #     return points.reshape(-1,self.dim)
        # mvee = LowResMVEE(pts=points.T, max_n_verts=self.n_vertices)
        hull_pts = self.hull.points[self.hull.vertices]
        logging.info(f"[Fusion] Hull: {hull_pts.shape[0]} points.")
        if hull_pts.shape[0] < self.dim:
            logging.warning(f"[Fusion] Hull has too few points ({hull_pts.shape[0]}), returning as is.")
            return hull_pts.reshape(-1, self.dim)
        elif hull_pts.shape[0]> self.n_vertices:
            # if too many points, low res mvee
            mvee = LowResMVEE(pts=hull_pts.T, max_n_verts=self.n_vertices)
            results = mvee.vertices.reshape(-1, self.dim)
            logging.info(f"[Fusion] Approximate hull: {results.shape[0]} vertices after low-res MVEE.")
            return results
        else:
            # if enough points, return them as is
            logging.info(f"[Fusion] Hull is good: {hull_pts.shape[0]} vertices.")
            return hull_pts.reshape(-1, self.dim)

    
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

    def fuse(self, new_hull: ConvexHull, ins: np.ndarray):
        # checkl if the new hull is degenerate
        if new_hull is None or new_hull.equations.shape[0] < self.dim:
            # could be that ins is empty, in this case, skip
            if ins is None or ins.shape[0] == 0:
                logging.warning("[Fusion] Skipping fusion: hull is degenerate (no points).")
                return
            # if ins not empty, then we can use it the way it is as selected points
            # attempt to create a convex hull
            ins = self.filter_points(ins)
            if ins.shape[0] < self.dim:
                logging.warning(f"[Fusion] Skipping fusion: not enough points ({ins.shape[0]}) to form convex hull.")
                return
            hull = ConvexHull(ins)
            if hull.equations.shape[0] < self.dim:
                logging.warning(f"[Fusion] Skipping fusion: hull is degenerate ({hull.equations.shape[0]} vertices).")
                return
                
        if self.hull is not None:
            p_B = point_in_hull_likelihood(self.X, new_hull.equations, p=self.p, steepness=self.steepness)
            # If the new likelihood is too flat, skip
            if np.max(p_B) - np.min(p_B) < EPS:
                logging.info("[Fusion] Skipping fusion: Likelihood update is too weak (flat).")
                return
            curr_p_map = fuse_numba(new_info_log=p_B, prior_log=self.curr_p_map, forget=self.forget)
            selected_pts, selected_probs = sample_conf_region_from_points(self.X, curr_p_map, cumprob=self.fusion_prob)
            stable_indices = self.stable_points_idx(selected_pts)
        else:
            curr_p_map = point_in_hull_likelihood(self.X, new_hull.equations, p=self.p, steepness=self.steepness)
            logging.info(f"[Fusion] Initializing convex hull with {curr_p_map.shape[0]} points.")
            selected_pts, selected_probs = sample_conf_region_from_points(self.X, curr_p_map, cumprob=self.fusion_prob)
            stable_indices = self.stable_points_idx(selected_pts)
           

        selected_pts = selected_pts[stable_indices]
        selected_probs = selected_probs[stable_indices]
        try:
            logging.info(f"[Fusion] Selected {selected_pts.shape[0]} points for convex hull.")
            self.hull = ConvexHull(selected_pts)
            self.curr_p_map = curr_p_map
        except Exception:
            try: 
                logging.error(f"[Fusion] Failed to create convex hull")
                mvee = LowResMVEE(pts=selected_pts.T, max_n_verts=self.n_vertices)
                logging.info(f"[Fusion] Using low-res MVEE with {mvee.vertices.shape[0]} vertices.")
                mvee_verts = np.concatenate([mvee.vertices.reshape(-1, self.dim), selected_pts], axis=0)
                logging.info(f"[Fusion] Using low-res MVEE modified {mvee_verts.shape[0]} vertices instead.")
            except Exception:
                self.best_points_reached = True
                self.selected_pts = selected_pts
                logging.error(f"[Fusion] Failed to create mvee from selected points")
                return
            try:
                self.hull = ConvexHull(mvee_verts)
                self.curr_p_map = curr_p_map
            except Exception:
                self.best_points_reached = True
                self.selected_pts = selected_pts
                logging.error(f"[Fusion] Failed to create convex hull from MVEE")
                return
           

        logging.info(f"[Fusion] Convex hull created with {self.hull.points.shape[0]} points and {self.hull.equations.shape[0]} vertices.")
        self.hull_pmap = selected_probs
        self.k_probable = self.get_K_probable_points(selected_pts, selected_probs, K=self.K)
        # plotting values
        plot_points = self.pca.transform(selected_pts) if self.dim >= 3 else selected_pts
        # ensure unique points
        if plot_points.shape[0] > MAX_POINTS_TO_PLOT:
            plot_points, selected_probs = sample_conf_region_from_points_and_prob(plot_points, selected_probs, cumprob=0.60)
            # sort by probabilities and get top MAX_POINTS_TO_PLOT
            if plot_points.shape[0] > MAX_POINTS_TO_PLOT:
                idx = np.argsort(selected_probs)[::-1]
                plot_points = plot_points[idx][:MAX_POINTS_TO_PLOT]
                selected_probs = selected_probs[idx][:MAX_POINTS_TO_PLOT]

        self.plot_points = plot_points
        self.plot_probs = log_probs_to_normalized_probs(selected_probs)
        self.step+=1
        self.new_update = True

    def reset_used_idx_set(self):
        """
        Reset the used index set to allow re-sampling of points.
        """
        self.used_idx_mask = np.zeros(self.X.shape[0], dtype=bool)
        if self.curr_p_map is not None:
            # lets first increase the density of points in the hull by 2
            self.curr_p_map_sorted_idx = np.argsort(self.curr_p_map)[::-1]  # descending order of log-probs
        logging.info("[Fusion] Used index set reset.")

    def get_K_probable_points(self, points: np.ndarray, prob, K: int = 10):
        """
        Get K most probable points based on the provided probabilities.
        Uses np.argpartition for fast top-K selection.
        """
        if points.shape[0] < K:
            logging.warning(f"[Fusion] Not enough points to select {K} most probable points. Returning all {points.shape[0]} points.")
            return points.reshape(-1, self.dim)

        # Fast partial selection of top-K*2 candidates
        max_k = min(K*2, points.shape[0])
        idx = np.argpartition(prob, -max_k)[-max_k:]
        
        # Sort candidates by descending probability
        idx = idx[np.argsort(prob[idx])[::-1]]
        candidate_points = points[idx]

        # Filter the candidate points
        filtered_points = candidate_points #self.filter_points(candidate_points)

        # Fallback: nothing passed filtering
        if filtered_points.shape[0] == 0:
            idx = np.argpartition(prob, -K)[-K:]
            logging.warning(f"[Fusion] No points passed stability check. Returning top {K} points based on probabilities.")
            return points[idx].reshape(-1, self.dim)

        # Return as many filtered points as possible (up to K)
        return filtered_points[:K].reshape(-1, self.dim)
    
    def stable_points_idx(self, points: np.ndarray):
        """
        Get indices of points that pass the stability check.
        """
        if self.fn_stability_check is None:
            return np.arange(points.shape[0])
        stable_indices = []
        for i, pt in enumerate(points):
            if self.fn_stability_check(pt):
                stable_indices.append(i)
        return np.array(stable_indices)

    def unstable_points_idx(self, points: np.ndarray):
        """
        Get indices of points that do not pass the stability check.
        """
        if self.fn_stability_check is None:
            return np.array([])
        unstable_indices = []
        for i, pt in enumerate(points):
            if not self.fn_stability_check(pt):
                unstable_indices.append(i)
        return unstable_indices

    def filter_points(self, points: np.ndarray):
        """
        Filter points based on the stability callback function.
        """
        if self.fn_stability_check is None:
            return points
        filtered_points = []
        for pt in points:
            if self.fn_stability_check(pt):
                filtered_points.append(pt)
        return np.array(filtered_points).reshape(-1, self.dim)
    
    def sample_points(self):
        """
        Sample points inside and outside the convex hull.
        """
        if self.hull is None:
            raise ValueError("Convex hull not defined. Call fuse() first.")
        selected_pts = self.hull.points
        n = selected_pts.shape[0]
        n_inside = int(self.p*self.n_centers)
        outside = self.n_centers - n_inside

        outside = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(outside, self.dim))
        if n>self.n_centers:
            # inside points
            idx = np.random.choice(n, size=n_inside, replace=False)
            inside = selected_pts[idx].reshape(-1, self.dim)
        else:
            combined_pts = np.concatenate([selected_pts, self.hull.points[self.hull.vertices]], axis=0)
            inside = np.random.choice(combined_pts.shape[0], size=n_inside, replace=False)
            inside = combined_pts[inside].reshape(-1, self.dim)
            
        return inside, outside

    def sample_points_inside(self, alpha=0.7):
        """
        Sample points inside the convex hull using a blended score of log-probabilities
        and geometric centrality, avoiding previously used points.
        """
        if self.hull is None:
            raise ValueError("Convex hull not defined. Call fuse() first.")

        selected_pts = self.hull.points
        n = selected_pts.shape[0]

        if self.hull_pmap is None or len(self.hull_pmap) != n:
            raise ValueError("Invalid or missing hull_pmap (log-probabilities).")

        # Step 1: Get available indices
        all_indices = np.arange(n)
        mask = np.array([i not in self.used_idx_set for i in all_indices])
        available_indices = all_indices[mask]

        if len(available_indices) == 0:
            self.use_random_centers = True
            raise NoCentersInHullError("No available points left to sample from the convex hull.")

        available_points = selected_pts[available_indices]
        available_logp = self.hull_pmap[available_indices]

        # Step 2: Scoring
        logp_probs = log_probs_to_normalized_probs(available_logp)
        centroid = np.mean(selected_pts, axis=0)
        centrality = compute_centrality(available_points, centroid)
        blended_scores = blend_scores(logp_probs, centrality, alpha)
        blended_scores /= np.sum(blended_scores)

        # Step 3: Sample
        sample_size = min(self.n_centers, len(available_indices))
        sampled_idx_local = np.random.choice(len(available_indices), size=sample_size, replace=False, p=blended_scores)
        sampled_global_idx = available_indices[sampled_idx_local]

        # Step 4: Update used index set
        self.used_idx_set.update(sampled_global_idx.tolist())

        # check if we have used all points
        if len(self.used_idx_set) >= n:
            self.use_random_centers = True
            logging.info("[Fusion] All points used, switching to random centers sampling in next iteration.")

        # Step 5: Return points
        inside = selected_pts[sampled_global_idx].reshape(-1, self.dim)
        return inside, sampled_global_idx


    # def sample_points_inside(self):
    #     """
    #     Sample points inside the convex hull using a log-probability map
    #     that biases sampling toward central points.
    #     """
    #     if self.hull is None:
    #         raise ValueError("Convex hull not defined. Call fuse() first.")

    #     selected_pts = self.hull.points
    #     n = selected_pts.shape[0]

    #     if self.hull_pmap is None or len(self.hull_pmap) != n:
    #         raise ValueError("Invalid or missing hull_pmap (log-probabilities).")

    #     # Convert log-probabilities to normalized probabilities
    #     probs = log_probs_to_normalized_probs(self.hull_pmap)

    #     # Sample indices based on probabilities
    #     if n >= self.n_centers:
    #         idx = np.random.choice(n, size=self.n_centers, replace=False, p=probs)
    #         inside = selected_pts[idx].reshape(-1, self.dim)
    #     else:
    #         # if not enough points, order the points by probabilities
    #         idx = np.argsort(probs)[::-1][:self.n_centers]
    #         inside = selected_pts[idx].reshape(-1, self.dim)

    #     return inside

    def reset_last_successful_centers(self):
        """
        Reset the last successful centers and their indices.
        This allows re-sampling of centers in the next choose_random_centers call.
        """
        self.last_succesfull_centers = None
        self.last_succesfull_centers_idx = None
        logging.info("[Fusion] Last successful centers reset.")

    def choose_random_centers(self):
        """
        Choose random centers from the convex hull vertices.
        """
        if self.hull is None:
            raise ValueError("Convex hull not defined. Call fuse() first.")
        # inside, outside = self.sample_points_(self.n_centers)
        # center_pts = np.concatenate([inside, outside], axis=0)
        if self.last_succesfull_centers is not None:
            # use the last successful centers if available
            self.center_pts = self.last_succesfull_centers
            self.last_succesfull_centers = None
            if self.last_succesfull_centers_idx is not None:
                self.used_idx_mask[self.last_succesfull_centers_idx] = True
            logging.info(f"[Fusion] Using last successful centers: {self.center_pts.shape}")
            return

        center_pts, last_succesfull_centers_idx, self.used_idx_mask = sample_from_sorted_pmap_numba(points=self.X,
                                                                    log_probs=self.curr_p_map,
                                                                    used_mask=self.used_idx_mask,
                                                                    sorted_indices=self.curr_p_map_sorted_idx,
                                                                    n_samples=self.n_centers)
        
        stable_centers_idx = self.stable_points_idx(center_pts)
        if stable_centers_idx.shape[0] == 0:
            raise NoCentersError("No stable centers found after random sampling.")
        else:
            self.center_pts = center_pts[stable_centers_idx]
            self.last_succesfull_centers_idx = last_succesfull_centers_idx[stable_centers_idx]
            logging.info(f"[Fusion] Chose {self.center_pts.shape} stable centers from probability.")


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
            self.ax.grid(True)
            self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=10,)
            self.line_collection = LineCollection([], colors='r', linewidths=2)
            self.ax.add_collection(self.line_collection)

        elif self.dim >= 3:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("SPS Region (PCA Projection)")
            self.ax.set_xlabel("PC1")
            self.ax.set_ylabel("PC2")

            # Initialize PCA scatter with projected coordinates
            self.scatter = self.ax.scatter([],[], c=[], cmap='viridis', s=10)

            # Optional: set axis limits based on PCA-projected X
            if hasattr(self, "proj"):
                xlim = [np.min(self.proj[:, 0]), np.max(self.proj[:, 0])]
                ylim = [np.min(self.proj[:, 1]), np.max(self.proj[:, 1])]
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

        self.colorbar = self.fig.colorbar(self.scatter, ax=self.ax, label='Fused Posterior Probabilities')

        self.fig.canvas.draw()
        plt.show(block=False)

    def plot_curr_region(self):
        if not self.new_update or self.curr_p_map is None:
            plt.pause(1.0)  # keep GUI responsive
            return  # No update needed or nothing to plot

        vmin = np.min(self.plot_probs)
        vmax = np.max(self.plot_probs)
        plot_probs = self.plot_probs
        plot_points = self.plot_points

        if self.dim == 2:
            # for 2d we also want to plot the convex hull edges
            segments = [
                [self.hull.points[simplex[0]], self.hull.points[simplex[1]]]
                for simplex in self.hull.simplices
            ]
            self.line_collection.set_segments(segments)

        self.scatter.set_offsets(plot_points)
        self.scatter.set_array(plot_probs)
        
        # # readjust axis limits if needed
        # xmin, ymin = np.min(plot_points, axis=0)*2
        # xmax, ymax = np.max(plot_points, axis=0)*2
        # if self.xmin is None or self.xmax is None or self.ymin is None or self.ymax is None:
        #     self.xmin, self.ymin = xmin, ymin
        #     self.xmax, self.ymax = xmax, ymax
        # else:
        #     self.xmin = min(self.xmin, xmin)
        #     self.ymin = min(self.ymin, ymin)
        #     self.xmax = max(self.xmax, xmax)
        #     self.ymax = max(self.ymax, ymax)
        # self.ax.set_xlim(self.xmin, self.xmax)
        # self.ax.set_ylim(self.ymin, self.ymax)

        self.scatter.set_clim(vmin, vmax)
        self.colorbar.update_normal(self.scatter)
        self.ax.figure.canvas.draw_idle()
        self.new_update = False
        plt.pause(0.005)

        

@numba.njit(inline='always')
def logsumexp_two(a, b):
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

@numba.njit(cache=True, fastmath=True)
def cumulative_logsumexp(log_probs):
    n = log_probs.shape[0]
    cum_logsumexp = np.empty(n, dtype=np.float64)
    cum_logsumexp[0] = log_probs[0]
    for i in range(1, n):
        cum_logsumexp[i] = logsumexp_two(cum_logsumexp[i-1], log_probs[i])
    return cum_logsumexp

@numba.njit(cache=True, fastmath=True)
def log_probs_to_normalized_probs(log_probs):
    max_log = np.max(log_probs)
    stabilized = np.exp(log_probs - max_log)
    s = np.sum(stabilized)
    return stabilized / s

# @numba.njit(cache=True, fastmath=True)
# def sample_conf_region_from_points(points: np.ndarray, probs: np.ndarray, cumprob: float = 0.95):
#     """
#     Select points above a certain percentile threshold from unnormalized probabilities.

#     Args:
#         points: ndarray of shape (N, D)
#         probs: ndarray of shape (N,) — unnormalized likelihoods
#         cumprob: float — desired retention percentile (e.g., 0.95 keeps top 5%)

#     Returns:
#         selected_points: ndarray of shape (M, D)
#     """
#     assert 0.0 < cumprob <= 1.0

#     # Find the percentile threshold (top `cumprob` mass)
#     threshold = np.percentile(probs, 100 * (1 - cumprob))

#     # Select points above this threshold
#     mask = probs >= threshold
#     return points[mask]

@numba.njit(cache=True, fastmath=True)
def sample_conf_region_from_points_and_prob(points: np.ndarray, log_probs: np.ndarray, cumprob: float = 0.95):
    probs = np.exp(log_probs - np.max(log_probs))  # normalize for stability

    threshold = np.percentile(probs, cumprob * 100)
    mask = probs >= threshold
    return points[mask], probs[mask]

# @numba.njit(cache=True, fastmath=True)
# def sample_conf_region_from_points(points: np.ndarray, log_probs: np.ndarray, cumprob: float = 0.95):
#     probs = np.exp(log_probs - np.max(log_probs))  # normalize for stability

#     threshold = np.percentile(probs, cumprob * 100)
#     mask = probs >= threshold
#     return points[mask]

# @numba.njit(cache=True, fastmath=True, parallel=True)
# def sample_conf_region_from_points(points: np.ndarray, log_probs: np.ndarray, cumprob: float = 0.95):
#     # Normalize for numerical stability
#     probs = np.exp(log_probs - np.max(log_probs))
    
#     # Replace np.percentile using sorting
#     n = probs.shape[0]
#     sorted_probs = np.empty_like(probs)
#     for i in prange(n):
#         sorted_probs[i] = probs[i]
    
#     # In-place sort (Numba supports ndarray.sort())
#     sorted_probs.sort()

#     # Calculate the index corresponding to the desired percentile
#     k = int(np.floor((cumprob) * n))
#     if k < 0:
#         k = 0
#     elif k >= n:
#         k = n - 1

#     threshold = sorted_probs[k]

#     # Apply the mask as before
#     mask = probs >= threshold
#     return points[mask], log_probs[mask]



@numba.njit(cache=True, fastmath=True)
def sample_from_sorted_pmap_numba(points, log_probs, used_mask, sorted_indices, n_samples):
    """
    Numba-compatible version to sample points by descending log-probability,
    skipping already-used points.

    Args:
        points: (N, D) array
        log_probs: (N,) array
        used_mask: (N,) boolean array
        sorted_indices: (N,) array, sorted descending by log_probs
        n_samples: int 

    Returns:
        selected_points: (M, D)
        selected_indices: (M,)
        updated used_mask
    """
    N, D = points.shape

    probs = np.exp(log_probs - np.max(log_probs))
    probs /= np.sum(probs)

    max_output =  min(n_samples, N)
    selected_indices = np.empty(max_output, dtype=np.int32)
    selected_points = np.empty((max_output, D), dtype=points.dtype)

    count = 0
    cum_sum = 0.0

    for i in range(N):
        idx = sorted_indices[i]
        if used_mask[idx]:
            continue

        selected_indices[count] = idx
        selected_points[count] = points[idx]
        used_mask[idx] = True
        cum_sum += probs[idx]
        count += 1

        if count >= n_samples:
            break
  

    return selected_points[:count], selected_indices[:count], used_mask

@numba.njit(cache=True, fastmath=True)
def sample_conf_region_from_points(points: np.ndarray, log_probs: np.ndarray, cumprob: float = 0.95):
    n = points.shape[0]
    probs = log_probs_to_normalized_probs(log_probs)

    # Sort descending by prob
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_points = points[sorted_indices]
    sorted_log_probs = log_probs[sorted_indices]

    # Cumulative sum cutoff (A)
    cum_probs = np.cumsum(sorted_probs)
    cutoff_idx_A = np.searchsorted(cum_probs, cumprob, side='right')
    if cutoff_idx_A >= n:
        cutoff_idx_A = n - 1


    # Percentile threshold cutoff (B)
    k = int(np.floor(cumprob * n))
    if k < 0:
        k = 0
    elif k >= n:
        k = n - 1

    count = min(cutoff_idx_A, k)+1

    selected_points = sorted_points[:count]
    selected_log_probs = sorted_log_probs[:count]

    return selected_points, selected_log_probs

@numba.njit(cache=True, fastmath=True, parallel=True)
def point_in_hull_likelihood(points, equations, p, steepness=100.0):
    N, d = points.shape
    M = equations.shape[0]
    log_probs = np.empty(N, dtype=np.float64)

    log_p = np.log(max(p, EPS))  # base log prob inside hull

    for i in prange(N):
        max_violation = -np.inf
        for j in range(M):
            val = 0.0
            for k in range(d):
                val += equations[j, k] * points[i, k]
            val += equations[j, -1]
            if val > max_violation:
                max_violation = val

        if max_violation <= 0.0:
            # inside hull, uniform score
            log_probs[i] = log_p
        else:
            # outside hull, decaying sigmoid
            val = max(MIN_LOG, min(MAX_LOG, steepness * max_violation))
            prob = 1.0 / (1.0 + np.exp(val))
            prob = max(prob, EPS)
            log_probs[i] = np.log(prob)

    return log_probs


@numba.njit(cache=True, fastmath=True)
def fuse_numba(new_info_log: np.ndarray, prior_log: np.ndarray, forget=0.0):
    new_info_log = np.clip(new_info_log, MIN_LOG, MAX_LOG)
    prior_log = np.clip(prior_log, MIN_LOG, MAX_LOG)
    if forget == 0.0:
        # Clamp to avoid overflow/underflow
        return new_info_log + prior_log

    prior = np.exp(prior_log)
    mean_prior = np.mean(prior)
    modified_prior = mean_prior * forget + prior * (1 - forget)
    modified_prior = np.maximum(modified_prior, EPS)
    modified_prior_log = np.log(modified_prior)
    modified_prior_log = np.clip(modified_prior_log, MIN_LOG, MAX_LOG)

    posterior_log = modified_prior_log + new_info_log
    return posterior_log

@numba.njit(cache=True, fastmath=True)
def compute_centrality(points, centroid):
    n = points.shape[0]
    dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist = 0.0
        for j in range(points.shape[1]):
            diff = points[i, j] - centroid[j]
            dist += diff * diff
        dists[i] = np.sqrt(dist)
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max - d_min < 1e-8:
        return np.ones(n, dtype=np.float64)  # All points are equally central
    
    norm_dists = (dists - d_min) / (d_max - d_min)
    return 1.0 - norm_dists

@numba.njit(cache=True, fastmath=True)
def blend_scores(logp_probs, centrality, alpha):
    return alpha * logp_probs + (1 - alpha) * centrality