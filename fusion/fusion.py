import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging

from indirect_identification.sps_indirect import OpenLoopStabilityError, SPS_indirect_model
from indirect_identification.d_tfs import d_tfs
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import _is_stable
from dB.sim_db import SPSType

PDF_FLOOR = 1e-16   # to avoid numerical stability issues, this is the new "zero" value

def fuse_ranks(new_ranks: np.ndarray, old_ranks: np.ndarray, forget=0):

    # Ensure forget parameter is valid
    logger = logging.getLogger(__name__)
    if not 0.0 <= forget <= 1.0:
        forget = max(min(forget, 1), 0)
        logger.warning(f" Warning: invalid forget parameter (0 <= forget <= 1 not satisfied). Proceeding with forget={forget}.")

    # Convert to probabilities
    new_info = 1 - new_ranks
    prior = 1 - old_ranks
    
    # Apply the floor to avoid floating point errors
    new_info = np.maximum(PDF_FLOOR, new_info)
    prior = np.maximum(PDF_FLOOR, prior)

    # Flatten prior according to "forget"
    modified_prior = np.mean(prior) * np.ones(prior.shape) * forget + prior * (1-forget)

    # Fuse and normalise to form the new PDF
    fused_probs = np.multiply(new_info, modified_prior)
    fused_probs = fused_probs / fused_probs.max()

    # Convert back to ranks representation
    fused_ranks = 1 - fused_probs

    return fused_ranks

def get_ranks(grid_axes, model: SPS_indirect_model, n_a, n_b, C, L, F, Y, R, epsilon=1e-10):
    """
    Compute the SPS ranks over a grid of plant parameters. NB: only for SISO.

    Parameters
    ----------
    grid_axes : list of ndarray
        Each entry represents a 1D array of values for one plant parameter.
        Order: [a1 range, a2 range, ..., an range, b1 range, b2 range, ..., bn range]
        Search occurs with B = [0,b1,b2] and A = [1,-a1,-a2] and so forth
    model : object
        Model object that implements `transform_to_open_loop` and `open_loop_sps` methods.
    n_a : int
        Number of denominator (A) coefficients excluding the leading 1.
    n_b : int
        Number of numerator (B) coefficients excluding the leading 0.

    Returns
    -------
    rank_tensor : ndarray
        Tensor of SPS float ranks with the same shape as the grid defined by `grid_axes`.
    """

    mesh = np.meshgrid(*grid_axes, indexing='ij')
    grid_shape = mesh[0].shape
    rank_tensor = np.zeros(grid_shape)

    # Loop through all plant params and store the SPS ranks
    for idx in np.ndindex(grid_shape):
        # Current plant params
        pt = np.array([axis[idx] for axis in mesh])

        # A = np.concatenate(([1], -pt[:n_a]))
        # B = np.concatenate(([0],  pt[n_a:]))
        # G = (B, A)
        # H = (C, A)

        A = np.array(pt[:n_a])
        A = np.hstack([1.0, A])
        if not _is_stable(A, epsilon=epsilon):
            return False

        B = np.array(pt[n_a:])
        B = np.hstack([0.0, B])

        G = d_tfs((B, A))
        H = d_tfs((C, A))

        F = d_tfs((F, [1.0]))
        L =  d_tfs((L, [1.0]))
        # Transform to open loop
        try:
            # G_0, H_0 = model.transform_to_open_loop(G, H, F, L)
            # in_sps, float_rank, S1 = model.open_loop_sps(G_0, H_0, Y, R, n_a, n_b, return_rank=True)
            in_sps, float_rank = model.sps_indicator(G=G, H=H, A=A, B=B, C=C,
                                                     F = F, L = L,
                                                     Y_t = Y.reshape(1,-1),
                                                     U_t = None,
                                                     R_t = R.reshape(1,-1),
                                                     sps_type = SPSType.CLOSED_LOOP,
                                                     return_rank=True)
            rank_tensor[idx] = float_rank
        except OpenLoopStabilityError as e:
            rank_tensor[idx] = 1.0 # Unstable open loop system
    
    return rank_tensor

def sample_fused_rank_tensor(rank_tensor, grid_axes, p=0.95):
    """
    Extracts grid points from a rank tensor that fall within a specified confidence threshold.

    Parameters:
    -----------
    rank_tensor : np.ndarray
        A tensor of fused rank values ranging from 0 (most likely) to 1 (least likely).
    grid_axes : list of np.ndarray
        A list containing arrays for each grid dimension, corresponding to the axes used
        to generate the rank_tensor.
    p : float, optional
        Threshold percentile (default is 0.95). Points with rank <= p are included.

    Returns:
    --------
    np.ndarray
        An array of shape (N, D), where N is the number of points within the confidence region
        and D is the number of dimensions. Each row represents one grid point.
    """

    mask = np.where(rank_tensor <= p)
    mask = np.array(mask)
    grid_vals = np.meshgrid(*grid_axes, indexing='ij')
    grid_vals = np.array(grid_vals)
    n_dims = grid_vals.shape[0]
    in_conf_region = []
    for idx in zip(*mask):  # idx is a tuple (x, y)
        plant_params = [grid_vals[i][idx] for i in range(n_dims)]
        in_conf_region.append(plant_params)

    return np.array(in_conf_region)

def scatter3d(conf_region, ax, plane_idx):
    
    ax.scatter(conf_region[:,0], conf_region[:,1], conf_region[:,2])

    if plane_idx == 0:
        title = 'xy plane'
        ax.set_xlabel('a_1')
        ax.set_ylabel('a_2')

    elif plane_idx == 1:
        title = 'xz plane'
        ax.set_xlabel('a_1')
        ax.set_zlabel('b_1')

    elif plane_idx == 2:
        title = 'yz plane'
        ax.set_ylabel('a_2')
        ax.set_zlabel('b_1')
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    elev = 90 if plane_idx == 0 else 0
    azim = 0  if plane_idx == 2 else -90
    ax.view_init(elev=elev, azim=azim, roll=0)

def sample_fused_conf_region(p_tensor, grid_axes, cumprob=0.95):
    """
    Find the confidence region for arbitrary-dimensional p_tensor using specified grid_axes.
    Returns selected points and actual cumulative probability achieved.
    
    Args:
        p_tensor: ndarray of probabilities (not necessarily 2D).
        grid_axes: list of arrays for each axis (e.g., [a_vals, b_vals, ...]).
        cumprob: desired cumulative probability (default: 0.95).
        
    Returns:
        selected_points: np.ndarray of shape (N, D), D = number of dimensions
        p: actual cumulative probability (float)
    """
    flat_p = p_tensor.flatten()
    sorted_indices = np.argsort(flat_p)[::-1]  # sort high to low
    sorted_probs = flat_p[sorted_indices]

    # Get unique candidate thresholds from highest to lowest
    candidate_thresholds = np.sort(np.unique(sorted_probs))[::-1]
    cumsums = []
    for threshold in candidate_thresholds:
        above_threshold = sorted_probs >= threshold
        cumsum = np.sum(sorted_probs[above_threshold])
        cumsums.append(cumsum)
    cumsums = np.array(cumsums)

    # Find threshold whose cumulative probability is closest to target
    idx = np.argmin(np.abs(cumsums - cumprob))
    if idx == len(cumsums) - 1:
        idx -= 1
    threshold = candidate_thresholds[idx]
    p = cumsums[idx]

    # Indices of elements in tensor above the threshold
    top_indices = np.where(sorted_probs >= threshold)[0]
    in_set_flat = sorted_indices[top_indices]
    unravelled = np.array(np.unravel_index(in_set_flat, p_tensor.shape)).T  # shape (N, D)

    # Map each index to its corresponding coordinate
    selected_points = np.stack(
        [grid_axes[d][unravelled[:, d]] for d in range(len(grid_axes))],
        axis=1
    )

    return selected_points, p

def fuse(new_info: np.ndarray, prior: np.ndarray, forget = 0.0):
    # https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation#Model
    #  - not sure how to theoretically justify the forgetting factor part

    # forget: forgetting factor such that
    #   1 = disregard all past data
    #   0 = assume past data is always relevant (no change in plant over time)

    logger = logging.getLogger(__name__)
    if not 0.0 <= forget <= 1.0:
        forget = max(min(forget, 1), 0)
        logger.warning(f" Warning: invalid forget parameter (0 <= forget <= 1 not satisfied). Proceeding with forget={forget}.")
    
    modified_prior = np.mean(prior) * np.ones(prior.shape) * forget + prior * (1-forget)
    posterior = np.multiply(modified_prior, new_info) # Hadamard product
    posterior /= np.sum(np.abs(posterior)) # normalisation
    posterior = np.maximum(PDF_FLOOR, posterior) # to avoid floating point error, this is the new "zero" value
    return posterior

def plot_fused_conf_region(p_matrix, fig, ax, a_values, b_values, true_a=None, true_b=None, title=None, colorbar=True):
    im = ax.imshow(p_matrix, cmap="PuBu_r", origin="lower", norm=colors.LogNorm(vmin=1e-5, vmax=1e-1), extent=[b_values[0], b_values[-1], a_values[0], a_values[-1]])
    ax.set_aspect('auto')
    ax.set_xticks(b_values[::10])
    ax.set_yticks(a_values[::10])
    ax.set_xlabel('b')
    ax.set_ylabel('a')
    if colorbar:
        fig.colorbar(im, ax=ax)
    if true_a is not None and true_b is not None:
        ax.plot(true_b, true_a, 'rx')  # Plot the true values as a red dot
    ax.set_title(title)


def construct_p_tensor(points_in_conf_region, grid_axes, p=0.95, eps=1e-6):
    """
    Constructs a PDF p_tensor over the grid defined by grid_axes.
    Gridpoints which are INSIDE or OUTSIDE the confidence region are set to have probability 'p' or '1-p' respectively.
    This is then normalised to form a PDF.

    Parameters:
    -----------
    points_in_conf_region : np.ndarray
        An array of shape (N, D) containing the D-dimensional grid points that lie within
        the confidence region (e.g., from sample_fused_rank_tensor()).
    grid_axes : list of np.ndarray
        A list of arrays representing the grid dimensions.
    p : float, optional
        Probability assigned to points in the confidence region (default is 0.95).
        All other points receive probability (1 - p).
    eps : float, optional
        Tolerance used in `np.isclose()` when checking if a grid point is in the confidence region.
        Default is 1e-6. Should be adjusted according to application specifics.

    Returns:
    --------
    np.ndarray
        A tensor of the same shape as the grid, containing normalized probabilities
        summing to 1 over the entire grid.
    """

    mesh = np.meshgrid(*grid_axes, indexing='ij')
    grid_shape = mesh[0].shape
    n_dims = len(grid_axes)

    # Initialize full p_tensor with p_bar = 1-p
    p_tensor = np.ones(grid_shape) * (1 - p)

    # Loop through all indices in the grid, amending p=0.95 for points in confidence region
    for idx in np.ndindex(grid_shape):
        pt = np.array([axis[idx] for axis in mesh])
        if np.any(np.all(np.isclose(points_in_conf_region, pt, atol=eps), axis=1)):
            p_tensor[idx] = p

    # Normalize
    p_tensor /= np.sum(np.abs(p_tensor))
    return p_tensor


def get_conf_region(search_ranges, search_resolutions, model: SPS_indirect_model, n_a, n_b, C: np.ndarray, L, F, Y, R, epsilon=1e-10):
    
    assert n_a + n_b == len(search_ranges)
    assert n_a + n_b == len(search_resolutions)
    
    # Create a meshgrid for A and B values
    grid_axes = [np.linspace(search_ranges[i][0], search_ranges[i][1], search_resolutions[i]) for i in range(n_a+n_b)]
    grid_vals = np.meshgrid(*grid_axes, indexing='xy')
    grid_vals = np.array(grid_vals)

    # Flatten the meshgrid and iterate through each (A,B) candidate
    flat_param_vals = [grid_vals[i].ravel() for i in range(n_a+n_b)]
    flat_param_vals = np.array(flat_param_vals).T
    pts_in_conf_region = []
    n_pts = flat_param_vals.shape[0]

    F = d_tfs((F, [1.0]))
    L = d_tfs((L, [1.0]))

    for i in range(n_pts):

        # Initialise this candidate's transfer functions
        point = flat_param_vals[i]
        
        # A = np.concatenate(([1], point[:n_a]))
        # B = np.concatenate(([0], point[n_a:]))
        # G = (B, A)
        # H = (C, A)

        A = np.array(point[:n_a])
        A = np.hstack([1.0, A])
        if not _is_stable(A, epsilon=epsilon):
            continue        

        B = np.array(point[n_a:])
        B = np.hstack([0.0, B])

        G = d_tfs((B, A))
        H = d_tfs((C, A))

        # Check the SPS indicator and store the parameters if true
        try:
            in_sps = model.sps_indicator(G=G, H=H, A=A, B=B, C=C,
                                         F = F, L = L,
                                         Y_t = Y.reshape(1,-1),
                                         U_t = None,
                                         R_t = R.reshape(1,-1),
                                         sps_type = SPSType.CLOSED_LOOP)
            if in_sps: pts_in_conf_region.append(point)
        except OpenLoopStabilityError:
            # Unstable open loop system - ignore this infeasible parameter combination
            continue
        except Exception as e:
            print(f"Error: {e}")
    
    return np.array(pts_in_conf_region)

def plot_pts_in_conf_region(pts_in_conf_region, x_dim, y_dim, fig=None, ax=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots()
        ax.set_title('Points for which closed_loop_sps returns True')
    elif fig is None and ax is not None:
        raise Exception("fig,ax must both be specified")
    elif fig is not None and ax is None:
        raise Exception("fig,ax must both be specified")
    
    try:
        results = np.asnumpy(results)  # convert to np.ndarray
    except AttributeError:
        pass  # already an np.ndarray

    ax.plot(pts_in_conf_region[:, x_dim], pts_in_conf_region[:, y_dim], 'k.', markersize=3)
    
    return