import numpy as np
from numba import njit, jit, prange
import matplotlib.pyplot as plt
from itertools import combinations


from search.hull_sutherland_hodgman import PolygonClipper
from scipy.spatial import ConvexHull

# TODO: JIT THESE
def expand_convex_hull(hull_points, expansion_factor=0.05):
    """Expand the convex hull by a small factor to make it slightly larger."""
    hull_center = np.mean(hull_points, axis=0)
    expanded_points = []
    
    for point in hull_points:
        direction = point - hull_center
        expanded_point = point + direction * expansion_factor
        expanded_points.append(expanded_point)
    
    return np.array(expanded_points)

def compare_hulls(hull1, hull2):
    # Sort the points
    hull1 = np.asarray(hull1)
    center1 = hull1.mean(axis=0)
    
    # Compute angles from center
    angles = np.arctan2(hull1[:, 1] - center1[1], hull1[:, 0] - center1[0])
    
    # Sort points by angle in descending order for anticlockwise
    sort_order = np.argsort(angles)
    hull1 = hull1[sort_order]
    hull2 = np.asarray(hull2)
    center2 = hull2.mean(axis=0)
    angles = np.arctan2(hull2[:, 1] - center2[1], hull2[:, 0] - center2[0])
    
    sort_order = np.argsort(angles)
    hull2 = hull2[sort_order]

    clipper = PolygonClipper(warn_if_empty=True)
    clipped_polygon = clipper(hull1, hull2)
    return clipped_polygon


# Returns a list of ConvexHull instances, one for each permutation of the dimensions provided.
def get_hull_points_n_dim(points):
    hull_coordinates = []

    n_dims = points.shape[1]
    indices = np.arange(n_dims)

    options = list(combinations(indices, 2))
    for i in range(len(options)):
        subset = points[:, options[i]]
        hull_coordinates.append(np.array(subset[ConvexHull(subset).vertices]))

    return hull_coordinates


def compare_hulls_n_dim(in_points_1, in_points_2, plot_results=False) -> float:
    """Returns the average of IoU measurements for each 2-dimensional permutation of the data"""
    ious = []

    n_dims = in_points_1.shape[1]
    indices = np.arange(n_dims)

    options = list(combinations(indices, 2))

    for i in range(len(options)):
        ins_1 = in_points_1[:, options[i]]
        ins_2 = in_points_2[:, options[i]]

        hull_intersect = compare_hulls(ins_1[::-1], ins_2[::-1])
        if len(hull_intersect) == 0:
            ious.append(0)
            continue
        intersection_area = ConvexHull(hull_intersect).area

        ins_concat = np.concatenate([ins_1, ins_2])
        union_hull = ConvexHull(ins_concat)
        union_points = ins_concat[union_hull.vertices]

        hull_union = ConvexHull(union_points)
        union_area = hull_union.area

        iou = float(intersection_area / union_area)
        ious.append(iou)

        if plot_results:
            fig, axs = plt.subplots(2,1, figsize=(10, 8))
            axs[0].plot(ins_1[:, 0], ins_1[:,1], color='blue', label="dataset_1")
            axs[0].fill(ins_1[:, 0], ins_1[:,1], color='blue', alpha=0.2)
            axs[0].plot(ins_2[:, 0], ins_2[:,1], color='red', label="dataset_2")
            axs[0].fill(ins_2[:, 0], ins_2[:,1], color='red', alpha=0.2)
            axs[0].legend()
            
            axs[1].plot(union_points[:, 0], union_points[:,1], color='blue', label="union")
            axs[1].fill(union_points[:, 0], union_points[:,1], color='blue', alpha=0.2)
            axs[1].plot(hull_intersect[:, 0], hull_intersect[:,1], color='red', label="intersection")
            axs[1].fill(hull_intersect[:, 0], hull_intersect[:,1], color='red', alpha=0.2)
            axs[0].set_title("Plot for dimensions %d-%d - IoU = %.2f" % (options[i][0], options[i][1], iou))
            axs[1].set_title("Plot for dimensions %d-%d - IoU = %.2f" % (options[i][0], options[i][1], iou))
            axs[1].legend()

    if plot_results:
        plt.show()
        
    return np.mean(ious)


# https://stackoverflow.com/questions/57500001/numba-failure-with-np-mean
@jit(parallel=True)
def mean_numba(a):

    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)
