import numpy as np
from numba import njit, jit, prange
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA
from shapely.geometry import Polygon

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

def sort_clockwise(hull_points, center=np.zeros(2)):
    """Sorts a list of x,y points clockwise according to arctan2(y,x) with an optional center point"""
    # Sort the points
    hull_points = np.asarray(hull_points)

    # Compute angles from center
    angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
    angles[angles < 0] += 2*np.pi

    # Sort points by angle in descending order for clockwise sorting
    sort_order = np.argsort(angles)

    return hull_points[sort_order][::-1]


def compare_hulls(hull1, hull2):
    """@DEPRECATED since the PolygonClipper method seems bugged"""
    # Sort the points
    hull1 = sort_clockwise(hull1)
    hull2 = sort_clockwise(hull2)

    clipper = PolygonClipper(warn_if_empty=True)
    clipped_polygon = clipper(hull1, hull2)
    return clipped_polygon

# def compare_pca_shapely(pca1, pca2):
#     p = Polygon(pca1)
#     q = Polygon(pca2)

#     return compare_hulls_shapely(p.convex_hull, q.convex_hull)

def compare_hulls_shapely(hull1, hull2):
    p = Polygon(hull1).convex_hull
    q = Polygon(hull2).convex_hull
    intersect = p.intersection(q)
    union = Polygon(sort_clockwise(np.vstack([hull1, hull2]))).convex_hull

    intersect_coords = np.array(intersect.boundary.coords.xy).transpose()
    union_coords = np.array(union.boundary.coords.xy).transpose()

    iou = intersect.area / union.area

    return intersect_coords, union_coords, iou

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


def get_2d_pca_hull_n_dim(in_points_1, in_points_2):
    pca = PCA(n_components=2)

    pca_1 = sort_clockwise(pca.fit_transform(in_points_1))
    hull_1 = ConvexHull(pca_1)

    pca_2 = sort_clockwise(pca.fit_transform(in_points_2))
    hull_2 = ConvexHull(pca_2)

    # Returns (pca_1, pca_2, intersection, union, iou_score)
    return sort_clockwise(pca_1[hull_1.vertices]), sort_clockwise(pca_2[hull_2.vertices]), *compare_hulls_shapely(pca_1, pca_2)


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

def plot_pca_iou(pca_orig, pca_comparison, intersection, union, iou_score):
    plt.figure()
    plt.plot(pca_orig[:,0], pca_orig[:,1], label="original")
    plt.plot(pca_comparison[:,0], pca_comparison[:,1], label="sorted")
    plt.fill(intersection[:,0], intersection[:,1], alpha=0.2, color="green", label="intersect")
    plt.fill(union[:,0], union[:,1], alpha=0.2, color="blue", label="union")
    plt.title("IoU score: %.2f" % (iou_score))
    plt.legend()
    plt.show()

# https://stackoverflow.com/questions/57500001/numba-failure-with-np-mean
@jit(parallel=True)
def mean_numba(a):

    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)
