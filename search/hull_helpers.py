import numpy as np
from numba import njit, jit, prange


from search.hull_sutherland_hodgman import PolygonClipper

# TODO: JIT THESE
def expand_convex_hull(hull_points, expansion_factor=0.05):
    """Expand the convex hull by a small factor to make it slightly larger."""
    hull_center = np.mean(hull_points, axis=0)
    expanded_points = []
    
    for point in hull_points:
        direction = point - hull_center
        expanded_point = point + direction * expansion_factor
        expanded_points.append(expanded_point)
        # expanded_points.append(expanded_point)
    
    return np.array(expanded_points)

def compare_hulls(hull1, hull2):
    clipper = PolygonClipper(warn_if_empty=True)
    clipped_polygon = clipper(hull1, hull2)
    return clipped_polygon


# https://stackoverflow.com/questions/57500001/numba-failure-with-np-mean
@jit(parallel=True)
def mean_numba(a):

    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)
