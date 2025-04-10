import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, diagflat
from numpy.linalg import inv, norm, eig
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from scipy.spatial import ConvexHull

class LowResMVEE:
    def __init__(self, pts, max_n_verts=6):
        """
        pts (array): set of points in the confidence region, shape (ndim,npoints).
        """
        self.pts = np.array(pts)
        self.nD = self.pts.shape[0]
        self.n_pts = self.pts.shape[1]
        self.khachiyan_tol = 1e-1
        self.vertices = self.compute_vertices(max_n_verts)

    def compute_vertices(self, max_n_verts):
        """
        Compute vertices of the low-res MVEE that encloses self.pts.
        Returns the vertices as an array with shape (ndim, npoints).
        """
        ##### Adjust mesh_density s.t. the approximation is as high-res as possible:
        #       - no more than max_n_verts vertices, and
        #       - precomputed scaling factor exists for this (nD, mesh_density) pair.
        # Minimum mesh_density is 3, which corresponds to a cross-polytope (diamond/octahedron/...).
        mesh_density = 3
        while compute_n_vertices(self.nD, mesh_density+1) <= max_n_verts:
            mesh_density += 1
        while mesh_density >= 3:
            try:
                sf = scaling_factor(self.nD, mesh_density)
                break
            except Exception:
                mesh_density -= 1
        else:
            raise RuntimeError("Unable to find precomputed scaling factor. Check nD and max_n_verts are realistic.")

        ##### Compute the MVEE using Khachiyan's Algorithm
        A, c = min_vol_ellipse(self.pts,self.khachiyan_tol)
        [D, P] = eig(A)
        radii = 1 / sqrt(D)

        #### Scale this MVEE so that it actually encloses the points
        # Undo the transformation defined by (A,c) s.t. the ellipse is mapped to the unit circle. Apply this same transformation to our data.
        tm = diagflat(sqrt(D)) @ P.T
        pts_mapped_to_unit_sphere = tm @ (self.pts-c)

        # Scale the unit circle so it encompasses the points.
        radial_scaling_to_correct_undershoot = np.max(norm(pts_mapped_to_unit_sphere,axis=0))
        A_sf = 1/radial_scaling_to_correct_undershoot**2

        # Apply the scaling factor to the ellipse
        new_D = A_sf * D
        new_D = diagflat(new_D)

        #### Compute the low-res polyhedron that bounds the corrected ellipse
        V = get_mesh_vertices(self.nD,mesh_density) # polyhedron coordinates on a unit sphere
        D_sf = scaling_factor(self.nD,mesh_density) * radial_scaling_to_correct_undershoot

        D_sqrt_inv = diagflat(1/sqrt(D))
        tm = P @ (D_sf * D_sqrt_inv)
        V = (tm @ V.T) + c

        return V


def plot_polyhedron(V,ax=None):
    """
    Convenient function for plotting a polytope.
    V (array): vertices of the polytope, with shape (number of dimensions, number of vertices).
    """
    nD = V.shape[0]
    V = V.T # passed argument has vertices has column vectors, but ConvexHull expects them as row vectors
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=None if nD < 3 else '3d')
        ax.axis('equal')
        ax.grid(True)

    if nD == 1:
        ax.scatter([np.min(V), np.max(V)], [0, 0])
        ax.plot([np.min(V), np.max(V)], [0, 0])
    elif nD == 2:
        hull = ConvexHull(V)
        ax.fill(V[hull.vertices, 0], V[hull.vertices, 1], color='b', alpha=0.2)
    elif nD == 3:
        hull = ConvexHull(V)
        # ax.plot_trisurf(V[hull.vertices, 0], V[hull.vertices, 1], V[hull.vertices, 2], color='#0F72BD', alpha=0.2)
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=hull.simplices, cmap='coolwarm', alpha=0.4)
    else:
        print("nD > 3: cannot plot polyhedron.")


def scaling_factor(nD, mesh_density):
    """
    Using get_mesh_vertices(), the mesh has unit-magnitude vertices: it is ENCLOSED BY the unit circle/sphere/n-sphere.
    We want it the other way around: the mesh should enclose the unit circle/sphere/n-sphere.
    Scaling the vertices by scaling_factor() will ensure the mesh does just that.
    """
    scaling_factors = {
        "nD" : {
            2  : {'md' : {3: 1.4143, 4: 1.1548, 5: 1.0824, 6: 1.0515, 7: 1.0353, 8: 1.0258, 9: 1.0196, 10: 1.0155}},
            3  : {'md' : {3: 1.7072, 4: 1.3334, 5: 1.1586, 6: 1.1056, 7: 1.0694, 8: 1.0521, 9: 1.0389}},
            4  : {'md' : {3: 1.9143, 4: 1.5397, 5: 1.2289, 6: 1.1625, 7: 1.1023, 8: 1.0792}},
            5  : {'md' : {3: 2.0607, 4: 1.7778, 5: 1.2939, 6: 1.2223, 7: 1.1341}},
            6  : {'md' : {3: 2.1643, 4: 2.0529, 5: 1.3539, 6: 1.2852}},
            7  : {'md' : {3: 2.2375, 4: 2.3704, 5: 1.4093}},
            8  : {'md' : {3: 2.2893, 4: 2.7371}},
            9  : {'md' : {3: 2.3259}},
            10 : {'md' : {3: 2.3518}},
            11 : {'md' : {3: 2.3701}},
            12 : {'md' : {3: 2.3830}},
            13 : {'md' : {3: 2.3922}},
            14 : {'md' : {3: 2.3986}},
            15 : {'md' : {3: 2.4032}},
            16 : {'md' : {3: 2.4065}},
        }
    }

    try:
        return scaling_factors['nD'][nD]['md'][mesh_density]
    except:
        raise Exception("No precomputed scaling factor: check nD and mesh_density are realistic.")


def compute_n_vertices(n_states,m):
    """
    Computes the number of vertices of a mesh computed using get_mesh_vertices().
    Recursive formula.

    """
    if n_states == 2:
        return 2*m - 2
    else:
        return compute_n_vertices(n_states-1,m)*(m-2) + 2
    

def get_mesh_vertices(n, mesh_density):
    """
    Computes a mesh approximating an n-sphere.
    Returns the vertices of this mesh.
    Use generalised spherical coordinates.

    mesh_density: controls the level of approximation.
        e.g. if nD == 2, with mesh_density = 3 the circle is approximated as a diamond
        e.g. if nD == 2, with mesh_density = 4 the circle is approximated as a hexagon
        e.g. if nD == 2, with mesh_density = 5 the circle is approximated as an octagon
    mesh_density == 3 always returns the most basic shape, an n-dimensional cross-polytope.
    mesh_density controls the number of divisions from 0 to pi, or 0 to 2pi for the final spherical coordinate.
        e.g. mesh_density == 3 gives [0, pi/2, pi] for all theta_i except the final one, which will be [0, pi/2, pi, 3pi/2]
        e.g. mesh_density == 5 similarly gives [0, pi/4, pi/2, 3pi/4, pi] and [0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4]
    
    """

    theta = [np.linspace(0, np.pi, mesh_density, endpoint=True) for _ in range(n-1)]
    theta[-1] = np.linspace(0, 2*np.pi, 2*(mesh_density-1), endpoint=False)

    Theta = np.meshgrid(*theta, indexing='xy')
    Theta_flat = [t.flatten() for t in Theta]
    n_coords = len(Theta_flat[0])

    sin_terms = np.ones((n_coords, n-1))
    for i in range(n-1):
        if i==0:
            sin_terms[:,0] = np.sin(Theta_flat[0])
        else:
            sin_terms[:,i] = np.sin(Theta_flat[i]) * sin_terms[:,i-1]

    X = np.ones((n_coords, n))
    for i in range(n):
        if i==0:
            X[:,i] = np.cos(Theta_flat[0])
        elif i==n-1:
            X[:,i] = sin_terms[:,i-1]
        else:
            X[:,i] = np.cos(Theta_flat[i]) * sin_terms[:,i-1]
    
    return uniquetol(X)


def uniquetol(V, decimals=3):
    """
    Returns the unique rows of V, with some tolerance allowed.

    """
    rounded = V.round(decimals)
    return np.unique(rounded,axis=0)


def scatter_nD(pts, ax=None):
    """
    Convenient function for scatter plots of 1D/2D/3D data.

    """
    nD = pts.shape[0]
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=None if nD < 3 else '3d')
        ax.axis('equal')
        ax.grid(True)

    if nD == 1:
        ax.scatter(pts[0,:], np.zeros_like(pts[0,:]))
    elif nD == 2:
        ax.scatter(pts[0,:], pts[1,:]) #, c='b', marker='o')
    elif nD == 3:
        ax.scatter(pts[0,:], pts[1,:], pts[2,:]) #, c='b', marker='o')
    else:
        Warning("nD > 3: cannot scatter plot.")
    

def min_vol_ellipse(pts, tolerance=1e-1):
    """
    Uses Khachiyan's Algorithm to find the MVEE enclosing a set of points.
    NB: this is only an approximate algorithm, with no guarantee that it is enclosing, so some post-processing is required.

    Original MATLAB script by Nima Moshtagh (nima@seas.upenn.edu), University of Pennsylvania.
    https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid

    """

    d, N = pts.shape # d == dimensionality, N == number of points
    Q = np.zeros((d+1,N))
    Q[0:d,:] = pts[0:d,0:N]
    Q[d,:] = np.ones((1,N))

    count = 1
    err = 1
    u = 1/N * np.ones((N,1))

    while err > tolerance:
        X = Q @ diagflat(u) @ Q.T;       # X = \sum_i ( u_i * q_i * q_i')  is a (d+1)x(d+1) matrix
        M = np.diag(Q.T @ inv(X) @ Q) # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d -1)/((d+1)*(maximum-1))
        new_u = (1 - step_size)*u
        new_u[j] = new_u[j] + step_size
        count = count + 1
        err = norm(new_u - u)
        u = new_u
    U = diagflat(u)
    A = (1/d) * inv(pts @ U @ pts.T - (pts @ u) @ (pts @ u).T)
    c = pts @ u
    return A,c


def ellipse_plot(A, C, ax=None, N=20):
    """
    ellipse_plot(A, C, N) plots a 2D ellipse or a 3D ellipsoid 
    represented in the "center" form:
            
            (x-C)' A (x-C) <= 1

    A and C could be the outputs of the function: "MinVolEllipse" in MATLAB,
    which computes the minimum volume enclosing ellipsoid containing a set of points.

    Inputs: 
    A: a 2x2 or 3x3 matrix.
    C: a 2D or 3D vector representing the center of the ellipsoid.
    N: the number of grid points for plotting the ellipse; Default: N = 20.

    Original MATLAB script by Nima Moshtagh (nima@seas.upenn.edu), University of Pennsylvania.
    https://au.mathworks.com/matlabcentral/fileexchange/13844-plot-an-ellipse-in-center-form

    """

    if len(C) == 3:
        Type = '3D'
    elif len(C) == 2:
        Type = '2D'
    elif len(C) == 1:
        Type = '1D'
    else:
        Warning('Cannot plot an ellipse with more than 3 dimensions!')
        return

    U, D, Vt = svd(A)
    V = Vt.T  # transpose to get the correct orientation

    if Type == '1D':
        a = 1 / sqrt(D[0])

        if ax==None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axis('equal')
            ax.grid(True)
        ax.plot([C[0] - a, C[0] - a], [-0.1, 0.1], 'k--')
        ax.plot([C[0] + a, C[0] + a], [-0.1, 0.1], 'k--')
        # ax.plot(C[0], 0, 'r*')

    elif Type == '2D':
        # Get the major and minor axes
        a = 1 / sqrt(D[0])
        b = 1 / sqrt(D[1])

        theta = np.arange(0, 2*np.pi + 1/N, 1/N)
        state = np.array([a * np.cos(theta), b * np.sin(theta)])
        X = np.dot(V, state)
        X[0, :] += C[0]
        X[1, :] += C[1]

        if ax==None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axis('equal')
            ax.grid(True)
        ax.plot(X[0, :], X[1, :])
        # ax.plot(C[0], C[1], 'r*')

    elif Type == '3D':
        a = 1 / sqrt(D[0])
        b = 1 / sqrt(D[1])
        c = 1 / sqrt(D[2])

        u = np.linspace(0, 2 * np.pi, N + 1)
        v = np.linspace(0, np.pi, N + 1)
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones(np.size(u)), np.cos(v))

        # Rotate and center the ellipsoid to the actual center point
        XX, YY, ZZ = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        for i in range(len(x)):
            for j in range(len(x)):
                point = np.array([x[i,j], y[i,j], z[i,j]])
                P = np.dot(V, point)
                XX[i,j] = P[0] + C.squeeze()[0]
                YY[i,j] = P[1] + C.squeeze()[1]
                ZZ[i,j] = P[2] + C.squeeze()[2]

        # Plot the 3D ellipsoid
        if ax==None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, color='b', alpha=0.5)
        # ax.scatter(C[0], C[1], C[2], color='r', marker='*')