import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.linalg import svd

# class Polyhedron:
#     def __init__(self, points, tol=1e-6, mesh_density=5):
#         self.points = np.array(points)
#         self.tol = tol
#         self.mesh_density = mesh_density
#         self.nD = self.points.shape[1]
#         self.V = None
#         self.n_pts = None
#         self.compute_MVEE()

#     def compute_MVEE(self):
#         A, c = self.min_vol_ellipse(self.points, self.tol)
#         self.V, self.n_pts = self.get_lowres_MVEE(A, c)
    
def plot_polyhedron(V,ax=None):
    nD = V.shape[1]
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
        ax.plot_trisurf(V[hull.vertices, 0], V[hull.vertices, 1], V[hull.vertices, 2], color='#0F72BD', alpha=0.2)
    else:
        print("nD > 3: cannot plot polyhedron.")

def scaling_factor(nD, mesh_density):
    scaling_factors = {
        "nD" : {
            2 : {'md' : {3: 1.4143, 4: 1.1548, 5: 1.0824, 6: 1.0515, 7: 1.0353, 8: 1.0258, 9: 1.0196, 10: 1.0155}},
            3 : {'md' : {3: 1.7072, 4: 1.3334, 5: 1.1586, 6: 1.1056, 7: 1.0694, 8: 1.0521, 9: 1.0389}},
            4 : {'md' : {3: 1.9143, 4: 1.5397, 5: 1.2289, 6: 1.1625, 7: 1.1023, 8: 1.0792}},
            5 : {'md' : {3: 2.0607, 4: 1.7778, 5: 1.2939, 6: 1.2223, 7: 1.1341}},
            6 : {'md' : {3: 2.1643, 4: 2.0529, 5: 1.3539, 6: 1.2852}},
            7 : {'md' : {3: 2.2375, 4: 2.3704, 5: 1.4093}},
            8 : {'md' : {3: 2.2893, 4: 2.7371}}
        }
    }

    try:
        return scaling_factors['nD'][nD]['md'][mesh_density]
    except:
        raise Exception("No precomputed scaling factor: check nD and mesh_density are realistic.")
    
def compute_n_vertices(n_states,m):
    if n_states == 2:
        return 2*m - 2
    else:
        return compute_n_vertices(n_states-1,m)*(m-2) + 2
    
def get_mesh_vertices(n, mesh_density):
    # Generate mesh vertices for n-dimensional space
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
            sin_terms[:,i] = np.sin(Theta_flat[i-1]) * np.sin(Theta_flat[i])

    X = np.ones((n_coords, n))
    for i in range(n):
        if i==0:
            X[:,i] = np.cos(Theta_flat[0])
        elif i==n-1:
            X[:,i] = sin_terms[:,n-2]
        else:
            X[:,i] = np.cos(Theta_flat[i]) * sin_terms[:,i-1]
    
    return X # np.unique(X, axis=0) probably not good, need tol

def scatter_nD(pts, ax=None):
    nD = pts.shape[1]
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=None if nD < 3 else '3d')
        ax.axis('equal')
        ax.grid(True)

    if nD == 1:
        ax.scatter(pts[:,0], np.zeros_like(pts[:,0]))
    elif nD == 2:
        ax.scatter(pts[:,0], pts[:,1]) #, c='b', marker='o')
    elif nD == 3:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2]) #, c='b', marker='o')
    else:
        Warning("nD > 3: cannot scatter plot.")
    
def min_vol_ellipse(pts, tolerance):
    """
    Original MATLAB script available at
        https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
    Author:
        Nima Moshtagh (nima@seas.upenn.edu)
        University of Pennsylvania
    """

    d, N = pts.shape # d == dimensionality, N == number of points
    Q = np.zeros((d+1,N))
    Q[0:d,:] = pts[0:d,0:N]
    Q[d,:] = np.ones((1,N))

    count = 1
    err = 1
    u = 1/N * np.ones((N,1))

    tolerance = 1e-1

    while err > tolerance:
        X = Q @ np.diagflat(u) @ Q.T;       # X = \sum_i ( u_i * q_i * q_i')  is a (d+1)x(d+1) matrix
        M = np.diag(Q.T @ np.linalg.inv(X) @ Q) # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d -1)/((d+1)*(maximum-1))
        new_u = (1 - step_size)*u
        new_u[j] = new_u[j] + step_size
        count = count + 1
        err = np.linalg.norm(new_u - u)
        u = new_u
    U = np.diagflat(u)
    A = (1/d) * np.linalg.inv(pts @ U @ pts.T - (pts @ u) @ (pts @ u).T)
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

    Original MATLAB script available at
        https://au.mathworks.com/matlabcentral/fileexchange/13844-plot-an-ellipse-in-center-form
    Author:
        Nima Moshtagh (nima@seas.upenn.edu)
        University of Pennsylvania
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
        a = 1 / np.sqrt(D[0])

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
        a = 1 / np.sqrt(D[0])
        b = 1 / np.sqrt(D[1])

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
        a = 1 / np.sqrt(D[0])
        b = 1 / np.sqrt(D[1])
        c = 1 / np.sqrt(D[2])

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