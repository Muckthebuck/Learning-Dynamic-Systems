import sys
import numpy as np
import mosek.fusion as mf
import mosek.fusion.pythonic # Provides operators +, -, @, .T, slicing etc.
import logging
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.linalg import eigvals, solve_discrete_lyapunov, solve_discrete_are
from lowres_MVEE import *

np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

def get_optimal_controller(A_list, B_list,
                           Q=None, R=None, x_0=None,
                           uncertainty_region_method=None, simstab_method=None, minimize_wrt=None):
    """
    Computes the min-max optimal controller K_opt.

    Parameters
    ----------
        A_list: ndarray of shape (n_plants, n_states, n_states)
            Set of N discrete-time state matrices A.
        B_list: ndarray of shape (n_plants, n_states, n_inputs)
            Set of N discrete-time input matrices B.

        Q: ndarray of shape (n_states, n_states)
            State cost matrix. Must pe positive semidefinite.
        R: ndarray of shape (n_inputs, n_inputs)
            Input cost matrix. Must be positive definite.
        x_0: ndarray
            Specifies the initial state or covariance matrix used to compute cost J.
            - If x_0 is a vector, J = x₀ᵀ P x₀.
            - If x_0 is a square matrix, J = trace(P @ x₀) i.e. x₀ is a matrix describing the covariance of the initial condition.
            - If x_0 is None, J = trace(P) i.e. assume identity covariance.

        uncertainty_region_method : {'all_plants', 'convhull_oliveira', 'convhull_peaucelle', 'lowresMVEE'}
            Determines how the set of plants is reduced into a more tractable convex region.
            - If None, the default setting specified in OptimalControlSolver class is used.
        simstab_method : {'LMI', 'riccati'}
            Determines how the initial stabilizing controller K_stab is computed.
            - 'LMI' uses a Lyapunov-based feasibility test.
            - 'riccati' uses per-plant Riccati solutions.
            - If None, the default setting specified in OptimalControlSolver class is used.
        minimize_wrt : {'vertex_plants', 'all_plants'}
            Determines how the worst-case J is computed during optimization.
            - 'vertex_plants' evaluates J using the plants at the vertices of the convex region.
            - 'all_plants' evaluates J across the full set of identified plants.
            - If None, the default setting specified in OptimalControlSolver class is used.

    Returns
    -------
        K_opt : ndarray of shape (n_inputs, n_states) or None
            Optimal feedback gain minimizing the worst-case LQR cost.
            Returns None if controller synthesis fails for any reason.
    """

    ocSolver = OptimalControlSolver(A_list, B_list)

    if uncertainty_region_method is not None: ocSolver.uncertainty_region_method = uncertainty_region_method
    if simstab_method is not None:            ocSolver.simstab_method = simstab_method
    if minimize_wrt is not None:              ocSolver.minimize_wrt = minimize_wrt
    if Q is not None:                         ocSolver.Q = Q
    if R is not None:                         ocSolver.R = R
    if x_0 is not None:                       ocSolver.x_0 = x_0
    
    try:
        ocSolver.solve()
        return ocSolver.K_opt
    except Exception as e:
        print(f"Error during controller synthesis: {e}")
        return None


class OptimalControlSolver:

    def __init__(self, A_list, B_list, logger: logging.Logger = None):

        self.set_plants(A_list,B_list)
        
        # Parameters specifying the method for computing K
        self.simstab_method            = 'LMI'
        self.uncertainty_region_method = 'convhull_peaucelle'
        self.minimize_wrt              = 'vertex_plants'
        self.x_0                       = np.eye(self.n_states) # this can be an initial condition x_0, or a covariance matrix Sigma
        self.Q                         = np.eye(self.n_states)
        self.R                         = np.eye(self.n_inputs)

        # Parameters related to simultaneously stabilising K
        self.K_stab = None
        if self.simstab_method == 'LMI': self.set_mosek_params()

        # Parameters related to LQR-optimal K
        self.K_opt = None
        self.Q = np.eye(self.n_states)
        self.R = np.eye(self.n_inputs)
        self.delta_J_tol = 1e-2

        self.logger = logger if logger else logging.getLogger(__name__)

    def set_mosek_params(self):
        self.pd_eps = 1e-3
        self.pd_defn = self.pd_eps * np.eye(2 * self.n_states)
        self.acceptableSoln = mf.AccSolutionStatus.Feasible
        self.prosta = None

    def set_plants(self,A_list,B_list):

        # Ensure numpy arrays, and that the number of plants is consistent
        A_list = np.array(A_list)
        B_list = np.array(B_list)
        assert A_list.shape[0] == B_list.shape[0] 

        # Reshape scalar plants as required
        n_plants = B_list.shape[0]
        if np.isscalar(A_list[0]):
            A_list = A_list.reshape([n_plants,1,1])
        if np.isscalar(B_list[0]):
            B_list = B_list.reshape([n_plants,1,1])

        # Set plants and extract dimensions
        self.A_list = A_list
        self.B_list = B_list
        self.n_plants = self.B_list.shape[0]
        self.n_states = self.B_list.shape[1]
        self.n_inputs = self.B_list.shape[2]

    def solve(self):
        self.compute_vertices()
        self.get_K_stab()
        self.optimise_K_stab()
    
    def compute_vertices(self):

        if self.uncertainty_region_method == 'all_plants':
            self.A_verts = self.A_list
            self.B_verts = self.B_list

        elif self.uncertainty_region_method == 'convhull_oliveira':

            # Compute hulls A,B individually
            (A_verts, n_A, _) = get_hull_verts(self.A_list)
            (B_verts, n_B, _) = get_hull_verts(self.B_list)
            self.logger.info(f" Convex hull of A set has {n_A} vertices")
            self.logger.info(f" Convex hull of B set has {n_B} vertices")
            self.logger.info(f" This will lead to {n_A*n_B} LMI constraints")            

            # Enumerate to get the combined vertices (A_ij, B_ij)
            A_idx, B_idx = np.meshgrid(np.arange(n_A), np.arange(n_B), indexing='ij')
            A_idx = A_idx.flatten()
            B_idx = B_idx.flatten()
            A_paired = [A_verts[i] for i in A_idx]
            B_paired = [B_verts[i] for i in B_idx]

            self.A_verts = np.array(A_paired)
            self.B_verts = np.array(B_paired)        

        elif self.uncertainty_region_method == 'convhull_peaucelle':

            # Compute AB hull
            AB_list = [np.hstack((A, B)) for A, B in zip(self.A_list, self.B_list)]
            AB_list = np.array(AB_list)
            (AB_verts, _, _) = get_hull_verts(AB_list)
            self.A_verts = AB_verts[:, :, :self.n_states]
            self.B_verts = AB_verts[:, :,  self.n_states:]
            
        elif self.uncertainty_region_method == 'lowresMVEE':

            # Pair the A and B
            AB_list = [np.hstack((A, B)) for A, B in zip(self.A_list, self.B_list)]
            AB_list = np.array(AB_list)

            # Vectorise the uncertain entries and compute the enclosing polytope
            pts, where_unc = vectorise_unc_entries(AB_list)
            lrMVEE = LowResMVEE(pts.T, max_n_verts=self.max_n_verts)
            v = lrMVEE.vertices.T
            n = v.shape[0]

            # Map back to matrix space
            M0 = AB_list[0]
            (AB_verts, _, _) = map_v_to_mat(M0, where_unc, v, n)
            self.A_verts = AB_verts[:, :, :self.n_states]
            self.B_verts = AB_verts[:, :,  self.n_states:]

    def get_K_stab(self):

        if self.simstab_method == 'LMI':
            self.solve_LMIs()

        elif self.simstab_method == 'riccati':

            K = solve_riccati(self.A_verts, self.B_verts, self.Q, self.R, self.x_0)
            J = compute_worst_case_J(K, self.A_list, self.B_list, self.Q, self.R, self.x_0)

            if J == np.inf:
                self.logger.info(" Riccati heuristic failed to find a simultaneously stabilising feedback gain!")
                self.K_stab = None
            else:
                self.logger.info(f" Simultaneously stabilising feedback gain is K = {K}")
                self.K_stab = K

    def solve_LMIs(self):
        M, G, L = self.define_LMI_problem()
        self.solve_LMI_problem(M, G, L)
    
    def define_LMI_problem(self):
        assert self.A_verts.shape[0] == self.B_verts.shape[0]
        n_verts = self.A_verts.shape[0]

        M = mf.Model("Simultaneous stabilisation LMI solver")
        G = M.variable("G", [self.n_states, self.n_states], mf.Domain.unbounded())
        L = M.variable("L", [self.n_inputs, self.n_states], mf.Domain.unbounded())
        P_i = [M.variable(f"P_{i}", [self.n_states, self.n_states], mf.Domain.inPSDCone()) for i in range(n_verts)]

        for i in range(n_verts):
            A, B = self.A_verts[i], self.B_verts[i]
            P = P_i[i]
            AG_plus_BL = A @ G + B @ L
            LMI_i = mf.Expr.vstack(
                mf.Expr.hstack(P, AG_plus_BL),
                mf.Expr.hstack(AG_plus_BL.T, G + G.T - P)
            )
            M.constraint(LMI_i - self.pd_defn, mf.Domain.inPSDCone(2 * self.n_states))

        self.logger.info(f" Defined {n_verts} LMI constraints")
        return M, G, L

    def solve_LMI_problem(self, M, G, L):
        try:
            M.acceptedSolutionStatus(self.acceptableSoln)
            M.solve()
            solsta = M.getPrimalSolutionStatus()

            G_val = np.array(G.level()).reshape(self.n_states, self.n_states)
            L_val = np.array(L.level()).reshape(self.n_inputs, self.n_states)
            self.K_stab = -L_val @ np.linalg.inv(G_val)

            self.logger.info(f" {solsta}, simultaneously stabilising feedback gain is K = {self.K_stab}")

        except:
            self.logger.info(f" Could not find stabilising controller!")
            self.prosta = M.getProblemStatus()
            if self.prosta == mf.ProblemStatus.PrimalInfeasible:
                self.logger.info(" Infeasible problem: primal infeasibility certificate found.")
            elif self.prosta == mf.ProblemStatus.DualInfeasible:
                self.logger.info(" Infeasible problem: dual infeasibility certificate found.")
            elif self.prosta == mf.ProblemStatus.Unknown:
                self.logger.info(" MOSEK returned unknown problem status (invalid licence, or solver ran into stall/numerical issues).")
            else:
                self.logger.info(f"Unknown error has occurred! MOSEK ProblemStatus: {self.prosta}")
    
    def optimise_K_stab(self):
        
        assert self.K_stab is not None
        assert compute_worst_case_J(self.K_stab, self.A_list, self.B_list, self.Q, self.R, self.x_0) != np.inf

        if self.minimize_wrt == 'vertex_plants':
            objective = lambda K: compute_worst_case_J(K.reshape(self.n_inputs,self.n_states), self.A_verts, self.B_verts, self.Q, self.R, self.x_0)
        elif self.minimize_wrt == 'all_plants':
            objective = lambda K: compute_worst_case_J(K.reshape(self.n_inputs,self.n_states), self.A_list, self.B_list, self.Q, self.R, self.x_0)

        # Suppress runtime warnings related to invalid values (J==Inf will be encountered during optimisation)
        np.seterr(invalid='ignore')
        results = minimize(objective, self.K_stab.flatten(), method="BFGS", tol=self.delta_J_tol, options={"disp": False})
        np.seterr(invalid='warn')
        
        self.K_opt = results.x.reshape(self.n_inputs,self.n_states)
        self.logger.info(f" Optimised the simultaneously stabilising feedback gain, yielding K = {self.K_opt}")
        
        worst_J = compute_worst_case_J(self.K_stab, self.A_list, self.B_list, self.Q, self.R, self.x_0)
        worst_J_optimized = compute_worst_case_J(self.K_opt, self.A_list, self.B_list, self.Q, self.R, self.x_0)
        self.logger.info(f" Worst J using initial K: {worst_J:.4f}")
        self.logger.info(f" Worst J using optimized K: {worst_J_optimized:.4f}")
        
    def set_max_n_verts(self, n):
        self.max_n_verts = n

def dlqr(A, B, Q, R):
    """
    Solves the infinte horizon problem for discrete state-space system (A,B), with costs defined by (Q,R).
    Returns the optimal full-state feedback controller K and Riccati solution P.
    """
    P = solve_discrete_are(A, B, Q, R)
    if np.isscalar(A):
        K = (R + B * P * B)**-1 * (B * P * A)
        K = K.squeeze()
        P = P.squeeze()
    else:
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def solve_riccati(A_in_set, B_in_set, Q, R, x_0=None):
    """
    Heuristic for finding the min-max optimal controller.
    Evaluates the worst-case Riccati solution P, and returns the LQR controller associated with this plant.
    This method is often faster than the LMI-based solution, but the controller is not guaranteed to be stabilising or optimal, particularly with large parameter uncertainty.
    """

    # Ensure numpy arrays
    A_in_set = np.array(A_in_set)
    B_in_set = np.array(B_in_set)
    Q = np.array(Q)
    R = np.array(R)
    x_0 = np.array(x_0) if x_0 is not None else None
    
    # Ensure number of plants is consistent, and reshape scalar plants as required
    assert A_in_set.shape[0] == B_in_set.shape[0]
    n_plants = B_in_set.shape[0]
    
    if np.isscalar(A_in_set[0]):
        A_in_set = A_in_set.reshape([n_plants,1,1])
    if np.isscalar(B_in_set[0]):
        B_in_set = B_in_set.reshape([n_plants,1,1])

    n_states = B_in_set.shape[1]
    n_inputs = B_in_set.shape[2]

    # Solve
    K_set = np.zeros([n_plants, n_inputs, n_states])
    P_set = np.zeros([n_plants, n_states, n_states])
    J_set = np.zeros(n_plants)

    for i in range(n_plants):
        K, P = dlqr(A_in_set[i], B_in_set[i], Q, R)
        K_set[i] = K
        P_set[i] = P

        if x_0 is None:
            J_set[i] = np.trace(P) # Assume identity covariance in x_0
        elif x_0.shape[0] == x_0.shape[1]:
            Sigma = x_0
            J_set[i] = np.trace(P @ Sigma) # Assume covariance in x_0 described by Sigma
        else:
            J_set[i] = x_0.T @ P @ x_0 # Calculate J w.r.t. a specific x_0

    max_J_idx = np.argmax(J_set)
    optimal_K = K_set[max_J_idx]

    return optimal_K

def compute_worst_case_J(K, A_in_set, B_in_set, Q, R, x_0=None):

    # Given u=-Kx, find the worst-case J across the plants (A_in_set, B_in_set)
    max_J = -np.inf
    for A, B in zip(A_in_set, B_in_set):
        max_J = max(calc_J(A, B, K, Q, R, x_0), max_J)
    return max_J

def calc_J(A, B, K, Q, R, x_0=None):
    
    """
    Compute the LQR cost J, for a given plant (A,B) under feedback u_k = -K x_k.
    """

    stability_radius = 1 - 1e-6  # Slightly stricter stability criterion

    Phi = A - B @ K
    is_scalar_R = np.isscalar(R) or (isinstance(R, np.ndarray) and R.ndim == 0)
    Q_tilde = Q + K.T @ (R * K) if is_scalar_R else Q + K.T @ R @ K

    if np.all(np.abs(eigvals(Phi)) < stability_radius):
        S = solve_discrete_lyapunov(Phi.T, Q_tilde)
        if x_0 is None:
            # Calculate assuming identity covariance in x_0
            J = np.trace(S)
        elif x_0.shape[0] == x_0.shape[1]:
            # Calculate assuming covariance in x_0 described by Sigma
            Sigma = x_0
            J = np.trace(S @ Sigma)
        else:
            # Calculate J w.r.t. a specific x_0
            J = x_0.T @ S @ x_0
        return J.item()
    else:
        return np.inf

def is_uncertain_parameter(matrix_list):
    unc = np.max(matrix_list,axis=0) - np.min(matrix_list,axis=0)
    return unc > 0

def vectorise_unc_entries(matrix_list: np.ndarray):
    n_plants = matrix_list.shape[0]
    where_unc = is_uncertain_parameter(matrix_list)
    unc_entries = [matrix_list[i][where_unc] for i in range(n_plants)]
    return np.array(unc_entries), where_unc

def compute_convhull(vectors):

    n_d = vectors.shape[1]

    if n_d > 1:
        vertex_indices = ConvexHull(vectors).vertices # NB: qhull_options='QJ' results in a less precise hull, but is faster and often more reliable computation
        vertex_indices = np.unique(vertex_indices)
        vertices = vectors[vertex_indices, :]
        n_vertices = vertices.shape[0]
        return vertices, n_vertices
    
    elif n_d == 1:
        # Special case: scalar uncertainty
        max_idx = np.argmax(vectors)
        min_idx = np.argmin(vectors)
        max = vectors[max_idx]
        min = vectors[min_idx]
        vertices = np.array([min, max])
        n_vertices = 2
        return vertices, n_vertices

def map_v_to_mat(M0, where_unc, v, n):

    # All matrices are variants of M0
    matrix_verts = [M0.copy() for _ in range(n)]

    for i in range(n):
        matrix_verts[i][where_unc] = v[i]

    return np.array(matrix_verts), n, where_unc

def get_hull_verts(matrix_list: np.ndarray):

    # Find and vectorise the uncertain entries
    unc_entries_vec, where_unc = vectorise_unc_entries(matrix_list)

    # Handle special case: no uncertain params
    if np.all(~where_unc):
        vertices = matrix_list[0]
        n_vertices = 1
        return vertices, n_vertices, where_unc

    # Compute the convex hull in vector space
    v, n = compute_convhull(unc_entries_vec)

    # Map back to matrix space
    M0 = matrix_list[0]
    return map_v_to_mat(M0, where_unc, v, n)
    
