import sys
import numpy as np
import mosek.fusion as mf
import mosek.fusion.pythonic # Provides operators +, -, @, .T, slicing etc.
import logging
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.linalg import eigvals, solve_discrete_lyapunov, solve_discrete_are

np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

def get_optimal_controller(A_list, B_list, method='LMI', Q=None, R=None, x_0=None):
    ocSolver = OptimalControlSolver(A_list, B_list)
    if Q is not None: ocSolver.Q = Q
    if R is not None: ocSolver.R = R
    if x_0 is not None: ocSolver.x_0 = x_0
    
    ocSolver.solve(method=method)
    if ocSolver.prosta == None:
        return ocSolver.K_opt

class OptimalControlSolver:
    def __init__(self, A_list, B_list, logger: logging.Logger = None):

        self.logger = logger if logger else logging.getLogger(__name__)
        
        self.set_plants(A_list,B_list)

        # Parameters related to stabilising K
        self.pd_eps = 1e-3
        self.acceptableSoln = mf.AccSolutionStatus.Feasible
        self.prosta = None
        self.K_stab = None

        # Parameters related to LQR-optimal K
        self.Q = np.eye(self.n_states)
        self.R = np.eye(self.n_inputs)
        self.x_0 = None
        self.delta_J_tol = 1e-2
        self.K_opt = None
    
    def set_plants(self,A_list,B_list):
        A_list = np.array(A_list)
        B_list = np.array(B_list)

        # Ensure number of plants is consistent, and reshape scalar plants as required
        n_plants = B_list.shape[0]
        assert A_list.shape[0] == B_list.shape[0]
        if np.isscalar(A_list[0]):
            A_list = A_list.reshape([n_plants,1,1])
        if np.isscalar(B_list[0]):
            B_list = B_list.reshape([n_plants,1,1])

        self.A_list = A_list
        self.B_list = B_list
        self.n_plants = self.B_list.shape[0]
        self.n_states = self.B_list.shape[1]
        self.n_inputs = self.B_list.shape[2]

    def get_plant_hulls(self):
        # Using ConvexHull method for now - MVEE method will be implemented later as needed
        self.A_verts, self.n_A_verts, self.unc_A = get_hull_matrices(self.A_list)
        self.B_verts, self.n_B_verts, self.unc_B = get_hull_matrices(self.B_list)
        self.logger.info(f" Convex hull of A set has {self.n_A_verts} vertices")
        self.logger.info(f" Convex hull of B set has {self.n_B_verts} vertices")
        self.logger.info(f" This will lead to {self.n_A_verts*self.n_B_verts} LMI constraints")
    
    def solve(self, method='LMI'):
        if method == 'riccati':
            self.solve_heuristically()
        else:
            self.solve_LMIs()
            if self.prosta == None:
                self.optimise_K()

    def solve_LMIs(self):

        self.get_plant_hulls()

        M = mf.Model("Simultaneous stabilisation LMI solver")
        # M.setLogHandler(sys.stdout) # enable for verbose MOSEK status updates
        G = M.variable("G", [self.n_states, self.n_states], mf.Domain.unbounded())
        L = M.variable("L", [self.n_inputs, self.n_states], mf.Domain.unbounded())
        P_ij = [[M.variable(f"P_{i}_{j}", [self.n_states, self.n_states], mf.Domain.inPSDCone()) for j in range(self.n_B_verts)] for i in range(self.n_A_verts)]

        for i in range(self.n_A_verts):
            for j in range(self.n_B_verts):
                # Construct the ij'th LMI constraint
                A, B = self.A_verts[i], self.B_verts[j]
                P = P_ij[i][j]
                AG_plus_BL =  A @ G + B @ L
                LMI_ij = mf.Expr.vstack(
                    mf.Expr.hstack(P, AG_plus_BL),
                    mf.Expr.hstack(AG_plus_BL.T, G + G.T - P)
                )
                pd_defn = self.pd_eps * np.eye(2 * self.n_states)
                M.constraint(LMI_ij - pd_defn, mf.Domain.inPSDCone(2 * self.n_states))
        self.logger.info(f" Defined {self.n_A_verts*self.n_B_verts} LMI constraints")

        try:
            M.acceptedSolutionStatus(self.acceptableSoln)
            M.solve()
            solsta = M.getPrimalSolutionStatus()

            G_val = np.array(G.level()).reshape(self.n_states, self.n_states)
            L_val = np.array(L.level()).reshape(self.n_inputs, self.n_states)
            self.K_stab = -L_val @ np.linalg.inv(G_val) # K = -L * inv(G)

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
        
    
    def optimise_K(self):
        
        if self.K_stab is None:
            self.solve_LMIs()

        assert compute_worst_case_J(self.K_stab, self.A_list, self.B_list, self.Q, self.R, self.x_0) != np.inf

        # objective      = lambda K: compute_worst_case_J(K.reshape(self.n_inputs,self.n_states), self.A_list, self.B_list, self.Q, self.R, self.x_0)
        objective      = lambda K: compute_worst_case_J(K.reshape(self.n_inputs,self.n_states), self.A_verts, self.B_verts, self.Q, self.R, self.x_0)
        disp_current_J = lambda K: None # self.logger.info(f"\t Current worst-case J: {compute_worst_case_J(K.reshape(self.n_inputs,self.n_states), self.A_list, self.B_list, self.Q, self.R, self.x_0)}")

        np.seterr(invalid='ignore') # Suppress runtime warnings related to invalid values (J==Inf)
        results = minimize(objective, self.K_stab.flatten(), method="BFGS", tol=self.delta_J_tol, options={"disp": True}, callback=disp_current_J)
        np.seterr(invalid='warn')
        
        self.K_opt = results.x.reshape(self.n_inputs,self.n_states)
        self.logger.info(f" Optimised the simultaneously stabilising feedback gain, yielding K = {self.K_opt}")
        
        worst_J = compute_worst_case_J(self.K_stab, self.A_list, self.B_list, self.Q, self.R, self.x_0)
        worst_J_optimized = compute_worst_case_J(self.K_opt, self.A_list, self.B_list, self.Q, self.R, self.x_0)
        self.logger.info(f" Worst J using initial K: {worst_J:.4f}")
        self.logger.info(f" Worst J using optimized K: {worst_J_optimized:.4f}")
    
    def solve_heuristically(self):
        K_soln = get_optimal_controller_riccati_method(self.A_list, self.B_list, self.Q, self.R, self.x_0)
        self.logger.info(f" Computed solution via Riccati heuristic, K = {K_soln}")

        worst_J = compute_worst_case_J(K_soln, self.A_list, self.B_list, self.Q, self.R, self.x_0)
        if worst_J == np.inf:
            self.logger.info(" Riccati heuristic did not return a stabilising controller!")
        else:
            self.logger.info(f" Worst J using Riccati heuristic K: {worst_J:.4f}")
            self.K_stab = K_soln
            self.optimise_K()


def is_uncertain_parameter(M):
    M = np.array(M)
    unc = np.max(M,axis=0) - np.min(M,axis=0)
    return unc > 0

def get_unc_entries(M):
    M = np.array(M)
    n_plants = M.shape[0]
    where_unc = is_uncertain_parameter(M)
    unc_entries = [M[i][where_unc] for i in range(n_plants)]
    return np.array(unc_entries)

def get_hull(M_list):
    M_list = np.array(M_list)
    unc_entries = get_unc_entries(M_list)

    n_unc_entries = unc_entries.shape[1]
    if n_unc_entries > 1:
        v_idx = np.unique(ConvexHull(unc_entries, qhull_options='QJ').vertices) # Extract unique vertices of the convex hull
        vertices = [unc_entries[i] for i in v_idx]
        vertices = np.array(vertices)
        n_vertices = vertices.shape[0]
        return vertices, n_vertices
    elif n_unc_entries == 1:
        # Special case: scalar
        max_idx = np.argmax(unc_entries)
        min_idx = np.argmin(unc_entries)
        max = unc_entries[max_idx]
        min = unc_entries[min_idx]
        vertices = np.array([min, max])
        n_vertices = 2
        return vertices, n_vertices
    elif n_unc_entries == 0:
        # Special case: no uncertain params
        vertices = np.array(M_list[0])
        n_vertices = 1
        return vertices, n_vertices

def get_hull_matrices(M):
    v, n = get_hull(M)

    # Map back to matrix space
    M0 = M[0]
    where_unc = is_uncertain_parameter(M)
    M_verts = [M0.copy() for _ in range(n)]
    for i in range(n):
        M_verts[i][where_unc] = v[i]
    return np.array(M_verts), n, where_unc

def compute_worst_case_J(K, A_in_set, B_in_set, Q, R, x_0=None):
    max_J = -np.inf
    for A, B in zip(A_in_set, B_in_set):
        max_J = max(calc_J(A, B, K, Q, R, x_0), max_J)
    return max_J

def calc_J(A, B, K, Q, R, x_0=None):
    
    stability_radius = 1 - 1e-6  # Slightly stricter stability criterion

    Phi = A - B @ K
    is_scalar_R = np.isscalar(R) or (isinstance(R, np.ndarray) and R.ndim == 0)
    Q_tilde = Q + K.T @ (R * K) if is_scalar_R else Q + K.T @ R @ K

    if np.all(np.abs(eigvals(Phi)) < stability_radius):
        S = solve_discrete_lyapunov(Phi.T, Q_tilde)
        J = np.trace(S) if x_0 is None else x_0.T @ S @ x_0 
        return J.item()
    else:
        return np.inf

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def get_optimal_controller_riccati_method(A_in_set, B_in_set, Q, R, x_0=None):
    """
    Heuristic Riccati-based method.
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
        J_set[i] = np.trace(P) if x_0 is None else x_0.T @ P @ x_0

    max_J_idx = np.argmax(J_set)
    optimal_K = K_set[max_J_idx]

    return optimal_K

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~