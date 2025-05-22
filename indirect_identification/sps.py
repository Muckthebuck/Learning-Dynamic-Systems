from typing import Tuple, List
import numpy as np
from search.search_radial import RadialSearch
from indirect_identification.sps_indirect import SPS_indirect_model, OpenLoopStabilityError
from indirect_identification.d_tfs import d_tfs, apply_tf_matrix
from dB.sim_db import Database, SPSType
from types import SimpleNamespace
from scipy.spatial import ConvexHull
import argparse
import ast
import logging
import time
from indirect_identification.sps_utils import get_construct_ss_from_params_method, get_max_radius, get_construct_ss_from_params_method_siso
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import _is_stable
from scipy import optimize
import threading
from fusion.SPSFusion import Fusion


SEARCH_METHODS = {"radial": RadialSearch,}
OPTIMAL_N_VECTORS = {2: 100, 3: 400, 4: 1500}
SPS_MAX = 1e10
class SPS:
    """
    Sign Perturbed Sum (SPS) class.
    """


    def __init__(self, Lambda: np.ndarray, bounds: np.ndarray, n_A: int = None, n_B: int = None,
                 n_states: int=2, n_inputs: int = 1, n_outputs: int = 1, n_refs: int = 1, forget:float=0.0,
                 C_obs: np.ndarray = None, n_noise_order: int = 1, n_points: List[int] = [11, 21, 11],
                 m: int=100, q: int=5, N: int = 50, db: Database = None, search_type: str = "radial",
                 debug: bool = False, epsilon: float = 1e-10, random_centers: int = 50, 
                 logger: logging.Logger = None):
        """"
        Initialize the SPS model search.
        For SISO models. ensure n_states represents the max delay in A,B
        so the order of A is n_states, order of B is n_states otherwise provide n_A, n_B
        """
        self.Lambda = Lambda
        self.logger = logger if logger else logging.getLogger(__name__)
        self.debug = debug
        self.epsilon = epsilon
        self.m = m
        self.q = q
        self.N = N
        # counts
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_refs = n_refs
        self.n_noise_order = n_noise_order
        if n_inputs==1 and n_outputs==1 and n_A and n_B:
            self.nA = n_A
            self.nB = n_B
        else:
            self.nA = n_states
            self.nB = n_states*n_inputs
        self.nH = n_outputs*(n_noise_order+1)
        self.n_params = self.nA + self.nB + self.nH
        # C matrix: n_output x n_state matrix
        self.C = C_obs
        # assert the shape of C
        assert self.C.shape == (self.n_outputs, self.n_states)

        # empty data and controller
        self.data = None
        self.controller = None

        #database
        self.db = db
        self.db.subscribe("data", self.data_callback)
        self.db.subscribe("controller", self.controller_callback)
        self.model = SPS_indirect_model(m=self.m, q=self.q, N=self.N, epsilon=self.epsilon,
                                        n_states=self.n_states, n_inputs=self.n_inputs, 
                                        n_outputs=self.n_outputs, n_noise=self.n_noise_order)
        if self.n_inputs==1 and self.n_outputs==1:
            self._construct_ss_from_params = get_construct_ss_from_params_method_siso(n_A=self.nA,
                                                                                      n_B=self.nB)
        else:
            self._construct_ss_from_params = get_construct_ss_from_params_method(n_states=self.n_states, 
                                                                                n_inputs=self.n_inputs,
                                                                                n_outputs=self.n_outputs,
                                                                                C=self.C)
        if search_type not in SEARCH_METHODS:
            raise RuntimeError(f"{search_type} not supported.")
        self.search_type = search_type
        self.bounds = bounds
        self.max_radius = get_max_radius(bounds)
        self.center = None
        self.controller = SimpleNamespace()
        self.controller.F = None
        self.controller.L = None
        # initialise Fusion
        p=1-q/m
        self.fusion = Fusion(bounds=bounds, 
                             num_points=n_points, 
                             dim=self.n_params, 
                             p=p, forget=forget, random_centers=random_centers)

        
    def search_factory(self, search_type: str, center: np.ndarray, test_cb: callable):
        match search_type:
            case "radial":
                search = RadialSearch(n_dimensions=self.n_params,
                                      n_vectors=OPTIMAL_N_VECTORS[self.n_params],
                                      center_options=center,
                                      test_cb=test_cb,
                                      max_radius=self.max_radius
                                      )
            case _:
                raise RuntimeError(f"{search_type} not supported.")
        return search 

    def data_callback(self, data):
        """
        data attributes: y, u, r, sps_type
        """
        self.logger.info("[SPS] Data callback")
        self.data = self.db._deserialize(data)
        self.data.y = np.array(self.data.y, dtype=np.float64)
        self.data.u = np.array(self.data.u, dtype=np.float64)
        if self.data.r is not None:
            self.data.r = np.array(self.data.r, dtype=np.float64)
        self.run_in_thread(self.update_sps_region)

    
    def controller_callback(self, controller):
        """
        controller attributes: F, L
        """
        self.logger.info("[SPS] Controller callback")
        controller = self.db._deserialize(controller)
        self.run_in_thread(self._convert_controller_to_dtfs, controller)
     
    
    def run_in_thread(self, target_fn, *args):
        def wrapper():
            try:
                target_fn(*args)
            except Exception as e:
                self.logger.exception(f"Exception in thread for {target_fn.__name__}: {e}")
        threading.Thread(target=wrapper, daemon=True).start()



    def _convert_controller_to_dtfs(self, controller):
        controller.L = np.array(controller.L, dtype=np.float64)
        controller.F = np.array(controller.F, dtype=np.float64)
        
        one = np.array([1.0], dtype=np.float64)

        # SISO case: SPS wants just d_tfs objects; L and F must be 1D
        if self.n_inputs == 1 and self.n_outputs == 1:
            F = d_tfs((controller.F.flatten(), one))
            L = d_tfs((controller.L.flatten(), one))
        else:
            # F: (n_inputs, n_outputs)
            controller.F = controller.F.reshape(self.n_inputs, self.n_outputs)
            F = np.empty((self.n_inputs, self.n_outputs), dtype=object)
            for i in range(self.n_inputs):
                for j in range(self.n_outputs):
                    F[i, j] = d_tfs((np.atleast_1d(controller.F[i, j]), one))
            
            # L: (n_inputs, n_refs)
            controller.L = controller.L.reshape(self.n_inputs, self.n_refs)
            L = np.empty((self.n_inputs, self.n_refs), dtype=object)
            for i in range(self.n_inputs):
                for j in range(self.n_refs):
                    L[i, j] = d_tfs((np.atleast_1d(controller.L[i, j]), one))
        
        self.controller.F = F
        self.controller.L = L

            

        
    def update_sps_region(self):
        self.logger.info("[SPS] Starting Search")
        self.logger.info(f"[SPS] Data: {self.data.sps_type}")
        self.logger.info(f"[SPS] Number of params: {self.n_params}")
        
        if self.data.sps_type == SPSType.OPEN_LOOP:
            self.fusion.center_pts = self.get_lse(Y=self.data.y, U=self.data.u)
            self.logger.info(f"center {self.fusion.center_pts}")

        if self.fusion.center_pts is None:
            self.logger.warning(f"No center provided sps will fail.")
            return
        
        if self.data.r is not None:
            self.logger.info(f" y: {self.data.y.shape}, u: {self.data.u.shape}, r: {self.data.u.shape}")
        else:
            self.logger.info(f" y: {self.data.y.shape}, u: {self.data.u.shape}")
        while True:
            try:
                if self.fusion.hull:
                    # closed loop, randomly select next set of points
                    self.fusion.choose_random_centers()
                search = self.search_factory(search_type=self.search_type, 
                                            center=self.fusion.center_pts, 
                                            test_cb=self._get_search_fn(self.data.sps_type))
                self.logger.info(f"[SPS] Search Initialized")
                ins, outs, boundaries, hull, expanded_hull = search.search()
                self.logger.info(f"[SPS] Search Finished")
                break
            except Exception as e:
                self.logger.info(f"{e}")
                self.logger.info(f"[SPS] retrying")

        # fuse
        self.fusion.fuse(new_hull=hull)
        self.logger.info(f"[SPS] Fused Regions")

        # get the results
        results = self.fusion.approximate_hull()
        A, B, C, D = self.get_ss_region(results=results)
        if self.db is not None:
            self.write_state_space_to_db(A, B, C, D)
        

        
    
    def plot_sps_region(self):
        while True:
            #plt.pause if happening in the function to keep things resposnive
            self.fusion.plot_curr_region()


    def _get_search_fn(self, sps_type: SPSType):
        """
        get the search function based on the SPS type.
        """
        if sps_type == SPSType.OPEN_LOOP:
            return self.open_loop_sps_search_fn
        elif sps_type == SPSType.CLOSED_LOOP:
            return self.closed_loop_sps_search_fn
        else:
            self.logger.error(f"Unknown SPS type: {sps_type}")
            raise ValueError(f"Unknown SPS type: {sps_type}")
        
    def get_ss_region(self, results:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Get the state space region from the given parameters.
        """
        # Construct state space matrices from the given parameters
        if results is None:
            raise ValueError("No results found")

        self.logger.info(f"[SPS] Results shape: {results.shape}")
        if results.ndim==2 and results.shape[0] == 0:
            # no results found
            raise ValueError("No results found")
        
        
        results = [self._construct_ss_from_params(np.ascontiguousarray(row)) for row in results]

        # Unpack first 4 outputs, ignore the rest
        A_list, B_list, C_list, D_list, *_ = zip(*results)

        # Convert to numpy arrays
        A = np.array(A_list)
        B = np.array(B_list)
        C = np.array(C_list)
        D = np.array(D_list)
        return A,B,C,D
        

    def construct_gh_from_params(self, params):
        """
        Return G,H filter matrices and A,B,C polynomials matrices
        Construct state space matrices from the given parameters.
        State space matrices are in observable canonical form.
        Also checks all assumptions are satisfied.
        assumptions:
            1. G(0) = 0
            2. H(0) = 1
            3. G and H are stable transfer functions
            4. H is invertible transfer function (H(0) != 0)
        """
        A_obs, B_obs, C_obs, D_obs, A,B = self._construct_ss_from_params(params)

        # G Transfer function matrix, no need to check value assumption, guaranteed to be 0
        if self.n_inputs==1 and self.n_outputs==1:
            G = d_tfs((B,A))
        else: 
            G = d_tfs.ss_to_tf(A_obs, B_obs, C_obs, D_obs, check_assumption=False, epsilon=self.epsilon)
        if not _is_stable(A, epsilon=self.epsilon):
            raise OpenLoopStabilityError
        
        try: 
            H, C = self._get_H_from_params(params, A)
        except Exception:
            raise OpenLoopStabilityError

        return G, H, A, B, C
    
    def _get_H_from_params(self, params, A):
        # H Transfer function matrix
        if self.n_noise_order == -1:
            # if noise order is -1 then H is a scalar transfer function
            if self.n_outputs == 1:
                C = np.array([1.0])
                H = d_tfs((C, A))
            else:
                C = np.empty((self.n_outputs, 1))
                H = np.zeros((self.n_outputs, self.n_outputs), dtype=object)
                for i in range(self.n_outputs):
                    C[i]=np.array([1.0])
                    H[i,i]=d_tfs((np.array([1.0]),A))
        else:
            # rest of params form H matrix, H is a n_state x n_state matrix
            # the diagonals of H are the noise transfer functions
            # each tf has numerator and denominator of order n_noise_order
            if self.n_outputs == 1:
                # if only one output, H is a scalar transfer function
                # H = np.zeros((self.n_states, self.n_states), dtype=object)
                j = self.n_states + self.n_states*self.n_inputs
                num = params[j:j+(self.n_noise_order+1)]
                den = A
                H = d_tfs((num, den))
                C = num
                # test assumptions
                d_tfs.sps_assumption_check(tf=H, value=1, epsilon=self.epsilon)
                d_tfs.sps_assumption_check(tf=d_tfs((den,num)), value=1, epsilon=self.epsilon)
            else:
                C = np.empty((self.n_outputs, self.n_noise_order+1))
                H = np.zeros((self.n_outputs, self.n_outputs), dtype=object)
                for i in range(self.n_outputs):
                    j = self.n_states + self.n_states*self.n_inputs + i*(self.n_noise_order+1)
                    # numerator and denominator of the transfer function
                    num = params[j:j+self.n_noise_order+1]
                    den = A
                    H[i,i] = d_tfs((num, den))
                    C[i] = num
                    # test assumptions
                    d_tfs.sps_assumption_check(tf=H[i,i], value=1, epsilon=self.epsilon)
                    d_tfs.sps_assumption_check(tf=d_tfs((den,num)), value=1, epsilon=self.epsilon, check_stability=True)
        return H, C


    def open_loop_sps_search_fn(self, params):
        """
        Perform open loop SPS search for the given parameters.
        """
        in_sps = False
        try:
            G, H , A, B, C = self.construct_gh_from_params(params)
        except Exception as e:
            self.logger.debug(f"[SPS] not valid param. {e}")
            return False

        # Check the condition and store the result if true
        
        if self.data is None:
            raise RuntimeError("Data not found in database")
        try:
            in_sps = self.model.sps_indicator(G=G, H=H, A=A, B=B, C=C,
                                             Y_t=self.data.y, U_t=self.data.u, 
                                             sps_type=SPSType.OPEN_LOOP, Lambda=None)
        except Exception as e:
            self.logger.debug("SPS Failed")
            self.logger.debug(f"{e}")
            in_sps = False
        self.logger.debug(f"[SPS] Params: {params}, SPS: {in_sps}")
        return in_sps
    
    def closed_loop_sps_search_fn(self, params):
        """
        Perform closed loop SPS search for the given parameters.
        """
        in_sps = False
        try:
            G, H, A, B, C = self.construct_gh_from_params(params)
            if self.controller is None:
                raise RuntimeError("Controller not found in database")
        except Exception as e:
            self.logger.warning(f"[SPS] not valid param. {e}")
            return False
        except RuntimeError:
            if self.debug:
                self.logger.debug("Controller not found... skipping")
            raise
        if self.data is None:
            raise RuntimeError("Data not found in database")
        
        try:
            
            in_sps = self.model.sps_indicator(G=G, H=H, A=A, B=B, C=C,
                                             Y_t=self.data.y, U_t=self.data.u, R_t=self.data.r,
                                             sps_type=SPSType.CLOSED_LOOP,
                                             F=self.controller.F, L=self.controller.L, Lambda=None)
        except Exception as e:
            self.logger.warning("SPS Failed")
            self.logger.warning(f"{e}")
            in_sps = False
        
        return in_sps
    
    def get_error_norm(self, params, Y, U):
        try:
            G, H , A, B, C = self.construct_gh_from_params(params)
        except Exception:
            return SPS_MAX
        if self.n_inputs==1 and self.n_outputs==1:
            Hinv = d_tfs((A, C))
            YGU = Y - G*U
            N = Hinv*YGU
            error_norm = np.linalg.norm(N@N.T)
        else:
            YGU = Y - apply_tf_matrix(G,U)
            N = apply_tf_matrix(Hinv,YGU)
            error_norm = np.linalg.norm(N@N.T)
        return error_norm
    def get_lse(self, Y,U):
        x0 = np.zeros(self.n_params)
        res = optimize.least_squares(self.get_error_norm, x0, args=(Y,U))
        params_ls = res.x
        return params_ls

    def write_state_space_to_db(self, A, B, C, D):
        try:
            ss = SimpleNamespace(
                A=A,
                B=B,
                C=C,
                D=D
            )
            self.db.write_ss(ss=ss)
        except Exception as e:
            self.logger.error(f"Failed to write state space: {ss}")
            self.logger.exception(f"Error writing to database: {e}")
            raise RuntimeError(f"Error writing to database: {e}")



# Function to parse the input string into a NumPy array
def parse_array(input_string):
    try:
        # Safely evaluate the string into a list using ast.literal_eval
        parsed_list = ast.literal_eval(input_string)
        # Convert the list into a NumPy array
        return np.array(parsed_list)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Input must be a valid list-like string (e.g. [1, 2, 3])")

def parse_args(raw_args=None, db:Database = None):
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="SPS Indirect Identification")
    
    # Define arguments
    parser.add_argument("--Lambda", type=parse_array, help="Weight matrix output by output as a list-like string")
    parser.add_argument("--n_A", type=int, default=2, help="Polynomial A order")
    parser.add_argument("--n_B", type=int, default=2, help="Polynomical B order")
    parser.add_argument("--n_states", type=int, required=True, help="Number of states")
    parser.add_argument("--n_inputs", type=int, required=True, help="Number of inputs")
    parser.add_argument("--n_outputs", type=int, required=True, help="Number of references")
    parser.add_argument("--n_refs", type=int, required=True, help="Number of outputs")
    parser.add_argument("--C", type=parse_array, help="C matrix as a list-like string")
    parser.add_argument("--n_noise_order", type=int, default=3, help="Noise order")
    parser.add_argument("--n_points", type=parse_array, help="Number of points as a list-like string")
    parser.add_argument("--m", type=int, default=100, help="Number of samples")
    parser.add_argument("--q", type=int, default=5, help="Number of points")
    parser.add_argument("--N", type=int, default=50, help="N samples")
    parser.add_argument("--random_centers", type=int, default=50, help="random samples to take in hull for center")
    parser.add_argument("--dB", type=str, default="sim.db", help="Database file name")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Epsilon value for stability check")
    parser.add_argument("--search",type=str, choices=["grid", "radial"], required=True, help="Search strategy" )
    parser.add_argument("--bounds", type=parse_array, required=True, 
                        help="Bound of the region for each direction. For example [[-2, 2], [-2, 2]]")
    parser.add_argument("--forget", type=float, default=0.0, 
                        help="fusion forgetting factor")
    args = parser.parse_args(raw_args)
    if db is None:
        db = Database(args.dB)

    return args, db

def run_sps(raw_args=None, db:Database = None, args: argparse.Namespace = None, logger: logging.Logger = None):
    """
    Run the SPS Indirect Identification.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args is None:
        args, db = parse_args(raw_args=raw_args, db=db)
    sps = SPS(
        Lambda=args.Lambda,
        forget=args.forget,
        n_A=args.n_A,
        n_B=args.n_B,
        n_states=args.n_states,
        n_inputs=args.n_inputs,
        n_outputs=args.n_outputs,
        n_refs=args.n_refs,
        C_obs=args.C,
        n_noise_order=args.n_noise_order,
        n_points=args.n_points,
        m=args.m,
        q=args.q,
        N=args.N,
        db=db,
        debug=args.debug,
        epsilon=args.epsilon,
        search_type=args.search,
        bounds=args.bounds,
        random_centers=args.random_centers,
        logger=logger
    )
    # sps.update_sps_region(data)
    sps.plot_sps_region()

if __name__ == "__main__":
    run_sps()
    