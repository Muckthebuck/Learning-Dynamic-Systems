from typing import Tuple, List, Union
import numpy as np
from search.search import SPSSearch
from indirect_identification.sps_indirect import SPS_indirect_model
from indirect_identification.d_tfs import d_tfs
from dB.sim_db import Database, SPSType
from types import SimpleNamespace
import scipy.signal as signal
import matplotlib.pyplot as plt
import argparse
import ast
import logging
import time
import asyncio
class SPS:
    """
    Sign Perturbed Sum (SPS) class.
    """


    def __init__(self, n_states: int=2, n_inputs: int = 1, n_output: int = 1, 
                 C: np.ndarray = None, n_noise_order: int = 3, n_points: List[int] = [11, 21, 11],
                 m: int=100, q: int=5, N: int = 50, db: Database = None, 
                 debug: bool = False, epsilon: float = 1e-10, logger: logging.Logger = None):
        """"
        Initialize the SPS model search."
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.debug = debug
        self.epsilon = epsilon
        self.m = m
        self.q = q
        self.N = N
        # counts
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_output = n_output
        self.n_noise_order = n_noise_order
        self.n_params = n_states + n_states*n_inputs + n_output*2*n_noise_order
        self.nA = n_states
        self.nB = n_states*n_inputs
        self.nH = n_output*2*n_noise_order
        self.n_points = np.repeat(n_points, [self.nA, self.nB, self.nH])
        # C matrix: n_output x n_state matrix
        self.C = C
        # assert the shape of C
        assert self.C.shape == (self.n_output, self.n_states)

        # empty data and controller
        self.data = None
        self.controller = None

        #database
        self.db = db
        self.db.subscribe("data", self.data_callback)
        self.db.subscribe("controller", self.controller_callback)
        self.model = SPS_indirect_model(m, q, N)
        
        
    
    def data_callback(self, data):
        """
        data attributes: y, u, r, sps_type
        """
        self.logger.info("[SPS] Data callback")
        self.data = self.db._deserialize(data)
        self.update_sps_region()
    
    def controller_callback(self, controller):
        """
        controller attributes: F, L
        """
        self.logger.info("[SPS] Controller callback")
        self.controller = self.db._deserialize(controller)

        
    def update_sps_region(self):
        self.logger.info("[SPS] Starting Search")
        self.logger.debug(f"[SPS] Data: {self.data.sps_type}")
        self.logger.debug(f"[SPS] Number of params: {self.n_params}")
        search = SPSSearch(
                mins=[0]*self.n_params,
                maxes=[1]*self.n_params,
                n_dimensions=self.n_params,
                n_points=self.n_points,
                test_cb=self._get_search_fn(self.data.sps_type),
                logger=self.logger
            )
        self.logger.debug(f"[SPS] Search Initialized")
        search.go()
        # get the results
        results = search.get_results()
        A, B, C, D = self.get_ss_region(results=results)
        self.logger.warning(A, B, C, D)
        if self.db is not None:
            self.write_state_space_to_db(A, B, C, D)
        pass

    def plot_sps_region(self):
        """
        Plot the SPS region.
        """
        # TODO: implement the plotting of the SPS region
        # run infinite loop for now with some sleep time
        while True:
            time.sleep(0.0001)

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
        
    def get_ss_region(self, results:np.ndarray) -> np.ndarray:
        """
        Get the state space region from the given parameters.
        """
        # Construct state space matrices from the given parameters
        if results is None:
            raise ValueError("No results found")
        # if results is a 1D array, means only one parameter set is given
        self.logger.debug(f"[SPS] Results: {results}")
        self.logger.debug(f"[SPS] Results shape: {results.shape}")
        self.logger.debug(f"[SPS] Results dim: {results.ndim}")
        if results.shape[1] == 0:
            # no results found
            raise ValueError("No results found")
        
        # results is a 2D array of shape (n_points, n_params)
        result_ss = np.apply_along_axis(self._construct_ss_from_params, 0, results)
        return result_ss
        
    def _construct_ss_from_params(self, params):
        # A: n_state x n_state matrix
        A = np.vstack([np.hstack([np.zeros((self.n_states-1, 1)), 
                                  np.eye(self.n_states-1)]), -params[:self.n_states]])
        # B: n_state x n_input matrix
        B = params[self.n_states:self.n_states*self.n_inputs].reshape(self.n_states, self.n_inputs)
        # C: n_output x n_state matrix
        C = self.C
        # D: n_output x n_input matrix: zero matrix for now
        D = np.zeros((self.n_output, self.n_inputs))

        return A, B, C, D

    def construct_gh_from_params(self, params):
        """
        Construct state space matrices from the given parameters.
        State space matrices are in observable canonical form.
        Also checks all assumptions are satisfied.
        assumptions:
            1. G(0) = 0
            2. H(0) = 1
            3. G and H are stable transfer functions
            4. H is invertible transfer function (H(0) != 0)
        """
        A, B, C, D = self._construct_ss_from_params(params)

        # G Transfer function matrix
        G = d_tfs.ss_to_tf(A, B, C, D, check_assumption=True, epsilon=self.epsilon)


        # H Transfer function matrix
        # rest of params form H matrix, H is a n_state x n_state matrix
        # the diagonals of H are the noise transfer functions
        # each tf has numerator and denominator of order n_noise_order
        if self.n_output == 1:
            # if only one output, H is a scalar transfer function
            # H = np.zeros((self.n_states, self.n_states), dtype=object)
            j = self.n_states + self.n_states*self.n_inputs
            num = params[j:j+self.n_noise_order]
            den = params[j+self.n_noise_order:j+2*self.n_noise_order]
            H = d_tfs((num, den))
            # test assumptions
            d_tfs.sps_assumption_check(tf=H, value=1, epsilon=self.epsilon)
            d_tfs.sps_assumption_check(tf=d_tfs((den,num)), value=1, epsilon=self.epsilon)
        else:
            H = np.zeros((self.n_output, self.n_output), dtype=object)
            for i in range(self.n_output):
                j = self.n_states + self.n_states*self.n_inputs + i*self.n_noise_order
                # numerator and denominator of the transfer function
                num = params[j:j+self.n_noise_order]
                den = params[j+self.n_noise_order:j+2*self.n_noise_order]
                H[i,i] = d_tfs((num, den))
                # test assumptions
                d_tfs.sps_assumption_check(tf=H[i,i], value=1, epsilon=self.epsilon)
                d_tfs.sps_assumption_check(tf=d_tfs((den,num)), value=1, epsilon=self.epsilon)
        return G, H


    def open_loop_sps_search_fn(self, params):
        """
        Perform open loop SPS search for the given parameters.
        """
        # Transform to open loop
        try:
            G, H = self.construct_gh_from_params(params)
        except ValueError:
            # if self.debug:
            #     self.logger.debug("Assumptions not satisfied... skipping")
            return False

        # Check the condition and store the result if true
        
        if self.data is None:
            raise RuntimeError("Data not found in database")
        try:
            in_sps, _ = self.model.open_loop_sps(G_0=G, H_0=H, 
                                                Y_t=self.data.y, U_t=self.data.u, 
                                                n_a=self.nA, n_b=self.nB)
        except ValueError:
            self.logger.warning("SPS Failed")
            in_sps = False
        self.logger.debug(f"[SPS] Params: {params}, SPS: {in_sps}")
        return in_sps
    
    def closed_loop_sps_search_fn(self, params):
        """
        Perform closed loop SPS search for the given parameters.
        """
        try:
            G, H = self.construct_gh_from_params(params)
    
            if self.controller is None:
                raise RuntimeError("Controller not found in database")
            G_0, H_0 = self.model.transform_to_open_loop(G, H, self.controller.F, self.controller.L)
        except ValueError:
            if self.debug:
                self.logger.debug("Assumptions not satisfied... skipping")
            return False
        except RuntimeError:
            if self.debug:
                self.logger.debug("Controller not found... skipping")
            raise
        if self.data is None:
            raise RuntimeError("Data not found in database")
        in_sps, _ = self.model.open_loop_sps(G_=G_0, H=H_0, 
                                             Y_t=self.data.y, U_t=self.data.r, 
                                             n_a=self.nA, n_b=self.nB)
        return in_sps
    


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
    parser.add_argument("--n_states", type=int, default=2, help="Number of states")
    parser.add_argument("--n_inputs", type=int, default=1, help="Number of inputs")
    parser.add_argument("--n_output", type=int, default=1, help="Number of outputs")
    parser.add_argument("--C", type=parse_array, help="C matrix as a list-like string")
    parser.add_argument("--n_noise_order", type=int, default=3, help="Noise order")
    parser.add_argument("--n_points", type=parse_array, help="Number of points as a list-like string")
    parser.add_argument("--m", type=int, default=100, help="Number of samples")
    parser.add_argument("--q", type=int, default=5, help="Number of points")
    parser.add_argument("--N", type=int, default=50, help="N samples")
    parser.add_argument("--dB", type=str, default="sim.db", help="Database file name")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Epsilon value for stability check")
    
    args = parser.parse_args(raw_args)
    if db is None:
        db = Database(args.dB)

    return args, db

def run_sps(raw_args=None, db:Database = None, args: argparse.Namespace = None, logger: logging.Logger = None):
    """
    Run the SPS Indirect Identification.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if args is None:
        args, db = parse_args(raw_args=raw_args, db=db)
    sps = SPS(
        n_states=args.n_states,
        n_inputs=args.n_inputs,
        n_output=args.n_output,
        C=args.C,
        n_noise_order=args.n_noise_order,
        n_points=args.n_points,
        m=args.m,
        q=args.q,
        N=args.N,
        db=db,
        debug=args.debug,
        epsilon=args.epsilon,
        logger=logger
    )
    # sps.update_sps_region(data)
    sps.plot_sps_region()

if __name__ == "__main__":
    run_sps()
    