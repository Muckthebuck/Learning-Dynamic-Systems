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
class SPS:
    """
    Sign Perturbed Sum (SPS) class.
    """

    def __init__(self, n_states: int=2, n_inputs: int = 1, n_output: int = 1, 
                 C: np.ndarray = None, n_noise_order: int = 3, n_points: List[int] = [11, 21, 11],
                 m: int=100, q: int=5, N: int = 50, db: str = "sim.db", 
                 debug: bool = False, epsilon: float = 1e-10):
        """"
        Initialize the SPS model search."
        """
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

        self.db = Database(db)
        self.db.subscribe("data", self.data_callback)
        self.model = SPS_indirect_model(m, q, N)
        
    
    def data_callback(self, data):
        """
        data attributes: y, u, r, sps_type
        """
        self.update_sps_region(data=data)
        

    def update_sps_region(self, data):
        search = SPSSearch(
                lower_bounds=[0]*self.n_params,
                upper_bounds=[1]*self.n_params,
                n_dimensions=self.n_params,
                n_points=self.n_points,
                test_cb=self._get_search_fn(data.sps_type)
            )
        search.go()
        # get the results
        results = search.get_results()
        A, B, C, D = self.get_ss_region(results=results)
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
            plt.pause(0.1)

    def _get_search_fn(self, sps_type: SPSType):
        """
        get the search function based on the SPS type.
        """
        if sps_type == SPSType.OPEN_LOOP:
            return self.open_loop_sps_search_fn
        elif sps_type == SPSType.CLOSED_LOOP:
            return self.closed_loop_sps_search_fn
        else:
            raise ValueError(f"Unknown SPS type: {sps_type}")
        
    def get_ss_region(self, results:np.ndarray) -> np.ndarray:
        """
        Get the state space region from the given parameters.
        """
        # Construct state space matrices from the given parameters
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
            if self.debug:
                print("Assumptions not satisfied... skipping")
            return False

        # Check the condition and store the result if true
        data = self.db.get_latest_data()
        if data is None:
            raise RuntimeError("Data not found in database")
        in_sps, _ = self.model.open_loop_sps(G_0=G, H_0=H, 
                                             Y_t=data.y, U_t=data.u, 
                                             n_a=self.nA, n_b=self.nB)
        return in_sps
    
    def closed_loop_sps_search_fn(self, params):
        """
        Perform closed loop SPS search for the given parameters.
        """
        try:
            G, H = self.construct_gh_from_params(params)
            ctrl = self.db.get_latest_ctrl()
            if ctrl is None:
                raise RuntimeError("Controller not found in database")
            G_0, H_0 = self.model.transform_to_open_loop(G, H, ctrl.F, ctrl.L)
        except ValueError:
            if self.debug:
                print("Assumptions not satisfied... skipping")
            return False
        except RuntimeError:
            if self.debug:
                print("Controller not found... skipping")
            raise
        data = self.db.get_latest_data()
        if data is None:
            raise RuntimeError("Data not found in database")
        in_sps, _ = self.model.open_loop_sps(G_=G_0, H=H_0, 
                                             Y_t=data.y, U_t=data.r, 
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
            print(f"Failed to write state space: {ss}")
            raise RuntimeError(f"Error writting to database: {e}")



# Function to parse the input string into a NumPy array
def parse_array(input_string):
    try:
        # Safely evaluate the string into a list using ast.literal_eval
        parsed_list = ast.literal_eval(input_string)
        # Convert the list into a NumPy array
        return np.array(parsed_list)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Input must be a valid list-like string (e.g. [1, 2, 3])")

def parse_args():
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
    parser.add_argument("--db", type=str, default="sim.db", help="Database file name")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="Enable debug mode")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Epsilon value for stability check")
    # Parse the arguments
    return parser.parse_args()

if __name__ == "__main__":
    # parse args from command line
    args = parse_args()

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
        db=args.db,
        debug=args.debug,
        epsilon=args.epsilon
    )
    # sps.update_sps_region(data)
    sps.plot_sps_region()