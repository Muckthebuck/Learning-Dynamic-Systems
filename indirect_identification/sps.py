from typing import Tuple, List, Union
import numpy as np
from search.search import SPSSearch
from indirect_identification.sps_indirect import SPS_indirect_model
from indirect_identification.d_tfs import d_tfs
from dB.sim_db import Database, SPSType
from types import SimpleNamespace
import scipy.signal as signal

class SPS:
    """
    Sign Perturbed Sum (SPS) class.
    """

    def __init__(self, n_states: int=2, n_inputs: int = 1, n_output: int = 1, 
                 C: np.ndarray = None,
                 m: int=100, q: int=5, N: int = 50, db: str = "sim.db"):
        """"
        Initialize the SPS model search."
        """
        self.m = m
        self.q = q
        self.N = N
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_output = n_output
        self.n_params = n_states + n_states*n_inputs + n_output**2
        self.C = C
        # assert the shape of C
        assert self.C.shape == (self.n_output, self.n_states)

        self.db = Database(db)
        self.db.subscribe("data", self.data_callback)
        
    
    def data_callback(self, data):
        """
        data attributes: y, u, r, sps_type
        """
        self.update_sps_region(data=data)
        pass

    def update_sps_region(self, data):
        if data.sps_type == SPSType.OPEN_LOOP:
            pass
        if data.sps_type == SPSType.CLOSED_LOOP:
            pass
        A=B=C=D=None
        if self.db is not None:
            self.write_state_space_to_db(A, B, C, D)
        pass

    def construct_ss_from_params(self, params):
        """
        Construct state space matrices from the given parameters.
        State space matrices are in observable canonical form.
        """
        # A: n_state x n_state matrix
        A = np.vstack([np.hstack([np.zeros((self.n_states-1, 1)), 
                                  np.eye(self.n_states-1)]), -params[:self.n_states]])
        # B: n_state x n_input matrix
        B = params[self.n_states:self.n_states*self.n_inputs].reshape(self.n_states, self.n_inputs)
        # C: n_output x n_state matrix
        C = self.C
        # D: n_output x n_input matrix: zero matrix for now
        D = np.zeros((self.n_output, self.n_inputs))

        # G Transfer function matrix
        G = d_tfs.ss_to_tf(A, B, C, D)
        # rest of params form H matrix, last n_output x n_output matrix
        H = params[-self.n_output**2:].reshape(self.n_output, self.n_output)
        return G, H


        




    def open_loop_sps_search(self, )

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

    def plot_current_sps_region(self):
        pass