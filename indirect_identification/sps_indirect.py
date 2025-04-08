
from typing import Tuple, List, Union
import numpy as np
from scipy.signal import lfilter
from dB.sim_db import Database, SPSType
from indirect_identification.d_tfs import d_tfs
from types import SimpleNamespace

np.random.seed(42)

class SPS_indirect_model:
    """
    Indirect Sign Perturbed Sum (SPS) model class.
    """
    def __init__(self, m: int, q: int, N: int = 50):
        """
        Initialize the SPS model.
        
        Parameters:
        m (int): The number of perturbations.
        q (int): The number of retained perturbations.
        N (int): The length of the perturbation sequence.
        """
        self.N = N
        self.m = m
        self.q = q
        self.alpha = np.random.randn(m,N)
        self.alpha = np.sign(self.alpha)
        self.alpha[0, :] = 1
        self.pi_order = np.random.permutation(np.arange(m))
 
    def create_phi_optimized(self, Y: np.ndarray, U: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
        """
        Create the phi matrix optimized for the given inputs.
        
        Parameters:
        Y (array): The output sequence.
        U (array): The input sequence.
        n_a (int): The number of lags for the output sequence.
        n_b (int): The number of lags for the input sequence.
        
        Returns:
        array: The phi matrix.
        """
        m, t = Y.shape
        
        # Initialize phi with zeros
        phi = np.zeros((m, t, n_a + n_b), dtype=Y.dtype)
        
        # Handle Y lags
        for i in range(1, n_a + 1):
            phi[:, i:, i-1] = -1*Y[:, :-i]
        
        # Handle U lags
        for i in range(1, n_b + 1):
            phi[:, i:, n_a+i-1] = U[:-i]
        
        return phi

    def transform_to_open_loop(self, 
                                G: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]], 
                                H: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]], 
                                F: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]], 
                                L: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]],
                                check_stability: bool = True) -> Tuple['d_tfs', 'd_tfs']:
        """
        Transform the closed-loop system to an open-loop system.
        
        Parameters:
        G (tuple): The transfer function G.
        H (tuple): The transfer function H.
        F (tuple): The transfer function F.
        L (tuple): The transfer function L.
        
        Returns:
        tuple: The open-loop transfer functions G_0 and H_0.
        """
        if not isinstance(G, d_tfs):
            G = d_tfs(G)
        if not isinstance(H, d_tfs):
            H = d_tfs(H)
        if not isinstance(F, d_tfs):
            F = d_tfs(F)
        if not isinstance(L, d_tfs):
            L = d_tfs(L)

        GF_plus_I = (G * F) + 1
        i_GF_plus_I = 1/GF_plus_I
        
        if check_stability:
            if not all(tf.is_stable() for tf in [L, G, H, 1/H, i_GF_plus_I] if isinstance(tf, d_tfs)):
                raise ValueError(f"Error transforming to open loop: stability conditions not satisfied.")
        
        G_0 = i_GF_plus_I * G * L
        H_0 = i_GF_plus_I * H
        return G_0, H_0
    
    def open_loop_sps(self, G_0: 'd_tfs', H_0: 'd_tfs', Y_t: np.ndarray, U_t: np.ndarray, n_a: int, n_b: int) -> Tuple[bool, np.ndarray]:
        """
        Perform open-loop SPS.
        
        Parameters:
        G_0 (d_tfs): The open-loop transfer function G_0.
        H_0 (d_tfs): The open-loop transfer function H_0.
        Y_t (array): The output sequence.
        U_t (array): The input sequence.
        n_a (int): The number of lags for the output sequence.
        n_b (int): The number of lags for the input sequence.
        
        Returns:
        tuple: A boolean indicating if the rank is within the threshold and the S values.
        """
        try:
            Y_t = np.asarray(Y_t)
            U_t = np.asarray(U_t)
            YGU = Y_t - G_0*U_t
            N_hat = (1/H_0)*YGU
            
            # Extract relevant segments
            N_hat_par = N_hat[-self.N:]
            U_t_par = U_t[-self.N-1:-1]
            perturbed_N_hat = np.multiply(self.alpha, N_hat_par)
            
            # Compute y_bar vectorized
            y_bar = G_0*U_t_par[None, :] + H_0*(perturbed_N_hat[:, None])
            y_bar = y_bar.transpose(1, 0, 2)[0]
            # Compute phi_tilde
            phi_tilde = self.create_phi_optimized(y_bar, U_t_par, n_a, n_b)
            # Compute Cholesky decomposition and norm squared
            R = np.matmul(phi_tilde.transpose(0, 2, 1), phi_tilde) / len(Y_t)
            L = np.linalg.cholesky(R)
            Q, R = np.linalg.qr(L)
            R_root_inv = np.linalg.solve(R, Q.transpose(0, 2, 1))
            weighted_sum = np.matmul(phi_tilde.transpose(0, 2, 1), perturbed_N_hat[:, :, None])
            S = np.sum(np.square(np.matmul(R_root_inv, weighted_sum)), axis=(1, 2))
            # Ranking
            combined = np.array(list(zip(self.pi_order, S)))
            order = np.lexsort(combined.T)
            rank_R = np.where(order == 0)[0][0] + 1
            
            return rank_R <= self.m - self.q, S
        except Exception as e:
            raise ValueError(f"Error in open-loop SPS: {e}")
