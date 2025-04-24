
from typing import Tuple, List, Union, Optional
import numpy as np
from dB.sim_db import SPSType
from indirect_identification.d_tfs import d_tfs, invert_matrix, apply_tf_matrix
from types import SimpleNamespace
from indirect_identification.sps_utils import get_phi_method, compute_phi_phiT, compute_phi_Y
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import lfilter_numba

np.random.seed(42)

class SPS_indirect_model:
    """
    Indirect Sign Perturbed Sum (SPS) model class.
    """
    def __init__(self, m: int, q: int, N: int = 50, 
                 n_states: int = 1, n_inputs: int = 1, n_outputs: int = 1, 
                 n_noise: int = -1, epsilon: float = 1e-6):
        """
        Initialize the SPS model.
        
        Parameters:
        m (int): .
        q (int): .
        N (int): The lngeth of sequences to be used for the SPS.
        n_states (int): Number of states in the system.
        n_inputs (int): Number of inputs in the system.
        n_outputs (int): Number of outputs in the system.
        n_noise (int): order of the noise polynomial. -1 if the C(s) = 1. 
        """
        self.N = N
        self.m = m
        self.q = q
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_noise = n_noise
        self.is_siso = n_inputs == 1 and n_outputs == 1
        self.epsilon = epsilon
        self.create_phi_method = get_phi_method(n_inputs=n_inputs, n_outputs=n_outputs, n_noise=n_noise)
        self._initialise_alpha_phi_order()

    def _initialise_alpha_phi_order(self):
        """
        Initialize the alpha matrix and the permutation order for the SPS.
        """
        if self.is_siso:
            self.alpha = np.random.randn(self.m,self.N)
            self.alpha = np.sign(self.alpha)
        else:
            self.alpha = np.sign(np.random.randn(self.m,self.n_outputs,self.N))

        self.alpha[0, :] = 1
        self.pi_order = np.random.permutation(np.arange(self.m))

    def sps_indicator(self, Y_t: np.ndarray, U_t: np.ndarray,
                  A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                  G: Union['d_tfs', np.ndarray], H: Union['d_tfs', np.ndarray],
                  sps_type: SPSType,
                  F: Optional[Union['d_tfs', np.ndarray]] = None,
                  L: Optional[Union['d_tfs', np.ndarray]] = None,
                  R_t: Optional[np.ndarray] = None
                  ) -> Tuple[bool, np.ndarray]:
    
        if sps_type == SPSType.CLOSED_LOOP:
            if F is None or L is None or R_t is None:
                raise ValueError("F and L must be provided for closed-loop SPS.")
            G_0, H_0 = self.transform_to_open_loop(G, H, F, L)
            U = R_t
        elif sps_type == SPSType.OPEN_LOOP:
            G_0, H_0 = G, H
            U = U_t
        else:
            raise ValueError("Invalid SPS type. Use SPSType.CLOSED_LOOP or SPSType.OPEN_LOOP.")
        if self.is_siso:
            phi_U_t = U_t[-self.N-1:-1]
        else:
            phi_U_t = U_t[:, -self.N-1:-1]
        
        return self.open_loop_sps(G_0=G_0, H_0=H_0, 
                           Y_t=Y_t, U_t=U, phi_U_t=phi_U_t, 
                           A=A, B=B, C=C)

    def transform_to_open_loop(self, G, H, F, L):
        if self.is_siso:
            return self.transform_to_open_loop_siso(G, H, F, L)
        else:
            return self.transform_to_open_loop_mimo(G, H, F, L)
    
    def transform_to_open_loop_siso(self, 
                               G: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]], 
                               H: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]], 
                               F: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]], 
                               L: Union['d_tfs', Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]]) -> Tuple['d_tfs', 'd_tfs']:
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
        
        if not all(tf.is_stable() for tf in [L, i_GF_plus_I] if isinstance(tf, d_tfs)):
            raise ValueError(f"Error transforming to open loop: stability conditions not satisfied.")
        
        G_0 = i_GF_plus_I * G * L
        H_0 = i_GF_plus_I * H
        return G_0, H_0
    
    def transform_to_open_loop_mimo(self, G, H, F, L): 
        # L assumptions
        for tf in L:
            d_tfs.sps_assumption_check(tf=tf, value_check=False, stability_check=True)
        GF_plus_I = (G @ F) + np.eye(G.shape[0])
        i_GF_plus_I = invert_matrix(GF_plus_I)

        for tf in i_GF_plus_I:
            d_tfs.sps_assumption_check(tf=tf, value_check=False, stability_check=True)
        
        G_0 = i_GF_plus_I @ G @ L
        H_0 = i_GF_plus_I @ H
        return G_0, H_0
    
    def open_loop_sps(self, G_0: Union['d_tfs', np.ndarray], H_0: Union['d_tfs', np.ndarray],
                       Y_t: np.ndarray, U_t: np.ndarray, phi_U_t: np.ndarray,
                          A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        if self.is_siso:
            return self.open_loop_sps_siso(G_0, H_0, Y_t, U_t, phi_U_t, A, B, C)
        else:
            return self.open_loop_sps_mimo(G_0, H_0, Y_t, U_t, phi_U_t, A, B, C)

    def open_loop_sps_mimo(self, G_0: np.ndarray, H_0: np.ndarray,
                       Y_t: np.ndarray, U_t: np.ndarray, phi_U_t: np.ndarray,
                          A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        
        try:
            # create the perturbed noise, y, u
            YGU = Y_t - apply_tf_matrix(G_0, U_t)
            H_invert = np.zeros_like(H_0)
            for i in range(self.n_outputs):
                H_invert[i,i]=d_tfs((A, C[i]))
            N_hat = apply_tf_matrix(H_invert,YGU)
            N_hat_par = N_hat[:, -self.N:]
            U_t_par = U_t[:, -self.N-1:-1]
            perturbed_N_hat = np.multiply(self.alpha, N_hat_par)
            y_bar = apply_tf_matrix(G_0, U_t_par) + apply_tf_matrix(H_0, perturbed_N_hat[:]) 

            # compute S
            phi_tilde = self.create_phi_method(Y=y_bar, U=phi_U_t, A=A, B=B, C=C)
            R = compute_phi_phiT(phi_tilde)
            L = np.linalg.cholesky(R)
            Q, R = np.linalg.qr(L)
            R_root_inv = np.linalg.solve(R, Q.transpose(0, 2, 1))
            weighted_sum = compute_phi_Y(phi_tilde, perturbed_N_hat)
            S = np.sum(np.square(np.matmul(R_root_inv, weighted_sum)), axis=(1, 2))

            # Ranking
            combined = np.array(list(zip(self.pi_order, S)))
            order = np.lexsort(combined.T)
            rank_R = np.where(order == 0)[0][0] + 1
            return rank_R <= self.m - self.q
        except Exception as e:
            raise ValueError(f"Error in open-loop SPS: {e}")

    def open_loop_sps_siso(self, G_0: 'd_tfs', H_0: 'd_tfs',
                       Y_t: np.ndarray, U_t: np.ndarray, phi_U_t: np.ndarray, 
                      A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        """
        Perform open-loop SPS.
        
        Parameters:
        G_0 (d_tfs): The open-loop transfer function G_0.
        H_0 (d_tfs): The open-loop transfer function H_0.
        Y_t (array): The output sequence.
        U_t (array): The input sequence.
        A (array): The A polynomial coefficients.
        B (array): The B polynomial coefficients.
        C (array): The C polynomial coefficients.
        
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
            phi_tilde = self.create_phi_method(Y=y_bar, U=phi_U_t, A=A, B=B, C=C)
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
            
            return rank_R <= self.m - self.q
        except Exception as e:
            raise ValueError(f"Error in open-loop SPS: {e}")
