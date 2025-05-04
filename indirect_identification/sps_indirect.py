
from typing import Tuple, List, Union, Optional
import numpy as np
from dB.sim_db import SPSType
from indirect_identification.d_tfs import d_tfs, invert_matrix, apply_tf_matrix
from types import SimpleNamespace
from indirect_identification.sps_utils import *
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import lfilter_numba

np.random.seed(42)

class OpenLoopStabilityError(Exception):
    """Raised when stability conditions are not satisfied during open-loop transformation."""
    pass

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
        n_states (int): Number of states in the system. if siso, it is the max order of the system.
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
        self.max_order = max(self.n_states, self.n_inputs, self.n_outputs, self.n_noise)
        self.N_max = self.N + self.max_order + 1
        self.create_phi_method = get_phi_method(n_inputs=n_inputs, n_outputs=n_outputs, n_noise=n_noise)
        self._initialise_alpha_phi_order()

    def _initialise_alpha_phi_order(self):
        """
        Initialize the alpha matrix and the permutation order for the SPS.
        """
        self.alpha = np.sign(np.random.randn(self.m,self.N_max))
        self.alpha[0, :] = 1
        if not self.is_siso:
            self.alpha = np.broadcast_to(self.alpha,(self.n_outputs,self.m,self.N_max)).transpose(1,0,2)

        self.pi_order = np.random.permutation(np.arange(self.m))

    def sps_indicator(self, Y_t: np.ndarray, U_t: np.ndarray,
                  A: np.ndarray, B: np.ndarray, C: np.ndarray,
                  G: Union['d_tfs', np.ndarray], H: Union['d_tfs', np.ndarray],
                  sps_type: SPSType,
                  F: Optional[Union['d_tfs', np.ndarray]] = None,
                  L: Optional[Union['d_tfs', np.ndarray]] = None,
                  R_t: Optional[np.ndarray] = None, Lambda: Optional[np.ndarray] = None,
                  return_rank: bool=False) -> Union[bool, Tuple[bool, float]]:
        """
        SPS indicator, returns True if the tested transfer functions G and H are
        in the SPS region. The function handles both open-loop and closed-loop scenarios.

        Parameters:
            Y_t (np.ndarray): Output array n_outputs x t.
            U_t (np.ndarray): Input array  n_inputs x t.
            A (np.ndarray): A polynomial np array.
            B (np.ndarray): B polynomial/s np array.
            C (np.ndarray): C polynomial/s np array.
            G (Union['d_tfs', np.ndarray]): Transfer function G.
            H (Union['d_tfs', np.ndarray]): Transfer function H.
            sps_type (SPSType): Type of SPS (e.g., whether system is running in closed loop or open).
            F (Optional[Union['d_tfs', np.ndarray]]): Transfer function F (required for closed loop).
            L (Optional[Union['d_tfs', np.ndarray]]): Transfer function L (required for closed loop).
            R_t (Optional[np.ndarray]): Reference array (required for closed loop).
            Lambda (Optional[np.ndarray]): Regularization or weighting parameter of shape n_output x n_output.
                                          If not provided, Lambda will be estimated from the estimated noise.

        Returns:
            Union[bool, Tuple[bool, float]]: A tuple where the first element is a boolean indicating
            whether the system is in the SPS region (True or False), and the second element
            is the normalised SPS rank.
        """
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
            phi_U_t = U[0]
            if sps_type == SPSType.CLOSED_LOOP:
                # check if L is a np array if its extract it
                if isinstance(L, np.ndarray):
                    L = L[0]
                if isinstance(F, np.ndarray):
                    F = F[0]
                phi_U_t = L * phi_U_t
            phi_U_t = phi_U_t[-self.N_max:]
        else:
            phi_U_t = U
            if sps_type == SPSType.CLOSED_LOOP:
                phi_U_t = apply_tf_matrix(L, phi_U_t)
            phi_U_t = phi_U_t[:, -self.N_max:]

        phi_U_t = np.broadcast_to(phi_U_t, (self.m, *phi_U_t.shape))

        return self.indirect_sps(G_0=G_0, H_0=H_0,
                           Y_t=Y_t, U_t=U, phi_U_t=phi_U_t,
                           A=A, B=B, C=C, F=F, sps_type=sps_type, Lambda=Lambda, return_rank=return_rank)


    def transform_to_open_loop(self, G, H, F, L):
        if self.is_siso:
            return self.transform_to_open_loop_siso(G, H, F, L)
        else:
            return self.transform_to_open_loop_mimo(G, H, F, L)

    def transform_to_open_loop_siso(self,
                               G: Union['d_tfs', np.float64],
                               H: Union['d_tfs', np.float64],
                               F: Union['d_tfs', np.float64],
                               L: Union['d_tfs', np.float64]) -> Tuple['d_tfs', 'd_tfs']:
        """
        Transform the closed-loop system to an open-loop system.

        This function takes the closed-loop system transfer functions G, H, F, and L, and
        returns the corresponding open-loop transfer functions G_0 and H_0.

        Parameters:
            G (tuple): The transfer function G (closed-loop).
            H (tuple): The transfer function H (closed-loop).
            F (tuple): The transfer function F (closed-loop).
            L (tuple): The transfer function L (closed-loop).

        Returns:
            tuple: A tuple containing the open-loop transfer functions G_0 and H_0.
        """
        GF_plus_I = (G * F) + 1
        i_GF_plus_I = 1/GF_plus_I
        for tf in [L, i_GF_plus_I]:
            if isinstance(tf, d_tfs):
                if not tf.is_stable():
                    raise OpenLoopStabilityError(f"Error transforming to open loop: stability conditions not satisfied. Failed for {tf}")

        G_0 = i_GF_plus_I * G * L
        H_0 = i_GF_plus_I * H
        return G_0, H_0

    def transform_to_open_loop_mimo(self, G, H, F, L: np.ndarray):
        # L assumptions
        for tf in L.flat:
            d_tfs.sps_assumption_check(tf=tf, value_check=False, check_stability=True)
        GF_plus_I = (G @ F) + np.eye(G.shape[0])
        i_GF_plus_I = invert_matrix(GF_plus_I)
        for tf in i_GF_plus_I.flat:
                d_tfs.sps_assumption_check(tf=tf, value_check=False, check_stability=True)

        G_0 = i_GF_plus_I @ G @ L
        H_0 = i_GF_plus_I @ H
        return G_0, H_0

    def indirect_sps(self, G_0: Union['d_tfs', np.ndarray], H_0: Union['d_tfs', np.ndarray],
                       Y_t: np.ndarray, U_t: np.ndarray, phi_U_t: np.ndarray,
                          A: np.ndarray, B: np.ndarray, C: np.ndarray,
                          F: Union['d_tfs', np.ndarray], sps_type: SPSType, Lambda: np.ndarray, return_rank: bool=False) -> Union[bool, Tuple[bool, float]]:
        if self.is_siso:
            return self.indirect_sps_siso(G_0, H_0, Y_t, U_t, phi_U_t, A, B, C, F, sps_type, return_rank)
        else:
            return self.indirect_sps_mimo(G_0, H_0, Y_t, U_t, phi_U_t, A, B, C, F, sps_type, Lambda, return_rank)

    def indirect_sps_mimo(self, G_0: np.ndarray, H_0: np.ndarray,
                       Y_t: np.ndarray, U_t: np.ndarray, phi_U_t: np.ndarray,
                          A: np.ndarray, B: np.ndarray, C: np.ndarray,
                          F: np.ndarray, sps_type: SPSType, Lambda: np.ndarray, return_rank: bool=False) -> Union[bool, Tuple[bool, float]]:

        try:
            # create the perturbed noise, y, u
            YGU = Y_t - apply_tf_matrix(G_0, U_t)
            H_invert = np.zeros_like(H_0)
            for i in range(self.n_outputs):
                H_invert[i,i]=d_tfs((A, C[i]))
            N_hat = apply_tf_matrix(H_invert,YGU)
            perturbed_N_hat, U_t_par, N_hat_par = get_U_perturbed_nhat(N_hat, U_t, self.alpha, self.N_max)
            y_bar = apply_tf_matrix(G_0, U_t_par) + apply_tf_matrix(H_0, perturbed_N_hat[:])
            if sps_type == SPSType.CLOSED_LOOP:
                FY = apply_tf_matrix(F, y_bar)
                phi_U_t = phi_U_t - FY
            phi_tilde = self.create_phi_method(Y=y_bar, U=phi_U_t, A=A, B=B, C=C)
            # compute S
            S = compute_S(N_hat_par, perturbed_N_hat, phi_tilde, self.N, Lambda)
            # Ranking
            combined = np.array(list(zip(self.pi_order, S)))
            order = np.lexsort(combined.T)
            rank_R = np.where(order == 0)[0][0] + 1

            if return_rank:
                    return rank_R <= self.m - self.q, rank_R/self.m
            else:
                return rank_R <= self.m - self.q
        except Exception as e:
            raise ValueError(f"Error in open-loop SPS: {e}")

    def indirect_sps_siso(self, G_0: 'd_tfs', H_0: 'd_tfs',
                       Y_t: np.ndarray, U_t: np.ndarray, phi_U_t: np.ndarray,
                      A: np.ndarray, B: np.ndarray, C: np.ndarray,
                      F: 'd_tfs', sps_type: SPSType, return_rank: bool=False) -> Union[bool, Tuple[bool, float]]:
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
            YGU = Y_t - G_0*U_t
            N_hat = (1/H_0)*YGU

            # Extract relevant segments
            N_hat_par = N_hat[:, -self.N_max:]
            U_t_par = U_t[:, -self.N_max:]
            perturbed_N_hat = np.multiply(self.alpha, N_hat_par)
            # Compute y_bar vectorized
            y_bar = G_0*U_t_par[None, :] + H_0*(perturbed_N_hat[:, None])
            y_bar = y_bar.transpose(1, 0, 2)[0]
            if sps_type == SPSType.CLOSED_LOOP:
                FY = F*y_bar
                phi_U_t = phi_U_t - FY
            # Compute phi_tilde
            phi_tilde = self.create_phi_method(Y=y_bar, U=phi_U_t, A=A, B=B, C=C)
            phi_tilde = phi_tilde[:,-self.N:,:]
            perturbed_N_hat = perturbed_N_hat[:,-self.N:]
            # Compute Cholesky decomposition and norm squared
            R = np.matmul(phi_tilde.transpose(0, 2, 1), phi_tilde) / self.N
            L = np.linalg.cholesky(R)
            Q, R = np.linalg.qr(L)
            R_root_inv = np.linalg.solve(R, Q.transpose(0, 2, 1))
            weighted_sum = np.matmul(phi_tilde.transpose(0, 2, 1), perturbed_N_hat[:, :, None])
            S = np.sum(np.square(weighted_sum), axis=(1, 2))
            # Ranking
            combined = np.array(list(zip(self.pi_order, S)))
            order = np.lexsort(combined.T)
            rank_R = np.where(order == 0)[0][0] + 1

            if return_rank:
                return rank_R <= self.m - self.q, rank_R/self.m
            else:
                return rank_R <= self.m - self.q
        except Exception as e:
            raise ValueError(f"Error in open-loop SPS: {e}")
