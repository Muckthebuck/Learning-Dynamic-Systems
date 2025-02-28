import os
os.environ['CUPY_ACCELERATORS'] = 'cub'

import os
import torch
from typing import Tuple, List, Union
try:
    torch.cuda.current_device()
    import cupy as cp
    from cupyx.scipy.signal import lfilter
except:
    # Fall back to unoptimised versions
    import numpy as cp
    from scipy.signal import lfilter

cp.random.seed(42)

class d_tfs:
    """
    Discrete Transfer Function System class.
    """
    def __init__(self, A: Tuple[Union[List[float], cp.ndarray], Union[List[float], cp.ndarray]]):
        """
        Initialize the transfer function with numerator and denominator coefficients.
        
        Parameters:
        A (tuple): A tuple containing two lists or arrays, the numerator and denominator coefficients.
        """
        self.epsilon = cp.float64(1e-10)
        self.num = cp.asarray(A[0]).astype(self.epsilon.dtype)  # Ensure CuPy array
        self.den = cp.asarray(A[1]).astype(self.epsilon.dtype)   # Ensure CuPy array

    def __mul__(self, other: 'd_tfs') -> 'd_tfs':
        """
        Multiply two transfer functions.
        
        Parameters:
        other (d_tfs): Another transfer function to multiply with.
        
        Returns:
        d_tfs: The resulting transfer function after multiplication.
        """
        num = cp.convolve(self.num, other.num)  # Use cp.convolve for efficiency
        den = cp.convolve(self.den, other.den)
        return d_tfs((num, den))

    def add_scalar(self, scalar: float) -> 'd_tfs':
        """
        Add a scalar to the transfer function.
        
        Parameters:
        scalar (float): The scalar value to add.
        
        Returns:
        d_tfs: The resulting transfer function after adding the scalar.
        """
        # Multiply the denominator by the scalar
        scalar_denom = scalar * self.den
        # Ensure both polynomials have the same length by padding with zeros at the back
        if len(self.num) > len(scalar_denom):
            scalar_denom = cp.pad(scalar_denom, (0, len(self.num) - len(scalar_denom)))
        elif len(self.num) < len(scalar_denom):
            self.num = cp.pad(self.num, (0, len(scalar_denom) - len(self.num)))
        # Add the polynomials
        num_with_scalar = cp.polyadd(self.num, scalar_denom)
        return d_tfs((num_with_scalar, self.den))

    def __truediv__(self, other: 'd_tfs') -> 'd_tfs':
        """
        Divide two transfer functions.
        
        Parameters:
        other (d_tfs): Another transfer function to divide by.
        
        Returns:
        d_tfs: The resulting transfer function after division.
        """
        num = cp.convolve(self.num, other.den)
        den = cp.convolve(self.den, other.num)
        return d_tfs((num, den))

    def __invert__(self) -> 'd_tfs':
        """
        Invert the transfer function.
        
        Returns:
        d_tfs: The inverted transfer function.
        """
        return d_tfs((self.den, self.num))

    def __str__(self) -> str:
        """
        String representation of the transfer function.
        
        Returns:
        str: The string representation of the transfer function.
        """
        return f"Transfer Function: num={self.num}, den={self.den}"

    def __repr__(self) -> str:
        """
        Representation of the transfer function.
        
        Returns:
        str: The string representation of the transfer function.
        """
        return self.__str__()

    def __call__(self, z: float) -> float:
        """
        Evaluate the transfer function at a given point.
        
        Parameters:
        z (float): The point at which to evaluate the transfer function.
        
        Returns:
        float: The value of the transfer function at the given point.
        """
        return cp.polyval(self.num, z) / cp.polyval(self.den, z)

    def apply_shift_operator(self, U_t: Union[List[float], cp.ndarray]) -> cp.ndarray:
        """
        Apply the shift operator to the input sequence.
        
        Parameters:
        U_t (array): The input sequence.
        
        Returns:
        array: The output sequence after applying the shift operator.
        """
        try:
            # Ensure input is a CuPy array
            U_t = cp.asarray(U_t)
            self.num += self.epsilon  # Add epsilon to avoid division by zero
            self.den += self.epsilon  # Add epsilon to avoid division by zero
            Y_t = lfilter(self.num, self.den, U_t)  # Use lfilter for filtering
            return cp.asarray(Y_t)
        except Exception as e:
            raise ValueError(f"Error applying shift operator: {e}")


class SPS_indirect_model:
    """
    Indirect Stochastic Process Simulation (SPS) model class.
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
        self.alpha = cp.random.randn(m,N)
        self.alpha = cp.sign(self.alpha)
        self.alpha[0, :] = 1
        self.pi_order = cp.random.permutation(cp.arange(m))

    def create_phi_optimized(self, Y: cp.ndarray, U: cp.ndarray, n_a: int, n_b: int) -> cp.ndarray:
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
        phi = cp.zeros((m, t, n_a + n_b), dtype=Y.dtype)
        
        # Handle Y lags
        for i in range(1, n_a + 1):
            phi[:, i:, i-1] = -1*Y[:, :-i]
        
        # Handle U lags
        for i in range(1, n_b + 1):
            phi[:, i:, n_a+i-1] = U[:-i]
        
        return phi

    def transform_to_open_loop(self, G: Tuple[Union[List[float], cp.ndarray], Union[List[float], cp.ndarray]], 
                               H: Tuple[Union[List[float], cp.ndarray], Union[List[float], cp.ndarray]], 
                               F: Tuple[Union[List[float], cp.ndarray], Union[List[float], cp.ndarray]], 
                               L: Tuple[Union[List[float], cp.ndarray], Union[List[float], cp.ndarray]]) -> Tuple['d_tfs', 'd_tfs']:
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
        G = d_tfs(G)
        H = d_tfs(H)
        F = d_tfs(F)
        L = d_tfs(L)

        GF_plus_I = (G * F).add_scalar(1)
        i_GF_plus_I = ~GF_plus_I
        G_0 = i_GF_plus_I * G * L
        H_0 = i_GF_plus_I * H
        return G_0, H_0
    
    def open_loop_sps(self, G_0: 'd_tfs', H_0: 'd_tfs', Y_t: cp.ndarray, U_t: cp.ndarray, n_a: int, n_b: int) -> Tuple[bool, cp.ndarray]:
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
            Y_t = cp.asarray(Y_t)
            U_t = cp.asarray(U_t)
            YGU = Y_t - G_0.apply_shift_operator(U_t)
            N_hat = (~H_0).apply_shift_operator(YGU)
            
            # Extract relevant segments
            N_hat_par = N_hat[-self.N:]
            U_t_par = U_t[-self.N-1:-1]
            perturbed_N_hat = cp.multiply(self.alpha, N_hat_par)
            
            # Compute y_bar vectorized
            y_bar = G_0.apply_shift_operator(U_t_par)[None, :] + H_0.apply_shift_operator(perturbed_N_hat[:, None])
            y_bar = y_bar.transpose(1, 0, 2)[0]
            # Compute phi_tilde
            phi_tilde = self.create_phi_optimized(y_bar, U_t_par, n_a, n_b)
            # Compute Cholesky decomposition and norm squared
            R = cp.matmul(phi_tilde.transpose(0, 2, 1), phi_tilde) / len(Y_t)
            L = cp.linalg.cholesky(R)
            Q, R = cp.linalg.qr(L)
            R_root_inv = cp.linalg.solve(R, Q.transpose(0, 2, 1))
            weighted_sum = cp.matmul(phi_tilde.transpose(0, 2, 1), perturbed_N_hat[:, :, None])
            S = cp.sum(cp.square(cp.matmul(R_root_inv, weighted_sum)), axis=(1, 2))
            # Ranking
            combined = cp.array(list(zip(self.pi_order, S)))
            order = cp.lexsort(combined.T)
            rank_R = cp.where(order == 0)[0][0] + 1
            
            return rank_R <= self.m - self.q, S
        except Exception as e:
            raise ValueError(f"Error in open-loop SPS: {e}")
