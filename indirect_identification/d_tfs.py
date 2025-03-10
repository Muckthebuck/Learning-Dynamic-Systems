import numpy as np
from numba.experimental import jitclass
from numba import njit, jit, float32 
from collections import OrderedDict
from typing import Tuple, Union, List
from scipy.signal import lfilter
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import *


def commutative_ops(cls):
    """Decorator that automatically adds reverse methods for commutative operations."""
    for op in ["add", "mul"]:  # Only for commutative operations
        if hasattr(cls, f"__{op}__"):
            setattr(cls, f"__r{op}__", getattr(cls, f"__{op}__"))
    return cls

@commutative_ops
class d_tfs:
    """
    Discrete Transfer Function System class.
    """
    def __init__(self, A: Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]):
        """
        Initialize the transfer function with numerator and denominator coefficients.
        
        Parameters:
        A (tuple): A tuple containing two lists or arrays, the numerator and denominator coefficients.
        """
        self.epsilon = np.float64(1e-10)
        self.num = np.asarray(A[0]).astype(self.epsilon.dtype)  # Ensure CuPy array
        self.den = np.asarray(A[1]).astype(self.epsilon.dtype)   # Ensure CuPy array

    def __add__(self, other: Union['d_tfs', Union[int, float, np.float32, np.float64]]) -> 'd_tfs':
        """
        Perform addition of the current transfer function with another transfer function or a scalar.

        Parameters:
        other (Union[d_tfs, int, float, np.float32, np.float64]): 
            - A `d_tfs` instance representing another transfer function.  
            - A scalar value (int, float, np.float32, np.float64) to be added to the transfer function.

        Returns:
        d_tfs: A new transfer function representing the sum.

        Raises:
        NotImplementedError: If the operation is attempted with an unsupported type.
        """
        if isinstance(other, d_tfs):
            return d_tfs(_add_tfs(self.num, self.den, other.num, other.den))
        elif isinstance(other, (int, float, np.float32, np.float64)):
            return d_tfs(_add_scalar(self.num, self.den, other))
        else:
            return NotImplementedError(f"Addition with {type(other)} is not supported")

    def __sub__(self, other: Union['d_tfs', Union[int, float, np.float32, np.float64]]) -> 'd_tfs':
        """
        Subtract another transfer function or a scalar from the current transfer function.

        Parameters:
        other (Union[d_tfs, int, float, np.float32, np.float64]): 
            - A `d_tfs` instance representing another transfer function to subtract.
            - A scalar value (int, float, np.float32, np.float64) to subtract from the transfer function.

        Returns:
        d_tfs: A new transfer function representing the result of the subtraction.

        Raises:
        NotImplementedError: If the operation is attempted with an unsupported type.
        """
        if isinstance(other, d_tfs):
            return d_tfs(_sub_tfs(self.num, self.den, other.num, other.den))
        elif isinstance(other, (int, float, np.float32, np.float64)):
            return d_tfs(_sub_scalar(self.num, self.den, other))
        else:
            return NotImplementedError(f"Subtraction with {type(other)} is not supported")
        
    def __rsub__(self, other: Union[int, float, np.float32, np.float64]) -> 'd_tfs':
        """
        Subtract the current transfer function from a scalar (right-hand side).

        Parameters:
        other (Union[int, float, np.float32, np.float64]): 
            A scalar value (int, float, np.float32, np.float64) from which the transfer function will be subtracted.

        Returns:
        d_tfs: A new transfer function representing the result of the subtraction.

        Raises:
        NotImplementedError: If the operation is attempted with an unsupported type.
        """
        if isinstance(other, (int, float, np.float32, np.float64)):
            return d_tfs(_rsub_scalar(self.num, self.den, other))
        else:
            return NotImplementedError(f"Subtraction with {type(other)} is not supported")

    def __mul__(self, other: Union['d_tfs', Union[int, float, np.float32, np.float64], np.ndarray]) -> Union['d_tfs', np.ndarray]:
        """
        Multiply the current transfer function with another transfer function, a scalar, or a time-series array.

        Parameters:
        other (Union[d_tfs, Union[int, float, np.float32, np.float64], np.ndarray]): 
            - A `d_tfs` instance representing another transfer function.
            - A scalar value (int, float, np.float32, np.float64) to multiply with the transfer function.
            - A NumPy array representing a discrete-time signal to which the transfer function is applied using `scipy.lfilter(self.num, self.den, U_t)`.

        Returns:
        Union[d_tfs, np.ndarray]: 
            - If multiplied with another `d_tfs` or scalar, returns a new `d_tfs` instance representing the resulting transfer function.
            - If multiplied with a NumPy array, returns the filtered output signal.

        Raises:
        NotImplementedError: If the operation is attempted with an unsupported type.
        """
        if isinstance(other, d_tfs):
            return d_tfs(_mul_tfs(self.num, self.den, other.num, other.den))
        elif isinstance(other, (int, float, np.float32, np.float64)):
            return d_tfs(_mul_scalar(self.num, self.den, other))
        elif isinstance(other, np.ndarray):
            return self._apply_shift_operator(U_t=other)
        else:
            return NotImplementedError(f"Multiplication with {type(other)} is not supported")

    def __truediv__(self, other: Union['d_tfs', Union[int, float, np.float32, np.float64]]) -> 'd_tfs':
        """
        Divide two transfer functions or divide a transfer function by a scalar.
        
        Parameters:
        other (d_tfs): Another transfer function to divide by.
        or
        other (scalar): A scalar value to divide the transfer function by.
        
        Returns:
        d_tfs: The resulting transfer function after division.
        
        Raises:
        NotImplementedError: If the operation is attempted with an unsupported type.
        """
        if isinstance(other, d_tfs):
            return d_tfs(_div_tfs(self.num, self.den, other.num, other.den))
        elif isinstance(other, (int, float, np.float32, np.float64)):
            return d_tfs(_mul_scalar(self.num, self.den, 1/other))
        else:
            raise NotImplementedError(f"Division with {type(other)} is not supported")

    def __rtruediv__(self, other: Union[int, float, np.float32, np.float64]) -> 'd_tfs':
        """
        Divide a scalar by a transfer function.
        
        Parameters:
        other (scalar): A scalar value to divide the transfer function by.
        
        Returns:
        d_tfs: The resulting transfer function after division.
        
        Raises:
        NotImplementedError: If the operation is attempted with an unsupported type.
        """
        if isinstance(other, (int, float, np.float32, np.float64)):
            return d_tfs(_mul_scalar(self.den, self.num, other))
        else:
            raise NotImplementedError(f"Division with {type(other)} is not supported")

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
        return np.polyval(self.num, z) / np.polyval(self.den, z)

    def _apply_shift_operator(self, U_t: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Apply the shift operator to the input sequence.
        
        Parameters:
        U_t (array): The input sequence.
        
        Returns:
        array: The output sequence after applying the shift operator.
        """
        try:
            # Ensure input is a np array
            U_t = np.asarray(U_t)
            self.num += self.epsilon  # Add epsilon to avoid division by zero
            self.den += self.epsilon  # Add epsilon to avoid division by zero
            Y_t = lfilter(self.num, self.den, U_t)  # Use lfilter for filtering
            return np.asarray(Y_t)
        except Exception as e:
            raise ValueError(f"Error applying shift operator: {e}")

