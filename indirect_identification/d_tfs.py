import numpy as np
from typing import Tuple, Union, List
from scipy.signal import lfilter
from indirect_identification.tf_methods.fast_tfs_methods_fast_math import *

__all__  = [
    'd_tfs',
    'apply_tf_matrix',
    'invert_matrix'
]

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
    def __new__(cls, A):
        """
        Control instance creation: If numerator and denominator are both scalars, return the scalar result.
        """
        num, den = np.asarray(A[0]), np.asarray(A[1])
        
        # If both numerator and denominator are scalars (size=1), return the scalar value directly
        if num.size == 1 and den.size == 1:
            return num[0] / den[0]  # Convert to scalar float
        
        return super().__new__(cls)  # Otherwise, create a normal instance
    
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
            - A NumPy array of Tfs 
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
            if other.dtype == object:
                if np.any([isinstance(x, d_tfs) for x in other.flat]):  # Check if any element is d_tfs
                    return np.array(self)*other
                else:
                    raise TypeError("Multiplication with an ndarray containing objects is not supported. Supported objects d_tfs")
            else:  
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
    
    def __neg__(self):
        """Implement unary negation (-A) by negating the numerator."""
        return d_tfs((-self.num, self.den))

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
    
    def __eq__(self, other: Union['d_tfs', Union[int, float, np.float32, np.float64]]) -> bool:
        """Check if two transfer function matrices are equal."""
        if isinstance(other, d_tfs):
            return np.allclose(self.num, other.num) and np.allclose(self.den, other.den)
        elif isinstance(other, (int, float, np.float32, np.float64)):
            if self.num.size == 1 and self.den.size ==1:
                # only case where tf is basically a number 
                return np.allclose(self.num/self.den, other)
            else:
                return False
        else:
            raise NotImplementedError(f"Division with {type(other)} is not supported")


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
            Y_t = lfilter(self.num + self.epsilon, self.den + self.epsilon, U_t)  # Use lfilter for filtering
            return np.asarray(Y_t)
        except Exception as e:
            raise ValueError(f"Error applying shift operator: {e}")


def apply_tf_matrix(G: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Applies a matrix of transfer functions G to an input matrix U.

    This function performs element-wise multiplication between each transfer function 
    in G and the corresponding row in U, then accumulates the results along the rows.

    Parameters:
        G (np.ndarray): A (m x n) matrix where each element is a transfer function 
            (or a scalar/matrix that supports multiplication with U).
        U (np.ndarray): A (n x k) matrix representing n input signals, each of length k.

    Returns:
        np.ndarray: A (m x k) matrix where each row represents the output of 
        applying the corresponding transfer functions from G to the inputs in U.

    Example:
        >>> G = np.array([[tf1, tf2], [tf3, tf4]])  # Some transfer functions
        >>> U = np.array([[u1, u2, u3], [v1, v2, v3]])  # Two input signals
        >>> Y = apply_tf_matrix(G, U)
        >>> print(Y.shape)  # (2, 3), output has same time length as U
    """
    m, n = G.shape  # m outputs, n inputs
    k = U.shape[1]  # Time length of each input sequence
    Y = np.zeros((m, k))  # Initialize output matrix

    for i in range(m):
        for j in range(n):
            Y[i, :] += G[i, j] * U[j, :]  # Apply transfer function multiplication

    return Y


def _lu_decomposition(A):
    """Performs LU decomposition of a square matrix A.

    The function decomposes the matrix `A` into a lower triangular matrix `L`
    and an upper triangular matrix `U` such that A = L * U.

    Parameters:
        A (np.ndarray): A square (n, n) matrix to be decomposed.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple (L, U) where:
            - L (np.ndarray) is an (n, n) lower triangular matrix.
            - U (np.ndarray) is an (n, n) upper triangular matrix.
    """
    n = A.shape[0]
    L = np.eye(n, dtype=type(A))
    U = A.copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor*U[i, i:]
    return L, U

def _forward_substitution(L, b):
    """Solves the lower triangular system Ly = b using forward substitution.

    Parameters:
        L (np.ndarray): An (n, n) lower triangular matrix.
        b (np.ndarray): A right-hand side (n,) vector.

    Returns:
        np.ndarray: A solution vector `y` of shape (n,).
    """
    n = L.shape[0]
    y = np.zeros_like(b, dtype=type(L))
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def _backward_substitution(U, y):
    """Solves the upper triangular system Ux = y using backward substitution.

    Parameters:
        U (np.ndarray): An (n, n) upper triangular matrix.
        y (np.ndarray): A right-hand side (n,) vector obtained from forward substitution.

    Returns:
        np.ndarray: A solution vector `x` of shape (n,).
    """
    n = U.shape[0]
    x = np.zeros_like(y, dtype=type(U))
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def invert_matrix(A):
    """Computes the inverse of a square matrix A made up of d_tfs using LU decomposition.

    The function decomposes `A` into L and U matrices using LU decomposition,
    then solves for each column of the identity matrix to construct the inverse.

    Parameters:
        A (np.ndarray): A square (n, n) matrix to be inverted.

    Returns:
        np.ndarray: The inverse of `A`, an (n, n) matrix.
    """
    n = A.shape[0]
    L, U = _lu_decomposition(A)
    inv_A = np.zeros_like(A)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        y = _forward_substitution(L, e)
        x = _backward_substitution(U, y)
        inv_A[:, i] = x
    return inv_A

