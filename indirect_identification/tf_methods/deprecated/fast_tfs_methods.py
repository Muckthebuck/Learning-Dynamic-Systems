import numpy as np
from numpy.polynomial.polynomial import polydiv
from numpy.polynomial.polyutils import trimseq
from numba import njit

from typing import Tuple, Union, List


# __all__ = [
#     '_add_tfs',
#     '_add_scalar',
#     '_sub_tfs',
#     '_mul_tfs',

# ]

_epsilon = 1e-10

@njit
def _simplify_array(arr: np.ndarray, epsilon: float = _epsilon) -> np.ndarray:
    if arr.size == 1:
        return arr
    arr[np.abs(arr) < epsilon] = 0
    arr = trimseq(arr)
    return arr

@njit
def _simplify(num, den, epsilon: float = _epsilon) -> Tuple[np.ndarray, np.ndarray]:
    return _simplify_array(num, epsilon),  _simplify_array(den, epsilon)


@njit
def _ensure_same_size(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    size1, size2 = p1.size, p2.size
    max_size = max(size1, size2)

    if size1 < max_size:
        p1 = np.concatenate((p1, np.zeros(max_size - size1, dtype=p1.dtype)))
    if size2 < max_size:
        p2 = np.concatenate((p2, np.zeros(max_size - size2, dtype=p2.dtype)))

    return p1, p2

@njit
def _convolve(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return _simplify_array(np.convolve(arr1, arr2))

@njit
def _add_tfs(num1: np.ndarray, den1: np.ndarray, num2: np.ndarray, den2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_self = _convolve(num1, den2) 
    num_other = _convolve(num2, den1)
    num_self, num_other = _ensure_same_size(num_self, num_other)
    num = num_self + num_other
    # new common denom
    den = _convolve(den1, den2)
    
    num, den = _reduce_fraction_numpy(num, den)
    return num, den
@njit
def _add_scalar(num1: np.ndarray, den1: np.ndarray, scalar: Union[int, np.float32, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
    num, den = _ensure_same_size(num1, scalar*den1)
    return _reduce_fraction_numpy(num+den, den)

@njit
def _sub_tfs(num1: np.ndarray, den1: np.ndarray, num2: np.ndarray, den2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num, den = _add_tfs(num1, den1, -num2, den2)
    return num, den

@njit
def _mul_tfs(num1: np.ndarray, den1: np.ndarray, num2: np.ndarray, den2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num = _convolve(num1, num2)
    den = _convolve(den1, den2)
    return _reduce_fraction_numpy(num, den)

@njit
def _mul_scalar(num1: np.ndarray, den1: np.ndarray, scalar: Union[int, np.float32, np.float64]) -> Tuple[np.ndarray, np.ndarray]:
    return num1*scalar, den1

@njit
def _div_tfs(num1: np.ndarray, den1: np.ndarray, num2: np.ndarray, den2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _mul_tfs(num1, den1, den2, num2)



@njit
def _poly_divide(a, b):
    """Divide polynomial a(x) by b(x), return quotient and remainder."""
    quotient, remainder = polydiv(a, b)
    quotient, remainder = _simplify(quotient, remainder)
    return quotient, remainder

@njit
def _poly_sub(p, q):
    """Subtract p-q"""
    p,q = _ensure_same_size(p, q)
    return p-q

@njit
def _eGCD(f, g, tolerance=_epsilon):
    """Compute the GCD of two polynomials and the coefficients u(x) and v(x)."""

    # Initialize u(x) and v(x)
    u0, v0 = np.array([1.0]), np.array([0.0])
    u1, v1 = np.array([0.0]), np.array([1.0])

    while np.any(np.abs(g) > tolerance):
        q, r = _poly_divide(f, g)

        f, g = g, r  # Update for next iteration
        if np.all(np.abs(g) < tolerance):
            g = np.zeros_like(g)

        # Update u and v
        u2 = _poly_sub(u0, np.convolve(q, u1))
        v2 = _poly_sub(v0, np.convolve(q, v1))

        u2, _ = _ensure_same_size(u2, u0)
        v2, _ = _ensure_same_size(v2, v0)

        # Ensure correct sizes
        u0, u1 = u1, u2
        v0, v1= v1, v2

    gcd = f  # The remaining f is the GCD
    return gcd, u0, v0

@njit
def _reduce_fraction_numpy(num: np.ndarray, den: np.ndarray, tol: float = _epsilon):
    """Reduce a polynomial fraction by dividing by the GCD."""
    gcd, _, _ = _eGCD(num, den)  # Compute GCD
    if np.all(np.abs(gcd) < tol):  # If GCD is too small, return the original fraction
        return num, den
    # Perform polynomial division
    num_reduced, _ = polydiv(num, gcd)
    den_reduced, _ = polydiv(den, gcd)

    # ensure the first number is 1
    gcd = den_reduced[np.argmax(den != 0)]
    num_reduced = num_reduced / gcd
    den_reduced = den_reduced / gcd
    return _simplify(num_reduced, den_reduced)
