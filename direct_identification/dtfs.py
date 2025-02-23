import os
import torch
try:
    torch.cuda.current_device()
    import cupy as cp
    from cupyx.scipy.signal import lfilter
except:
    # Fall back to unoptimised versions
    import numpy as cp
    from scipy.signal import lfilter

class d_tfs:
    def __init__(self, A):
        self.epsilon = cp.float64(1e-10)
        self.num = cp.asarray(A[0]).astype(self.epsilon.dtype)  # Ensure CuPy array
        self.den = cp.asarray(A[1]).astype(self.epsilon.dtype)   # Ensure CuPy array

    def __mul__(self, other):
        num = cp.convolve(self.num, other.num)  # Use cp.convolve for efficiency
        den = cp.convolve(self.den, other.den)
        return d_tfs((num, den))

    def add_scalar(self, scalar):
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

    def __truediv__(self, other):
        num = cp.convolve(self.num, other.den)
        den = cp.convolve(self.den, other.num)
        return d_tfs((num, den))

    def __invert__(self):
        return d_tfs((self.den, self.num))

    def __str__(self):
        return f"Transfer Function: num={self.num}, den={self.den}"

    def __repr__(self):
        return self.__str__()

    def __call__(self, z):
        return cp.polyval(self.num, z) / cp.polyval(self.den, z)

    def apply_shift_operator(self, U_t):
        try:
            # Ensure input is a CuPy array
            U_t = cp.asarray(U_t)
            self.num += self.epsilon  # Add epsilon to avoid division by zero
            self.den += self.epsilon  # Add epsilon to avoid division by zero
            Y_t = lfilter(self.num, self.den, U_t)  # Use lfilter for filtering
            return cp.asarray(Y_t)
        except Exception as e:
            raise ValueError(f"Error applying shift operator: {e}")

