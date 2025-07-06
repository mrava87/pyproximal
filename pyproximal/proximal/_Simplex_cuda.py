from numba import cuda # type: ignore # Numba is not available in this environment
import numpy as np # For np.ndarray type hint, though numba uses its own array types internally
from typing import Any # For device arrays, which don't have a standard Python type hint


# Note: Type hints for Numba JIT functions are primarily for Python-side static analysis.
# Numba has its own type inference and system for CUDA.
# For device functions and kernels, 'Any' or specific Numba types (if available for hinting)
# might be used for array-like structures passed to/from CUDA.
# Using np.ndarray here as a placeholder for "CUDA array-like".

@cuda.jit(device=True)
def fun_jit_cuda(mu: float, x: Any, # CUDA device array (e.g., numba.cuda.cudadrv.devicearray.DeviceNDArray)
                 coeffs: Any, scalar: float,
                 lower: float, upper: float) -> float:
    """Bisection function"""
    p: float = 0.0 # Ensure p is float
    for i in range(coeffs.shape[0]):
        # Numba's min/max handle scalars, direct translation
        val_inside: float = x[i] - mu * coeffs[i]
        clipped_val: float
        if val_inside < lower:
            clipped_val = lower
        elif val_inside > upper:
            clipped_val = upper
        else:
            clipped_val = val_inside
        p += coeffs[i] * clipped_val
    return p - scalar


@cuda.jit(device=True)
def bisect_jit_cuda(x: Any, coeffs: Any, scalar: float, # CUDA device arrays
                    lower: float, upper: float,
                    bisect_lower: float, bisect_upper: float,
                    maxiter: int, ftol: float, xtol: float) -> float:
    """Bisection method (See _Simplex_numba for details).

    """
    a: float = bisect_lower
    b: float = bisect_upper
    fa: float = fun_jit_cuda(a, x, coeffs, scalar, lower, upper)
    # fa_sign: float # To store sign of fa if needed, Numba handles division by abs fine
    
    for iiter in range(maxiter):
        c: float = (a + b) / 2.
        if (b - a) / 2. < xtol: # Ensure float division
            return c
        fc: float = fun_jit_cuda(c, x, coeffs, scalar, lower, upper)
        if abs(fc) < ftol:
            return c
        
        # Check signs: (fc / abs(fc)) == (fa / abs(fa)) can be problematic if fc or fa is zero.
        # Numba might handle this, but safer: np.sign(fc) == np.sign(fa)
        # However, direct translation of original logic:
        fc_sign: float = 0.0
        if fc != 0: fc_sign = fc / abs(fc) # Avoid division by zero
        
        fa_sign: float = 0.0
        if fa != 0: fa_sign = fa / abs(fa)

        if fc_sign == fa_sign: # Same sign (or both zero)
            a = c
            fa = fc
        else:
            b = c
    return c # Or (a+b)/2.0 if maxiter reached without convergence


@cuda.jit
def simplex_jit_cuda(x: Any, coeffs: Any, scalar: float, # CUDA device arrays
                     lower: float, upper: float,
                     maxiter: int, ftol: float, xtol: float,
                     y: Any): # Output array (CUDA device array)
    """Simplex proximal

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Input vector
    coeffs : :obj:`numpy.ndarray`
        Vector of coefficients used in the definition of the hyperplane
    scalar : :obj:`float`
        Scalar used in the definition of the hyperplane
    lower : :obj:`float` or :obj:`numpy.ndarray`, optional
        Lower bound of Box
    upper : :obj:`float` or :obj:`numpy.ndarray`, optional
        Upper bound of Box
    maxiter : :obj:`int`, optional
        Maximum number of iterations
    ftol : :obj:`float`, optional
        Function tolerance
    xtol : :obj:`float`, optional
        Solution absolute tolerance
    y : :obj:`numpy.ndarray`
        Output vector

    """
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < x.shape[0]:
        bisect_lower = -1
        while fun_jit_cuda(bisect_lower, x[i], coeffs, scalar, lower, upper) < 0:
            bisect_lower *= 2
        bisect_upper = 1
        while fun_jit_cuda(bisect_upper, x[i], coeffs, scalar, lower, upper) > 0:
            bisect_upper *= 2

        c = bisect_jit_cuda(x[i], coeffs, scalar, lower, upper,
                            bisect_lower, bisect_upper, maxiter, ftol, xtol)

        for j in range(coeffs.shape[0]):
            y[i][j] = min(max(x[i][j] - c * coeffs[j], lower), upper)