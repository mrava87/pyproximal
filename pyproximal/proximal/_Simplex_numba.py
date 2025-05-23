import os
import numpy as np
from numba import jit, prange # type: ignore # Numba is not available in this environment
from typing import Any # For Numba JIT compiled functions, array types are specific

# detect whether to use parallel or not
numba_threads_str: str = os.getenv('NUMBA_NUM_THREADS', '1')
numba_threads: int = int(numba_threads_str)
parallel: bool = True if numba_threads != 1 else False


# Note: Type hints for Numba JIT functions are primarily for Python-side static analysis.
# Numba has its own type inference. Using np.ndarray as a general hint for array-like inputs.
# Numba actually uses its own array types internally.

@jit(nopython=True)
def fun_jit(mu: float, x: np.ndarray, coeffs: np.ndarray, scalar: float,
            lower: float, upper: float) -> float:
    """Bisection function"""
    # Numba will infer types for internal variables like p.
    # The dot product of two 1D arrays or compatible arrays returns a scalar.
    # minimum/maximum are element-wise.
    return np.dot(coeffs, np.minimum(np.maximum(x - mu * coeffs, lower), upper)) - scalar


@jit(nopython=True, nogil=True)
def bisect_jit(x: np.ndarray, coeffs: np.ndarray, scalar: float,
               lower: float, upper: float,
               bisect_lower: float, bisect_upper: float,
               maxiter: int, ftol: float, xtol: float) -> float:
    """Bisection method

    Parameters
    ----------
    x : :obj:`np.ndarray`
        Input vector
    coeffs : :obj:`np.ndarray`
        Vector of coefficients used in the definition of the hyperplane
    scalar : :obj:`float`
        Scalar used in the definition of the hyperplane
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound of Box
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound of Box
    bisect_lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower end of bisection
    bisect_upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper end of bisection
    maxiter : :obj:`int`, optional
        Maximum number of iterations
    ftol : :obj:`float`, optional
        Function tolerance
    xtol : :obj:`float`, optional
        Solution absolute tolerance

    """
    a: float = bisect_lower
    b: float = bisect_upper
    fa: float = fun_jit(a, x, coeffs, scalar, lower, upper)
    
    for iiter in range(maxiter):
        c: float = (a + b) / 2.
        if (b - a) / 2. < xtol: # Ensure float division
            return c
        fc: float = fun_jit(c, x, coeffs, scalar, lower, upper)
        if np.abs(fc) < ftol:
            return c
        # Numba's np.sign behaves like Python's math.copysign(1, x) for x!=0, returns 0 for x=0.
        # This comparison should be safe.
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
    return c # Or (a+b)/2.0 if maxiter reached


@jit(nopython=True, parallel=parallel, nogil=True)
def simplex_jit(x: np.ndarray, coeffs: np.ndarray, scalar: float,
                lower: float, upper: float,
                maxiter: int, ftol: float, xtol: float) -> np.ndarray:
    """Simplex proximal

    Parameters
    ----------
    x : :obj:`np.ndarray`
        Input vector
    coeffs : :obj:`np.ndarray`
        Vector of coefficients used in the definition of the hyperplane
    scalar : :obj:`float`
        Scalar used in the definition of the hyperplane
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound of Box
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound of Box
    maxiter : :obj:`int`, optional
        Maximum number of iterations
    ftol : :obj:`float`, optional
        Function tolerance
    xtol : :obj:`float`, optional
        Solution absolute tolerance

    """
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        bisect_lower = -1
        while fun_jit(bisect_lower, x[i], coeffs, scalar, lower, upper) < 0:
            bisect_lower *= 2
        bisect_upper = 1
        while fun_jit(bisect_upper, x[i], coeffs, scalar, lower, upper) > 0:
            bisect_upper *= 2
        c = bisect_jit(x[i], coeffs, scalar, lower, upper,
                       bisect_lower, bisect_upper, maxiter, ftol, xtol)
        y[i] = np.minimum(np.maximum(x[i] - c * coeffs, lower), upper)
    return y
