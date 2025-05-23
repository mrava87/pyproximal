import logging
import numpy as np
from typing import Optional, Tuple, Callable, Any # Added imports

from pylops.utils.backend import get_array_module, to_cupy_conditional # type: ignore[import-untyped]
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import SimplexProj

# Numba and CUDA related imports are conditional, handle their types carefully
jit: Optional[Callable[..., Any]]
bisect_jit: Optional[Callable[..., Any]]
simplex_jit: Optional[Callable[..., Any]]
fun_jit: Optional[Callable[..., Any]]
bisect_jit_cuda: Optional[Callable[..., Any]]
simplex_jit_cuda: Optional[Callable[..., Any]]
fun_jit_cuda: Optional[Callable[..., Any]]
jit_message: str # Define jit_message to ensure it's always available

try:
    from numba import jit # type: ignore[no-redef]
    # Assuming these Numba functions are correctly typed in their own files if possible,
    # or treat as Any if their signatures are too dynamic or complex for here.
    from ._Simplex_numba import bisect_jit, simplex_jit, fun_jit # type: ignore[no-redef]
    from ._Simplex_cuda import bisect_jit_cuda, simplex_jit_cuda, fun_jit_cuda # type: ignore[no-redef]
except ModuleNotFoundError:
    jit = None
    bisect_jit = None
    simplex_jit = None
    fun_jit = None
    bisect_jit_cuda = None
    simplex_jit_cuda = None
    fun_jit_cuda = None
    jit_message = 'Numba not available, reverting to numpy.'
    logging.warning(jit_message) 
except Exception as e:
    jit = None
    bisect_jit = None
    simplex_jit = None
    fun_jit = None
    bisect_jit_cuda = None
    simplex_jit_cuda = None
    fun_jit_cuda = None
    jit_message = 'Failed to import numba (error:%s), use numpy.' % e
    logging.warning(jit_message) 

# logging.basicConfig should ideally be called only once at application level.
# If this module is imported multiple times, it could cause issues.
# For a library, it's better to get a logger instance: logger = logging.getLogger(__name__)
# and let the application configure logging.
# However, to match original behavior of printing warning:
if 'jit_message' not in locals() and jit is None : # If imports failed silently before message assignment
    jit_message = 'Numba/CUDA JIT functions not loaded (reason unknown), reverting to numpy if applicable.'
    logging.warning(jit_message)

# Ensure logging is configured if it hasn't been (e.g. if run as script or imported early)
# This is a bit of a workaround for library logging.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class _Simplex(ProxOperator):
    """Simplex operator (numpy version)
    """
    def __init__(self, n: int, radius: float,
                 dims: Optional[Tuple[int, int]] = None, axis: int = -1,
                 maxiter: int = 100, xtol: float = 1e-8, call: bool = True):
        super().__init__(None, False) # Op is None, hasgrad is False
        if dims is not None and len(dims) != 2:
            raise ValueError('provide only 2 dimensions, or None')
        self.n: int = n
        self.radius: float = radius
        self.dims: Optional[Tuple[int, int]] = dims
        self.axis: int = axis
        self.otheraxis: int = 1 if axis == 0 else 0
        self.maxiter: int = maxiter
        self.xtol: float = xtol
        self.call_enabled: bool = call # Ensuring self.call was renamed to self.call_enabled

        self.simplex: SimplexProj = SimplexProj(self.n if dims is None else dims[self.axis], # type: ignore[index]
                                   self.radius, maxiter=self.maxiter,
                                   xtol=self.xtol)

    def __call__(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        if not self.call_enabled: # Ensuring this uses self.call_enabled
            return False
        
        is_in_set: bool
        if self.dims is None: # Vector case
            sum_check: bool = np.abs(np.sum(x) - self.radius) < tol
            non_negative_check: bool = np.all(x >= -tol) 
            is_in_set = sum_check and non_negative_check
        else: # Matrix case
            x_reshaped: np.ndarray = x.reshape(self.dims)
            if self.axis == 0: 
                x_reshaped = x_reshaped.T
            
            all_vectors_in_set: bool = True
            # Ensure self.dims is not None before accessing self.dims[self.otheraxis]
            if self.dims is not None: # Redundant check if initial logic path is correct, but safe
                for i in range(x_reshaped.shape[0]): 
                    vec: np.ndarray = x_reshaped[i]
                    sum_check_vec: bool = np.abs(np.sum(vec) - self.radius) < tol
                    non_negative_check_vec: bool = np.all(vec >= -tol)
                    if not (sum_check_vec and non_negative_check_vec):
                        all_vectors_in_set = False
                        break
                is_in_set = all_vectors_in_set
            else: 
                is_in_set = False 
        return is_in_set

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        y_result: np.ndarray # Renamed y to y_result
        if self.dims is None: # Vector case
            y_result = self.simplex(x)
        else: # Matrix case
            x_reshaped: np.ndarray = x.reshape(self.dims)
            if self.axis == 0: 
                x_reshaped = x_reshaped.T
            
            y_processed_shape: np.ndarray = np.zeros_like(x_reshaped)
            
            # Ensure self.dims is not None
            if self.dims is not None: # Redundant check if initial logic path is correct, but safe
                 for i in range(x_reshaped.shape[0]): 
                    y_processed_shape[i] = self.simplex(x_reshaped[i])
            
            if self.axis == 0: 
                y_processed_shape = y_processed_shape.T
            y_result = y_processed_shape
        return y_result.ravel()


class _Simplex_numba(_Simplex):
    """Simplex operator (numba version)
   """
    def __init__(self, n: int, radius: float,
                 dims: Optional[Tuple[int, int]] = None, axis: int = -1,
                 maxiter: int = 100, ftol: float = 1e-8, xtol: float = 1e-8,
                 call: bool = False): # Changed default call to False as in original
        # Call _Simplex.__init__ first to set up common attributes like self.n, self.dims etc.
        # Note: _Simplex __init__ expects 'call' but not 'ftol'.
        # We are overriding __init__, so we need to ensure all necessary attributes are set.
        # ProxOperator.__init__ is called by _Simplex.__init__
        # super().__init__(n, radius, dims=dims, axis=axis, maxiter=maxiter, xtol=xtol, call=call)
        # Manual super call to ProxOperator directly, then set attributes.
        ProxOperator.__init__(self, None, False) # type: ignore[misc] # Op is None, hasgrad is False
        if dims is not None and len(dims) != 2:
            raise ValueError('provide only 2 dimensions, or None')
        self.n: int = n
        self.radius: float = radius
        self.dims: Optional[Tuple[int, int]] = dims
        self.axis: int = axis
        self.otheraxis: int = 1 if axis == 0 else 0
        self.maxiter: int = maxiter
        self.xtol: float = xtol # _Simplex init uses xtol
        self.call_enabled: bool = call # _Simplex init uses call (renamed to call_enabled)

        # Numba specific attributes
        self.coeffs: np.ndarray = np.ones(self.n if dims is None else dims[self.axis])
        self.ftol: float = ftol # Numba version uses ftol

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used
        y_result: np.ndarray # Renamed y to y_result
        # Ensure Numba functions are available
        if fun_jit is None or bisect_jit is None or simplex_jit is None:
            logging.warning("Numba JIT functions not available, operation will be slow or fail if NumPy fallback not implemented in caller.")
            # Fallback to numpy version's prox logic (or could call super().prox if it did the same)
            # This would require self.simplex to be initialized as in _Simplex.
            # For now, raising error as Numba path implies Numba functions are critical.
            raise RuntimeError("Numba JIT functions required for _Simplex_numba.prox but not found.")


        if self.dims is None: # Vector case
            bisect_lower: float = -1.0
            # Assuming fun_jit takes (val, x, coeffs, radius, lower_bound, upper_bound)
            while fun_jit(bisect_lower, x, self.coeffs, self.radius, 0., 1e10) < 0: # type: ignore
                bisect_lower *= 2
            bisect_upper: float = 1.0
            while fun_jit(bisect_upper, x, self.coeffs, self.radius, 0., 1e10) > 0: # type: ignore
                bisect_upper *= 2
            
            # Assuming bisect_jit takes (x, coeffs, radius, lower_b, upper_b, bisect_low, bisect_up, maxiter, ftol, xtol)
            c_val: float = bisect_jit(x, self.coeffs, self.radius, 0., 1e10, # type: ignore
                                      bisect_lower, bisect_upper, self.maxiter,
                                      self.ftol, self.xtol)
            y_result = np.minimum(np.maximum(x - c_val * self.coeffs, 0.), 1e10)
        else: # Matrix case
            x_reshaped: np.ndarray = x.reshape(self.dims)
            if self.axis == 0:
                x_reshaped = x_reshaped.T
            
            # Assuming simplex_jit processes the matrix row-wise (or col-wise after transpose)
            y_processed: np.ndarray = simplex_jit(x_reshaped, self.coeffs, self.radius, 0., 1e10, # type: ignore
                                                 self.maxiter, self.ftol, self.xtol)
            if self.axis == 0:
                y_processed = y_processed.T
            y_result = y_processed
        return y_result.ravel()


class _Simplex_cuda(_Simplex):
    """Simplex operator (cuda version)

    This implementation is adapted from https://github.com/DIG-Kaust/HPC_Hackathon_DIG.

   """
    def __init__(self, n: int, radius: float,
                 dims: Optional[Tuple[int, int]] = None, axis: int = -1,
                 maxiter: int = 100, ftol: float = 1e-8, xtol: float = 1e-8,
                 call: bool = False, num_threads_per_blocks: int = 32):
        # Similar to _Simplex_numba, call ProxOperator.__init__ then set attributes
        ProxOperator.__init__(self, None, False) # type: ignore[misc] # Op is None, hasgrad is False
        if dims is not None and len(dims) != 2:
            raise ValueError('provide only 2 dimensions, or None')
        self.n: int = n
        self.radius: float = radius
        self.dims: Optional[Tuple[int, int]] = dims
        self.axis: int = axis
        self.otheraxis: int = 1 if axis == 0 else 0
        self.maxiter: int = maxiter
        self.xtol: float = xtol
        self.call_enabled: bool = call # Renamed to avoid conflict

        # CUDA specific attributes
        self.coeffs_np: np.ndarray = np.ones(self.n if dims is None else dims[self.axis]) # Store NumPy version
        self.ftol: float = ftol
        self.num_threads_per_blocks: int = num_threads_per_blocks
        self._coeffs_gpu: Optional[Any] = None # To store cupy array if used, type Any for cupy array

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used
        ncp = get_array_module(x) # Should be cupy for this path
        if get_module_name(ncp) != 'cupy': # type: ignore[attr-defined]
             logging.warning("CUDA engine selected but input array is not CuPy. Performance may suffer or it may fail.")
             # Or raise error: raise TypeError("Input array must be CuPy array for CUDA engine.")

        if simplex_jit_cuda is None: # Check if CUDA function is available
            raise RuntimeError("CUDA JIT function 'simplex_jit_cuda' not available.")
        if self.dims is None: # Should not happen if factory ensures dims for CUDA path, but defensive
            raise ValueError("_Simplex_cuda requires 'dims' to be specified.")

        x_reshaped: np.ndarray = x.reshape(self.dims)
        if self.axis == 0:
            x_reshaped = x_reshaped.T
        
        # Ensure coeffs is on the same device as x
        if self._coeffs_gpu is None or get_array_module(self._coeffs_gpu) != ncp: # type: ignore[attr-defined]
            self._coeffs_gpu = ncp.asarray(self.coeffs_np)

        y_processed: np.ndarray = ncp.empty_like(x_reshaped)
        num_blocks: int = (x_reshaped.shape[0] + self.num_threads_per_blocks - 1) // self.num_threads_per_blocks
        
        # Call CUDA kernel
        simplex_jit_cuda[num_blocks, self.num_threads_per_blocks]( # type: ignore
            x_reshaped, self._coeffs_gpu, self.radius,
            0., 1e10, self.maxiter,
            self.ftol, self.xtol, y_processed
        )
        
        if self.axis == 0:
            y_processed = y_processed.T
        return y_processed.ravel()


def Simplex(n: int, radius: float,
            dims: Optional[Tuple[int, int]] = None, axis: int = -1,
            maxiter: int = 100, ftol: float = 1e-8, xtol: float = 1e-8,
            call: bool = True, engine: str = 'numpy') -> ProxOperator: # Verified signature
    r"""Simplex proximal operator.

    Proximal operator of a Simplex: :math:`\Delta_n(r) = \{ \mathbf{x}:
    \sum_i x_i = r,\; x_i \geq 0 \}`. This operator can be applied to a
    single vector as well as repeatedly to a set of vectors which are
    defined as the rows (or columns) of a matrix obtained by reshaping the
    input vector as defined by the ``dims`` and ``axis`` parameters.

    Parameters
    ----------
    n : :obj:`int`
        Number of elements of input vector
    radius : :obj:`float`
        Radius
    dims : :obj:`tuple`, optional
        Dimensions of the matrix onto which the input vector is reshaped
    axis : :obj:`int`, optional
        Axis along which simplex is repeatedly applied when ``dims`` is not
        provided
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by bisection
    ftol : :obj:`float`, optional
        Function tolerance in bisection (only with ``engine='numba'`` or ``engine='cuda'``)
    xtol : :obj:`float`, optional
        Solution absolute tolerance in bisection
    call : :obj:`bool`, optional
        Evalutate call method (``True``) or not (``False``)
    engine : :obj:`str`, optional
        Engine used for simplex computation (``numpy``, ``numba``or ``cuda``).

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba`` nor ``cuda``
    ValueError
        If ``dims`` is provided as a list (or tuple) with more or less than
        2 elements

    Notes
    -----
    As the Simplex is an indicator function, the proximal operator corresponds
    to its orthogonal projection (see :class:`pyproximal.projection.SimplexProj`
    for details.

    Note that ``tau`` does not have effect for this proximal operator, any
    positive number can be provided.

    """
    if not engine in ['numpy', 'numba', 'cuda']:
        raise KeyError('engine must be numpy or numba or cuda')

    if engine == 'numba' and jit is not None:
        s = _Simplex_numba(n, radius, dims=dims, axis=axis,
                           maxiter=maxiter, ftol=ftol, xtol=xtol, call=call)
    elif engine == 'cuda' and jit is not None:
        s = _Simplex_cuda(n, radius, dims=dims, axis=axis,
                          maxiter=maxiter, ftol=ftol, xtol=xtol, call=call)
    else:
        if engine == 'numba' and jit is None:
            logging.warning(jit_message)
        s = _Simplex(n, radius, dims=dims, axis=axis,
                     maxiter=maxiter, xtol=xtol, call=call)
    return s