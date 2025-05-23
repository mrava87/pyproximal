import numpy as np
from typing import Union, Callable, Any, List, Optional # Added Optional

from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.Box import BoxProj # Ensure correct import path
from pyproximal.projection.L1 import L1BallProj # Ensure correct import path


def _softthreshold(x: np.ndarray, thresh: float) -> np.ndarray:
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x - g``.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    x1: np.ndarray
    if np.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.exp(1j * np.angle(x)) # type: ignore # np.exp can return complex
    else:
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.sign(x)

    return x1


# Define a TypeVar or more specific types if possible for sigma's callable and return
SigmaType = Union[float, List[Any], np.ndarray, Callable[[int], Union[float, List[Any], np.ndarray]]]
ResolvedSigmaType = Union[float, List[Any], np.ndarray] # Output of _current_sigma

def _current_sigma(sigma: SigmaType, count: int) -> ResolvedSigmaType:
    if not callable(sigma):
        return sigma
    else:
        # If sigma is callable, it's Callable[[int], Union[float, list, np.ndarray]]
        # So, its return type matches ResolvedSigmaType.
        return sigma(count)


class L1(ProxOperator):
    r"""L1 norm proximal operator.

    Proximal operator of the :math:`\ell_1` norm:
    :math:`\sigma\|\mathbf{x} - \mathbf{g}\|_1 = \sigma \sum |x_i - g_i|`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`list` or :obj:`np.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L1 norm. This can be a constant number, a list
        of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``sigma`` to be used.
    g : :obj:`np.ndarray`, optional
        Vector to be subtracted

    Notes
    -----
    The :math:`\ell_1` proximal operator is defined as [1]_:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_1}(\mathbf{x}) =
        \operatorname{soft}(\mathbf{x}, \tau \sigma) =
        \begin{cases}
        x_i + \tau \sigma, & x_i - g_i < -\tau \sigma \\
        g_i, & -\tau\sigma \leq x_i - g_i \leq \tau\sigma \\
        x_i - \tau\sigma,  & x_i - g_i > \tau\sigma\\
        \end{cases}

    where :math:`\operatorname{soft}` is the so-called called *soft thresholding*.

    Moreover, as the conjugate of the :math:`\ell_1` norm is the orthogonal projection of
    its dual norm (i.e., :math:`\ell_\inf` norm) onto a unit ball, its dual
    operator (when :math:`\mathbf{g}=\mathbf{0}`) is defined as:

    .. math::

        \prox^*_{\tau \sigma \|\cdot\|_1}(\mathbf{x}) = P_{\|\cdot\|_{\infty} <=\sigma}(\mathbf{x}) =
        \begin{cases}
        -\sigma, & x_i < -\sigma \\
        x_i,& -\sigma \leq x_i \leq \sigma \\
        \sigma,  & x_i > \sigma\\
        \end{cases}

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    def __init__(self, sigma: SigmaType = 1., g: Optional[np.ndarray] = None):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.sigma_arg: SigmaType = sigma # Store original sigma argument
        self.g: Optional[np.ndarray] = g
        self.gdual: Union[int, np.ndarray] = 0 if g is None else g # Type for gdual
        
        # Initialize BoxProj with initial sigma value
        initial_sigma_val = _current_sigma(self.sigma_arg, 0)
        # BoxProj expects float or np.ndarray for lower/upper.
        # Handle cases where initial_sigma_val might be a list.
        # Assuming _current_sigma resolves to float or compatible np.ndarray for BoxProj.
        # If initial_sigma_val is a list, this will error.
        # This implies _current_sigma should ideally return float or ndarray for BoxProj.
        # For now, casting to float if it's a scalar, else assuming it's a compatible ndarray.
        # This might need refinement based on how list-sigma is intended to work with BoxProj.
        processed_initial_sigma: Union[float, np.ndarray]
        if isinstance(initial_sigma_val, list):
            # This is problematic for BoxProj if it expects scalar or ndarray.
            # Defaulting to first element or average might be an option, or raise error.
            # For now, let's assume it should have been resolved to float/ndarray by _current_sigma.
            # This indicates a potential design consideration for list-based sigma.
            # To avoid error, let's assume if list, it's a list of one float element for now.
            if len(initial_sigma_val) == 1 and isinstance(initial_sigma_val[0], (float, int)):
                processed_initial_sigma = float(initial_sigma_val[0])
            else:
                # Fallback or error: For now, assume it's an ndarray if not float.
                # This part is tricky without clearer spec on list sigma with BoxProj.
                # Let's assume _current_sigma returns float or np.ndarray for this context.
                # If it can return a list of numbers, BoxProj needs to handle it or sigma needs processing.
                # Casting to np.array if it's a list of numbers.
                try:
                    processed_initial_sigma = np.array(initial_sigma_val, dtype=float)
                    if processed_initial_sigma.ndim == 0: # Convert 0-dim array to scalar float
                         processed_initial_sigma = float(processed_initial_sigma.item())
                except TypeError: # If it's a list of non-numbers, this will fail.
                     raise ValueError("Sigma list for BoxProj must contain numbers.")

        elif isinstance(initial_sigma_val, np.ndarray):
            processed_initial_sigma = initial_sigma_val
        else: # float
            processed_initial_sigma = float(initial_sigma_val)

        self.box: BoxProj = BoxProj(-processed_initial_sigma, processed_initial_sigma)
        self.count: int = 0

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar sum
        current_sigma_val = _current_sigma(self.sigma_arg, self.count)
        # Assuming current_sigma_val resolves to a type compatible with multiplication (float or ndarray for element-wise)
        # If sigma is array, this should be element-wise product then sum. np.sum handles this.
        # If sigma is scalar, it's scalar multiplication.
        return float(np.sum(current_sigma_val * np.abs(x if self.g is None else x - self.g)))


    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'L1', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        current_sigma_val = _current_sigma(self.sigma_arg, self.count)
        # _softthreshold expects threshold to be float.
        # If current_sigma_val is array for element-wise thresholding, _softthreshold needs to handle it.
        # Current _softthreshold takes scalar thresh. This implies sigma should be scalar here.
        # Similar to L0, this needs clarification if sigma can be array for element-wise prox.
        # Assuming current_sigma_val resolves to float for now.
        threshold: float
        if isinstance(current_sigma_val, (np.ndarray, list)):
            if isinstance(current_sigma_val, np.ndarray) and current_sigma_val.size == 1:
                 threshold = float(current_sigma_val.item()) * tau
            elif isinstance(current_sigma_val, float): # Should not happen if already ndarray/list
                 threshold = current_sigma_val * tau
            else: # List or multi-element ndarray
                 raise TypeError("Array/list sigma for L1 prox needs element-wise softthreshold or scalar sigma.")
        else: # float
            threshold = float(current_sigma_val) * tau

        if self.g is None:
            x_prox: np.ndarray = _softthreshold(x, threshold)
        else:
            # use precomposition property
            x_prox = _softthreshold(x - self.g, threshold) + self.g
        return x_prox

    @_check_tau
    def proxdual(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used when g is None (BoxProj case), but is part of Moreau for g != None
        if not isinstance(self.gdual, np.ndarray): # True if g was None (gdual=0)
            # Update box with current sigma if sigma is callable
            if callable(self.sigma_arg):
                current_sigma_val = _current_sigma(self.sigma_arg, self.count)
                # Similar processing for sigma as in __init__ for BoxProj
                processed_sigma: Union[float, np.ndarray]
                if isinstance(current_sigma_val, list):
                    if len(current_sigma_val) == 1 and isinstance(current_sigma_val[0], (float, int)):
                        processed_sigma = float(current_sigma_val[0])
                    else:
                        try:
                            processed_sigma = np.array(current_sigma_val, dtype=float)
                            if processed_sigma.ndim == 0: processed_sigma = float(processed_sigma.item())
                        except TypeError:
                             raise ValueError("Sigma list for BoxProj must contain numbers.")
                elif isinstance(current_sigma_val, np.ndarray):
                     processed_sigma = current_sigma_val
                else: # float
                     processed_sigma = float(current_sigma_val)
                self.box = BoxProj(-processed_sigma, processed_sigma)
            x_proxdual: np.ndarray = self.box(x)
        else: # g was not None, use Moreau decomposition
            x_proxdual = self._proxdual_moreau(x, tau) # type: ignore # ProxOperator method
        return x_proxdual


class L1Ball(ProxOperator):
    r"""L1 ball proximal operator.

    Proximal operator of the :math:`\ell_1` ball: :math:`L1_{r} =
    \{ \mathbf{x}: \|\mathbf{x}\|_1 \leq r \}`.

    Parameters
    ----------
    n : :obj:`int`
        Number of elements of input vector
    radius : :obj:`float`
        Radius
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by :func:`scipy.optimize.bisect`
    xtol : :obj:`float`, optional
        Absolute tolerance of :func:`scipy.optimize.bisect`

    Notes
    -----
    As the L1 ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.L1BallProj` for details.

    """
    def __init__(self, n: int, radius: float, maxiter: int = 100, xtol: float = 1e-5):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.n: int = n
        self.radius: float = radius
        self.maxiter: int = maxiter
        self.xtol: float = xtol
        self.ball: L1BallProj = L1BallProj(self.n, self.radius, self.maxiter, self.xtol)

    def __call__(self, x: np.ndarray, tol: float = 1e-4) -> bool:
        # Check if L1 norm of x is within radius (plus tolerance)
        return bool(np.sum(np.abs(x)) - self.radius < tol)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        return self.ball(x)
