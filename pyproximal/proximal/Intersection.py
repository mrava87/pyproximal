import numpy as np
from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.Intersection import IntersectionProj # Ensure correct import path
from typing import Union, Any # Any for Optional Op in super().__init__


class Intersection(ProxOperator):
    r"""Intersection of multiple convex sets operator.

    Parameters
    ----------
    k : :obj:`int`
        Size of vector to be projected
    n : :obj:`int`
        Number of vectors to be projected simultaneously
    sigma : :obj:`np.ndarray` or :obj:`int`
        Matrix of distances of size :math:`k \times k` (or single value in the
        case of constant matrix)
    k : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Toleance of update
    call : :obj:`bool`, optional
        Evalutate call method (``True``) or not (``False``)

    Notes
    -----
    As the Intersection is an indicator function, the proximal operator
    corresponds to its orthogonal projection (see
    :class:`pyproximal.projection.IntersectionProj` for details.

    """
    def __init__(self, k: int, n: int, sigma: Union[float, np.ndarray],
                 niter: int = 100, tol: float = 1e-5, call: bool = True):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.k: int = k
        self.n: int = n
        self.sigma: np.ndarray
        if isinstance(sigma, np.ndarray):
            self.sigma = sigma
        else: # sigma is float or int
            self.sigma = float(sigma) * np.ones((k, k))
        self.call_enabled: bool = call # Renamed to avoid conflict with __call__
        self.ic: IntersectionProj = IntersectionProj(k, n, self.sigma, niter=niter, tol=tol)

    def __call__(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        if not self.call_enabled:
            # For indicator functions, typically 0 (in set) or np.inf (not in set) is returned.
            # Or, if strictly boolean, False could mean "not in set" or "evaluation disabled".
            # Assuming here it means "evaluation disabled" leading to "not in set" interpretation.
            return False # Or raise an error, or return np.inf
        
        x_reshaped: np.ndarray = x.reshape(self.k, self.n)
        for i in range(self.n): # Iterate over columns (vectors)
            for i1 in range(self.k - 1): # Iterate over elements in a vector
                for i2 in range(i1 + 1, self.k):
                    if np.abs(x_reshaped[i1, i] - x_reshaped[i2, i]) > self.sigma[i1, i2] + tol:
                        return False # Condition not met, x is not in the set
        return True # All conditions met for all vectors, x is in the set

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        return self.ic(x)