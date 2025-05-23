import numpy as np
from pyproximal.projection.Box import HyperPlaneBoxProj # Ensure correct import path
from typing import Any # HyperPlaneBoxProj might be considered Any if not typed


class SimplexProj:
    r"""Simplex projection.

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
    Given a Simplex set defined as:

    .. math::

        \Delta_n(r) = \{ \mathbf{x}: \sum_i x_i = r,\; x_i \geq 0 \}

    its orthogonal projection is simply given by projection of the intersection
    between a hyperplane and a box with chosen radius and bounds equal to 0
    and inf (see :class:`pyproximal.projection.HyperPlaneBoxProj` for more
    details).

    """
    def __init__(self, n: int, radius: float, maxiter: int = 100, xtol: float = 1e-5):
        self.simplex: HyperPlaneBoxProj = HyperPlaneBoxProj(np.ones(n), radius,
                                                            lower=0., upper=np.inf, # Ensure float for lower/upper
                                                            maxiter=maxiter, xtol=xtol)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply Simplex projection

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        maxiter : :obj:`int`, optional
            Maximum number of iterations used by :func:`scipy.optimize.bisect`
        xtol : :obj:`float`, optional
            Absolute tolerance of :func:`scipy.optimize.bisect`

        """
        return self.simplex(x)