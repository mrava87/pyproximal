import numpy as np
from pyproximal.projection.Simplex import SimplexProj # Ensure correct import path
from typing import Union


class L1BallProj:
    r""":math:`L_1` ball projection.

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
    Given an L1 ball defined as:

    .. math::

        L1_{r} = \{ \mathbf{x}: ||\mathbf{x}||_1 \leq r \}

    its orthogonal projection is:

    .. math::

        P_{L1_{r}} (\mathbf{x}) = sign(\mathbf{x}) P_{\operatorname{Simplex}(r)}(\mathbf{x})

    Note that this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{L1_{r}}`.

    """
    def __init__(self, n: int, radius: float, maxiter: int = 100, xtol: float = 1e-5):
        self.n: int = n
        self.radius: float = radius
        self.simplex: SimplexProj = SimplexProj(n, radius, maxiter, xtol)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(x):
            # For complex numbers, projection involves phase of x and simplex projection of |x|
            return np.exp(1j * np.angle(x)) * self.simplex(np.abs(x))
        else:
            # For real numbers, projection involves sign of x and simplex projection of |x|
            return np.sign(x) * self.simplex(np.abs(x))

