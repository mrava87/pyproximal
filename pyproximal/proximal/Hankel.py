import numpy as np
from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.Hankel import HankelProj # Ensure correct import path
from typing import Tuple, Any # Any for Optional Op in super().__init__


class Hankel(ProxOperator):
    r"""Hankel proximal operator.

    Proximal operator of the Hankel matrix indicator function.

    Parameters
    ----------
    dim : :obj:`tuple`
        Dimension of the Hankel matrix.

    Notes
    -----
    As the Hankel Operator is an indicator function, the proximal operator corresponds to
    its orthogonal projection (see :class:`pyproximal.projection.HankelProj` for
    details).

    """
    def __init__(self, dim: Tuple[int, ...]):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.dim: Tuple[int, ...] = dim
        self.hankel_proj: HankelProj = HankelProj()

    def __call__(self, x: np.ndarray) -> bool:
        # Reshape x to the specified dimensions to treat it as a matrix X
        X: np.ndarray = x.reshape(self.dim)
        # Check if X is a Hankel matrix by comparing it with its projection onto the Hankel set
        return bool(np.allclose(X, self.hankel_proj(X)))

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        # Reshape x to the matrix dimensions
        X: np.ndarray = x.reshape(self.dim)
        # Project X onto the set of Hankel matrices and then flatten back to a vector
        return self.hankel_proj(X).ravel()
