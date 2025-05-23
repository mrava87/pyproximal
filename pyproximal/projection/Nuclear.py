import numpy as np
from pyproximal.projection.L1 import L1BallProj # Ensure correct import path
from typing import Tuple


class NuclearBallProj:
    r"""Nuclear ball projection

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
    Given a Nuclear ball defined as:

    .. math::

        N_{r} = \{ \mathbf{X}: ||\mathbf{X}||_* \leq r \}

    its orthogonal projection is:

    .. math::

        P_{N_{r}} (\mathbf{X}) = \mathbf{U} diag(P_{L1_{r}}
                                 (\sigma(\mathbf{X}))) \mathbf{V}^H

    where :math:`\mathbf{U} diag(\sigma(\mathbf{X})) \mathbf{V}^H`
    is the SVD decomposition of :math:`\mathbf{X}`.

    """
    def __init__(self, n: int, radius: float, maxiter: int = 100, xtol: float = 1e-5):
        self.n: int = n
        self.radius: float = radius
        self.l1ball: L1BallProj = L1BallProj(n, radius, maxiter, xtol)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        U: np.ndarray
        S_vec: np.ndarray # Singular values vector
        Vh: np.ndarray
        U, S_vec, Vh = np.linalg.svd(X, full_matrices=False)
        
        # Project the singular values onto the L1 ball
        S_proj_vec: np.ndarray = self.l1ball(S_vec)
        
        # Reconstruct the matrix
        # np.diag creates a 2D array from a 1D array of singular values.
        # Ensure correct matrix multiplication order and dimensions.
        # U is (M, K), diag(S_proj_vec) is (K, K), Vh is (K, N)
        # where K = min(M, N)
        X_projected: np.ndarray = U @ np.diag(S_proj_vec) @ Vh
        return X_projected