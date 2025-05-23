import numpy as np
from scipy.linalg import hankel
from typing import Tuple


class HankelProj:
    r"""Hankel matrix projection.

    Solves the least squares problem

        .. math::
          \min_{X\in\mathcal{H}} \|X-X_0\|_F^2

    where :math:`\mathcal{H}` is the set of Hankel matrices.

    Notes
    -----
    The solution to the above-mentioned least squares problem is given by a Hankel matrix,
    where the (constant) anti-diagonals are the average value along the corresponding
    anti-diagonals of the original matrix :math:`X_0`.
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        m: int
        n: int
        m, n = X.shape
        # hankel function can return ndarray of various types depending on input,
        # but here it's used for indexing, so int is expected.
        ind: np.ndarray = hankel(np.arange(m, dtype=np.int32),
                                 m - 1 + np.arange(n, dtype=np.int32)) # type: ignore
        
        # np.bincount requires 1D input for weights and minlength.
        # ind.ravel() ensures it's 1D.
        # mean_values will be float if X is float.
        mean_values: np.ndarray = np.bincount(ind.ravel(), weights=X.ravel()) / \
                                  np.bincount(ind.ravel())
        
        # The result should have the same dtype as mean_values (promoted by division).
        projected_X: np.ndarray = mean_values[ind]
        return projected_X
