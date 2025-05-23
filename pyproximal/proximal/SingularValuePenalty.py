import numpy as np
from typing import Tuple, Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator


class SingularValuePenalty(ProxOperator):
    r"""Proximal operator of a penalty acting on the singular values.

    Generic regularizer :math:`\mathcal{R}_f` acting on the singular values of a matrix,

    .. math::

        \mathcal{R}_f(\mathbf{X}) = f(\boldsymbol\lambda)

    where :math:`\mathbf{X}` is a matrix of size :math:`M \times N` and
    :math:`\boldsymbol\lambda` is the corresponding singular value vector.

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of matrix :math:`\mathbf{X}`.
    penalty : :obj:`pyproximal.ProxOperator`
        Function acting on the singular values.

    Notes
    -----
    The pyproximal implementation allows ``penalty`` to be any
    :class:`pyproximal.ProxOperator` acting on the singular values; however, not all
    penalties will result in a mathematically accurate proximal operator defined this
    way. Given a penalty :math:`f`, the proximal operator is assumed to be

    .. math::

        \prox_{\tau \mathcal{R}_f}(\mathbf{X}) =
        \mathbf{U} \diag\left( \prox_{\tau f}(\boldsymbol\lambda)\right) \mathbf{V}^H

    where :math:`\mathbf{X} = \mathbf{U}\diag(\boldsymbol\lambda)\mathbf{V}^H`, is an
    SVD of :math:`\mathbf{X}`. It is the user's responsibility to check that this is
    true for their particular choice of ``penalty``.
    """

    def __init__(self, dim: Tuple[int, ...], penalty: ProxOperator):
        super().__init__(None, False) # Op is None, hasgrad depends on penalty, but prox is implemented
        self.dim: Tuple[int, ...] = dim
        self.penalty: ProxOperator = penalty

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar value
        X: np.ndarray = x.reshape(self.dim)
        # Singular values are sqrt of eigenvalues of X.H @ X or X @ X.H
        # Using X.T @ X assumes X is real or we are interested in X.T rather than X.H
        # For general case (complex matrices), X.conj().T @ X or X.H @ X is preferred.
        # np.linalg.eigvalsh assumes Hermitian matrix, so X.conj().T @ X is appropriate.
        # If X is real, X.T @ X is fine.
        # Assuming X can be complex, use X.conj().T
        XTX: np.ndarray = X.conj().T @ X
        eigs: np.ndarray = np.linalg.eigvalsh(XTX)
        eigs[eigs < 0] = 0  # Ensure all eigenvalues are non-negative
        singular_values: np.ndarray = np.sqrt(eigs)
        
        # self.penalty.__call__ should return a scalar if penalty is scalar-valued for vector input
        # Or, if penalty.elementwise exists and returns array, sum would be applied.
        # Assuming self.penalty.__call__ on singular_values (a vector) returns a scalar sum.
        penalty_val: Any = self.penalty(singular_values)
        return float(penalty_val) # Ensure it's float

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        X: np.ndarray = x.reshape(self.dim)
        U: np.ndarray
        S_vec: np.ndarray # Singular values vector
        Vh: np.ndarray # Note: Vh is V.conj().T
        U, S_vec, Vh = np.linalg.svd(X, full_matrices=False)
        
        # Apply proximal operator of the penalty to the singular values
        S_prox: np.ndarray = self.penalty.prox(S_vec, tau)
        
        # Reconstruct the matrix: U @ diag(S_prox) @ Vh
        # np.dot(U * S_prox, Vh) is equivalent to U @ np.diag(S_prox) @ Vh
        # U is (M, K), S_prox is (K,), Vh is (K, N), where K = min(M,N)
        # U * S_prox creates (M, K) by broadcasting S_prox along rows of U.
        X_prox: np.ndarray = np.dot(U * S_prox, Vh)
        return X_prox.ravel()
