import numpy as np
from typing import Optional, Any, Union # Added Union

# Remove unused scipy.sparse.linalg.lsqr and pylops.MatrixMult, pylops.Identity
from pylops.LinearOperator import LinearOperator # For Q type
from pyproximal.ProxOperator import _check_tau, ProxOperator


class Orthogonal(ProxOperator):
    r"""Proximal operator of any function of the product between an orthogonal
    matrix and a vector (plus summation of a vector).

    Proximal operator of any quadratic function
    :math:`g(\mathbf{x})=f(\mathbf{Qx} + \mathbf{b})` where :math:`\mathbf{Q}`
    is an orthogonal operator and :math:`\mathbf{b}` is a vector in the data
    space of :math:`\mathbf{Q}`.

    Parameters
    ----------
    f : :obj:`pyproximal.ProxOperator`
        Proximal operator
    Q : :obj:`pylops.LinearOperator`
        Orthogonal operator
    partial : :obj:`bool`, optional
        Partial (``True``) of full (``False``) orthogonality
    b : :obj:`np.ndarray`, optional
        Vector
    alpha : :obj:`float`, optional
        Positive coefficient for partial orthogonality. It will be ignored if
        ``partial=False``

    Notes
    -----
    The Proximal operator of any function of the form
    :math:`g(\mathbf{x}) = f(\mathbf{Qx} + \mathbf{b})` with the operator
    :math:`\mathbf{Q}` satisfying the following condition
    :math:`\mathbf{Q}\mathbf{Q}^T = \alpha \mathbf{I}` (partial orthogonality)
    is [1]_:

    .. math::

        \prox_{\tau g}(x) = \frac{1}{\alpha} ((\alpha \mathbf{I} -
        \mathbf{Q}^H \mathbf{Q}) \mathbf{x} + \mathbf{Q}^H
        (\prox_{\alpha \tau f}(\mathbf{Qx} + \mathbf{b}) - \mathbf{b}))

    A special case arises when :math:`\mathbf{Q}\mathbf{Q}^T =
    \mathbf{Q}^T\mathbf{Q} = \mathbf{I}`
    (full orthogonality), and the proximal operator reduces to:

    .. math::

        \prox_{\tau g}(x) = \mathbf{Q}^H (\prox_{\tau f}(\mathbf{Qx} +
        \mathbf{b}) - \mathbf{b}))

    .. [1] Daniel O'Connor, D., and Vandenberghe, L., "Primal-Dual
        Decomposition by Operator Splitting and Applications to Image
        Deblurring", SIAM J. Imaging Sciences, vol. 7, pp. 1724–1754. 2014.

    """
    def __init__(self, f: ProxOperator, Q: LinearOperator,
                 partial: bool = False, b: Optional[np.ndarray] = None,
                 alpha: float = 1.):
        super().__init__(None, False) # Op is None, hasgrad depends on f, but prox is implemented
        self.f: ProxOperator = f
        self.Q: LinearOperator = Q
        self.partial: bool = partial
        self.alpha: float = alpha
        self.b_offset: Union[np.ndarray, float] = b if b is not None else 0. # Renamed to avoid conflict

    def __call__(self, x: np.ndarray) -> Any: # Return type depends on self.f.__call__
        y: np.ndarray = self.Q.matvec(x)
        y_offset: np.ndarray = y + self.b_offset # Apply offset
        f_val: Any = self.f(y_offset) # Call the wrapped proximal operator's __call__
        return f_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        y: np.ndarray = self.Q.matvec(x)
        z: np.ndarray
        if self.partial:
            # Ensure Q.rmatvec is callable and returns ndarray
            # Ensure f.prox is callable and returns ndarray
            # Ensure b_offset is compatible for subtraction if it's an array
            prox_f_input: np.ndarray = y + self.b_offset
            prox_f_output: np.ndarray = self.f.prox(prox_f_input, self.alpha * tau)
            term_in_rmatvec: np.ndarray = prox_f_output - self.b_offset
            
            z = (1. / self.alpha) * \
                (self.alpha * x - self.Q.rmatvec(y) + # type: ignore
                 self.Q.rmatvec(term_in_rmatvec)) # type: ignore
        else: # Full orthogonality
            y_offset_full: np.ndarray = y + self.b_offset
            prox_f_output_full: np.ndarray = self.f.prox(y_offset_full, tau)
            term_for_rmatvec_full: np.ndarray = prox_f_output_full - self.b_offset
            z = self.Q.rmatvec(term_for_rmatvec_full) # type: ignore
        return z
