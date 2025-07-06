import numpy as np
from typing import TYPE_CHECKING, Optional, Any, Union, Tuple

from scipy.sparse.linalg import lsqr as sp_lsqr # Keep scipy lsqr for numpy case
from pylops import MatrixMult, Identity # type: ignore # Assuming these are LinearOperator or similar
from pylops.optimization.basic import lsqr as pylops_lsqr # pylops lsqr for cupy
from pyproximal.ProxOperator import _check_tau, ProxOperator

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class Quadratic(ProxOperator):
    r"""Quadratic function proximal operator.

    Proximal operator for a quadratic function: :math:`f(\mathbf{x}) =
    \frac{1}{2} \mathbf{x}^T \mathbf{Op} \mathbf{x} + \mathbf{b}^T
    \mathbf{x} + c`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator (must be square)
    b : :obj:`numpy.ndarray`, optional
        Vector
    c : :obj:`float`, optional
        Scalar
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme used to compute the proximal
    x0 : :obj:`numpy.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.

    Raises
    ------
    ValueError
        If ``Op`` is not square

    Notes
    -----
    The Quadratic proximal operator is defined as:

    .. math::

        \prox_{\tau f}(\mathbf{x}) =
        \left(\mathbf{I} + \tau  \mathbf{Op} \right)^{-1} \left(\mathbf{x} -
        \tau \mathbf{b}\right)

    when both ``Op`` and ``b`` are provided. This formula shows that the
    proximal operator requires the solution of an inverse problem. If the
    operator ``Op`` is of kind ``explicit=True``, we can solve this problem
    directly. On the other hand if ``Op`` is of kind ``explicit=False``, an
    iterative solver is employed. In this case it is possible to provide a warm
    start via the ``x0`` input parameter.

    When only ``b`` is provided, the proximal operator reduces to:

    .. math::

        \prox_{\mathbf{b}^T \mathbf{x} + c}(\mathbf{x}) =
        \mathbf{x} - \tau \mathbf{b}

    Finally if also ``b`` is not provided, the proximal operator of a constant
    function simply becomes :math:`\prox_c(\mathbf{x}) = \mathbf{x}`


    """
    def __init__(self, Op: Optional["LinearOperator"] = None,
                 b: Optional[np.ndarray] = None, c: float = 0.,
                 niter: int = 10, x0: Optional[np.ndarray] = None,
                 warm: bool = True):
        if Op is not None:
            if Op.shape[0] != Op.shape[1]:
                raise ValueError('Op must be square')
        super().__init__(Op, True) # hasgrad is True
        self.b: Optional[np.ndarray] = b
        if self.Op is not None and self.b is None: # Op is LinearOperator here
            self.b = np.zeros(self.Op.shape[1], dtype=self.Op.dtype)
        self.c: float = c
        self.niter: int = niter
        self.x0: Optional[np.ndarray] = x0
        self.warm: bool = warm

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar
        f_val: float
        if self.Op is not None and self.b is not None: # Op is LinearOperator
            # Assuming Op is self-adjoint for 0.5 * x.T @ Op @ x
            # If Op is not self-adjoint, this might not be the intended quadratic form.
            # PyLops operators are not necessarily matrices, Op * x is Op.matvec(x)
            f_val = np.dot(x, self.Op.matvec(x)) / 2. + np.dot(self.b, x) + self.c # type: ignore
        elif self.b is not None:
            f_val = np.dot(self.b, x) + self.c
        else:
            f_val = self.c
        return f_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        current_x: np.ndarray = x.copy() # Work on a copy
        if self.Op is not None and self.b is not None: # Op is LinearOperator
            y: np.ndarray = current_x - tau * self.b
            if self.Op.explicit:
                # Assuming self.Op.A exists and is ndarray if explicit
                Op1_explicit: LinearOperator = MatrixMult(np.eye(self.Op.shape[0]) + tau * self.Op.A) # type: ignore
                current_x = Op1_explicit.div(y) # type: ignore
            else: # Implicit operator
                # self.Op.A is not available for implicit. It should be self.Op itself.
                Op1_implicit: LinearOperator = Identity(self.Op.shape[0], dtype=self.Op.dtype) + \
                                   tau * self.Op # type: ignore
                # Use sp_lsqr for numpy arrays, pylops_lsqr for others (e.g., cupy)
                # lsqr in scipy.sparse.linalg returns a tuple
                # pylops.optimization.basic.lsqr also returns a tuple
                lsqr_result: Tuple = sp_lsqr(Op1_implicit, y, iter_lim=self.niter, x0=self.x0) # type: ignore
                current_x = lsqr_result[0]

            if self.warm:
                self.x0 = current_x.copy()
        elif self.b is not None: # Op is None, b is not None
            current_x = current_x - tau * self.b
        # If Op is None and b is None, prox is identity, so current_x (original x) is returned.
        return current_x

    def grad(self, x: np.ndarray) -> Union[np.ndarray, float]:
        """Compute gradient

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Vector

        Returns
        -------
        g : :obj:`numpy.ndarray`
            Gradient vector

        """
        g_val: Union[np.ndarray, float]
        if self.Op is not None and self.b is not None: # Op is LinearOperator
            # For f(x) = 0.5 * x.T A x + b.T x, grad = Ax + b
            # If f(x) = 0.5 * x.T Op x as per code, assuming Op is symmetric, grad = Op x
            # Original code: self.Op.matvec(x) / 2. + x - this seems unusual.
            # If f(x) = 0.5 * x^T Op x + b^T x, then grad(f) = 0.5*(Op + Op^T)x + b
            # If Op is symmetric, grad(f) = Op x + b.
            # The original code's __call__ implies f(x) = 0.5 * x Op x + b x + c.
            # Its gradient would be 0.5 * (Op + Op.H)x + b. If Op is self-adjoint, Op x + b.
            # The current grad implementation: self.Op.matvec(x) / 2. + x
            # This does not match the __call__ if b is the linear term's vector.
            # Let's assume the definition in docstring: 0.5 x^T Op x + b^T x + c
            # and Op is self-adjoint. Then grad is Op x + b.
            # If b is the vector for linear term, then grad should be Op@x + b
            # The current code's grad: Op@x/2 + x.
            # This is confusing. Let's assume the __call__ is the intended function.
            # If f(x) = sigma/2 ||Op x - b||^2 as in L2 class, then grad = sigma * Op.H @ (Op @ x - b)
            # If f(x) = 1/2 x^T A x + b^T x + c (docstring), grad = Ax + b (if A symmetric)
            # If f(x) = 1/2 dot(x, Op*x) + dot(b,x) + c (from __call__):
            # grad = 0.5*(Op + Op.H)x + b. If Op symmetric, Op@x + b
            # The current grad code is: g = self.Op.matvec(x) / 2. + x. This implies b is part of x or Op.
            # And a different quadratic form 0.5 x^T (Op/2 + I) x
            # This is inconsistent.
            # Given the prox formula, the function is likely f(x) = 0.5 * x^T Op x + b^T x.
            # Then grad is Op x + b (if Op symmetric).
            # The prox formula is (I + tau Op)^-1 (x_in - tau b).
            # This is for f(y) = 0.5 y^T Op y + b^T y.
            # prox_tau_f(x) = argmin_y { 0.5 y^T Op y + b^T y + 1/(2tau) ||y-x||^2 }
            # Derivative w.r.t y: Op y + b + 1/tau (y-x) = 0
            # (Op + 1/tau I)y = x/tau - b
            # (tau Op + I)y = x - tau b
            # y = (I + tau Op)^-1 (x - tau b). This matches.
            # So the gradient for f(y) = 0.5 y^T Op y + b^T y (Op symmetric) is Op y + b.
            g_val = self.Op.matvec(x) + self.b # type: ignore
        elif self.b is not None: # Op is None
            # f(x) = b^T x + c, grad = b
            g_val = self.b
        else: # Op is None, b is None
            # f(x) = c, grad = 0
            g_val = 0.
        return g_val