from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from scipy.sparse.linalg import cg as sp_cg
from pylops.optimization.basic import cg
from pylops.utils.backend import get_array_module, get_module_name

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class AffineSetProj:
    r"""Affine set projection.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear operator
    b : :obj:`numpy.ndarray`
        Data vector
    niter : :obj:`int`
        Number of iterations of iterative scheme used to compute the projection.

    Notes
    -----
    Given an Affine set defined as:

    .. math::

        \{ \mathbf{x} : \mathbf{Opx}=\mathbf{b} \}

    its orthogonal projection is:

    .. math::

       P_{\{\mathbf{y}:\mathbf{Opy}=\mathbf{b}\}} (\mathbf{x}) = \mathbf{x} -
       \mathbf{Op}^H(\mathbf{Op}\mathbf{Op}^H)^{-1}(\mathbf{Opx}-\mathbf{b})

    Note the this is the proximal operator of the corresponding
    indicator function :math:`I_{\{\mathbf{Opx}=\mathbf{b}\}}`

    """
    def __init__(self, Op: "LinearOperator", b: np.ndarray, niter: int):
        self.Op: "LinearOperator" = Op
        self.b: np.ndarray = b
        self.niter: int = niter

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Determine if running with NumPy or CuPy and choose appropriate cg
        xp = get_array_module(x)
        if get_module_name(xp) == 'numpy':
            # sp_cg returns a tuple (x, info)
            inv_result: Tuple[np.ndarray, int] = sp_cg(self.Op * self.Op.H, self.Op @ x - self.b, maxiter=self.niter)
            inv: np.ndarray = inv_result[0]
        else:
            # pylops.optimization.basic.cg also returns a tuple (x, info)
            inv_result_pylops: np.ndarray = cg(self.Op * self.Op.H, self.Op @ x - self.b, niter=self.niter)[0] # type: ignore
            inv = inv_result_pylops # Assuming it's similar structure or direct result
        # Ensure inv is 1D for proper broadcasting with Op.H if Op.H is a LinearOperator
        # However, Op.H * vector should handle dimensions correctly if inv is shaped as expected by Op.H
        y: np.ndarray = x - self.Op.H @ inv.ravel() # Using @ for matvec, ravel for safety
        return y