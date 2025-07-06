from typing import TYPE_CHECKING, Tuple

import numpy as np

from scipy.sparse.linalg import cg as sp_cg
from pylops.optimization.basic import cg
from pylops.utils.backend import get_array_module, get_module_name
from pylops.utils.typing import NDArray

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
    def __init__(self, Op: "LinearOperator", b: NDArray, niter: int):
        self.Op: "LinearOperator" = Op
        self.b: NDArray = b
        self.niter: int = niter

    def __call__(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        inv: NDArray
        if get_module_name(ncp) == 'numpy':
            xinv = sp_cg(self.Op * self.Op.H, self.Op @ x - self.b, maxiter=self.niter)[0]
        else:
            xinv = cg(self.Op * self.Op.H, self.Op @ x - self.b, niter=self.niter)[0]
        y: NDArray = x - self.Op.rmatvec(xinv.ravel()) # Using rmatvec, ravel for safety
        return y