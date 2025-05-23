import numpy as np
from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.AffineSet import AffineSetProj # Ensure correct import path
from pylops.LinearOperator import LinearOperator # For Op type
from typing import Optional, Any


class AffineSet(ProxOperator):
    r"""Affine set proximal operator.

    Proximal operator of an Affine set: :math:`\{ \mathbf{x} : \mathbf{Opx}=\mathbf{b} \}`.

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
    As the Affine set is an indicator function, the proximal operator corresponds to
    its orthogonal projection (see :class:`pyproximal.projection.AffineSetProj` for
    details.

    """
    def __init__(self, Op: LinearOperator, b: np.ndarray, niter: int):
        super().__init__(Op, False) # hasgrad is False for indicator functions
        self.b: np.ndarray = b
        self.niter: int = niter
        # self.Op is inherited from ProxOperator and should be LinearOperator
        self.affine: AffineSetProj = AffineSetProj(self.Op, self.b, self.niter)

    def __call__(self, x: np.ndarray) -> bool:
        # The __call__ for an indicator function usually returns 0 if x is in the set,
        # and np.inf if x is not. However, this implementation returns True/False.
        # For consistency with typical indicator function behavior in optimization,
        # this might need to return float (0.0 or np.inf).
        # Sticking to original True/False for now as per current code.
        
        # Op is guaranteed to be non-None by __init__ type hint for this subclass.
        # Add an assertion to help mypy and as a runtime safeguard.
        if self.Op is None:
            # This case should ideally not be reached if __init__ enforces Op is not None.
            # However, self.Op is typed as Optional in the base class.
            raise ValueError("AffineSet operator Op cannot be None for __call__.")
        
        # Op.matvec(x) will use the appropriate backend (numpy/cupy)
        return bool(np.allclose(self.Op.matvec(x), self.b)) # type: ignore[union-attr]

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function, but is part of signature
        # self.affine was initialized with self.Op, which __init__ guarantees is LinearOperator.
        # No explicit check for self.Op needed here as self.affine would fail in __init__ if Op was None.
        return self.affine(x)
