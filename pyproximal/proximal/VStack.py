from typing import TYPE_CHECKING, List, Optional

import numpy as np

from pylops.utils.typing import NDArray
from pyproximal.ProxOperator import _check_tau, ProxOperator

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class VStack(ProxOperator):
    r"""Vertical stacking.

    Stack a set of N proximal operators vertically. This operator can be used
    for separable inputs, where the overall proximal operator can be computed
    as the stack of proximal operators on parts of the input vector.

    Parameters
    ----------
    ops : :obj:`list`
        Proximal operators to be stacked
    nn : :obj:`list`, optional
        Size of each portion of the input vector (to be used when different
        portions are in consecutive order)
    restr : :obj:`list`, optional
        List of :class:`pylops.Restriction` operators extracting the subset of
        interest (to be used when different portions are not in consecutive
        order). It is user responsibility to ensure that all elements of the
        input vector are used exactly once)

    Notes
    -----
    Given an input vector :math:`\mathbf{x}` to which a number of :math:`N`
    functions are applied to different portions of the vector as:

    .. math::
        f(\mathbf{x}) = \sum_{i=1}^N f_i(\mathbf{x}_i)

    the related proximal operator becomes:

    .. math::
        \prox_{\tau f}(\mathbf{x}) = \left(
        \prox_{\tau f_1}(\mathbf{x}_1), \ldots,
        \tau f_N(\mathbf{x}_N) \right)

    """
    def __init__(
            self, 
            ops: List[ProxOperator],
            nn: Optional[List[int]] = None,
            restr: Optional[List["LinearOperator"]] = None,
        ) -> None:
        super().__init__(None, False) # Op is None, hasgrad depends on stacked ops

        self.nx: int
        self.ops: List[ProxOperator] = ops
        self.nn: Optional[List[int]] = None
        self.xin: Optional[NDArray] = None
        self.xend: Optional[NDArray] = None
        self.restr: Optional[List[LinearOperator]] = None

        if nn is not None:
            self.nn = nn
            cum_nn: NDArray = np.cumsum(nn)
            self.xin = np.insert(cum_nn[:-1], 0, 0)
            self.xend = cum_nn
            # store required size of input
            self.nx = int(cum_nn[-1])
        elif restr is not None:
            self.restr = restr
            self.nx = int(np.sum([r.shape[0] for r in self.restr])) 
        else:
            raise ValueError('provide either nn or restr')
        
    def __call__(self, x: NDArray) -> float:
        if x.size != self.nx:
            raise ValueError(f'x must have size {self.nx}, instead the provided x has size {x.size}')
        
        f: float = 0.
        if self.nn is not None and self.xin is not None and self.xend is not None:
            for iop, op in enumerate(self.ops):
                f += op(x[self.xin[iop]:self.xend[iop]])
        elif self.restr is not None:
            for op, r_op in zip(self.ops, self.restr):
                f += op(r_op.matvec(x))
        return f

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        if x.size != self.nx:
            raise ValueError(f'x must have size {self.nx}, instead the provided x has size {x.size}')
        
        prox: NDArray
        if self.nn is not None and self.xin is not None and self.xend is not None:
            prox = np.hstack([op.prox(x[self.xin[iop]:self.xend[iop]], tau)
                              for iop, op in enumerate(self.ops)])
        elif self.restr is not None:
            prox = np.zeros_like(x, dtype=x.dtype)
            for op, restr in zip(self.ops, self.restr):
                prox_part: NDArray = op.prox(restr.matvec(x), tau)
                prox[restr.iava] = prox_part
        return prox

    def grad(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.nx:
            raise ValueError(f'x must have size {self.nx}, instead the provided x has size {x.size}')

        grad: NDArray
        if self.nn is not None and self.xin is not None and self.xend is not None:
            grad = np.hstack([op.grad(x[self.xin[iop]:self.xend[iop]])
                              for iop, op in enumerate(self.ops)])
        elif self.restr is not None:
            grad = np.zeros_like(x, dtype=x.dtype)
            for op, restr in zip(self.ops, self.restr):
                grad_part: NDArray = op.grad(restr.matvec(x))
                grad[restr.iava] = grad_part
        return grad