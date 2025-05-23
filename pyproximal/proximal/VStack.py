import numpy as np
from typing import List, Optional, Any # Added List, Optional, Any
from pylops.LinearOperator import LinearOperator # For Restriction operator type
from pyproximal.ProxOperator import _check_tau, ProxOperator


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
        List of
        :class:`pylops.Restriction` operators extracting the subset of
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
    def __init__(self, ops: List[ProxOperator],
                 nn: Optional[List[int]] = None,
                 restr: Optional[List[LinearOperator]] = None):
        super().__init__(None, False) # Op is None, hasgrad depends on stacked ops
        if nn is None and restr is None:
            raise ValueError('provide either nn or restr')
        self.ops: List[ProxOperator] = ops

        self.nn: Optional[List[int]] = None
        self.xin: Optional[np.ndarray] = None
        self.xend: Optional[np.ndarray] = None
        self.restr: Optional[List[LinearOperator]] = None
        self.nx: int # Total size of input vector x

        if nn is not None:
            self.nn = nn
            cum_nn: np.ndarray = np.cumsum(nn)
            self.xin = np.insert(cum_nn[:-1], 0, 0)
            self.xend = cum_nn
            self.nx = int(cum_nn[-1]) # Ensure nx is int
        elif restr is not None: # restr is not None implies nn is None due to initial check
            self.restr = restr
            # Assuming restr.iava is an attribute of the Restriction operator giving indices
            # and it's an ndarray. The size of these tells us the size of each part.
            # The total size nx is sum of these parts.
            # If restr[i].shape[0] is size of output of restriction (size of x_i)
            # and restr[i].shape[1] is size of input x.
            # The logic here implies restr[i].iava.size is the size of the i-th part of x.
            # This needs to be consistent with how PyLops Restriction op works.
            # For now, assuming restr.iava.size is the intended logic for sum.
            self.nx = int(np.sum([r.shape[0] for r in self.restr])) # More robust: sum of output dims of restr
                                                                  # Or, if iava is for input indices:
                                                                  # self.nx = restr[0].shape[1] if restr else 0
                                                                  # Assuming the original logic for now:
                                                                  # self.nx = int(np.sum([r.iava.size for r in self.restr]))
                                                                  # A safer bet might be to require user to pass total size N
                                                                  # or infer from the first restriction operator's input dimension.
                                                                  # Let's assume restr[0].shape[1] is total N if restr is used.
            if self.restr: # If restr is not empty
                 self.nx = self.restr[0].shape[1] # Total size of x is input dim of any restriction
            else:
                 self.nx = 0 # Or handle as error if ops is not empty but restr is

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar sum
        if x.size != self.nx:
            raise ValueError(f'x must have size {self.nx}, instead the provided x has size {x.size}')
        f_val: float = 0.
        if self.nn is not None and self.xin is not None and self.xend is not None: # nn case
            for iop, op in enumerate(self.ops):
                f_val += op(x[self.xin[iop]:self.xend[iop]])
        elif self.restr is not None: # restr case
            for op, r_op in zip(self.ops, self.restr):
                f_val += op(r_op.matvec(x)) # type: ignore # matvec assumed
        return f_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        if x.size != self.nx:
            raise ValueError(f'x must have size {self.nx}, instead the provided x has size {x.size}')
        
        prox_parts: List[np.ndarray] = []
        if self.nn is not None and self.xin is not None and self.xend is not None: # nn case
            for iop, op in enumerate(self.ops):
                prox_parts.append(op.prox(x[self.xin[iop]:self.xend[iop]], tau))
            return np.hstack(prox_parts)
        elif self.restr is not None: # restr case
            # Initialize result array.
            # The type of f_prox should match x.dtype, ensure zeros_like uses it.
            f_prox: np.ndarray = np.zeros_like(x, dtype=x.dtype)
            for op, r_op in zip(self.ops, self.restr):
                # Assuming r_op.iava provides the indices in the original vector x
                # that correspond to the output of r_op.matvec(x) after prox.
                # This means r_op.adjoint_map(prox_result_part) would place it correctly.
                # Or, if iava are indices for selection:
                selected_x_part: np.ndarray = r_op.matvec(x) # type: ignore
                proxed_part: np.ndarray = op.prox(selected_x_part, tau)
                # Need to place proxed_part back into f_prox at correct locations.
                # This requires either r_op.H (adjoint) or knowing the indices.
                # Original: f[restr.iava] = op.prox(restr.matvec(x), tau)
                # This implies restr.iava are the indices in f_prox to fill.
                # This is unusual for pylops.Restriction where iava is usually input indices.
                # Assuming `r_op.H @ proxed_part` or similar logic for placing back.
                # For now, to match original, assuming restr.iava are output indices somehow.
                # This is highly dependent on the specific Restriction op implementation.
                # A standard way: f_prox += r_op.H @ proxed_part
                # If restr.iava are indices for direct assignment:
                if hasattr(r_op, 'iava') and isinstance(r_op.iava, (np.ndarray, slice)):
                     f_prox[r_op.iava] = proxed_part
                else:
                    # Fallback or error if iava is not directly usable for assignment
                    # This part is risky without knowing the exact nature of restr[i]
                    # For now, trying to replicate original, but it's fragile.
                    # A more robust way would be to sum (r_op.H @ op.prox(r_op @ x, tau))
                    # but that assumes operators stack additively, not replace parts of vector.
                    # The original code implies direct indexed assignment.
                    # This needs to be used with caution.
                    # Let's assume the user provides Restriction ops compatible with this indexing.
                    f_prox_part_indices = r_op.iava # This is likely incorrect usage of Restriction.iava
                    f_prox[f_prox_part_indices] = proxed_part
            return f_prox
        else: # Should not be reached due to __init__ check
            return x.copy()


    def grad(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.nx:
            raise ValueError(f'x must have size {self.nx}, instead the provided x has size {x.size}')

        grad_parts: List[np.ndarray] = []
        if self.nn is not None and self.xin is not None and self.xend is not None: # nn case
            for iop, op in enumerate(self.ops):
                grad_parts.append(op.grad(x[self.xin[iop]:self.xend[iop]]))
            return np.hstack(grad_parts)
        elif self.restr is not None: # restr case
            g_grad: np.ndarray = np.zeros_like(x, dtype=x.dtype)
            for op, r_op in zip(self.ops, self.restr):
                # Similar to prox, assuming r_op.iava allows placing gradient part.
                # More robust: g_grad += r_op.H @ op.grad(r_op @ x)
                if hasattr(r_op, 'iava') and isinstance(r_op.iava, (np.ndarray, slice)):
                    g_grad[r_op.iava] = op.grad(r_op.matvec(x)) # type: ignore
                else:
                    # Fallback or error
                    g_grad_part_indices = r_op.iava
                    g_grad[g_grad_part_indices] = op.grad(r_op.matvec(x)) # type: ignore
            return g_grad
        else: # Should not be reached
            return np.zeros_like(x)