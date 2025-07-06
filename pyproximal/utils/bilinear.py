from typing import TYPE_CHECKING, Optional, Any

import numpy as np

from abc import ABC, abstractmethod
from pylops.utils.typing import NDArray

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class BilinearOperator(ABC):
    r"""Common interface for bilinear operator of a function.

    Bilinear operator template class. A user
    must subclass it and implement the following methods:

    - ``gradx``: a method evaluating the gradient over :math:`\mathbf{x}`:
      :math:`\nabla_x H`
    - ``grady``: a method evaluating the gradient over :math:`\mathbf{y}`:
      :math:`\nabla_y H`
    - ``grad``: a method returning the stacked gradient vector over
      :math:`\mathbf{x},\mathbf{y}`: :math:`[\nabla_x H`, [\nabla_y H]`
      where the internal :math:`\mathbf{x}` variable is used in the 
      computation of the gradient of :math:`\mathbf{y}` and the internal 
      :math:`\mathbf{y}` variable is used in the  computation of the 
      gradient of :math:`\mathbf{x}` (not those provided)
    - ``lx``: Lipschitz constant of :math:`\nabla_x H`
    - ``ly``: Lipschitz constant of :math:`\nabla_y H`

    Two additional methods (``updatex`` and ``updatey``) are provided to
    update the :math:`\mathbf{x}` and :math:`\mathbf{y}` internal
    variables. It is user responsability to choose when to invoke such
    method (i.e., when to update the internal variables).

    Notes
    -----
    A bilinear operator is defined as a differentiable nonlinear function
    :math:`H(x,y)` that is linear in each of its components indipendently,
    i.e, :math:`\mathbf{H_x}(y)\mathbf{x}` and :math:`\mathbf{H_y}(y)\mathbf{x}`.

    """
    # Initialized by subclasses, but declared here for type checking
    x: NDArray
    y: NDArray
    sizex: int
    sizey: int

    def __init__(self) -> None:
        # Initialize sizex and sizey, e.g. to 0 or make them abstract if preferred
        # For now, allow subclasses to define them. Mypy will expect them.
        # However, to avoid runtime errors if not set by subclass and accessed:
        self.sizex = 0
        self.sizey = 0
        # Initialize x and y to empty arrays or placeholder of correct type
        # This depends on how they are first used. Subclasses like LowRankFactorizedMatrix
        # set them in their own __init__.
        # For the base class, if gradtest_bilinear accesses Op.x before it's properly set,
        # it could be an issue. Let's initialize to empty arrays.
        self.x: NDArray = np.array([])
        self.y: NDArray = np.array([])
        pass

    @abstractmethod
    def __call__(self, x: NDArray, y: Optional[NDArray] = None) -> Any:
        pass
    
    @abstractmethod
    def gradx(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def grady(self, y: NDArray) -> NDArray:
        pass

    @abstractmethod
    def grad(self, x_or_y: NDArray) -> NDArray:
        pass

    @abstractmethod
    def lx(self, x: NDArray) -> float:
        pass
    
    @abstractmethod
    def ly(self, y: NDArray) -> float:
        pass

    def updatex(self, x: NDArray) -> None:
        """Update x variable (to be used to update the internal variable x)
        """
        self.x = x

    def updatey(self, y: NDArray) -> None:
        """Update y variable (to be used to update the internal variable y)
        """
        self.y = y

    @abstractmethod
    def updatexy(self, xy: NDArray) -> None:
        pass

class LowRankFactorizedMatrix(BilinearOperator):
    r"""Low-Rank Factorized Matrix operator.

    Bilinear operator representing the L2 norm of a Low-Rank Factorized
    Matrix defined as: :math:`H(\mathbf{X}, \mathbf{Y}) =
    \frac{1}{2} \|\mathbf{Op}(\mathbf{X}\mathbf{Y}) - \mathbf{d}\|_2^2`,
    where :math:`\mathbf{X}` is a matrix of size  :math:`n \times k`,
    :math:`\mathbf{Y}` is a matrix of size :math:`k \times m`, and
    :math:`\mathbf{Op}` is a linear operator of size :math:`p \times n`.

    Parameters
    ----------
    X : :obj:`numpy.ndarray`
        Left-matrix of size :math:`n \times k`
    Y : :obj:`numpy.ndarray`
        Right-matrix of size :math:`k \times m`
    d : :obj:`numpy.ndarray`
        Data vector
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator

    Notes
    -----
    The Low-Rank Factorized Matrix operator has gradient with respect to x
    equal to:

    .. math::

        \nabla_x H(\mathbf{x};\ \mathbf{y}) =
        \mathbf{Op}^H(\mathbf{Op}(\mathbf{X}\mathbf{Y})
        - \mathbf{d})\mathbf{Y}^H

    and gradient with respect to y equal to:

    .. math::

        \nabla_y H(\mathbf{y}; \mathbf{x}) =
        \mathbf{X}^H \mathbf{Op}^H(\mathbf{Op}
        (\mathbf{X}\mathbf{Y}) - \mathbf{d})

    Note that in both cases, the currently stored :math`\mathbf{x}`/:math`\mathbf{y}` variable
    is used for the second variable within parenthesis (after ;).

    """
    def __init__(self, X: NDArray, Y: NDArray, d: NDArray,
                 Op: Optional["LinearOperator"] = None) -> None:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("The second dimension of x differs from the "
                             "first dimension of Y "
                             f"({X.shape[1]} != {Y.shape[0]:})")
        self.n: int
        self.k: int
        self.n, self.k = X.shape
        self.m: int = Y.shape[1]

        self.x: NDArray = X
        self.y: NDArray = Y
        self.d: NDArray = d
        self.Op: Optional["LinearOperator"] = Op
        self.sizex: int = self.n * self.k
        self.sizey: int = self.m * self.k

    def __call__(self, x: NDArray, y: Optional[NDArray] = None) -> float:
        # x_input can be concatenated [x,y] or just x if y_input is provided
        current_x: NDArray
        current_y: NDArray
        if y is None:
            current_x, current_y = x[:self.sizex], x[self.sizex:]
        else:
            current_x = x
            current_y = y
        
        # Store original self.x to restore after calculation, as _matvecy uses self.x
        original_x: NDArray = self.x.copy()

        # Temporarily update self.x for _matvecy
        self.updatex(current_x) 
        
        # Compute residual: note that _matvecy(y_vec) computes 
        # Op(self.x @ Y) where Y is from y_vec so 
        # res = d - Op(current_X @ current_Y)
        res: NDArray = self.d - self._matvecy(current_y)
        
        # Restore original self.x
        self.updatex(original_x)

        return float(np.linalg.norm(res) ** 2 / 2.)

    def _matvecx(self, x: NDArray) -> NDArray:
        # Recreate matrix from flattened x
        X: NDArray = x.reshape(self.n, self.k)
        # Recreate matrix from flattended self.y
        Y: NDArray = self.y.reshape(self.k, self.m)
        XY: NDArray = X @ Y
        if self.Op is not None:
            # Op acts on the vectorized version of XY
            XY: NDArray = self.Op @ XY.ravel() # type: ignore
        return XY.ravel()

    def _matvecy(self, y: NDArray) -> NDArray:
        # Recreate matrix from flattened y
        Y: NDArray = y.reshape(self.k, self.m)
        # Recreate matrix from flattended self.x
        X: NDArray = self.x.reshape(self.n, self.k)
        XY: NDArray = X @ Y
        if self.Op is not None:
            XY: NDArray = self.Op @ XY.ravel() # type: ignore
        return XY.ravel()

    def matvec(self, x: NDArray) -> NDArray:
        # Check that no ambiguous situation arises due to n==m
        if self.n == self.m:
            raise NotImplementedError('Since n=m, this method cannot distinguish '
                                      'automatically between _matvecx and _matvecy. '
                                      'Explicitly call either of those two methods.')
        y: NDArray
        if x.size == self.sizex:
            y = self._matvecx(x)
        elif x.size == self.sizey:
            y = self._matvecy(x)
        else:
            raise ValueError("Input vector size does not match sizex or sizey.")
        return y

    def lx(self, x: NDArray) -> float:
        if self.Op is not None:
            # Lipschitz constant for gradx H involves Op.H Op and Y Y.H.
            # This is non-trivial and depends on Op's norm.
            # For simplicity or if Op=I, this can be estimated.
            raise ValueError('lx cannot be computed automatically when using Op.')
        # If Op is None, H = 0.5 * ||XY - d||^2. gradx H = (XY-d)Y.H
        # Lipschitz of gradx H involves ||Y Y.H||_F or ||Y||_2^2
        # The original code calculates norm(X.H @ X, 'fro') which 
        # seems to be for a different formulation.
        # Let's assume self.y (stored Y) is used for Lipschitz constant.
        Y: NDArray = self.y.reshape(self.k, self.m)
        # L_x = ||Y^H Y||_2 or similar. For Frobenius norm of Y Y.H:
        return float(np.linalg.norm(Y @ Y.conj().T, 'fro'))

    def ly(self, y_vec: np.ndarray) -> float: # y_vec is the flattened Y matrix for which we want Ly
        if self.Op is not None:
            raise ValueError('ly cannot be computed automatically when using Op an external operator.')
        # If Op is None, H = 0.5 * ||XY - d||^2. grady H = X.H (XY-d)
        # Lipschitz of grady H involves ||X.H X||_F or ||X||_2^2
        # The original code uses self.x (stored X).
        X_mat_stored: np.ndarray = self.x.reshape(self.n, self.k)
        return float(np.linalg.norm(X_mat_stored.conj().T @ X_mat_stored, 'fro'))

    def gradx(self, x: NDArray) -> NDArray:
        """Compute gradient of H wrt x
        
        Compute grad_x H = Op.H ( Op(XY) - d ) Y.H
        Here, X comes from x, Y is self.y (stored).
        and _matvecx(x) computes Op(X @ Y).

        """
        # Compute residual
        res: NDArray = self._matvecx(x) - self.d
        
        # Apply adjoint of operator if present
        res_op: NDArray
        if self.Op is not None:
            res_op = self.Op.H @ res # type: ignore
        else: # Op is Identity
            res_op = res
        
        # Apply Y.H
        Y: NDArray = self.y.reshape(self.k, self.m)
        grad_x: NDArray = res_op.reshape(self.n, self.m) @ Y.conj().T
        
        return grad_x.ravel()

    def grady(self, y: NDArray) -> NDArray:
        """Compute gradient of H wrt y
        
        Compute grad_y H = X.H Op.H ( Op(XY) - d )
        Here, Y comes from y, X is self.x (stored).
        and _matvecy(y_vec) computes Op(X @ Y).

        """
        # Compute residual
        res: NDArray = self._matvecy(y) - self.d
        
        # Apply adjoint of operator if present
        res_op: NDArray
        if self.Op is not None:
            res_op = self.Op.H @ res # type: ignore
        else: # Op is Identity
            res_op = res
        
        # Apply X.H
        X: NDArray = self.x.reshape(self.n, self.k)
        grad_y: NDArray = X.conj().T @ res_op.reshape(self.n, self.m)

        return grad_y.ravel()

    def grad(self, xy: NDArray) -> NDArray:
        """Compute total gradient
                
        """
        x: NDArray = xy[:self.sizex]
        y: NDArray = xy[self.sizex:]
        
        # Computate gradients
        grad_x: NDArray = self.gradx(x)
        grad_y: NDArray = self.grady(y)
        
        # Stack gradients
        grad_stacked: NDArray = np.hstack([grad_x, grad_y])
        return grad_stacked

    def updatexy(self, xy: NDArray) -> None:
        self.updatex(xy[:self.sizex])
        self.updatey(xy[self.sizex:])