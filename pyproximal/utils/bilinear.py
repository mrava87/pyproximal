from typing import TYPE_CHECKING, Optional, Any, Tuple

import numpy as np

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class BilinearOperator:
    r"""Common interface for bilinear operator of a function.

    Bilinear operator template class. A user
    must subclass it and implement the following methods:

    - ``gradx``: a method evaluating the gradient over :math:`\mathbf{x}`:
      :math:`\nabla_x H`
    - ``grady``: a method evaluating the gradient over :math:`\mathbf{y}`:
      :math:`\nabla_y H`
    - ``grad``: a method returning the stacked gradient vector over
      :math:`\mathbf{x},\mathbf{y}`: :math:`[\nabla_x H`, [\nabla_y H]`
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
    # These will be initialized by subclasses, but good to declare them for type checking
    x: np.ndarray
    y: np.ndarray
    sizex: int # Added for gradtest_bilinear compatibility
    sizey: int # Added for gradtest_bilinear compatibility

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
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        pass

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Any:
        # Return type depends on subclass implementation
        raise NotImplementedError

    def gradx(self, x: np.ndarray) -> np.ndarray:
        # Return type depends on subclass implementation
        raise NotImplementedError

    def grady(self, y: np.ndarray) -> np.ndarray:
        # Return type depends on subclass implementation
        raise NotImplementedError

    def grad(self, x_or_y: np.ndarray) -> np.ndarray: # Argument can be x or y depending on context
        # Return type depends on subclass implementation
        raise NotImplementedError

    def lx(self, x: np.ndarray) -> float: # Lipschitz constant is scalar
        # Return type depends on subclass implementation
        raise NotImplementedError

    def ly(self, y: np.ndarray) -> float: # Lipschitz constant is scalar
        # Return type depends on subclass implementation
        raise NotImplementedError

    def updatex(self, x: np.ndarray) -> None:
        """Update x variable (to be used to update the internal variable x)
        """
        self.x = x

    def updatey(self, y: np.ndarray) -> None:
        """Update y variable (to be used to update the internal variable y)
        """
        self.y = y

    def updatexy(self, xy: np.ndarray) -> None:
        raise NotImplementedError


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
    def __init__(self, X: np.ndarray, Y: np.ndarray, d: np.ndarray,
                 Op: Optional["LinearOperator"] = None) -> None:
        super().__init__() # Call BilinearOperator's __init__
        self.n: int
        self.k: int
        self.n, self.k = X.shape
        self.m: int = Y.shape[1]

        self.x: np.ndarray = X # This is X_mat
        self.y: np.ndarray = Y # This is Y_mat
        self.d: np.ndarray = d
        self.Op: Optional["LinearOperator"] = Op
        self.sizex: int = self.n * self.k
        self.sizey: int = self.m * self.k

    def __call__(self, x_input: np.ndarray, y_input: Optional[np.ndarray] = None) -> float:
        # x_input can be concatenated [x,y] or just x if y_input is provided
        current_x_vec: np.ndarray
        current_y_vec: np.ndarray
        if y_input is None:
            current_x_vec, current_y_vec = x_input[:self.sizex], x_input[self.sizex:]
        else:
            current_x_vec = x_input
            current_y_vec = y_input
        
        # Store original self.x to restore after calculation, as _matvecy uses self.x
        original_self_x: np.ndarray = self.x.copy()
        self.updatex(current_x_vec) # Temporarily update self.x for _matvecy
        
        # H(X,Y) = 0.5 * || Op(XY) - d ||_2^2
        # _matvecy(y_vec) computes Op(self.x @ Y) where Y is from y_vec
        # So, res = d - Op(current_X @ current_Y)
        res: np.ndarray = self.d - self._matvecy(current_y_vec)
        
        self.updatex(original_self_x) # Restore original self.x
        return float(np.linalg.norm(res)**2 / 2.)

    def _matvecx(self, x_vec: np.ndarray) -> np.ndarray: # x_vec is the flattened X matrix
        X_mat: np.ndarray = x_vec.reshape(self.n, self.k)
        # Y_mat from self.y (which is the stored Y matrix)
        Y_mat_stored: np.ndarray = self.y.reshape(self.k, self.m)
        result_mat: np.ndarray = X_mat @ Y_mat_stored # X @ Y
        if self.Op is not None:
            # Op acts on the vectorized result_mat
            result_vec: np.ndarray = self.Op @ result_mat.ravel() # type: ignore
            return result_vec.ravel()
        return result_mat.ravel()

    def _matvecy(self, y_vec: np.ndarray) -> np.ndarray: # y_vec is the flattened Y matrix
        Y_mat: np.ndarray = y_vec.reshape(self.k, self.m)
        # X_mat from self.x (which is the stored X matrix)
        X_mat_stored: np.ndarray = self.x.reshape(self.n, self.k)
        result_mat: np.ndarray = X_mat_stored @ Y_mat # X @ Y
        if self.Op is not None:
            result_vec: np.ndarray = self.Op @ result_mat.ravel() # type: ignore
            return result_vec.ravel()
        return result_mat.ravel()

    def matvec(self, x_in: np.ndarray) -> np.ndarray: # x_in can be x_vec or y_vec
        # This method is ambiguous if n*k == m*k.
        # It's better to use _matvecx or _matvecy directly if sizes can be ambiguous.
        # The original code had n==m check, but it should be about total size.
        # sizex = n*k, sizey = m*k.
        if self.sizex == self.sizey and self.n != self.m : # Ambiguity if total sizes are same but underlying shapes differ
             pass # This case is still tricky, but less likely than n==m for ambiguity.

        if self.n == self.m and self.sizex == x_in.size and self.sizey == x_in.size : # True ambiguity
            raise NotImplementedError('Since n=m (and thus sizex=sizey if k is same), this method '
                                      'cannot distinguish automatically between _matvecx and _matvecy. '
                                      'Explicitly call either of those two methods.')
        
        result_vec: np.ndarray
        if x_in.size == self.sizex:
            result_vec = self._matvecx(x_in)
        elif x_in.size == self.sizey:
            result_vec = self._matvecy(x_in)
        else:
            raise ValueError("Input vector size does not match sizex or sizey.")
        return result_vec

    def lx(self, x_vec: np.ndarray) -> float: # x_vec is the flattened X matrix for which we want Lx
        if self.Op is not None:
            # Lipschitz constant for gradx H involves Op.H Op and Y Y.H.
            # This is non-trivial and depends on Op's norm.
            # For simplicity or if Op=I, this can be estimated.
            # Original code raises ValueError.
            raise ValueError('lx cannot be computed automatically when using Op an external operator.')
        # If Op is None, H = 0.5 * ||XY - d||^2. gradx H = (XY-d)Y.H
        # Lipschitz of gradx H involves ||Y Y.H||_F or ||Y||_2^2
        # The original code calculates norm(X.H @ X, 'fro') which seems to be for a different formulation.
        # Let's assume self.y (stored Y) is used for Lipschitz constant.
        Y_mat_stored: np.ndarray = self.y.reshape(self.k, self.m)
        # L_x = ||Y^H Y||_2 or similar. For Frobenius norm of Y Y.H:
        return float(np.linalg.norm(Y_mat_stored @ Y_mat_stored.conj().T, 'fro'))


    def ly(self, y_vec: np.ndarray) -> float: # y_vec is the flattened Y matrix for which we want Ly
        if self.Op is not None:
            raise ValueError('ly cannot be computed automatically when using Op an external operator.')
        # If Op is None, H = 0.5 * ||XY - d||^2. grady H = X.H (XY-d)
        # Lipschitz of grady H involves ||X.H X||_F or ||X||_2^2
        # The original code uses self.x (stored X).
        X_mat_stored: np.ndarray = self.x.reshape(self.n, self.k)
        return float(np.linalg.norm(X_mat_stored.conj().T @ X_mat_stored, 'fro'))


    def gradx(self, x_vec: np.ndarray) -> np.ndarray: # x_vec is the flattened X matrix
        # grad_x H = Op.H ( Op(XY) - d ) Y.H
        # Here, X comes from x_vec, Y is self.y (stored).
        # Need to compute Op(XY)-d first.
        # _matvecx(x_vec) computes Op(X_from_x_vec @ Y_stored).
        # So, Op(XY) is _matvecx(x_vec).
        
        # Temporarily update self.x to X_from_x_vec for consistent use in _matvecx if it relies on self.x for X part.
        # However, _matvecx is defined to take x (representing X) as input.
        # Let's ensure current self.x is the one for which this grad is calculated.
        original_self_x = self.x.copy()
        self.updatex(x_vec) # Ensures _matvecx uses the X from x_vec

        op_xy_vec: np.ndarray = self._matvecx(x_vec) # This is Op(X Y_stored)
        residual_vec: np.ndarray = op_xy_vec - self.d
        
        # r_intermediate is Op.H @ residual_vec
        r_intermediate: np.ndarray
        if self.Op is not None:
            r_intermediate = self.Op.H @ residual_vec # type: ignore
        else: # Op is Identity
            r_intermediate = residual_vec
            
        # Reshape r_intermediate to matrix form (n, m) to multiply with Y.H
        r_intermediate_mat: np.ndarray = r_intermediate.reshape(self.n, self.m)
        
        Y_mat_stored: np.ndarray = self.y.reshape(self.k, self.m)
        grad_x_mat: np.ndarray = r_intermediate_mat @ Y_mat_stored.conj().T
        
        self.updatex(original_self_x) # Restore original self.x
        return grad_x_mat.ravel()

    def grady(self, y_vec: np.ndarray) -> np.ndarray: # y_vec is the flattened Y matrix
        # grad_y H = X.H Op.H ( Op(XY) - d )
        # Here, Y comes from y_vec, X is self.x (stored).
        # Need Op(XY)-d.
        # _matvecy(y_vec) computes Op(X_stored @ Y_from_y_vec)
        
        original_self_y = self.y.copy()
        self.updatey(y_vec) # Ensure _matvecy uses Y from y_vec

        op_xy_vec: np.ndarray = self._matvecy(y_vec) # This is Op(X_stored Y)
        residual_vec: np.ndarray = op_xy_vec - self.d
        
        # r_intermediate is Op.H @ residual_vec
        r_intermediate: np.ndarray
        if self.Op is not None:
            r_intermediate = self.Op.H @ residual_vec # type: ignore
        else: # Op is Identity
            r_intermediate = residual_vec
            
        r_intermediate_mat: np.ndarray = r_intermediate.reshape(self.n, self.m)
        
        X_mat_stored: np.ndarray = self.x.reshape(self.n, self.k)
        grad_y_mat: np.ndarray = X_mat_stored.conj().T @ r_intermediate_mat
        
        self.updatey(original_self_y) # Restore
        return grad_y_mat.ravel()

    def grad(self, xy_vec: np.ndarray) -> np.ndarray: # xy_vec is concatenated [x_vec, y_vec]
        x_part: np.ndarray = xy_vec[:self.sizex]
        y_part: np.ndarray = xy_vec[self.sizex:]
        
        # Need to be careful: gradx uses self.y, grady uses self.x.
        # These should be the Y and X from the *current* point (x_part, y_part) at which grad is evaluated.
        # So, update self.x and self.y before calling gradx and grady.
        original_x = self.x.copy()
        original_y = self.y.copy()
        self.updatex(x_part)
        self.updatey(y_part)

        grad_x_val: np.ndarray = self.gradx(x_part) # gradx will use the updated self.y
        grad_y_val: np.ndarray = self.grady(y_part) # grady will use the updated self.x
        
        # Restore original self.x, self.y if they are meant to be persistent state beyond single grad call
        self.updatex(original_x)
        self.updatey(original_y)
        
        g_stacked: np.ndarray = np.hstack([grad_x_val, grad_y_val])
        return g_stacked

    def updatexy(self, xy_vec: np.ndarray) -> None: # xy_vec is concatenated [x_vec, y_vec]
        self.updatex(xy_vec[:self.sizex])
        self.updatey(xy_vec[self.sizex:])