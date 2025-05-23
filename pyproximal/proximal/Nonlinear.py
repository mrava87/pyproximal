import numpy as np
from typing import Tuple, Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator


class Nonlinear(ProxOperator):
    r"""Nonlinear function proximal operator.

    Proximal operator for a generic nonlinear function :math:`f`. This is a
    template class which a user must subclass and implement the following
    methods:

    - ``fun``: a method evaluating the generic function :math:`f`
    - ``grad``: a method evaluating the gradient of the generic function
      :math:`f`
    - ``fungrad``: a method evaluating both the generic function :math:`f` 
      and its gradient
    - ``optimize``: a method that solves the optimization problem associated
      with the proximal operator of :math:`f`. Note that the
      ``gradprox`` method must be used (instead of ``grad``) as this will
      automatically add the regularization term involved in the evaluation
      of the proximal operator

    Parameters
    ----------
    x0 : :obj:`np.ndarray`
        Initial vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme used to compute the proximal
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.

    Notes
    -----
    The proximal operator of a generic function requires solving the following
    optimization problem numerically

    .. math::

        prox_{\tau f} (\mathbf{x}) = arg \; min_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2

    which is done via the provided ``optimize`` method.

    """
    def __init__(self, x0: np.ndarray, niter: int = 10, warm: bool = True):
        super().__init__(None, True) # Op is None, hasgrad is True
        self.niter: int = niter
        self.x0: np.ndarray = x0 # Initial guess for iterative solver in optimize
        self.warm: bool = warm
        self.y: np.ndarray # Stores x for prox: prox_tf(x) = argmin_u f(u) + 1/(2t)||u-x||^2
        self.tau_prox: float # Stores tau for prox

    def __call__(self, x: np.ndarray) -> float: # Assuming fun returns a scalar value
        return self.fun(x)

    def _funprox(self, x: np.ndarray, tau: float) -> float:
        # This method seems to be for internal use by an optimization algorithm.
        # It evaluates f(x) + 1/(2*tau) * ||x - y||^2, where y is stored from prox call.
        return self.fun(x) + 1. / (2 * tau) * float(np.sum((x - self.y)**2))

    def _gradprox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # Gradient of f(x) + 1/(2*tau) * ||x - y||^2 w.r.t. x
        return self.grad(x) + 1. / tau * (x - self.y)

    def _fungradprox(self, x: np.ndarray, tau: float) -> Tuple[float, np.ndarray]:
        # Returns (value, gradient) of f(x) + 1/(2*tau) * ||x - y||^2
        f_val, g_val = self.fungrad(x)
        f_val_prox: float = f_val + 1. / (2 * tau) * float(np.sum((x - self.y)**2))
        g_val_prox: np.ndarray = g_val + 1. / tau * (x - self.y)
        return f_val_prox, g_val_prox

    # Abstract methods to be implemented by subclasses
    def fun(self, x: np.ndarray) -> float:
        raise NotImplementedError('The method fun has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('The method grad has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')

    def fungrad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        # Should return (f(x), grad f(x))
        raise NotImplementedError('The method fungrad has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.') # Corrected error message

    def optimize(self) -> np.ndarray:
        # This method should solve the proximal optimization problem using
        # self.y (the input to prox), self.tau_prox, self.x0 (initial guess), self.niter.
        # It would typically use self._funprox, self._gradprox, or self._fungradprox.
        raise NotImplementedError('The method optimize has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        self.y = x       # Store x (input to prox) as self.y for use in _funprox, etc.
        self.tau_prox = tau # Store tau for use in _funprox, etc.
        
        # Call the user-defined optimization method
        # The result of optimize() is the solution to the prox problem
        x_optimized: np.ndarray = self.optimize()
        
        if self.warm:
            self.x0 = x_optimized.copy() # Update initial guess for next prox call
        return x_optimized