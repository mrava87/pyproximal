import numpy as np
from typing import Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.proximal.L2 import L2 # Ensure correct import path
from pyproximal.proximal.L1 import L1 # Ensure correct import path


class Huber(ProxOperator):
    r"""Huber norm proximal operator.

    Proximal operator of the Huber norm defined as 
    :math:`H_\alpha(\mathbf{x}) = \sum_i H_\alpha(x_i)` where:

    .. math::

        H_\alpha(x_i) = 
        \begin{cases}
        \frac{|x_i|^2}{2 \alpha}, & |x_i| \leq \alpha \\
        |x_i| - \frac{\alpha}{2}, & |x_i| > \alpha
        \end{cases}

    which behaves like a :math:`\ell_2^2` norm for :math:`|x_i| \leq \alpha` and a
    :math:`\ell_1` norm for :math:`|x_i| > \alpha`.

    Parameters
    ----------
    alpha : :obj:`float`
        Huber parameter

    Notes
    -----
    The Huber proximal operator is defined as:

    .. math::

        \prox_{\tau H_\alpha(\cdot)}(\mathbf{x}) =
        \begin{cases}
        \prox_{\frac{\tau}{2 \alpha} |x_i|^2}(x_i), & |x_i| \leq \alpha \\
        \prox_{\tau |x_i|}(x_i), & |x_i| > \alpha
        \end{cases}
        
    """
    def __init__(self, alpha: float):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.alpha: float = alpha
        self.l2: L2 = L2(sigma=1. / self.alpha)
        self.l1: L1 = L1()

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar sum
        h: np.ndarray = np.zeros_like(x)
        xabs: np.ndarray = np.abs(x)
        mask: np.ndarray = xabs > self.alpha
        h[~mask] = xabs[~mask]**2 / (2. * self.alpha)
        h[mask] = xabs[mask] - self.alpha / 2.
        return float(np.sum(h))

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        y: np.ndarray = np.zeros_like(x)
        xabs: np.ndarray = np.abs(x)
        mask: np.ndarray = xabs > self.alpha
        
        # Elements where |x_i| <= alpha
        if np.any(~mask): # Check if there are any elements to process for L2 part
            y[~mask] = self.l2.prox(x[~mask], tau)
        
        # Elements where |x_i| > alpha
        if np.any(mask): # Check if there are any elements to process for L1 part
            y[mask] = self.l1.prox(x[mask], tau)
            
        # alternative from https://math.stackexchange.com/questions/1650411/
        # proximal-operator-of-the-huber-loss-function... currently commented
        # as it does not provide the same result
        # y = (1. - tau / np.maximum(np.abs(x), tau + self.alpha)) * x
        return y
    

class HuberCircular(ProxOperator):
    r"""Circular Huber norm proximal operator.

    Proximal operator of the Circular Huber norm defined as:

    .. math::

        H_\alpha(\mathbf{x}) =
        \begin{cases}
        \frac{\|\mathbf{x}\|_2^2}{2 \alpha}, & \|\mathbf{x}\|_2 \leq \alpha \\
        \|\mathbf{x}\|_2 - \frac{\alpha}{2}, & \|\mathbf{x}\|_2 > \alpha \\
        \end{cases}

    which behaves like a :math:`\ell_2^2` norm for :math:`\|\mathbf{x}\|_2 \leq \alpha` and a
    :math:`\ell_2` norm for :math:`\|\mathbf{x}\|_2 > \alpha`.

    Parameters
    ----------
    alpha : :obj:`float`
        Huber parameter

    Notes
    -----
    The Circular Huber proximal operator is defined as [1]_:

    .. math::

        \prox_{\tau H_\alpha(\cdot)}(\mathbf{x}) =
        \left( 1 - \frac{\tau}{\max\{\|\mathbf{x}\|_2, \tau + \alpha \} } \right) \mathbf{x}

    .. [1] O’Donoghue, B. and Stathopoulos, G. and Boyd, S. "A Splitting Method for Optimal Control", 
        In the IEEE Transactions on Control Systems Technology, 2013.
        
    """
    def __init__(self, alpha: float):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.alpha: float = alpha

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar
        l2_norm: float = float(np.linalg.norm(x)) # Cast to float
        h_val: float
        if l2_norm <= self.alpha:
            h_val = l2_norm**2 / (2 * self.alpha)
        else:
            h_val = l2_norm - self.alpha / 2.
        return h_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        norm_x: float = float(np.linalg.norm(x)) # Cast to float
        # Ensure no division by zero if norm_x and (tau + alpha) are both zero
        denominator: float = float(max(norm_x, tau + self.alpha)) # Cast max result
        if denominator == 0:
             # This case implies norm_x is 0 and tau + alpha is 0.
             # If x is zero vector, prox is x.
            return x
        
        x_prox: np.ndarray = (1. - tau / denominator) * x
        return x_prox
