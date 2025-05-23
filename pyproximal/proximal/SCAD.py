import numpy as np
from typing import Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator


class SCAD(ProxOperator):
    r"""Smoothly clipped absolute deviation (SCAD) penalty.

    The SCAD penalty is a concave function and is defined as

    .. math::

        \mathrm{SCAD}_{\sigma, a}(\mathbf{x}) =
        \begin{cases}
        \sigma x_i, & |x_i| \leq \sigma \\
        \frac{-x_i^2 + 2 a \sigma  - \sigma^2}{2 (a - 1)}, & \sigma < |x_i| \leq a\sigma \\
        \frac{(a + 1)\sigma^2}{2}, & |x_i| > a\sigma
        \end{cases}

    Parameters
    ----------
    sigma : :obj:`float`
        First threshold parameter (named :math:`\lambda` in the original paper [1]_)
    a : :obj:`float`, optional
        Second threshold parameter (must be larger than 2). Default is 3.7, see [1]_ for more information.

    Notes
    -----

    The SCAD penalty is continuous and differentiable and does not suffer from
    biasedness as the :math:`\ell_1`-norm, nor is discontinuous as the hard
    thresholding penalty. Thus, it fuses the most favourable properties of both
    penalties.

    The proximal operator is given by

    .. math::

        \prox_{\tau \mathrm{SCAD}_{\sigma, a}(\cdot)}(\mathbf{x}) =
        \begin{cases}
        \sgn(x_i)\max(0, |x_i| - \sigma), & |x_i| \leq \frac{\sigma(a - 1 - \tau + a\tau)}{a - 1} \\
        \frac{(a-1)x_i - \sgn(x_i)a\tau\sigma}{a-1-\tau}, & \frac{\sigma(a - 1 - \tau + a\tau)}{a - 1} < |x_i| \leq a\sigma \\
        x_i, & |x_i| > a\sigma
        \end{cases}

    .. [1] Fan, J. and Li, R. "Variable selection via nonconcave penalized likelihood and its oracle
        properties" Journal of the American Statistical Association, 96(456):1348–1360, 2001

    """

    def __init__(self, sigma: float, a: float = 3.7):
        super().__init__(None, False) # Op is None, hasgrad is False
        if sigma <= 0:
            raise ValueError('Variable "sigma" must be positive.')
        if a <= 2:
            raise ValueError('Variable "a" must be larger than two.')
        self.sigma: float = sigma
        self.a: float = a

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar sum
        return float(np.sum(self.elementwise(x)))

    def elementwise(self, x: np.ndarray) -> np.ndarray:
        f_val: np.ndarray = np.zeros_like(x)
        absx: np.ndarray = np.abs(x)
        
        # Case 1: |x_i| <= sigma
        ind1: np.ndarray = absx <= self.sigma
        f_val[ind1] = self.sigma * absx[ind1]
        
        # Case 2: sigma < |x_i| <= a*sigma
        ind2: np.ndarray = np.logical_and(self.sigma < absx, absx <= self.a * self.sigma)
        # Note: formula uses x_i^2, but for penalty it should be |x_i|^2 if x_i can be negative.
        # However, SCAD is typically defined for |x_i|.
        # The term -x[ind2]**2 in original code might be problematic if x is complex.
        # Assuming x is real based on typical SCAD usage.
        # If x can be complex, absx[ind2]**2 should be used. For real x, x[ind2]**2 is fine.
        f_val[ind2] = (-absx[ind2]**2 + 2 * self.a * self.sigma * absx[ind2] - self.sigma**2) / \
                      (2 * (self.a - 1))
                      
        # Case 3: |x_i| > a*sigma
        ind3: np.ndarray = absx > self.a * self.sigma
        f_val[ind3] = (self.a + 1) * self.sigma**2 / 2.0 # Ensure float division
        
        return f_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        theta_prox: np.ndarray = x.copy() # Result array, initialized with x (handles |x| > a*sigma case)
        absx: np.ndarray = np.abs(x)
        sign_x: np.ndarray = np.sign(x) # Store original signs
        
        # Denominator for first_threshold, ensure it's not zero
        denom_first_thresh: float = self.a - 1
        if denom_first_thresh == 0: # Should be caught by a > 2 check in __init__
            raise ValueError("Parameter 'a' must be greater than 2 to avoid division by zero.")

        first_threshold: float = self.sigma * (self.a - 1 - tau + tau * self.a) / denom_first_thresh
        
        # Case 1: |x_i| <= first_threshold
        # prox = sgn(x_i) * max(0, |x_i| - sigma*tau)
        ind1_prox: np.ndarray = absx <= first_threshold
        theta_prox[ind1_prox] = sign_x[ind1_prox] * np.maximum(0, absx[ind1_prox] - self.sigma * tau)
        
        # Case 2: first_threshold < |x_i| <= a*sigma
        # prox = ((a-1)x_i - sgn(x_i)*a*tau*sigma) / (a-1-tau)
        ind2_prox: np.ndarray = np.logical_and(first_threshold < absx, absx <= self.a * self.sigma)
        denom_case2: float = self.a - 1 - tau
        if denom_case2 == 0: # Avoid division by zero
            # This case implies a specific relationship between a and tau.
            # The paper might specify behavior here, or it's an edge case.
            # If denom is 0, the prox value might be x or tend to infinity.
            # For now, if it's zero, keep original x for these elements (as if |x|>a*sigma)
            # or handle as per specific SCAD prox derivation for this edge case.
            # Assuming this implies elements behave as if in the third region (no change).
            pass # theta_prox[ind2_prox] remains x[ind2_prox]
        else:
            theta_prox[ind2_prox] = ((self.a - 1) * x[ind2_prox] - 
                                     sign_x[ind2_prox] * self.a * self.sigma * tau) / denom_case2
            
        # Case 3: |x_i| > a*sigma
        # prox = x_i (already handled by theta_prox = x.copy())
        
        return theta_prox
