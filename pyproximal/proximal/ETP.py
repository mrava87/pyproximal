import numpy as np
from scipy.special import lambertw
from typing import Union, Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator


class ETP(ProxOperator):
    r"""Exponential-type penalty (ETP).

    The exponential-type penalty is defined as

    .. math::

        \mathrm{ETP}_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma}{1-e^{-\gamma}}(1-e^{-\gamma|x_i|})

    for :math:`{\sigma>0}`, and :math:`{\gamma>0}`.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.0.

    Notes
    -----
    As :math:`{\gamma\rightarrow 0}` the exponential-type penalty approaches the :math:`\ell_1`-penalty and when
    :math:`{\gamma\rightarrow\infty}` tends to the :math:`\ell_0`-penalty [1]_.

    As for the proximal operator, consider the one-dimensional case

    .. math::

        \prox_{\tau \mathrm{ETP}(\cdot)}(x) = \argmin_{z} \mathrm{ETP}(z) + \frac{1}{2\tau}(x - z)^2

    and assume that :math:`x\geq 0`. The minima can be obtained when :math:`z=0` or at a stationary point,
    where the latter must satisfy

    .. math::

        x = z + \frac{\gamma \sigma \tau}{1-e^{-\gamma}} e^{-\gamma z} .

    The solution to the above equation can be expressed using the *Lambert W function*.

    .. [1] Gao, C. et al. "A Feasible Nonconvex Relaxation Approach to Feature Selection",
        In the Proceedings of the Conference on Artificial Intelligence (AAAI), 2011.

    """

    def __init__(self, sigma: float, gamma: float = 1.0):
        super().__init__(None, False) # Op is None, hasgrad is False
        if sigma < 0:
            raise ValueError('Variable "sigma" must be positive.')
        if gamma <= 0:
            raise ValueError('Variable "gamma" must be strictly positive.')
        self.sigma: float = sigma
        self.gamma: float = gamma

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar sum
        return float(np.sum(self.elementwise(x)))

    def elementwise(self, x: np.ndarray) -> np.ndarray:
        return self.sigma / (1 - np.exp(-self.gamma)) * (1 - np.exp(-self.gamma * np.abs(x)))

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        k: float = tau * self.sigma / (1 - np.exp(-self.gamma))
        out: np.ndarray = np.zeros_like(x)

        # Get real-valued solutions to the Lambert W function
        # Ensure operations are on np.ndarray to allow boolean indexing
        abs_x_gamma: np.ndarray = np.abs(x) * self.gamma
        tmp: np.ndarray = np.exp(-abs_x_gamma) * k * self.gamma ** 2
        
        # idx should be a boolean array of the same shape as x (or tmp)
        idx: np.ndarray = tmp <= np.exp(-1)
        
        # Operations on slices
        x_idx: np.ndarray = x[idx]
        tmp_idx: np.ndarray = tmp[idx]
        
        # lambertw can return complex numbers, ensure we take real part.
        # The output of lambertw will have the same shape as tmp_idx.
        lambertw_result: np.ndarray = np.real(lambertw(-tmp_idx))
        
        stat_points: np.ndarray = np.sign(x_idx) * lambertw_result / self.gamma + x_idx

        # Check which stationary points are global minima
        # elementwise needs an array, ensure stat_points is correctly shaped if x_idx was empty
        if stat_points.size > 0: # Avoid error if stat_points is empty
            cost_stat_points: np.ndarray = tau * self.elementwise(stat_points) + \
                                       (stat_points - x_idx) ** 2 / 2
            cost_at_zero: np.ndarray = tau * self.elementwise(np.zeros_like(x_idx)) + \
                                   (np.zeros_like(x_idx) - x_idx) ** 2 / 2 # Cost if z=0
            # Original logic was comparing to x[idx]**2/2, which is cost at z=0 if ETP(0)=0.
            # ETP(0) is indeed 0 based on its definition.
            # So, comparing cost at stationary point vs cost at z=0.
            idx_minima: np.ndarray = cost_stat_points < cost_at_zero
            
            # Update idx based on idx_minima: only those stationary points that are indeed minima
            # Create a temporary boolean array of the same shape as idx
            final_idx_update = np.zeros_like(idx, dtype=bool)
            final_idx_update[idx] = idx_minima # Apply idx_minima only to the subset defined by original idx
            
            out[final_idx_update] = stat_points[idx_minima]
        
        # For elements not satisfying idx (i.e., tmp > exp(-1)) or where stat_point is not a minimum,
        # the solution is z=0, which is already set by np.zeros_like(x).
        return out
