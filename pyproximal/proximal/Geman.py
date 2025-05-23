import numpy as np
from typing import Tuple, Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator


class Geman(ProxOperator):
    r"""Geman penalty.

    The Geman penalty (named after its inventor) is a non-convex penalty [1]_.
    The pyproximal implementation considers a generalized model where

    .. math::

        \mathrm{Geman}_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma |x_i|}{|x_i| + \gamma}

    where :math:`{\sigma\geq 0}`, :math:`{\gamma>0}`.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.3.

    Notes
    -----
    In order to compute the proximal operator of the Geman penalty one must find the
    roots of a cubic polynomial. Consider the one-dimensional problem

    .. math::
        \prox_{\tau \mathrm{Geman}(\cdot)}(x) = \argmin_{z} \mathrm{Geman}(z) + \frac{1}{2\tau}(x - z)^2

    and assume :math:`{x\geq 0}`. Either the minimum is obtained when :math:`z=0` or
    when

    .. math::
        \tau\sigma\gamma + (z-x)(z+\gamma)^2 = 0 .

    The pyproximal implementation uses the closed-form solution for a cubic equation,
    and discards infeasible roots, to find the minimum.

    .. [1] Geman and Yang "Nonlinear image recovery with half-quadratic regularization",
        IEEE Transactions on Image Processing, 4(7):932 – 946, 1995.

    """

    def __init__(self, sigma: float, gamma: float = 1.3):
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
        abs_x: np.ndarray = np.abs(x)
        return self.sigma * abs_x / (abs_x + self.gamma)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        out: np.ndarray = np.zeros_like(x)
        abs_x: np.ndarray = np.abs(x)
        
        # Coefficients of the cubic polynomial derived from the derivative
        # P(z) = z^3 + b*z^2 + c*z + d = 0
        # where z is the potential stationary point (assuming z >= 0)
        b_coeffs: np.ndarray = 2 * self.gamma - abs_x
        c_coeffs: np.ndarray = self.gamma**2 - 2 * self.gamma * abs_x
        d_coeffs: np.ndarray = self.gamma * self.sigma * tau - (self.gamma**2) * abs_x
        
        # Find local minima by solving the cubic equation
        # idx_roots identifies where real roots exist that could be local minima
        idx_roots: np.ndarray
        loc_mins: np.ndarray
        idx_roots, loc_mins = self._find_local_minima(b_coeffs, c_coeffs, d_coeffs)
        
        # Compare cost at z=0 vs cost at local_minima
        # Cost function: E(z) = tau * Geman(z) + 0.5 * (z - |x|)^2
        # Geman(0) = 0, so cost at z=0 is 0.5 * |x|^2
        cost_at_zero: np.ndarray = 0.5 * abs_x[idx_roots]**2
        cost_at_loc_mins: np.ndarray = tau * self.elementwise(loc_mins) + \
                                  0.5 * (loc_mins - abs_x[idx_roots])**2
        
        # global_min_idx identifies which of the local_mins are actual global minima
        # by comparing their cost to the cost at z=0.
        global_min_idx: np.ndarray = cost_at_loc_mins < cost_at_zero
        
        # Update the output only for those elements where a valid local minimum
        # (which is also a global minimum compared to z=0) was found.
        # First, filter idx_roots based on global_min_idx
        final_update_indices: np.ndarray = np.zeros_like(x, dtype=bool)
        final_update_indices[idx_roots] = global_min_idx
        
        # Apply the sign of original x to the positive local minima
        out[final_update_indices] = np.sign(x[final_update_indices]) * loc_mins[global_min_idx]
        
        return out

    @staticmethod
    def _find_local_minima(b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Cardano's method for cubic equation: z^3 + bz^2 + cz + d = 0
        # This method finds real roots for the depressed cubic q(y) = y^3 + py + q = 0
        # where y = z + b/3.
        # p = c - b^2/3
        # q_cardano = (2b^3 - 9bc + 27d)/27
        # The f and g in original code are related to discriminant components.
        # f = -( (c - b^2/3)^3 ) / 27  (related to p^3/27)
        # g = (2b^3 - 9bc + 27d) / 27 (this is q_cardano)
        
        # Discriminant related terms (as in original code, but with clearer names)
        # Q_cubic = (b**2 - 3*c) / 9
        # R_cubic = (2*b**3 - 9*b*c + 27*d) / 54
        # discriminant_cubic = Q_cubic**3 - R_cubic**2
        # The original f and g are slightly different but achieve a similar check for num roots.
        
        # Using the original formulation for f and g for consistency:
        f_term: np.ndarray = -(c - b**2.0 / 3.0)**3.0 / 27.0
        g_term: np.ndarray = (2.0 * b**3.0 - 9.0 * b*c + 27.0 * d) / 27.0
        
        # Condition for three real roots (or multiple roots) for the depressed cubic
        # This corresponds to g_term^2 / 4 - f_term <= 0  (or R_cubic^2 - Q_cubic^3 <= 0)
        # Note: original code uses f, which is -Q_cubic^3. So g_term^2/4 + f_term <=0
        idx_three_real_roots: np.ndarray = g_term**2.0 / 4.0 + f_term <= 0
        
        # Only proceed for cases with three real roots, as we are looking for positive minima
        # from the Geman derivative which is a cubic.
        # Other cases (one real root) might not correspond to the positive solution we need or
        # might be handled by z=0 if that's the true minimum.
        
        b_filt: np.ndarray = b[idx_three_real_roots]
        g_filt: np.ndarray = g_term[idx_three_real_roots]
        f_filt: np.ndarray = f_term[idx_three_real_roots]

        # Avoid division by zero or sqrt of negative if f_filt is not strictly positive
        # For three real roots, f_term should be positive (as Q_cubic^3 > R_cubic^2 and f = -Q_cubic^3/27)
        # However, due to floating point, ensure sqrtf_val is on positive values.
        # The condition idx_three_real_roots already implies f_filt will be >= g_filt^2/4,
        # and for distinct real roots, f_filt > 0.
        # Let's assume f_filt is positive where idx_three_real_roots is true.
        sqrt_f_filt: np.ndarray = np.sqrt(np.abs(f_filt)) # abs for safety, though should be positive
        
        # Angle k for trigonometric solution
        # Argument to arccos must be in [-1, 1].
        # Clip for numerical stability, though g_filt^2/(4*f_filt) should be <= 1
        arccos_arg: np.ndarray = np.clip(-(g_filt / (2 * sqrt_f_filt + 1e-16)), -1.0, 1.0)
        k_angle: np.ndarray = np.arccos(arccos_arg)
        
        # Roots of the depressed cubic y^3 + py + q = 0 are:
        # y_k = 2 * sqrt(-p/3) * cos((arccos( (3q/2p)*sqrt(-3/p) ) - 2*pi*k_idx)/3) for k_idx=0,1,2
        # The original code's loc_mins formula corresponds to one of these roots
        # transformed back by z = y - b/3.
        # It specifically targets one root via cos(k_angle/3).
        # The Geman paper or its derivations usually show one specific root form
        # that corresponds to the positive solution z > 0.
        
        # loc_mins are solutions for z, from y = z + b/3
        loc_mins_found: np.ndarray = 2 * sqrt_f_filt**(1/3.0) * np.cos(k_angle / 3.0) - b_filt / 3.0
        
        # We are interested in positive solutions z > 0
        positive_loc_mins_mask: np.ndarray = loc_mins_found > 0
        
        # Filter loc_mins and corresponding original indices
        final_loc_mins: np.ndarray = loc_mins_found[positive_loc_mins_mask]
        
        # Update idx_three_real_roots to reflect only those that resulted in positive local minima
        idx_final_positive_mins = np.zeros_like(idx_three_real_roots, dtype=bool)
        temp_idx_for_filtering = np.where(idx_three_real_roots)[0] # Get integer indices from boolean mask
        idx_final_positive_mins[temp_idx_for_filtering[positive_loc_mins_mask]] = True
        
        return idx_final_positive_mins, final_loc_mins
