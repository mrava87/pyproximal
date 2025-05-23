import numpy as np
from pyproximal import ProxOperator # Keep this for ProxOperator base class
from pyproximal.ProxOperator import _check_tau # For the decorator
from typing import Any # Any for Optional Op in super().__init__


class L21_plus_L1(ProxOperator):
    r"""L21 + L1 norm proximal operator.

    Proximal operator of the :math:`L_{2,1} + L_1` mixed-norm:
    :math:`f(\mathbf{X}) = \sigma \rho \|\mathbf{X}\|_1 +
    \sigma (1 - \rho) \|\mathbf{X}\|_{2,1}`

    Parameters
    ----------
    sigma : :obj:`int`, optional
        Multiplicative coefficient of :math:`L_{2,1} + L_1` mixed-norm
    rho : :obj:`int`, optional
        Balancing between sparsity of :math:`L_1` and grouping of :math:`L_{2,1}`

    Notes
    -----
    The proximal operator of the :math:`L_{2,1} + L_1` mixed-norm is simply the
    product of each individual proximal operator [1]_.

    .. [1] Gramfort, Alexandre, Daniel Strohmeier, Jens Haueisen, Matti Hamalainen,
        and Matthieu Kowalski. "Functional brain imaging with M/EEG using structured
        sparsity in time-frequency dictionaries." In Biennial International Conference
        on Information Processing in Medical Imaging, pp. 600-611. Springer, Berlin,
        Heidelberg, 2011.
    """

    def __init__(self, sigma: float = 1.0, rho: float = 0.8):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.sigma: float = sigma
        self.rho: float = rho

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar
        # Assuming x is a 2D array for axis=0 in the L21 part
        # L1 part: rho * sigma * sum(|x_ij|)
        l1_term: float = self.rho * self.sigma * np.sum(np.abs(x))
        # L21 part: (1-rho) * sigma * sum_j (sqrt(sum_i x_ij^2))
        # np.sum(x**2, axis=0) gives sum of squares along columns
        # np.sqrt(...) gives L2 norm of columns
        # np.sum(...) sums these L2 norms
        l21_term: float = (1 - self.rho) * self.sigma * np.sum(np.sqrt(np.sum(x**2, axis=0)))
        return l1_term + l21_term

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, axis: int = 0) -> np.ndarray:
        thresh: float = self.sigma * tau
        
        # Soft thresholding part (L1)
        # l1_soft_thresh = sign(x) * max(0, |x| - thresh * rho)
        abs_x: np.ndarray = np.abs(x)
        l1_after_soft_thresh: np.ndarray = np.maximum(abs_x - thresh * self.rho, 0)
        
        # Group shrinkage part (L21)
        # This part operates on the result of L1 soft thresholding.
        # aux_l21 is the L2 norm of columns (or rows if axis=1) of the L1-thresholded result.
        # Ensure squared before sum for L2 norm calculation.
        l2_norm_of_l1_thresh_cols: np.ndarray = np.sqrt(np.sum(l1_after_soft_thresh**2, axis=axis))
        
        # Factor for group shrinkage: max(0, 1 - thresh * (1-rho) / ||(L1_thresh_cols)_j||_2 )
        # Add epsilon to denominator to avoid division by zero if a column norm is zero.
        group_shrinkage_factor: np.ndarray = np.maximum(
            0, 1 - thresh * (1 - self.rho) / (l2_norm_of_l1_thresh_cols + 1e-16)
        )
        
        # The final result combines both effects.
        # The term x / np.abs(x) is sign(x) for real x, or phase for complex x.
        # It's applied to the L1-thresholded values, which are then scaled by group_shrinkage_factor.
        # Need to handle shape of group_shrinkage_factor for broadcasting with l1_after_soft_thresh.
        # If axis=0, l1_after_soft_thresh is (N_dim, N_x), group_shrinkage_factor is (N_x,).
        # Need to expand dims of group_shrinkage_factor to (1, N_x) for broadcasting.
        if axis == 0 and x.ndim > 1 and group_shrinkage_factor.ndim == 1: # Ensure it's for matrix case
             group_shrinkage_factor = np.expand_dims(group_shrinkage_factor, axis=0)
        elif axis == 1 and x.ndim > 1 and group_shrinkage_factor.ndim == 1:
             group_shrinkage_factor = np.expand_dims(group_shrinkage_factor, axis=1)
        # If x is 1D, axis effectively doesn't change much, l1_after_soft_thresh is 1D,
        # group_shrinkage_factor will be scalar after sum/sqrt if not careful.
        # The original code implies x is likely a 2D matrix.
        
        # sign_x_or_phase handles both real and complex cases for x/|x|
        sign_x_or_phase: np.ndarray = np.zeros_like(x, dtype=x.dtype)
        non_zero_mask: np.ndarray = abs_x > 1e-16 # Avoid division by zero for sign/phase
        sign_x_or_phase[non_zero_mask] = x[non_zero_mask] / abs_x[non_zero_mask]
        
        x_prox: np.ndarray = sign_x_or_phase * l1_after_soft_thresh * group_shrinkage_factor
        
        # np.nan_to_num was in original, implies some divisions might result in nan.
        # This is handled by adding epsilon to denominators.
        # If x_prox can still have NaNs (e.g. if abs_x was zero leading to 0/0 for sign),
        # then nan_to_num is a safeguard.
        # With epsilon, 0/0 for sign is avoided. If l1_after_soft_thresh is 0, result is 0.
        return np.nan_to_num(x_prox)
