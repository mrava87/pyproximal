import numpy as np
from typing import Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator


class L21(ProxOperator):
    r""":math:`L_{2,1}` proximal operator.

    Proximal operator for :math:`L_{2,1}` matrix norm.

    Parameters
    ----------
    ndim : :obj:`int`
        Number of dimensions :math:`N_{dim}`. Used to reshape the input array
        in a matrix of size :math:`N_{dim} \times N'_{x}` where
        :math:`N'_x = \frac{N_x}{N_{dim}}`. Note that the input
        vector ``x`` should be created by stacking vectors from different
        dimensions.
    sigma : :obj:`float`, optional
        Multiplicative coefficient of :math:`L_{2,1}` norm

    Notes
    -----
    Given the :math:`L_{2,1}` norm of a matrix of size
    :math:`N_{dim} \times N'_x` defined as:

    .. math::

        \sigma \|\mathbf{X}\|_{2,1} = \sigma \sum_{j=0}^{N'_x} \|\mathbf{x}_j\|_2 =
        \sigma \sum_{j=0}^{N'_x} \sqrt{\sum_{i=0}^{N_{dim}}} |x_{ij}|^2

    the proximal operator is:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_{2,1}}(\mathbf{x}_j) =
        \left(1 - \frac{\sigma \tau}{max\{||\mathbf{x}_j||_2,
        \sigma \tau \}}\right) \mathbf{x}_j \quad \forall j

    Similar to the Euclidean norm, the dual operator is defined as:

    .. math::

        \prox^*_{\tau \sigma \||\cdot\|_{2,1}}(\mathbf{x}_j) =
        \frac{\sigma \mathbf{x}_j}{\max\{||\mathbf{x}_j||_2, \sigma\}}
        \quad \forall j

    Finally, we note that the :math:`L_{2,1}` norm is a separable function
    on each column on the matrix :math:`\mathbf{X}`. Taking advantage of the
    property of proximal operator of separable function [1]_, its proximal and
    dual proximal operators can be interpreted as a series of
    :class:`pyproximal.proximal.Euclidean` operators on each column
    of the matrix :math:`\mathbf{X}`.

    .. [1] N., Parikh, "Proximal Algorithms", Foundations and Trends
        in Optimization. 2013.

    """
    def __init__(self, ndim: int, sigma: float = 1.):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.ndim: int = ndim
        self.sigma: float = sigma

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar sum
        # Assuming x is 1D and its length is a multiple of self.ndim
        x_reshaped: np.ndarray = x.reshape(self.ndim, len(x) // self.ndim)
        # Sum of L2 norms of columns
        f_val: float = self.sigma * np.sum(np.sqrt(np.sum(x_reshaped**2, axis=0)))
        return f_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        x_reshaped: np.ndarray = x.reshape(self.ndim, len(x) // self.ndim)
        # L2 norm of each column
        col_norms: np.ndarray = np.sqrt(np.sum(x_reshaped**2, axis=0))
        
        # Threshold for shrinkage: tau * sigma
        # Max term for denominator: max(||x_j||_2, tau * sigma)
        # To do this element-wise for each column, we need to compare col_norms with tau * sigma.
        denominator: np.ndarray = np.maximum(col_norms, tau * self.sigma)
        
        # Avoid division by zero if a column norm and tau*sigma are both zero
        # If denominator is zero for a column, that column must be zero, and prox is zero for that column.
        # The multiplication (1 - factor) * x_j handles this naturally if x_j is zero.
        # If x_j is non-zero but denominator is zero (e.g. tau*sigma=0 and col_norm=0),
        # this implies x_j was zero, so it remains zero.
        # The only problematic case is if denominator is zero but col_norms was non-zero,
        # which cannot happen with np.maximum if tau*sigma >= 0.
        
        # Factor for shrinkage, needs to be applied column-wise
        # (1 - (tau * sigma) / denominator_j)
        # We need to be careful with shapes for broadcasting.
        # col_norms and denominator are 1D (shape N'_x). x_reshaped is (N_dim, N'_x).
        # We want to scale each column x_j by (1 - factor_j).
        
        # Create a scaling factor for each column
        # Add a small epsilon to denominator to prevent division by zero in edge cases
        # where a column and tau*sigma might be zero, though np.maximum should handle it.
        scaling_factor_per_column: np.ndarray = (1 - (tau * self.sigma) / (denominator + 1e-16))
        
        # Apply scaling to each column
        # x_reshaped * scaling_factor_per_column will broadcast scaling_factor_per_column
        # along axis 0 of x_reshaped.
        x_prox_reshaped: np.ndarray = x_reshaped * scaling_factor_per_column
        
        return x_prox_reshaped.ravel()

    @_check_tau
    def proxdual(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in this specific proxdual formula
        x_reshaped: np.ndarray = x.reshape(self.ndim, len(x) // self.ndim)
        col_norms: np.ndarray = np.sqrt(np.sum(x_reshaped**2, axis=0))
        
        denominator: np.ndarray = np.maximum(col_norms, self.sigma)
        
        # Similar broadcasting as in prox
        scaling_factor_per_column: np.ndarray = self.sigma / (denominator + 1e-16)
        
        x_proxdual_reshaped: np.ndarray = x_reshaped * scaling_factor_per_column
        return x_proxdual_reshaped.ravel()