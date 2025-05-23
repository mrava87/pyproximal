import numpy as np
from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.Euclidean import EuclideanBallProj # Ensure correct import path
from typing import Union, Any # Any for Optional Op in super().__init__


class Euclidean(ProxOperator):
    r"""Euclidean norm proximal operator.

    Proximal operator of the Euclidean norm: :math:`\sigma \|\mathbf{x}\|_2 =
    \sigma \sqrt{\sum x_i^2}`.

    Parameters
    ----------
    sigma : :obj:`int`, optional
        Multiplicative coefficient of :math:`L_{2}` norm

    Notes
    -----
    The Euclidean proximal operator is defined as:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_2}(\mathbf{x}) =
        \left(1 - \frac{\tau \sigma }{\max\{\|\mathbf{x}\|_2,
        \tau \sigma \}}\right) \mathbf{x}

    This operator is sometimes called *block soft thresholding*.

    Moreover, as the conjugate of the Euclidean norm is the orthogonal
    projection of its dual norm (i.e., Euclidean norm) onto a unit ball,
    its dual operator is defined as:

    .. math::

        \prox^*_{\tau \sigma \|\cdot\|_2}(\mathbf{x}) =
        \frac{\sigma \mathbf{x}}{\max\{\|\mathbf{x}\|_2, \sigma\}}

    """
    def __init__(self, sigma: float = 1.):
        super().__init__(None, True) # Op is None, hasgrad is True
        self.sigma: float = sigma

    def __call__(self, x: np.ndarray) -> float:
        return self.sigma * float(np.linalg.norm(x))

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        norm_x: float = float(np.linalg.norm(x)) # Cast to float
        # Ensure no division by zero if norm_x and tau * sigma are both zero
        denominator: float = float(max(norm_x, tau * self.sigma)) # Cast to float
        if denominator == 0:
            return x # if x is zero vector and tau*sigma is zero, prox is x itself
        
        x_prox: np.ndarray = (1. - (tau * self.sigma) / denominator) * x
        return x_prox

    @_check_tau
    def proxdual(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in this specific proxdual formula, but part of signature
        norm_x: float = float(np.linalg.norm(x)) # Cast to float
        # Ensure no division by zero if norm_x and sigma are both zero
        denominator: float = float(max(norm_x, self.sigma)) # Cast to float
        if denominator == 0:
            return x # if x is zero vector and sigma is zero, proxdual is x itself

        x_proxdual: np.ndarray = (self.sigma / denominator) * x
        return x_proxdual

    def grad(self, x: np.ndarray) -> np.ndarray:
        norm_x: float = float(np.linalg.norm(x)) # Cast to float
        if norm_x == 0:
            # Gradient is not defined at x=0.
            # Depending on convention, could return zero vector or raise error.
            # Following original logic, likely implies x!=0 or handled by caller.
            # For safety, return zero vector of same shape if norm_x is 0.
            return np.zeros_like(x)
        return self.sigma * x / norm_x


class EuclideanBall(ProxOperator):
    r"""Euclidean ball proximal operator.

    Proximal operator of the Euclidean ball: :math:`Eucl_{[c, r]} =
    \{ \mathbf{x}: ||\mathbf{x} - \mathbf{c}||_2 \leq r \}`.

    Parameters
    ----------
    center : :obj:`np.ndarray` or :obj:`float`
        Center of the ball
    radius : :obj:`float`
        Radius

    Notes
    -----
    As the Euclidean ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.EuclideanBallProj` for details.

    """
    def __init__(self, center: Union[np.ndarray, float], radius: float):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.center: Union[np.ndarray, float] = center
        self.radius: float = radius
        self.ball: EuclideanBallProj = EuclideanBallProj(self.center, self.radius)

    def __call__(self, x: np.ndarray) -> bool:
        # Ensure center is compatible with x for subtraction
        center_val: Union[np.ndarray, float]
        if isinstance(self.center, float) and isinstance(x, np.ndarray):
            # If x is an array and center is float, norm calculation needs compatible types.
            # Assuming x - self.center works by broadcasting self.center.
            center_val = self.center
        elif isinstance(self.center, np.ndarray) and not isinstance(x, np.ndarray): # x is scalar
             # This case is less common for vector norms but handle for completeness if x can be scalar.
            center_val = self.center # This might lead to issues if center is array and x scalar.
                                     # However, x is typed as np.ndarray.
        else: # Both are np.ndarray or both are float (though x is np.ndarray)
            center_val = self.center
        
        # np.linalg.norm can take scalar or array for x and center (if broadcastable)
        return bool(np.linalg.norm(x - center_val) <= self.radius)


    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        return self.ball(x)
