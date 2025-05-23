import numpy as np
from typing import Union


class EuclideanBallProj:
    r"""Euclidean ball projection.

    Parameters
    ----------
    center : :obj:`np.ndarray` or :obj:`float`
        Center of the ball
    radius : :obj:`float`
        Radius

    Notes
    -----
    Given an Euclidean ball defined as:

    .. math::

        \operatorname{Eucl}_{[c, r]} = \{ \mathbf{x}: l ||\mathbf{x} - \mathbf{c}||_2 \leq r \}

    its orthogonal projection is:

    .. math::

        P_{\operatorname{Eucl}_{[c, r]}} (\mathbf{x}) = \mathbf{c} + \frac{r}
        {max\{ ||\mathbf{x} - \mathbf{c}||_2^2, r\}}(\mathbf{x} - \mathbf{c})

    Note the this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{\operatorname{Eucl}_{[c, r]}}`.

    """
    def __init__(self, center: Union[np.ndarray, float], radius: float):
        self.center: Union[np.ndarray, float] = center
        self.radius: float = radius

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Ensure correct types for arithmetic operations
        center_arr: np.ndarray
        if isinstance(self.center, float):
            # If center is a float, x must be a scalar or np.array that can be operated with float.
            # For safety, and to match typical use with np.linalg.norm, convert center to array if x is array.
            if isinstance(x, np.ndarray):
                center_arr = np.full_like(x, self.center)
            else: # x is scalar, center is float
                center_arr = np.array(self.center) # Keep as scalar np.array
        else: # center is np.ndarray
            center_arr = self.center

        diff: np.ndarray = x - center_arr
        norm_diff: float = float(np.linalg.norm(diff)) # Ensure norm_diff is Python float
        
        # max function expects float arguments
        denominator: float = float(max(norm_diff, self.radius)) # Explicitly cast result to float
        
        # Ensure no division by zero if radius and norm_diff are both zero
        if denominator == 0:
            # If radius is 0 and x is at the center, projection is the center.
            # If radius > 0 and x is at the center, projection is the center.
            # If radius = 0 and x is not at the center, this case implies norm_diff > 0,
            # so denominator would be norm_diff. This if block handles norm_diff = 0.
            # If norm_diff is zero, diff is a zero vector.
            # The result is center_arr + 0 = center_arr.
            # If x was scalar, it would return a scalar np.array(self.center).
            return center_arr.copy() if isinstance(center_arr, np.ndarray) else np.array(center_arr)


        x_proj: np.ndarray = center_arr + (self.radius / denominator) * diff
        return x_proj
