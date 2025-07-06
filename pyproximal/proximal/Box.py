import numpy as np
from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.Box import BoxProj # Ensure correct import path
from typing import Union, Any, Tuple # Added Tuple just in case, though not directly for these errors


class Box(ProxOperator):
    r"""Box proximal operator.

    Proximal operator of a Box: :math:`\operatorname{Box}_{[l, u]} = \{ x: l \leq x\leq u \}`.

    Parameters
    ----------
    lower : :obj:`float` or :obj:`numpy.ndarray`, optional
        Lower bound
    upper : :obj:`float` or :obj:`numpy.ndarray`, optional
        Upper bound

    Notes
    -----
    As the Box is an indicator function, the proximal operator corresponds to
    its orthogonal projection (see :class:`pyproximal.projection.BoxProj` for
    details.

    """
    def __init__(self, lower: Union[float, np.ndarray] = -np.inf,
                 upper: Union[float, np.ndarray] = np.inf):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.lower: Union[float, np.ndarray] = lower
        self.upper: Union[float, np.ndarray] = upper
        self.box: BoxProj = BoxProj(self.lower, self.upper)

    def __call__(self, x: np.ndarray) -> float: # Changed return type to float
        # For indicator function, it should be 0.0 if in set, np.inf otherwise.
        # np.all can return np.bool_, ensure this is converted to Python bool for the if condition.
        in_set: bool = bool(np.all((x >= self.lower) & (x <= self.upper))) # Inclusive bounds for a box
        if in_set:
            return 0.0
        else:
            return np.inf


    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        return self.box(x)