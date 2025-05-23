import numpy as np
from typing import Tuple


class L0BallProj:
    r""":math:`L_0` ball projection.

    Parameters
    ----------
    radius : :obj:`int`
        Radius

    Notes
    -----
    Given an L0 ball defined as:

    .. math::

        L0_{r} = \{ \mathbf{x}: ||\mathbf{x}||_0 \leq r \}

    its orthogonal projection is computed by finding the :math:`r` highest
    largest entries of :math:`\mathbf{x}` (in absolute value), keeping those
    and zero-ing all the other entries.
    Note that this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{L0_{r}}`.

    """
    def __init__(self, radius: int):
        self.radius: int = int(radius)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xshape: Tuple[int, ...] = x.shape
        xf: np.ndarray = x.copy().flatten()
        # Ensure self.radius is not zero to avoid issues with [:-0] slicing if it means taking all elements
        if self.radius == 0:
            xf.fill(0)
        elif self.radius < len(xf): # Only apply argsort if radius is less than total elements
            indices_to_zero: np.ndarray = np.argsort(np.abs(xf))[:-self.radius]
            xf[indices_to_zero] = 0
        # If radius >= len(xf), all elements are kept, no change needed beyond copy.
        return xf.reshape(xshape)


class L01BallProj:
    r""":math:`L_{0,1}` ball projection.

    Parameters
    ----------
    radius : :obj:`int`
        Radius

    Notes
    -----
    Given an :math:`L_{0,1}` ball defined as:

    .. math::

        L_{0,1}^{r} =
        \{ \mathbf{x}: \text{count}([||\mathbf{x}_1||_1,
        ||\mathbf{x}_2||_1, ..., ||\mathbf{x}_1||_1] \ne 0) \leq r \}

    its orthogonal projection is computed by finding the :math:`r` highest
    largest entries of a vector obtained by applying the :math:`L_1` norm to each
    column of a matrix :math:`\mathbf{x}` (in absolute value), keeping those
    and zero-ing all the other entries.
    Note that this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{L_{0,1}^{r}}`.

    """
    def __init__(self, radius: int):
        self.radius: int = int(radius)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xc: np.ndarray = x.copy()
        # xf will be 1D array of L1 norms of columns
        xf: np.ndarray = np.linalg.norm(x, axis=0, ord=1)
        
        if self.radius == 0:
            xc.fill(0)
        elif self.radius < xf.shape[0]: # xf.shape[0] is the number of columns
            # Indices of columns to zero out
            indices_to_zero_cols: np.ndarray = np.argsort(np.abs(xf))[:-self.radius]
            xc[:, indices_to_zero_cols] = 0
        # If radius >= number of columns, all columns are kept.
        return xc