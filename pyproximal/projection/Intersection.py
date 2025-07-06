import numpy as np
from typing import Union


class IntersectionProj:
    r"""Intersection of multiple convex sets

    Parameters
    ----------
    k : :obj:`int`
        Size of vector to be projected
    n : :obj:`int`
        Number of vectors to be projected simultaneously
    sigma : :obj:`numpy.ndarray`
        Matrix of distances of size :math:`k \times k`
    k : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Tolerance of update

    Notes
    -----
    Given an Intersection of simple sets defined as:

    .. math::

        K = \bigcap_{1 \leq i_1 < i_2 \leq k} K_{i_1,i_2}, \quad
        K_{i_1,i_2}= \{ \mathbf{x}: |x_{i_2} - x_{i_1}| \leq \sigma_{i1, i2} \}

    its orthogonal projection can be obtained using the Dykstra's
    algorithm [1]_.

    .. [1] A., Chambolle, D., Cremers, and T., Pock, "A Convex Approach to
        Minimal Partitions", Journal of Mathematical, 2011.

    """
    def __init__(self, k: int, n: int, sigma: Union[float, np.ndarray],
                 niter: int = 100, tol: float = 1e-5):
        self.k: int = k
        self.n: int = n
        if isinstance(sigma, np.ndarray):
            self.sigma: np.ndarray = sigma
        else: # sigma is float
            self.sigma = sigma * np.ones((k, k))
        self.niter: int = niter
        self.tol: float = tol

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_reshaped: np.ndarray = x.reshape(self.k, self.n)
        # x12 stores the difference between components, initialize with zeros
        x12: np.ndarray = np.zeros((self.k, self.k, self.n))

        for iiter in range(self.niter):
            xold: np.ndarray = x_reshaped.copy()
            for i1 in range(self.k - 1):
                for i2 in range(i1 + 1, self.k):
                    xtilde: np.ndarray = x_reshaped[i2] - x_reshaped[i1] + x12[i1, i2]
                    xtildeabs: np.ndarray = np.abs(xtilde)
                    # Ensure no division by zero if xtildeabs is zero
                    # Add a small epsilon only where xtildeabs is zero to avoid changing other values
                    denominator: np.ndarray = xtildeabs + 1e-10 * (xtildeabs == 0)
                    xdtilde: np.ndarray = \
                        np.maximum(0, xtildeabs - self.sigma[i1, i2]) * \
                        xtilde / denominator
                    
                    update_val: np.ndarray = 0.5 * (xdtilde - x12[i1, i2])
                    x_reshaped[i1] = x_reshaped[i1] + update_val
                    x_reshaped[i2] = x_reshaped[i2] - update_val
                    x12[i1, i2] = xdtilde
            
            # Check for convergence
            # max() expects a single iterable, np.sum returns a scalar if axis is not None
            # or an array if axis is specified. Here sum over axis=0 gives per-column sum of abs diff.
            # Then max over these column sums.
            if np.max(np.sum(np.abs(x_reshaped - xold), axis=0)) < self.tol:
                break
        return x_reshaped.ravel()