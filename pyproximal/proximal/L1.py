from typing import Union, Callable, Any, List, Optional

import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import BoxProj, L1BallProj
from pyproximal import ProxOperator

# types for sigma
SigmaType = Union[float, np.ndarray, Callable[[int], float]]


def _softthreshold(x: np.ndarray, thresh: Union[float, np.ndarray]) -> np.ndarray:
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x``. If thresh is an array,
    the last dimension of ``x`` must match the size of ``thresh``.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float` or :obj:`numpy.ndarray`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    x1: np.ndarray
    if np.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.exp(1j * np.angle(x))
    else:
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.sign(x)

    return x1


def _current_sigma(sigma: SigmaType, count: int) -> Union[float, np.ndarray]:
    if not callable(sigma):
        return sigma
    else:
        return sigma(count)


class L1(ProxOperator):
    r"""L1 norm proximal operator.

    Proximal operator of the :math:`\ell_1` norm:
    :math:`\sigma\|\mathbf{x} - \mathbf{g}\|_1 = \sigma \sum |x_i - g_i|`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`numpy.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L1 norm. This can be a constant number, an array
        of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``sigma`` to be used.
    g : :obj:`numpy.ndarray`, optional
        Vector to be subtracted

    Notes
    -----
    The :math:`\ell_1` proximal operator is defined as [1]_:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_1}(\mathbf{x}) =
        \operatorname{soft}(\mathbf{x}, \tau \sigma) =
        \begin{cases}
        x_i + \tau \sigma, & x_i - g_i < -\tau \sigma \\
        g_i, & -\tau\sigma \leq x_i - g_i \leq \tau\sigma \\
        x_i - \tau\sigma,  & x_i - g_i > \tau\sigma\\
        \end{cases}

    where :math:`\operatorname{soft}` is the so-called called *soft thresholding*.

    Moreover, as the conjugate of the :math:`\ell_1` norm is the orthogonal projection of
    its dual norm (i.e., :math:`\ell_\inf` norm) onto a unit ball, its dual
    operator (when :math:`\mathbf{g}=\mathbf{0}`) is defined as:

    .. math::

        \prox^*_{\tau \sigma \|\cdot\|_1}(\mathbf{x}) = P_{\|\cdot\|_{\infty} <=\sigma}(\mathbf{x}) =
        \begin{cases}
        -\sigma, & x_i < -\sigma \\
        x_i,& -\sigma \leq x_i \leq \sigma \\
        \sigma,  & x_i > \sigma\\
        \end{cases}

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    def __init__(self, sigma: SigmaType = 1., g: Optional[np.ndarray] = None):
        super().__init__(None, False)
        self.sigma = sigma
        self.g = g
        self.gdual = 0 if g is None else g

        self.count = 0

    def __call__(self, x: np.ndarray) -> float:
        sigma = _current_sigma(self.sigma, self.count)
        return sigma * np.sum(np.abs(x if self.g is None else x - self.g))
    
    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self, *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        sigma = _current_sigma(self.sigma, self.count)

        if self.g is None:
            x = _softthreshold(x, tau * sigma)
        else:
            # use precomposition property
            x = _softthreshold(x - self.g, tau * sigma) + self.g
        return x

    @_increment_count
    @_check_tau
    def proxdual(self, x: np.ndarray, tau: float) -> np.ndarray:
        if not isinstance(self.gdual, np.ndarray):
            sigma = _current_sigma(self.sigma, self.count)
            box = BoxProj(-sigma, sigma)
            x = box(x)
        else:
            x = self._proxdual_moreau(x, tau)
        return x


class L1Ball(ProxOperator):
    r"""L1 ball proximal operator.

    Proximal operator of the :math:`\ell_1` ball: :math:`L1_{r} =
    \{ \mathbf{x}: \|\mathbf{x}\|_1 \leq r \}`.

    Parameters
    ----------
    n : :obj:`int`
        Number of elements of input vector
    radius : :obj:`float`
        Radius
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by :func:`scipy.optimize.bisect`
    xtol : :obj:`float`, optional
        Absolute tolerance of :func:`scipy.optimize.bisect`

    Notes
    -----
    As the L1 ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.L1BallProj` for details.

    """
    def __init__(self, n, radius, maxiter=100, xtol=1e-5):
        super().__init__(None, False)
        self.n = n
        self.radius = radius
        self.maxiter = maxiter
        self.xtol = xtol
        self.ball = L1BallProj(self.n, self.radius, self.maxiter, self.xtol)

    def __call__(self, x, tol=1e-4):
        return np.sum(np.abs(x)) - self.radius < tol

    @_check_tau
    def prox(self, x, tau):
        return self.ball(x)