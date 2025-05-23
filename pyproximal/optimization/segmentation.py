import time
import numpy as np
from typing import Callable, Optional, Any, Tuple, Dict, Union # Added Union

from pylops import Gradient, BlockDiag
from pyproximal import Simplex, L1, L21, VStack
from pyproximal.optimization.primaldual import PrimalDual
from pyproximal.ProxOperator import ProxOperator # For potential future use if proxs are passed


def Segment(y: np.ndarray, cl: np.ndarray, sigma: float, alpha: float,
            clsigmas: Optional[np.ndarray] = None,
            z: Optional[np.ndarray] = None, niter: int = 10,
            x0: Optional[np.ndarray] = None,
            callback: Optional[Callable[[np.ndarray], None]] = None,
            show: bool = False,
            kwargs_simplex: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    r"""Primal-dual algorithm for image segmentation

    Perform image segmentation over :math:`N_{cl}` classes using the
    general version of the first-order primal-dual algorithm [1]_.

    Parameters
    ----------
    y : :obj:`np.ndarray`
        Image to segment (must have 2 or more dimensions)
    cl : :obj:`numpy.ndarray`
        Classes
    sigma : :obj:`float`
        Positive scalar weight of the misfit term
    alpha : :obj:`float`
        Positive scalar weight of the regularization term
    clsigmas : :obj:`numpy.ndarray`, optional
        Classes standard deviations
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    x0 : :obj:`numpy.ndarray`, optional
        Initial vector
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    kwargs_simplex : :obj:`dict`, optional
        Arbitrary keyword arguments for
        :py:func:`pyproximal.Simplex` operator

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Classes probabilities. This is a vector of size :math:`N_{dim} \times
        N_{cl}` whose columns contain the probability for each pixel to be in
        the class :math:`c_i`
    cl : :obj:`numpy.ndarray`
        Estimated classes. This is a vector of the same size of the input data
        ``y`` with the selected classes at each pixel.

    Notes
    -----
    This solver performs image segmentation over :math:`N_{cl}` classes solving
    the following nonlinear minimization problem using the general version of
    the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} \frac{\sigma}{2} \mathbf{x}^T \mathbf{f} +
        \mathbf{x}^T \mathbf{z} + \frac{\alpha}{2}||\nabla \mathbf{x}||_{2,1}

    where :math:`X=\{ \mathbf{x}: \sum_{i=1}^{N_{cl}} x_i = 1,\; x_i \geq 0 \}`
    is a simplex and :math:`\mathbf{f}=[\mathbf{f}_1, ...,
    \mathbf{f}_{N_{cl}}]^T` with :math:`\mathbf{f}_i = |\mathbf{y}-c_i|^2/\sigma_i`.
    Here :math:`\mathbf{c}=[c_1, ..., c_{N_{cl}}]^T` and
    :math:`\mathbf{\sigma}=[\sigma_1, ..., \sigma_{N_{cl}}]^T` are vectors
    representing the optimal mean and standard deviations for each class.

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    current_kwargs_simplex: Dict[str, Any] = {} if kwargs_simplex is None else kwargs_simplex

    dims: Tuple[int, ...] = y.shape
    ndims: int = len(dims)
    dimsprod: int = np.prod(np.array(dims)).item() # Ensure scalar int
    ncl: int = len(cl)

    # Data (difference between image and center of classes)
    g_data: np.ndarray = sigma / 2. * (y.reshape(1, dimsprod) - cl[:, np.newaxis]) ** 2
    if clsigmas is not None:
        g_data /= clsigmas[:, np.newaxis]
    g_data = g_data.ravel()

    # Gradient operator
    sampling: float = 1.
    # Assuming Gradient and BlockDiag are correctly typed or can be treated as Any for now
    Gop: Any = Gradient(dims=dims, sampling=sampling, edge=False, # type: ignore
                       kind='forward', dtype='float64')
    Gop = BlockDiag([Gop] * ncl) # type: ignore

    # Simplex and L1 proximal operators
    simp: Simplex = Simplex(dimsprod * ncl, radius=1, dims=(ncl, dimsprod), axis=0, # type: ignore
                            **current_kwargs_simplex)
    #l1 = L1(sigma=0.5 * alpha)
    l21_op: VStack = VStack([L21(ndim=ndims, sigma=0.5 * alpha)] * ncl, # type: ignore
                            nn=[ndims * dimsprod] * ncl)

    # Steps
    L_const: float = 8. / sampling ** 2
    tau_step: float = 1.
    mu_step: float = 1. / (tau_step * L_const)

    # Inversion
    # PrimalDual returns Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], handle accordingly
    x_pd_result: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = \
        PrimalDual(simp, l21_op, Gop,
                   x0=np.zeros_like(g_data) if x0 is None else x0,
                   tau=tau_step, mu=mu_step,
                   z=g_data if z is None else g_data + z, theta=1.,
                   niter=niter, callback=callback, show=show)

    x_seg: np.ndarray
    if isinstance(x_pd_result, tuple): # if returny=True was passed implicitly or by mistake
        x_seg = x_pd_result[0]
    else:
        x_seg = x_pd_result

    x_reshaped: np.ndarray = x_seg.reshape(ncl, dimsprod).T
    cl_estimated: np.ndarray = np.argmax(x_reshaped, axis=1)
    cl_reshaped: np.ndarray = cl_estimated.reshape(dims)

    return x_reshaped, cl_reshaped
