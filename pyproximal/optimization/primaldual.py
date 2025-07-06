from typing import TYPE_CHECKING, Callable, Optional, Any, Union, Tuple

import time
import numpy as np

from pylops.utils.backend import get_array_module, to_numpy
from pyproximal.ProxOperator import ProxOperator

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


def PrimalDual(proxf: ProxOperator, proxg: ProxOperator, A: "LinearOperator",
               x0: np.ndarray, tau: Union[float, np.ndarray],
               mu: Union[float, np.ndarray], y0: Optional[np.ndarray] = None,
               z: Optional[np.ndarray] = None, theta: float = 1.,
               niter: int = 10, gfirst: bool = True,
               callback: Optional[Callable[..., None]] = None, # Can be callback(x) or callback(x,y)
               callbacky: bool = False, returny: bool = False,
               show: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Primal-dual algorithm

    Solves the following (possibly) nonlinear minimization problem using
    the general version of the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} g(\mathbf{Ax}) + f(\mathbf{x}) +
        \mathbf{z}^T \mathbf{x}

    where :math:`\mathbf{A}` is a linear operator, :math:`f`
    and :math:`g` can be any convex functions that have a known proximal
    operator.

    This functional is effectively minimized by solving its equivalent
    primal-dual problem (primal in :math:`f`, dual in :math:`g`):

    .. math::

        \min_{\mathbf{x} \in X} \max_{\mathbf{y} \in Y}
        \mathbf{y}^T(\mathbf{Ax}) + \mathbf{z}^T \mathbf{x} +
        f(\mathbf{x}) - g^*(\mathbf{y})

    where :math:`\mathbf{y}` is the so-called dual variable.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`numpy.ndarray`
        Stepsize of subgradient of :math:`f`. This can be constant 
        or function of iterations (in the latter cases provided as np.ndarray)
    mu : :obj:`float` or :obj:`numpy.ndarray`
        Stepsize of subgradient of :math:`g^*`. This can be constant 
        or function of iterations (in the latter cases provided as np.ndarray)
    z0 : :obj:`numpy.ndarray`
        Initial auxiliary vector
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    theta : :obj:`float`
        Scalar between 0 and 1 that defines the update of the
        :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
        special case that represents the semi-implicit classical Arrow-Hurwicz
        algorithm
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbacky : :obj:`bool`, optional
        Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
    returny : :obj:`bool`, optional
        Return also ``y``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Primal-dual algorithm can be expressed by the following recursion
    (``gfirst=True``):

    .. math::

        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k})\\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k+1} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k)

    where :math:`\tau \mu \lambda_{max}(\mathbf{A}^H\mathbf{A}) < 1`.

    Alternatively for ``gfirst=False`` the scheme becomes:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k) \\
        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k+1})

    .. [1] A., Chambolle, and T., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120-145. 2011.

    """
    ncp = get_array_module(x0) # ncp will be numpy or cupy
    # Ensure tau and mu are arrays
    tau_arr: np.ndarray
    mu_arr: np.ndarray
    fixedtau: bool = False
    fixedmu: bool = False

    if isinstance(tau, (int, float)):
        tau_arr = tau * ncp.ones(niter, dtype=x0.dtype)
        fixedtau = True
    elif isinstance(tau, np.ndarray):
        tau_arr = tau
        if tau.size != niter: # Ensure it has correct size if array
             raise ValueError("tau array must have size niter")
    else:
        raise TypeError("tau must be float, int, or numpy.ndarray")

    if isinstance(mu, (int, float)):
        mu_arr = mu * ncp.ones(niter, dtype=x0.dtype)
        fixedmu = True
    elif isinstance(mu, np.ndarray):
        mu_arr = mu
        if mu.size != niter: # Ensure it has correct size if array
            raise ValueError("mu array must have size niter")
    else:
        raise TypeError("mu must be float, int, or numpy.ndarray")


    if show:
        tstart: float = time.time()
        print('Primal-dual: min_x f(Ax) + x^T z + g(x)\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'Additional vector (z): %s\n'
              'tau = %s\t\tmu = %s\ntheta = %.2f\t\tniter = %d\n' %
              (type(proxf), type(proxg), type(A),
               None if z is None else 'vector',
               str(tau_arr[0]) if fixedtau else 'Variable',
               str(mu_arr[0]) if fixedmu else 'Variable', theta, niter))
        head: str = '   Itn       x[0]          f           g          z^x       J = f + g + z^x'
        print(head)

    x: np.ndarray = x0.copy()
    xhat: np.ndarray = x.copy()
    y: np.ndarray = y0.copy() if y0 is not None else ncp.zeros(A.shape[0], dtype=x.dtype)
    ATy: np.ndarray # Define ATy type

    for iiter in range(niter):
        xold: np.ndarray = x.copy()
        current_tau: float = tau_arr[iiter]
        current_mu: float = mu_arr[iiter]

        if gfirst:
            y = proxg.proxdual(y + current_mu * A.matvec(xhat), current_mu)
            ATy = A.rmatvec(y)
            if z is not None:
                ATy = ATy + z # Ensure ATy is updated correctly
            x = proxf.prox(x - current_tau * ATy, current_tau)
            xhat = x + theta * (x - xold)
        else:
            ATy = A.rmatvec(y)
            if z is not None:
                ATy = ATy + z # Ensure ATy is updated correctly
            x = proxf.prox(x - current_tau * ATy, current_tau)
            xhat = x + theta * (x - xold)
            y = proxg.proxdual(y + current_mu * A.matvec(xhat), current_mu)

        # run callback
        if callback is not None:
            if callbacky:
                callback(x, y)
            else:
                callback(x)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: Union[float, bool] = proxf(x)
                pg_val: Union[float, bool] = proxg(A.matvec(x))
                pf_val = 0. if isinstance(pf_val, bool) else pf_val
                pg_val = 0. if isinstance(pg_val, bool) else pg_val
                zx_val: float = 0. if z is None else float(ncp.dot(z, x))
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e      %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])), pf_val, pg_val, zx_val,
                       pf_val + pg_val + zx_val)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    if not returny:
        return x
    else:
        return x, y


def AdaptivePrimalDual(proxf: ProxOperator, proxg: ProxOperator, A: "LinearOperator",
                       x0: np.ndarray, tau: float, mu: float,
                       alpha: float = 0.5, eta: float = 0.95, s: float = 1.,
                       delta: float = 1.5, z: Optional[np.ndarray] = None,
                       niter: int = 10, tol: float = 1e-10,
                       callback: Optional[Callable[[np.ndarray], None]] = None,
                       show: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""Adaptive Primal-dual algorithm

    Solves the minimization problem in
    :func:`pyproximal.optimization.primaldual.PrimalDual`
    using an adaptive version of the first-order primal-dual algorithm of [1]_.
    The main advantage of this method is that step sizes :math:`\tau` and
    :math:`\mu` are changing through iterations, improving the overall speed
    of convergence of the algorithm.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Stepsize of subgradient of :math:`f`
    mu : :obj:`float`
        Stepsize of subgradient of :math:`g^*`
    alpha : :obj:`float`, optional
        Initial adaptivity level (must be between 0 and 1)
    eta : :obj:`float`, optional
        Scaling of adaptivity level to be multipled to the current alpha every
        time the norm of the two residuals start to diverge (must be between
        0 and 1)
    s : :obj:`float`, optional
        Scaling of residual balancing principle
    delta : :obj:`float`, optional
        Balancing factor. Step sizes are updated only when their ratio exceeds
        this value.
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    tol : :obj:`int`, optional
        Tolerance on residual norms
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    steps : :obj:`tuple`
        Tau, mu and alpha evolution through iterations

    Notes
    -----
    The Adative Primal-dual algorithm share the the same iterations of the
    original :func:`pyproximal.optimization.primaldual.PrimalDual` solver.
    The main difference lies in the fact that the step sizes ``tau`` and ``mu``
    are adaptively changed at each iteration leading to faster converge.

    Changes are applied by tracking the norm of the primal and dual
    residuals. When their mutual ratio increases beyond a certain treshold
    ``delta`` the step lenghts are updated to balance the minimization and
    maximization part of the overall optimization process.

    .. [1] T., Goldstein, M., Li, X., Yuan, E., Esser, R., Baraniuk, "Adaptive
        Primal-Dual Hybrid Gradient Methods for Saddle-Point Problems",
        ArXiv, 2013.

    """
    current_tau: float = tau
    current_mu: float = mu
    current_alpha: float = alpha

    if show:
        tstart: float = time.time()
        print('Adaptive Primal-dual: min_x f(Ax) + x^T z + g(x)\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'Additional vector (z): %s\n'
              'tau0 = %10e\tmu0 = %10e\n'
              'alpha0 = %10e\teta = %10e\n'
              's = %10e\tdelta = %10e\n'
              'niter = %d\t\ttol = %10e\n' %
              (type(proxf), type(proxg), type(A),
               None if z is None else 'vector', current_tau, current_mu,
               current_alpha, eta, s, delta, niter, tol))
        head: str = '   Itn       x[0]          f           g          z^x       J = f + g + z^x'
        print(head)

    # initialization
    x: np.ndarray = x0.copy()
    y: np.ndarray = np.zeros(A.shape[0], dtype=x.dtype)
    Ax: np.ndarray = np.zeros(A.shape[0], dtype=x.dtype) # Initialized, will be updated before use
    ATy: np.ndarray = np.zeros(A.shape[1], dtype=x.dtype) # Initialized, will be updated

    taus_hist: np.ndarray = np.zeros(niter + 1)
    mus_hist: np.ndarray =  np.zeros(niter + 1)
    alphas_hist: np.ndarray = np.zeros(niter + 1)
    taus_hist[0], mus_hist[0], alphas_hist[0] = current_tau, current_mu, current_alpha
    p_res: float = tol + 1. # Primal residual norm
    d_res: float = tol + 1. # Dual residual norm

    iiter: int = 0
    while iiter < niter and p_res > tol and d_res > tol:

        # store old values
        xold: np.ndarray = x.copy()
        yold: np.ndarray = y.copy()
        Axold: np.ndarray = Ax.copy()
        ATyold: np.ndarray = ATy.copy() # ATy from previous iteration

        # proxf
        ATy_step: np.ndarray = ATyold # Use ATy from previous step for x update
        if z is not None:
            ATy_step = ATy_step + z
        x = proxf.prox(xold - current_tau * ATy_step, current_tau) # x is x_k in paper, xold is x_{k-1}
        Ax = A.matvec(x) # Ax is A(x_k)
        Axhat: np.ndarray = 2 * Ax - Axold # Axhat is A(2x_k - x_{k-1})

        # proxg
        y = proxg.proxdual(yold + current_mu * Axhat, current_mu) # y is y_k
        ATy = A.rmatvec(y) # ATy is A^T(y_k)

        # update steps
        # Paper notation for residuals:
        # p_k = (x_{k-1} - x_k)/tau_k - (A^T y_{k-1} - A^T y_k)  (if z is None)
        # d_k = (y_{k-1} - y_k)/mu_k - (A x_{k-1} - A x_k)
        # Here, xold is x_{k-1}, x is x_k. yold is y_{k-1}, y is y_k.
        # ATyold is A^T y_{k-1}, ATy is A^T y_k.
        # Axold is A x_{k-1}, Ax is A x_k.
        term_x_update: np.ndarray = (xold - x) / current_tau
        term_ATy_update: np.ndarray = (ATyold - ATy)
        if z is not None: # In paper, z is part of f, so its gradient is included in A^T y
                          # Here it's treated separately. The residual for f's prox is (x_prev - x_curr)/tau - grad_f_at_x_prev
                          # grad_f_at_x_prev effectively includes A^T y_prev + z.
                          # So the change in gradient part is (A^T y_prev + z) - (A^T y_curr + z) = A^T y_prev - A^T y_curr
            p_res = np.linalg.norm(term_x_update - term_ATy_update)
        else:
            p_res = np.linalg.norm(term_x_update - term_ATy_update)

        d_res = np.linalg.norm((yold - y) / current_mu - (Axold - Ax))


        if p_res > s * d_res * delta:
            current_tau /= (1 - current_alpha)
            current_mu *= (1 - current_alpha)
            current_alpha *= eta
        elif p_res < s * d_res / delta:
            current_tau *= (1 - current_alpha)
            current_mu /= (1 - current_alpha)
            current_alpha *= eta

        # save history of steps
        taus_hist[iiter + 1] = current_tau
        mus_hist[iiter + 1] = current_mu
        alphas_hist[iiter + 1] = current_alpha
        iiter += 1

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0 : # iiter is already incremented
                pf_val: Union[float, bool] = proxf(x)
                pg_val: Union[float, bool] = proxg(A.matvec(x)) # Use current Ax for cost
                pf_val = 0. if isinstance(pf_val, bool) else pf_val
                pg_val = 0. if isinstance(pg_val, bool) else pg_val
                zx_val: float = 0. if z is None else float(np.dot(z, x))
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e      %10.3e' % \
                      (iiter, np.real(to_numpy(x[0])), pf_val, pg_val, zx_val, # Display current iiter
                       pf_val + pg_val + zx_val)
                print(msg)

    steps: Tuple[np.ndarray, np.ndarray, np.ndarray] = \
        (taus_hist[:iiter +1], mus_hist[:iiter+1], alphas_hist[:iiter+1]) # Correct slicing for history
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))

    return x, steps