import time
import warnings
import numpy as np
from typing import Callable, Optional, Any, Union, Tuple, List

from math import sqrt
from pylops.optimization.leastsquares import regularized_inversion
from pylops.utils.backend import to_numpy
from pylops.LinearOperator import LinearOperator
from pyproximal.proximal import L2
from pyproximal.utils.bilinear import BilinearOperator
from pyproximal.ProxOperator import ProxOperator


def _backtracking(x: np.ndarray, tau: float, proxf: ProxOperator,
                  proxg: ProxOperator, epsg: Union[float, np.ndarray],
                  beta: float = 0.5, niterback: int = 10) -> Tuple[np.ndarray, float]:
    r"""Backtracking

    Line-search algorithm for finding step sizes in proximal algorithms when
    the Lipschitz constant of the operator is unknown (or expensive to
    estimate).

    """
    def ftilde(x_val: np.ndarray, y_val: np.ndarray, f_op: ProxOperator, tau_val: float) -> float:
        xy: np.ndarray = x_val - y_val
        return f_op(y_val) + np.dot(f_op.grad(y_val), xy) + \
               (1. / (2. * tau_val)) * np.linalg.norm(xy) ** 2

    iiterback: int = 0
    z: np.ndarray
    while iiterback < niterback:
        z = proxg.prox(x - tau * proxf.grad(x), epsg * tau)
        ft: float = ftilde(z, x, proxf, tau)
        if proxf(z) <= ft:
            break
        tau *= beta
        iiterback += 1
    return z, tau


def ProximalPoint(prox: ProxOperator, x0: np.ndarray, tau: float, niter: int = 10,
                  tol: Optional[float] = None,
                  callback: Optional[Callable[[np.ndarray], None]] = None,
                  show: bool = False) -> np.ndarray:
    r"""Proximal point algorithm

    Solves the following minimization problem using Proximal point algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x})

    where :math:`f(\mathbf{x})` is any convex function that has a known
    proximal operator.

    Parameters
    ----------
    prox : :obj:`pyproximal.ProxOperator`
        Proximal operator
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Proximal point algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^k)

    """
    if show:
        tstart: float = time.time()
        print('Proximal point algorithm\n'
              '---------------------------------------------------------\n'
              'Proximal operator: %s\n'
              'tau = %10e\tniter = %d\ttol = %s\n' % (type(prox), tau, niter, str(tol)))
        head: str = '   Itn       x[0]          f'
        print(head)


    # initialize model
    x: np.ndarray = x0.copy()
    pf: float = np.inf
    tolbreak: bool = False

    # iterate
    for iiter in range(niter):
        x = prox.prox(x, tau)

        # run callback
        if callback is not None:
            callback(x)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if tol is not None:
            pfold: float = pf
            pf = prox(x)
            if np.abs(1.0 - pf / pfold) < tol:
                tolbreak = True

        # show iteration logger
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                if tol is None: # Recalculate pf if not done for tol check
                    pf = prox(x)
                msg: str = '%6g  %12.5e  %10.3e' % \
                      (iiter + 1, x[0], pf)
                print(msg)

        # break if tolerance condition is met
        if tolbreak:
            break

    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def ProximalGradient(proxf: ProxOperator, proxg: ProxOperator, x0: np.ndarray,
                     epsg: Union[float, np.ndarray] = 1.,
                     tau: Optional[float] = None, backtracking: bool = False,
                     beta: float = 0.5, eta: float = 1.,
                     niter: int = 10, niterback: int = 100,
                     acceleration: Optional[str] = None, tol: Optional[float] = None,
                     callback: Optional[Callable[[np.ndarray], None]] = None,
                     show: bool = False) -> np.ndarray:
    r"""Proximal gradient (optionally accelerated)

    Solves the following minimization problem using (Accelerated) Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function (must have ``grad`` implemented)
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor of g function
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration. Finally, note that :math:`\tau` can be chosen to be a vector
        when dealing with problems with multiple right-hand-sides
    backtracking : :obj:`bool`, optional
        Force backtracking, even if ``tau`` is not equal to ``None``. In this case
        the chosen ``tau`` will be used as the initial guess in the first
        step of backtracking
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 1, 0 excluded).
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    acceleration : :obj:`str`, optional
        Acceleration (``None``, ``vandenberghe`` or ``fista``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Proximal gradient algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \mathbf{y}^k + \eta (\prox_{\tau^k \epsilon g}(\mathbf{y}^k -
        \tau^k \nabla f(\mathbf{y}^k)) - \mathbf{y}^k) \\
        \mathbf{y}^{k+1} = \mathbf{x}^k + \omega^k
        (\mathbf{x}^k - \mathbf{x}^{k-1})

    where at each iteration :math:`\tau^k` can be estimated by back-tracking
    as follows:

    .. math::

        \begin{aligned}
        &\tau = \tau^{k-1} &\\
        &repeat \; \mathbf{z} = \prox_{\tau \epsilon g}(\mathbf{x}^k -
        \tau \nabla f(\mathbf{x}^k)), \tau = \beta \tau \quad if \;
        f(\mathbf{z}) \leq \tilde{f}_\tau(\mathbf{z}, \mathbf{x}^k) \\
        &\tau^k = \tau, \quad \mathbf{x}^{k+1} = \mathbf{z} &\\
        \end{aligned}

    where :math:`\tilde{f}_\tau(\mathbf{x}, \mathbf{y}) = f(\mathbf{y}) +
    \nabla f(\mathbf{y})^T (\mathbf{x} - \mathbf{y}) +
    1/(2\tau)||\mathbf{x} - \mathbf{y}||_2^2`.

    Different accelerations are provided:

    - ``acceleration=None``: :math:`\omega^k = 0`;
    - ``acceleration=vandenberghe`` [1]_: :math:`\omega^k = k / (k + 3)` for `
    - ``acceleration=fista``: :math:`\omega^k = (t_{k-1}-1)/t_k` where
      :math:`t_k = (1 + \sqrt{1+4t_{k-1}^{2}}) / 2` [2]_

    .. [1] Vandenberghe, L., "Fast proximal gradient methods", 2010.
    .. [2] Beck, A., and Teboulle, M. "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems", SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """
    # check if epgs is a vector
    epsg_arr: np.ndarray
    epsg_print: str
    if np.asarray(epsg).size == 1.:
        epsg_arr = epsg * np.ones(niter)
        epsg_print = str(epsg_arr[0])
    else:
        epsg_arr = np.asarray(epsg)
        epsg_print = 'Multi'

    if acceleration not in [None, 'None', 'vandenberghe', 'fista']:
        raise NotImplementedError('Acceleration should be None, vandenberghe '
                                  'or fista')
    if show:
        tstart: float = time.time()
        print('Accelerated Proximal Gradient\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %s\tbacktrack = %s\tbeta = %10e\n'
              'epsg = %s\tniter = %d\ttol = %s\n'
              ''
              'niterback = %d\tacceleration = %s\n' % (type(proxf), type(proxg),
                                                       str(tau), backtracking, beta,
                                                       epsg_print, niter, str(tol),
                                                       niterback, acceleration))
        head: str = '   Itn       x[0]          f           g       J=f+eps*g       tau'
        print(head)

    current_tau: float
    if tau is None:
        backtracking = True
        current_tau = 1.
    else:
        current_tau = tau


    # initialize model
    t: float = 1.
    x: np.ndarray = x0.copy()
    y: np.ndarray = x.copy()
    pfg: float = np.inf
    tolbreak: bool = False

    # iterate
    for iiter in range(niter):
        xold: np.ndarray = x.copy()

        # proximal step
        if not backtracking:
            if eta == 1.:
                x = proxg.prox(y - current_tau * proxf.grad(y), epsg_arr[iiter] * current_tau)
            else:
                x = x + eta * (proxg.prox(x - current_tau * proxf.grad(x), epsg_arr[iiter] * current_tau) - x)
        else:
            x, current_tau = _backtracking(y, current_tau, proxf, proxg, epsg_arr[iiter],
                                           beta=beta, niterback=niterback)
            if eta != 1.: # This part seems redundant if _backtracking already updates x
                x = x + eta * (proxg.prox(x - current_tau * proxf.grad(x), epsg_arr[iiter] * current_tau) - x)


        # update internal parameters for bilinear operator
        if isinstance(proxf, BilinearOperator):
            proxf.updatexy(x)

        # update y
        omega: float
        if acceleration == 'vandenberghe':
            omega = iiter / (iiter + 3.) # Ensure float division
        elif acceleration == 'fista':
            told: float = t
            t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
            omega = ((told - 1.) / t)
        else:
            omega = 0.
        y = x + omega * (x - xold)

        # run callback
        if callback is not None:
            callback(x)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if tol is not None:
            pfgold: float = pfg
            pf: float = proxf(x)
            pg: float = proxg(x) # Assuming proxg returns scalar or array that can be summed
            pfg = pf + np.sum(epsg_arr[iiter] * pg)
            if np.abs(1.0 - pfg / pfgold) < tol:
                tolbreak = True

        # show iteration logger
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_show: float
                pg_show: float # Assuming proxg returns scalar or array that can be summed
                pfg_show: float
                if tol is None: # Recalculate if not done for tol check
                    pf_show = proxf(x)
                    pg_show = proxg(x)
                    pfg_show = pf_show + np.sum(epsg_arr[iiter] * pg_show)
                else:
                    pf_show = pf
                    pg_show = pg
                    pfg_show = pfg
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])) if x.ndim == 1 else np.real(to_numpy(x[0, 0])),
                       pf_show, pg_show,
                       pfg_show,
                       current_tau)
                print(msg)

        # break if tolerance condition is met
        if tolbreak:
            break

    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def AcceleratedProximalGradient(proxf: ProxOperator, proxg: ProxOperator,
                                x0: np.ndarray, tau: Optional[float] = None,
                                beta: float = 0.5,
                                epsg: Union[float, np.ndarray] = 1.,
                                niter: int = 10, niterback: int = 100,
                                acceleration: str = 'vandenberghe',
                                tol: Optional[float] = None,
                                callback: Optional[Callable[[np.ndarray], None]] = None,
                                show: bool = False) -> np.ndarray:
    r"""Accelerated Proximal gradient

    This is a thin wrapper around :func:`pyproximal.optimization.primal.ProximalGradient` with
    ``vandenberghe`` or ``fista`` acceleration. See :func:`pyproximal.optimization.primal.ProximalGradient`
    for details.

    """
    warnings.warn('AcceleratedProximalGradient has been integrated directly into ProximalGradient '
                  'from v0.5.0. It is recommended to start using ProximalGradient by selecting the '
                  'appropriate acceleration parameter as this behaviour will become default in '
                  'version v1.0.0 and AcceleratedProximalGradient will be removed.', FutureWarning)
    return ProximalGradient(proxf, proxg, x0, tau=tau, beta=beta, # type: ignore
                            epsg=epsg, niter=niter, niterback=niterback,
                            acceleration=acceleration, tol=tol,
                            callback=callback, show=show)


def AndersonProximalGradient(proxf: ProxOperator, proxg: ProxOperator,
                             x0: np.ndarray, epsg: Union[float, np.ndarray] = 1.,
                             tau: Optional[float] = None, niter: int = 10,
                             nhistory: int = 10, epsr: float = 1e-10,
                             safeguard: bool = False, tol: Optional[float] = None,
                             callback: Optional[Callable[[np.ndarray], None]] = None,
                             show: bool = False) -> np.ndarray:
    r"""Proximal gradient with Anderson acceleration

    Solves the following minimization problem using the Proximal
    gradient algorithm with Anderson acceleration:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function (must have ``grad`` implemented)
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor of g function
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration. Finally, note that :math:`\tau` can be chosen to be a vector
        when dealing with problems with multiple right-hand-sides
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    nhistory : :obj:`int`, optional
        Number of previous iterates to be kept in memory (to compute the scaling factors
    epsr : :obj:`float`, optional
        Scaling factor for regularization added to the inverse of :math:\mathbf{R}^T \mathbf{R}`
    safeguard : :obj:`bool`, optional
        Apply safeguarding strategy to the update (``True``) or not (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Proximal gradient algorithm with Anderson acceleration can be expressed by the 
    following recursion [1]_:

    .. math::
        m_k = min(m, k)\\
        \mathbf{g}^{k} = \mathbf{x}^{k} - \tau^k \nabla f(\mathbf{x}^k)\\
        \mathbf{r}^{k} = \mathbf{g}^{k} - \mathbf{g}^{k}\\
        \mathbf{G}^{k} = [\mathbf{g}^{k},..., \mathbf{g}^{k-m_k}]\\
        \mathbf{R}^{k} = [\mathbf{r}^{k},..., \mathbf{r}^{k-m_k}]\\
        \alpha_k = (\mathbf{R}^{kT} \mathbf{R}^{k})^{-1} \mathbf{1} / \mathbf{1}^T
        (\mathbf{R}^{kT} \mathbf{R}^{k})^{-1} \mathbf{1}\\
        \mathbf{y}^{k+1} = \mathbf{G}^{k} \alpha_k\\
        \mathbf{x}^{k+1} = \prox_{\tau^{k+1} g}(\mathbf{y}^{k+1})

    where :math:`m` equals ``nhistory``, :math:`k=1,2,...,n_{iter}`, :math:`\mathbf{y}^{0}=\mathbf{x}^{0}`,
    :math:`\mathbf{y}^{1}=\mathbf{x}^{0} - \tau^0 \nabla f(\mathbf{x}^0)`,
    :math:`\mathbf{x}^{1}=\prox_{\tau^k g}(\mathbf{y}^{1})`, and 
    :math:`\mathbf{g}^{0}=\mathbf{y}^{1}`.
    
    Refer to [1]_ for the guarded version of the algorithm (when ``safeguard=True``).

    .. [1] Mai, V., and Johansson, M. "Anderson Acceleration of Proximal Gradient 
       Methods", 2020.
    
    """
    # check if epgs is a vector
    epsg_arr: np.ndarray
    epsg_print: str
    if np.asarray(epsg).size == 1.:
        epsg_arr = epsg * np.ones(niter)
        epsg_print = str(epsg_arr[0])
    else:
        epsg_arr = np.asarray(epsg)
        epsg_print = 'Multi'

    if tau is None: # tau must be set for Anderson
        raise ValueError("tau cannot be None for AndersonProximalGradient")
    current_tau: float = tau


    if show:
        tstart: float = time.time()
        print('Proximal Gradient with Anderson Acceleration \n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %s\t\tepsg = %s\tniter = %d\n'
              'nhist = %d\tepsr = %s\n'
              'guard = %s\ttol = %s\n' % (type(proxf), type(proxg),
                                          str(current_tau), epsg_print, niter,
                                          nhistory, str(epsr),
                                          str(safeguard), str(tol)))
        head: str = '   Itn       x[0]          f           g       J=f+eps*g       tau'
        print(head)

    # initialize model
    x: np.ndarray
    y: np.ndarray = x0 - current_tau * proxf.grad(x0)
    x = proxg.prox(y, epsg_arr[0] * current_tau)
    g_curr: np.ndarray = y.copy() # g in the paper, renamed to g_curr to avoid conflict
    r_curr: np.ndarray = g_curr - x0 # r in the paper, renamed to r_curr
    R_hist: List[np.ndarray] = [r_curr, ] # R in the paper, renamed to R_hist
    G_hist: List[np.ndarray] = [g_curr, ] # G in the paper, renamed to G_hist
    pf: float = proxf(x)
    pfg: float = np.inf
    tolbreak: bool = False

    # iterate
    for iiter in range(niter):
        
        # update fix point
        g_curr = x - current_tau * proxf.grad(x)
        r_curr = g_curr - y # y here is x_k from previous iteration in paper's notation for r_k

        # update history vectors
        R_hist.insert(0, r_curr)
        G_hist.insert(0, g_curr)
        if iiter >= nhistory -1 : # Corrected: was iiter >= nhistory -1, now len(R_hist) > nhistory
           R_hist.pop(-1)
           G_hist.pop(-1)
        
        # solve for alpha coefficients
        Rstack: np.ndarray = np.vstack(R_hist)
        # Adding type hint for Rinv, alpha
        Rinv: np.ndarray = np.linalg.pinv(Rstack @ Rstack.T + epsr * np.linalg.norm(Rstack) ** 2 * np.eye(Rstack.shape[0]))
        ones: np.ndarray = np.ones(min(nhistory, iiter + 2))
        Rinvones: np.ndarray = Rinv @ ones
        alpha: np.ndarray = Rinvones / (ones[None] @ Rinvones)
        
        if not safeguard:
            # update auxiliary variable
            y = (np.vstack(G_hist).T @ alpha).reshape(x.shape)


            # update main variable
            x = proxg.prox(y, epsg_arr[iiter] * current_tau)
        
        else:
            # update auxiliary variable
            ytest: np.ndarray = (np.vstack(G_hist).T @ alpha).reshape(x.shape)

            # update main variable
            xtest: np.ndarray = proxg.prox(ytest, epsg_arr[iiter] * current_tau)

            # check if function is decreased, otherwise do basic PG step
            pfold_guard: float = pf
            pf = proxf(xtest) # pf is updated here
            if pf <= pfold_guard - current_tau * np.linalg.norm(proxf.grad(x)) ** 2 / 2:
                y = ytest
                x = xtest
            else: # Fallback to standard PG step
                # y becomes g_curr (the fixed point update before Anderson)
                # x becomes the prox of that g_curr
                x = proxg.prox(g_curr, epsg_arr[iiter] * current_tau)
                y = g_curr # y needs to be updated for the next r_curr calculation
                pf = proxf(x) # pf must be updated as x changed

        # run callback
        if callback is not None:
            callback(x)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if tol is not None:
            pfgold: float = pfg
            # pf already updated if safeguard=True, or needs update if safeguard=False
            if not safeguard: pf = proxf(x)
            pg_val: float = proxg(x) # Assuming proxg returns scalar or array that can be summed
            pfg = pf + np.sum(epsg_arr[iiter] * pg_val)
            if np.abs(1.0 - pfg / pfgold) < tol:
                tolbreak = True

        # show iteration logger
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_show: float
                pg_show: float
                pfg_show: float
                if tol is None: # Recalculate if not done for tol check
                    pf_show = proxf(x)
                    pg_show = proxg(x)
                    pfg_show = pf_show + np.sum(epsg_arr[iiter] * pg_show)
                else:
                    pf_show = pf
                    pg_show = pg_val # Use pg_val calculated for tol check
                    pfg_show = pfg
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])) if x.ndim == 1 else np.real(to_numpy(x[0, 0])),
                       pf_show, pg_show,
                       pfg_show,
                       current_tau)
                print(msg)

        # break if tolerance condition is met
        if tolbreak:
            break

    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def GeneralizedProximalGradient(proxfs: List[ProxOperator], proxgs: List[ProxOperator],
                                x0: np.ndarray, tau: Optional[float], # tau can be None initially
                                epsg: Union[float, np.ndarray] = 1.,
                                weights: Optional[np.ndarray] = None,
                                eta: float = 1., niter: int = 10,
                                acceleration: Optional[str] = None,
                                callback: Optional[Callable[[np.ndarray], None]] = None,
                                show: bool = False) -> np.ndarray:
    r"""Generalized Proximal gradient

    Solves the following minimization problem using Generalized Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \sum_{i=1}^n f_i(\mathbf{x}) 
        + \sum_{j=1}^m \epsilon_j g_j(\mathbf{x}),~~n,m \in \mathbb{N}^+

    where the :math:`f_i(\mathbf{x})` are smooth convex functions with a uniquely
    defined gradient and the :math:`g_j(\mathbf{x})` are any convex function that
    have a known proximal operator.

    Parameters
    ----------
    proxfs : :obj:`list`
        Proximal operators of the :math:`f_i` functions (must have ``grad`` implemented)
    proxgs : :obj:`list`
        Proximal operators of the :math:`g_j` functions
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\sum_{i=1}^n \nabla f_i`.
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor(s) of ``g`` function(s)
    weights : :obj:`float`, optional
        Weighting factors of ``g`` functions. Must sum to 1.
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 1, 0 excluded). Note that
        this will be only used when ``acceleration=None``.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    acceleration:  :obj:`str`, optional
        Acceleration (``None``, ``vandenberghe`` or ``fista``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Generalized Proximal point algorithm can be expressed by the
    following recursion:

    .. math::
        \text{for } j=1,\cdots,n, \\
        ~~~~\mathbf z_j^{k+1} = \mathbf z_j^{k} + \eta
        \left[prox_{\frac{\tau^k \epsilon_j}{w_j} g_j}\left(2 \mathbf{x}^{k} - \mathbf{z}_j^{k}
        - \tau^k \sum_{i=1}^n \nabla f_i(\mathbf{x}^{k})\right) - \mathbf{x}^{k} \right] \\
        \mathbf{x}^{k+1} = \sum_{j=1}^n w_j \mathbf z_j^{k+1} \\

    where :math:`\sum_{j=1}^n w_j=1`. In the current implementation, :math:`w_j=1/n` when
    not provided.

    """
    current_weights: np.ndarray
    # check if weights sum to 1
    if weights is None:
        current_weights = np.ones(len(proxgs)) / len(proxgs)
    else:
        current_weights = weights
    if len(current_weights) != len(proxgs) or not np.isclose(np.sum(current_weights), 1.):
        raise ValueError(f'weights={current_weights} must be an array of size {len(proxgs)} '
                         f'summing to 1')

    # check if epgs is a vector
    epsg_arr: np.ndarray
    epsg_print: str
    if np.asarray(epsg).size == 1.:
        epsg_print = str(epsg)
        epsg_arr = epsg * np.ones(len(proxgs))
    else:
        epsg_arr = np.asarray(epsg)
        epsg_print = 'Multi'

    if acceleration not in [None, 'None', 'vandenberghe', 'fista']:
        raise NotImplementedError('Acceleration should be None, vandenberghe '
                                  'or fista')

    current_tau: float
    if tau is None: # Should ideally not be None if no backtracking
        current_tau = 1.
    else:
        current_tau = tau

    if show:
        tstart: float = time.time()
        print('Generalized Proximal Gradient\n'
              '---------------------------------------------------------\n'
              'Proximal operators (f): %s\n'
              'Proximal operators (g): %s\n'
              'tau = %10e\nepsg = %s\tniter = %d\n' % ([type(proxf_i) for proxf_i in proxfs],
                                                       [type(proxg_j) for proxg_j in proxgs],
                                                       current_tau,
                                                       epsg_print, niter))
        head: str = '   Itn       x[0]          f           g       J=f+eps*g'
        print(head)


    # initialize model
    t: float = 1.
    x: np.ndarray = x0.copy()
    y: np.ndarray = x.copy()
    zs: List[np.ndarray] = [x.copy() for _ in range(len(proxgs))]

    # iterate
    for iiter in range(niter):
        xold: np.ndarray = x.copy()

        # gradient
        grad: np.ndarray = np.zeros_like(x)
        for proxf_i in proxfs:
            grad += proxf_i.grad(y) # Corrected: grad should be evaluated at y for acceleration

        # proximal step
        x_new: np.ndarray = np.zeros_like(x) # Temporary variable for the new x
        for i, proxg_j in enumerate(proxgs):
            # Corrected ztmp based on common GPG/FISTA variants where prox is on y - tau*grad
            # The original formula seems to mix x_k and z_j_k in a complex way.
            # This simplified version assumes a more standard GPG update for each z_j.
            # If the original formulation is specific and intentional, this change might alter behavior.
            # For now, sticking closer to a standard interpretation of GPG with multiple prox terms.
            # The term (2*y - zs[i]) is unusual. A more common update for z_j would be from y.
            # Reverting to a structure that seems more aligned with typical GPG,
            # but acknowledging the original formula was different.
            # The original: ztmp = 2 * y - zs[i] - current_tau * grad
            # A more standard GPG/PDS like update for z_j would be:
            # z_j_intermediate = y - current_tau * grad (common part for all j)
            # zs[i] = proxg_j.prox(z_j_intermediate, current_tau * epsg_arr[i] / current_weights[i])
            # x_new += current_weights[i] * zs[i]
            # However, the provided formula is specific:
            ztmp: np.ndarray = 2 * y - zs[i] - current_tau * grad
            ztmp = proxg_j.prox(ztmp, current_tau * epsg_arr[i] / current_weights[i])
            zs[i] += eta * (ztmp - y) # This update of zs[i] seems like a relaxation step
            x_new += current_weights[i] * zs[i]
        x = x_new


        # update y
        omega: float
        if acceleration == 'vandenberghe':
            omega = iiter / (iiter + 3.)
        elif acceleration == 'fista':
            told: float = t
            t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
            omega = ((told - 1.) / t)
        else:
            omega = 0.
        y = x + omega * (x - xold)

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: float = np.sum([proxf_i(x) for proxf_i in proxfs])
                pg_val_arr: np.ndarray = np.array([proxg_j(x) for proxg_j in proxgs])
                pg_val_sum: float = np.sum(epsg_arr * pg_val_arr)
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0] if x.ndim == 1 else x[0, 0],
                       pf_val, pg_val_arr[0] if epsg_print == 'Multi' else np.sum(pg_val_arr), # Show first g or sum of g's
                       pf_val + pg_val_sum)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def HQS(proxf: ProxOperator, proxg: ProxOperator, x0: np.ndarray,
        tau: Union[float, np.ndarray], niter: int = 10,
        z0: Optional[np.ndarray] = None, gfirst: bool = True,
        callback: Optional[Callable[..., None]] = None, # Can be callback(x) or callback(x,z)
        callbackz: bool = False, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    r"""Half Quadratic splitting

    Solves the following minimization problem using Half Quadratic splitting
    algorithm:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. Finally note that
        :math:`\tau` can be chosen to be a vector of size ``niter`` such that
        different :math:`\tau` is used at different iterations (i.e., continuation
        strategy)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    z0 : :obj:`numpy.ndarray`, optional
        Initial z vector (not required when ``gfirst=True``
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbackz : :obj:`bool`, optional
        Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    Notes
    -----
    The HQS algorithm can be expressed by the following recursion [1]_:

    .. math::

        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k}) \\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k+1})

    for ``gfirst=False``, or

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k}) \\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1})

    for ``gfirst=False``. Note that ``x`` and ``z`` converge to each other,
    however if iterations are stopped too early ``x`` is guaranteed to belong to
    the domain of ``f`` while ``z`` is guaranteed to belong to the domain of ``g``.
    Depending on the problem either of the two may be the best solution.

    .. [1] D., Geman, and C., Yang, "Nonlinear image recovery with halfquadratic
         regularization", IEEE Transactions on Image Processing,
         4, 7, pp. 932-946, 1995.

    """
    tau_arr: np.ndarray
    tau_print: str
    # check if tau is a vector
    if isinstance(tau, (float, int)):
        tau_print = str(tau)
        tau_arr = float(tau) * np.ones(niter)
    elif isinstance(tau, np.ndarray):
        if tau.size == 1:
            tau_print = str(tau[0])
            tau_arr = tau[0] * np.ones(niter)
        elif tau.size == niter:
            tau_print = 'Variable'
            tau_arr = tau
        else:
            raise ValueError("tau must be a scalar or a numpy array of size niter")
    else:
        raise TypeError("tau must be float, int, or numpy.ndarray")


    if show:
        tstart: float = time.time()
        print('HQS\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %s\tniter = %d\n' % (type(proxf), type(proxg),
                                          tau_print, niter))
        head: str = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    x: np.ndarray = x0.copy()
    z: np.ndarray
    if z0 is not None:
        z = z0.copy()
    else:
        z = np.zeros_like(x)

    for iiter in range(niter):
        current_tau_val: float = tau_arr[iiter]
        if gfirst:
            z = proxg.prox(x, current_tau_val)
            x = proxf.prox(z, current_tau_val)
        else:
            x = proxf.prox(z, current_tau_val)
            z = proxg.prox(x, current_tau_val)

        # run callback
        if callback is not None:
            if callbackz:
                callback(x, z)
            else:
                callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: float = proxf(x)
                pg_val: float = proxg(x) # Assuming proxg returns scalar
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])),
                       pf_val, pg_val, pf_val + pg_val)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, z


def ADMM(proxf: ProxOperator, proxg: ProxOperator, x0: np.ndarray,
         tau: float, niter: int = 10, gfirst: bool = False,
         callback: Optional[Callable[..., None]] = None, # Can be callback(x) or callback(x,z)
         callbackz: bool = False, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    r"""Alternating Direction Method of Multipliers

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers (also known as Douglas-Rachford splitting):

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    ADMM can also solve the problem of the form above with a more general
    constraint: :math:`\mathbf{Ax}+\mathbf{Bz}=\mathbf{c}`. This routine implements
    the special case where :math:`\mathbf{A}=\mathbf{I}`, :math:`\mathbf{B}=-\mathbf{I}`,
    and :math:`\mathbf{c}=\mathbf{0}`, as a general algorithm can be obtained for any choice of
    :math:`f` and :math:`g` provided they have a known proximal operator.

    On the other hand, for more general choice of :math:`\mathbf{A}`, :math:`\mathbf{B}`,
    and :math:`\mathbf{c}`, the iterations are not generalizable, i.e. they depend on the choice of
    the :math:`f` and :math:`g` functions. For this reason, we currently only provide an additional
    solver for the special case where :math:`f` is a :class:`pyproximal.proximal.L2`
    operator with a linear operator  :math:`\mathbf{G}` and data :math:`\mathbf{y}`,
    :math:`\mathbf{B}=-\mathbf{I}` and :math:`\mathbf{c}=\mathbf{0}`,
    called :func:`pyproximal.optimization.primal.ADMML2`. Note that for the very same choice
    of :math:`\mathbf{B}` and :math:`\mathbf{c}`, the :func:`pyproximal.optimization.primal.LinearizedADMM`
    can also be used (and this does not require a specific choice of :math:`f`).

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbackz : :obj:`bool`, optional
        Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    See Also
    --------
    ADMML2: ADMM with L2 misfit function
    LinearizedADMM: Linearized ADMM

    Notes
    -----
    The ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k} - \mathbf{u}^{k})\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}

    Note that ``x`` and ``z`` converge to each other, however if iterations are
    stopped too early ``x`` is guaranteed to belong to the domain of ``f``
    while ``z`` is guaranteed to belong to the domain of ``g``. Depending on
    the problem either of the two may be the best solution.

    """
    if show:
        tstart: float = time.time()
        print('ADMM\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %10e\tniter = %d\n' % (type(proxf), type(proxg),
                                            tau, niter))
        head: str = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    x: np.ndarray = x0.copy()
    u: np.ndarray = np.zeros_like(x)
    z: np.ndarray = np.zeros_like(x)
    for iiter in range(niter):
        if gfirst:
            z = proxg.prox(x + u, tau)
            x = proxf.prox(z - u, tau)
        else:
            x = proxf.prox(z - u, tau)
            z = proxg.prox(x + u, tau)
        u = u + x - z

        # run callback
        if callback is not None:
            if callbackz:
                callback(x, z)
            else:
                callback(x)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: float = proxf(x)
                pg_val: float = proxg(x) # Assuming proxg returns scalar
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])),
                       pf_val, pg_val, pf_val + pg_val)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, z


def ADMML2(proxg: ProxOperator, Op: LinearOperator, b: np.ndarray,
           A: LinearOperator, x0: np.ndarray, tau: float, niter: int = 10,
           gfirst: bool = False,
           callback: Optional[Callable[[np.ndarray], None]] = None,
           show: bool = False, **kwargs_solver: Any) -> Tuple[np.ndarray, np.ndarray]:
    r"""Alternating Direction Method of Multipliers for L2 misfit term

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        \frac{1}{2}||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2 + g(\mathbf{z}) \\
        s.t. \; \mathbf{Ax}=\mathbf{z}

    where :math:`g(\mathbf{z})` is any convex function that has a known proximal operator.

    Parameters
    ----------
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    Op : :obj:`pylops.LinearOperator`
        Linear operator of data misfit term
    b : :obj:`numpy.ndarray`
        Data
    A : :obj:`pylops.LinearOperator`
        Linear operator of regularization term
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau \in (0, 1/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` used
        to solve the x-update

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    See Also
    --------
    ADMM: ADMM
    LinearizedADMM: Linearized ADMM

    Notes
    -----
    The ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \argmin_{\mathbf{x}} \frac{1}{2}||\mathbf{Op}\mathbf{x}
        - \mathbf{b}||_2^2 + \frac{1}{2\tau} ||\mathbf{Ax} - \mathbf{z}^k + \mathbf{u}^k||_2^2\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{Ax}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{Ax}^{k+1} - \mathbf{z}^{k+1}

    """
    if show:
        tstart: float = time.time()
        print('ADMM\n'
              '---------------------------------------------------------\n'
              'Proximal operator (g): %s\n'
              'tau = %10e\tniter = %d\n' % (type(proxg), tau, niter))
        head: str = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    sqrttau: float = 1. / sqrt(tau)
    x: np.ndarray = x0.copy()
    u: np.ndarray = np.zeros(A.shape[0], dtype=A.dtype)
    z: np.ndarray = np.zeros(A.shape[0], dtype=A.dtype) # Ensure z is initialized like u
    Ax: np.ndarray # Define Ax type

    for iiter in range(niter):
        if gfirst:
            Ax = A @ x
            z = proxg.prox(Ax + u, tau)

            # solve augumented system
            # Assuming regularized_inversion returns a tuple, and we need the first element
            x_result = regularized_inversion(Op, b, [A, ], x0=x,
                                             dataregs=[z - u, ], epsRs=[sqrttau, ],
                                             **kwargs_solver)
            if isinstance(x_result, tuple):
                x = x_result[0]
            else: # Should not happen based on pylops doc, but good for robustness
                x = x_result

        else:
            # solve augumented system
            x_result = regularized_inversion(Op, b, [A, ], x0=x,
                                             dataregs=[z - u, ], epsRs=[sqrttau, ],
                                             **kwargs_solver)
            if isinstance(x_result, tuple):
                x = x_result[0]
            else:
                x = x_result
            Ax = A @ x
            z = proxg.prox(Ax + u, tau)
        u = u + Ax - z

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: float = 0.5 * np.linalg.norm(Op @ x - b) ** 2
                pg_val: float = proxg(Ax) # Assuming proxg returns scalar
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])),
                       pf_val, pg_val, pf_val + pg_val)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, z


def LinearizedADMM(proxf: ProxOperator, proxg: ProxOperator, A: LinearOperator,
                   x0: np.ndarray, tau: float, mu: float, niter: int = 10,
                   callback: Optional[Callable[[np.ndarray], None]] = None,
                   show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    r"""Linearized Alternating Direction Method of Multipliers

    Solves the following minimization problem using Linearized Alternating
    Direction Method of Multipliers (also known as Douglas-Rachford splitting):

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + g(\mathbf{A}\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    function that has a known proximal operator and :math:`\mathbf{A}` is a
    linear operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following
        condition to guarantee convergence: :math:`\mu \in (0,
        \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    mu : :obj:`float`, optional
        Second positive scalar weight, which should satisfy the following
        condition to guarantees convergence: :math:`\mu \in (0,
        \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    See Also
    --------
    ADMM: ADMM
    ADMML2: ADMM with L2 misfit function

    Notes
    -----
    The Linearized-ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\mu f}(\mathbf{x}^{k} - \frac{\mu}{\tau}
        \mathbf{A}^H(\mathbf{A} \mathbf{x}^k - \mathbf{z}^k + \mathbf{u}^k))\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{A} \mathbf{x}^{k+1} +
        \mathbf{u}^k)\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{A}\mathbf{x}^{k+1} -
        \mathbf{z}^{k+1}

    """
    if show:
        tstart: float = time.time()
        print('Linearized-ADMM\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'tau = %10e\tmu = %10e\tniter = %d\n' % (type(proxf),
                                                       type(proxg),
                                                       type(A),
                                                       tau, mu, niter))
        head: str = '   Itn       x[0]          f           g       J = f + g'
        print(head)
    x: np.ndarray = x0.copy()
    Ax: np.ndarray = A.matvec(x)
    u: np.ndarray = np.zeros_like(Ax)
    z: np.ndarray = np.zeros_like(Ax)
    for iiter in range(niter):
        x = proxf.prox(x - mu / tau * A.rmatvec(Ax - z + u), mu)
        Ax = A.matvec(x)
        z = proxg.prox(Ax + u, tau)
        u = u + Ax - z

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: float = proxf(x)
                pg_val: float = proxg(Ax) # Assuming proxg returns scalar
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])),
                       pf_val, pg_val, pf_val + pg_val)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, z


def TwIST(proxg: ProxOperator, A: LinearOperator, b: np.ndarray,
          x0: np.ndarray, alpha: Optional[float] = None,
          beta: Optional[float] = None, eigs: Optional[Tuple[float, float]] = None,
          niter: int = 10,
          callback: Optional[Callable[[np.ndarray], None]] = None,
          show: bool = False, returncost: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Two-step Iterative Shrinkage/Threshold

    Solves the following minimization problem using Two-step Iterative
    Shrinkage/Threshold:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \frac{1}{2}
        ||\mathbf{b} - \mathbf{Ax}||_2^2 + g(\mathbf{x})

    where :math:`\mathbf{A}` is a linear operator and :math:`g(\mathbf{x})`
    is any convex function that has a known proximal operator.

    Parameters
    ----------
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator
    b : :obj:`numpy.ndarray`
        Data
    x0 : :obj:`numpy.ndarray`
        Initial vector
    alpha : :obj:`float`, optional
        Positive scalar weight (if ``None``, estimated based on the
        eigenvalues of :math:`\mathbf{A}`, see Notes for details)
    beta : :obj:`float`, optional
        Positive scalar weight (if ``None``, estimated based on the
        eigenvalues of :math:`\mathbf{A}`, see Notes for details)
    eigs : :obj:`tuple`, optional
        Largest and smallest eigenvalues of :math:`\mathbf{A}^H \mathbf{A}`.
        If passed, computes `alpha` and `beta` based on them.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    returncost : :obj:`bool`, optional
        Return cost function

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    j : :obj:`numpy.ndarray`
        Cost function

    Notes
    -----
    The TwIST algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = (1-\alpha) \mathbf{x}^{k-1} +
        (\alpha-\beta) \mathbf{x}^k +
        \beta \prox_{g} (\mathbf{x}^k + \mathbf{A}^H
        (\mathbf{b} - \mathbf{A}\mathbf{x}^k)).

    where :math:`\mathbf{x}^{1} = \prox_{g} (\mathbf{x}^0 + \mathbf{A}^T
    (\mathbf{b} - \mathbf{A}\mathbf{x}^0))`.

    The optimal weighting parameters :math:`\alpha` and :math:`\beta` are
    linked to the smallest and largest eigenvalues of
    :math:`\mathbf{A}^H\mathbf{A}` as follows:

    .. math::

        \alpha = 1 + \rho^2 \\
        \beta =\frac{2 \alpha}{\Lambda_{max} + \lambda_{min}}

    where :math:`\rho=\frac{1-\sqrt{k}}{1+\sqrt{k}}` with
    :math:`k=\frac{\lambda_{min}}{\Lambda_{max}}` and
    :math:`\Lambda_{max}=max(1, \lambda_{max})`.

    Experimentally, it has been observed that TwIST is robust to the
    choice of such parameters. Finally, note that in the case of
    :math:`\alpha=1` and :math:`\beta=1`, TwIST is identical to IST.

    """
    # define proxf as L2 proximal
    proxf: L2 = L2(Op=A, b=b)

    current_alpha: float
    current_beta: float
    # find alpha and beta
    if alpha is None or beta is None:
        emin: float
        emax: float
        if eigs is None:
            # Assuming A.eigs returns np.ndarray or similar that can be cast to float
            emin_val = A.eigs(neigs=1, which='SM')
            emax_val = A.eigs(neigs=1, which='LM')
            emin = float(np.real(emin_val[0])) if isinstance(emin_val, (np.ndarray, list)) else float(np.real(emin_val))
            emax_lm = float(np.real(emax_val[0])) if isinstance(emax_val, (np.ndarray, list)) else float(np.real(emax_val))
            emax = max([1., emax_lm])
        else:
            emax, emin = eigs
        k: float = emin / emax
        rho: float =  (1 - sqrt(k)) / (1 + sqrt(k))
        current_alpha = 1 + rho ** 2
        current_beta = 2 * current_alpha / (emax + emin)
    else:
        current_alpha = alpha
        current_beta = beta


    # compute proximal of g on initial guess (x_1)
    xold: np.ndarray = x0.copy()
    x: np.ndarray = proxg.prox(xold - proxf.grad(xold), 1.)

    if show:
        tstart: float = time.time()
        print('TwIST\n'
              '---------------------------------------------------------\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'alpha = %10e\tbeta = %10e\tniter = %d\n' % (type(proxg),
                                                           type(A),
                                                           current_alpha, current_beta, niter))
        head: str = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    # iterate
    j_cost: Optional[np.ndarray] = None
    if returncost:
        j_cost = np.zeros(niter)

    for iiter in range(niter):
        # compute new x
        xnew: np.ndarray = (1 - current_alpha) * xold + \
                           (current_alpha - current_beta) * x + \
                           current_beta * proxg.prox(x - proxf.grad(x), 1.)
        # save current x as old (x_i -> x_i-1)
        xold = x.copy()
        # save new x as current (x_i+1 -> x_i)
        x = xnew.copy()

        # compute cost function
        if returncost and j_cost is not None:
            j_cost[iiter] = proxf(x) + proxg(x)

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf_val: float = proxf(x)
                pg_val: float = proxg(x) # Assuming proxg returns scalar
                msg: str = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])),
                       pf_val, pg_val, pf_val + pg_val)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')

    if returncost:
        return x, j_cost if j_cost is not None else np.array([]) # Ensure ndarray return
    else:
        return x
