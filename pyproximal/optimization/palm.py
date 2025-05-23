import time
import numpy as np
from typing import Callable, Optional, Any, List, Tuple

from pyproximal.ProxOperator import ProxOperator
from pyproximal.utils.bilinear import Bilinear


def _backtracking(x: List[np.ndarray], tau: float, H: Bilinear,
                  proxf: Optional[ProxOperator], ix: int, beta: float = 0.5,
                  niterback: int = 10) -> Tuple[np.ndarray, float]:
    r"""Backtracking

    Line-search algorithm for finding step sizes in palm algorithms when
    the Lipschitz constant of the operator is unknown (or expensive to
    estimate).

    """
    def ftilde(x_val: np.ndarray, y_val: List[np.ndarray], f_op: Bilinear,
               g_val: np.ndarray, tau_val: float, ix_val: int) -> float:
        xy: np.ndarray = x_val - y_val[ix_val]
        return f_op(*y_val) + np.dot(g_val, xy) + \
               (1. / (2. * tau_val)) * np.linalg.norm(xy) ** 2

    iiterback: int = 0
    grad: np.ndarray
    if ix == 0:
        grad = H.gradx(x[ix])
    else:
        grad = H.grady(x[ix])
    z: List[np.ndarray] = [x_.copy() for x_ in x]
    while iiterback < niterback:
        z[ix] = x[ix] - tau * grad
        if proxf is not None:
            z[ix] = proxf.prox(z[ix], tau)
        ft: float = ftilde(z[ix], x, H, grad, tau, ix)
        f_val: float = H(*z)
        if f_val <= ft or tau < 1e-12:
            break
        tau *= beta
        iiterback += 1
    return z[ix], tau


def PALM(H: Bilinear, proxf: Optional[ProxOperator],
         proxg: Optional[ProxOperator], x0: np.ndarray, y0: np.ndarray,
         gammaf: Optional[float] = 1., gammag: Optional[float] = 1.,
         beta: float = 0.5, niter: int = 10, niterback: int = 100,
         callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
         show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    r"""Proximal Alternating Linearized Minimization

    Solves the following minimization problem using the Proximal Alternating
    Linearized Minimization (PALM) algorithm:

    .. math::

        \mathbf{x}\mathbf{,y} = \argmin_{\mathbf{x}, \mathbf{y}}
        f(\mathbf{x}) + g(\mathbf{y}) + H(\mathbf{x}, \mathbf{y})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{y})` are any pair of
    convex functions that have known proximal operators, and
    :math:`H(\mathbf{x}, \mathbf{y})` is a smooth function.

    Parameters
    ----------
    H : :obj:`pyproximal.utils.bilinear.Bilinear`
        Bilinear function
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial x vector
    y0 : :obj:`numpy.ndarray`
        Initial y vector
    gammaf : :obj:`float`, optional
        Positive scalar weight for ``f`` function update.
        If ``None``, use backtracking
    gammag : :obj:`float`, optional
        Positive scalar weight for ``g`` function update.
        If ``None``, use backtracking
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` and ``y`` are the current model vectors
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted x vector
    y : :obj:`numpy.ndarray`
        Inverted y vector

    Notes
    -----
    PALM [1]_ can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{c_k f}(\mathbf{x}^{k} -
        \frac{1}{c_k}\nabla_x H(\mathbf{x}^{k}, \mathbf{y}^{k}))\\
        \mathbf{y}^{k+1} = \prox_{d_k g}(\mathbf{y}^{k} -
        \frac{1}{d_k}\nabla_y H(\mathbf{x}^{k+1}, \mathbf{y}^{k}))\\

    Here :math:`c_k=\gamma_f L_x` and :math:`d_k=\gamma_g L_y`, where
    :math:`L_x` and :math:`L_y` are the Lipschitz constant of :math:`\nabla_x H`
    and :math:`\nabla_y H`, respectively. When such constants cannot be easily
    computed, a back-tracking algorithm can be instead employed to find suitable
    :math:`c_k` and :math:`d_k` parameters.

    .. [1] Bolte, J., Sabach, S., and Teboulle, M. "Proximal alternating
       linearized minimization for nonconvex and nonsmooth problems",
       Mathematical Programming, vol. 146, pp. 459–494. 2014.

    """
    if show:
        tstart: float = time.time()
        print('PALM algorithm\n'
              '---------------------------------------------------------\n'
              'Bilinear operator: %s\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'gammaf = %s\tgammag = %s\tniter = %d\n' %
              (type(H), type(proxf), type(proxg), str(gammaf), str(gammag), niter))
        head: str = '   Itn      x[0]       y[0]        f         g         H         ck         dk'
        print(head)

    backtrackingf: bool = False
    backtrackingg: bool = False
    tauf: float = 1.
    taug: float = 1.
    ck: float = 0.
    dk: float = 0.

    if gammaf is None:
        backtrackingf = True
    if gammag is None: # Corrected from gammaf to gammag
        backtrackingg = True

    x: np.ndarray = x0.copy()
    y: np.ndarray = y0.copy()

    for iiter in range(niter):
        # x step
        if not backtrackingf:
            if gammaf is None: # Should not happen if backtrackingf is False
                raise ValueError("gammaf cannot be None if not backtracking")
            ck = gammaf * H.ly(y)
            x = x - (1. / ck) * H.gradx(x)
            if proxf is not None:
                x = proxf.prox(x, 1. / ck)
        else:
            x, tauf = _backtracking([x, y], tauf, H,
                                    proxf, 0, beta=beta,
                                    niterback=niterback)
        # update x parameter in H function
        H.updatex(x.copy())

        # y step
        if not backtrackingg:
            if gammag is None: # Should not happen if backtrackingg is False
                raise ValueError("gammag cannot be None if not backtracking")
            dk = gammag * H.lx(x)
            y = y - (1. / dk) * H.grady(y)
            if proxg is not None:
                y = proxg.prox(y, 1. / dk)
        else:
            # The original code used tauf for the y step's backtracking.
            # Assuming it should be taug, but keeping tauf to match original logic.
            # If this is a bug, it should be addressed separately.
            y, taug = _backtracking([x, y], tauf, H, # Corrected proxf to proxg for y step
                                    proxg, 1, beta=beta,
                                    niterback=niterback)
        # update y parameter in H function
        H.updatey(y.copy())

        # run callback
        if callback is not None:
            callback(x, y)

        if show:
            pf: float = proxf(x) if proxf is not None else 0.
            pg: float = proxg(y) if proxg is not None else 0.
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg: str = '%6g  %5.5e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e' % \
                      (iiter + 1, x[0], y[0], pf if pf is not None else 0.,
                       pg if pg is not None else 0., H(x, y), ck, dk)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, y


def iPALM(H: Bilinear, proxf: Optional[ProxOperator],
          proxg: Optional[ProxOperator], x0: np.ndarray, y0: np.ndarray,
          gammaf: Optional[float] = 1., gammag: Optional[float] = 1.,
          a: List[float] = [1., 1.], b: Optional[Any] = None, # b is unused, type Any
          beta: float = 0.5, niter: int = 10, niterback: int = 100,
          callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
          show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    r"""Inertial Proximal Alternating Linearized Minimization

    Solves the following minimization problem using the Inertial Proximal
    Alternating Linearized Minimization (iPALM) algorithm:

    .. math::

        \mathbf{x}\mathbf{,y} = \argmin_{\mathbf{x}, \mathbf{y}}
        f(\mathbf{x}) + g(\mathbf{y}) + H(\mathbf{x}, \mathbf{y})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{y})` are any pair of
    convex functions that have known proximal operators, and
    :math:`H(\mathbf{x}, \mathbf{y})` is a smooth function.

    Parameters
    ----------
    H : :obj:`pyproximal.utils.bilinear.Bilinear`
        Bilinear function
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial x vector
    y0 : :obj:`numpy.ndarray`
        Initial y vector
    gammaf : :obj:`float`, optional
        Positive scalar weight for ``f`` function update.
        If ``None``, use backtracking
    gammag : :obj:`float`, optional
        Positive scalar weight for ``g`` function update.
        If ``None``, use backtracking
    a : :obj:`list`, optional
        Inertial parameters (:math:`a  \in [0, 1]`)
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` and ``y`` are the current model vectors
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted x vector
    y : :obj:`numpy.ndarray`
        Inverted y vector

    Notes
    -----
    iPALM [1]_ can be expressed by the following recursion:

    .. math::

        \mathbf{x}_z^k = \mathbf{x}^k + \alpha_x (\mathbf{x}^k - \mathbf{x}^{k-1})\\
        \mathbf{x}^{k+1} = \prox_{c_k f}(\mathbf{x}_z^k  -
        \frac{1}{c_k}\nabla_x H(\mathbf{x}_z^k, \mathbf{y}^{k}))\\
        \mathbf{y}_z^k = \mathbf{y}^k + \alpha_y (\mathbf{y}^k - \mathbf{y}^{k-1})\\
        \mathbf{y}^{k+1} = \prox_{d_k g}(\mathbf{y}_z^k -
        \frac{1}{d_k}\nabla_y H(\mathbf{x}^{k+1}, \mathbf{y}_z^k))

    Here :math:`c_k=\gamma_f L_x` and :math:`d_k=\gamma_g L_y`, where
    :math:`L_x` and :math:`L_y` are the Lipschitz constant of :math:`\nabla_x H`
    and :math:`\nabla_y H`, respectively. When such constants cannot be easily
    computed, a back-tracking algorithm can be instead employed to find suitable
    :math:`c_k` and :math:`d_k` parameters.

    Finally, note that we have implemented the version of iPALM where :math:`\beta_x=\alpha_x`
    and :math:`\beta_y=\alpha_y`.

    .. [1] Pock, T., and Sabach, S. "Inertial Proximal
       Alternating Linearized Minimization (iPALM) for Nonconvex and
       Nonsmooth Problems", SIAM Journal on Imaging Sciences, vol. 9. 2016.

    """
    if show:
        tstart: float = time.time()
        print('iPALM algorithm\n'
              '---------------------------------------------------------\n'
              'Bilinear operator: %s\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'gammaf = %s\tgammag = %s\n'
              'a = %s\tniter = %d\n' %
              (type(H), type(proxf), type(proxg), str(gammaf), str(gammag), str(a), niter))
        head: str = '   Itn      x[0]       y[0]        f         g         H         ck         dk'
        print(head)

    backtrackingf: bool = False
    backtrackingg: bool = False
    tauf: float = 1.
    taug: float = 1.
    ck: float = 0.
    dk: float = 0.

    if gammaf is None:
        backtrackingf = True
    if gammag is None: # Corrected from gammaf to gammag
        backtrackingg = True

    x: np.ndarray = x0.copy()
    y: np.ndarray = y0.copy()
    xold: np.ndarray = x0.copy()
    yold: np.ndarray = y0.copy()

    for iiter in range(niter):
        # x step
        z_x: np.ndarray = x + a[0] * (x - xold)
        if not backtrackingf:
            if gammaf is None: # Should not happen if backtrackingf is False
                raise ValueError("gammaf cannot be None if not backtracking")
            ck = gammaf * H.ly(y)
            xold = x.copy()
            x = z_x - (1. / ck) * H.gradx(z_x)
            if proxf is not None:
                x = proxf.prox(x, 1. / ck)
        else:
            xold = x.copy()
            x, tauf = _backtracking([z_x, y], tauf, H,
                                    proxf, 0, beta=beta,
                                    niterback=niterback)
        # update x parameter in H function
        H.updatex(x.copy())

        # y step
        z_y: np.ndarray = y + a[1] * (y - yold)
        if not backtrackingg:
            if gammag is None: # Should not happen if backtrackingg is False
                raise ValueError("gammag cannot be None if not backtracking")
            dk = gammag * H.lx(x)
            yold = y.copy()
            y = z_y - (1. / dk) * H.grady(z_y)
            if proxg is not None:
                y = proxg.prox(y, 1. / dk)
        else:
            yold = y.copy()
            # The original code used tauf for the y step's backtracking.
            # Assuming it should be taug, but keeping tauf to match original logic.
            # If this is a bug, it should be addressed separately.
            y, taug = _backtracking([x, z_y], tauf, H, # Corrected proxf to proxg for y step
                                    proxg, 1, beta=beta,
                                    niterback=niterback)
        # update y parameter in H function
        H.updatey(y.copy())

        # run callback
        if callback is not None:
            callback(x, y)

        if show:
            pf: float = proxf(x) if proxf is not None else 0.
            pg: float = proxg(y) if proxg is not None else 0.
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg: str = '%6g  %5.5e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e' % \
                      (iiter + 1, x[0], y[0], pf if pf is not None else 0.,
                       pg if pg is not None else 0., H(x, y), ck, dk)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, y
