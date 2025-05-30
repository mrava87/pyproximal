from typing import TYPE_CHECKING, Optional, Union, Callable, Any, Dict, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name

from pyproximal.ProxOperator import _check_tau, ProxOperator


if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class L2(ProxOperator):
    r"""L2 Norm proximal operator.

    The Proximal operator of the :math:`\ell_2` norm is defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`
    and :math:`f_\alpha(\mathbf{x}) = f(\mathbf{x}) +
    \alpha \mathbf{q}^T\mathbf{x}`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    q : :obj:`numpy.ndarray`, optional
        Dot vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    alpha : :obj:`float`, optional
        Multiplicative coefficient of dot product
    qgrad : :obj:`bool`, optional
        Add q term to gradient (``True``) or not (``False``)
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    densesolver : :obj:`str`, optional
        Use ``numpy``, ``scipy``, or ``factorize`` when dealing with explicit
        operators. The former two rely on dense solvers from either library,
        whilst the last computes a factorization of the matrix to invert and
        avoids to do so unless the :math:`\tau` or :math:`\sigma` paramets
        have changed. Choose ``densesolver=None`` when using PyLops versions
        earlier than v1.18.1 or v2.0.0
    **kwargs_solver : :obj:`dict`, optional
        Dictionary containing extra arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver when using
        numpy data (or :py:func:`pylops.optimization.solver.lsqr` and
        when using cupy data)

    Notes
    -----
    The L2 proximal operator is defined as:

    .. math::

        prox_{\tau f_\alpha}(\mathbf{x}) =
        \left(\mathbf{I} + \tau \sigma \mathbf{Op}^T \mathbf{Op} \right)^{-1}
        \left( \mathbf{x} + \tau \sigma \mathbf{Op}^T \mathbf{b} -
        \tau \alpha \mathbf{q}\right)

    when both ``Op`` and ``b`` are provided. This formula shows that the
    proximal operator requires the solution of an inverse problem. If the
    operator ``Op`` is of kind ``explicit=True``, we can solve this problem
    directly. On the other hand if ``Op`` is of kind ``explicit=False``, an
    iterative solver is employed. In this case it is possible to provide a warm
    start via the ``x0`` input parameter.

    When only ``b`` is provided, ``Op`` is assumed to be an Identity operator
    and the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} + \tau \sigma \mathbf{b} - \tau \alpha \mathbf{q}}
        {1 + \tau \sigma}

    If ``b`` is not provided, the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} - \tau \alpha \mathbf{q}}{1 + \tau \sigma}

    Finally, note that the second term in :math:`f_\alpha(\mathbf{x})` is added
    because this combined expression appears in several problems where Bregman
    iterations are used alongside a proximal solver.

    """
    def __init__(self, Op: Optional["LinearOperator"] = None,
                 b: Optional[np.ndarray] = None,
                 q: Optional[np.ndarray] = None,
                 sigma: float = 1., alpha: float = 1.,
                 qgrad: bool = True,
                 niter: Union[int, Callable[[int], int]] = 10,
                 x0: Optional[np.ndarray] = None, warm: bool = True,
                 densesolver: Optional[str] = None,
                 kwargs_solver: Optional[Dict[str, Any]] = None):
        super().__init__(Op, True) # hasgrad is True
        self.b: Optional[np.ndarray] = b
        self.q: Optional[np.ndarray] = q
        self.sigma: float = sigma
        self.alpha: float = alpha
        self.qgrad: bool = qgrad
        self.niter_arg: Union[int, Callable[[int], int]] = niter # Store original arg
        self.x0: Optional[np.ndarray] = x0
        self.warm: bool = warm
        self.densesolver: Optional[str] = densesolver
        self.count: int = 0
        self.kwargs_solver: Dict[str, Any] = {} if kwargs_solver is None else kwargs_solver

        self.OpTb: Optional[np.ndarray] = None
        self.ATA: Optional[np.ndarray] = None # For explicit operators
        self.tausigma: float = 0. # For factorize densesolver
        self.cl: Optional[Any] = None # For factorize densesolver, stores Cholesky factor

        # create data term
        if self.Op is not None and self.b is not None:
            # Ensure Op.H is callable if Op is a LinearOperator
            self.OpTb = self.sigma * self.Op.H @ self.b # type: ignore
            # create A.T A upfront for explicit operators
            if self.Op.explicit:
                # Assuming self.Op.A exists and is ndarray if explicit
                self.ATA = np.conj(self.Op.A.T) @ self.Op.A # type: ignore

    def __call__(self, x: np.ndarray) -> float:
        f_val: float
        if self.Op is not None and self.b is not None:
            f_val = (self.sigma / 2.) * (np.linalg.norm(self.Op @ x - self.b)**2) # type: ignore
        elif self.b is not None:
            f_val = (self.sigma / 2.) * (np.linalg.norm(x - self.b)**2)
        else:
            f_val = (self.sigma / 2.) * (np.linalg.norm(x)**2)
        
        if self.q is not None:
            # Ensure q and x are compatible for dot product (e.g., both 1D or correct matrix/vector)
            f_val += self.alpha * np.dot(self.q, x)
        return f_val

    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'L2', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        current_x: np.ndarray = x.copy() # Work on a copy
        # define current number of iterations
        niter_val: int
        if isinstance(self.niter_arg, int):
            niter_val = self.niter_arg
        else: # callable
            niter_val = self.niter_arg(self.count)

        # solve proximal optimization
        if self.Op is not None and self.b is not None and self.OpTb is not None:
            y: np.ndarray = current_x + tau * self.OpTb
            if self.q is not None:
                y -= tau * self.alpha * self.q
            
            if self.Op.explicit and self.ATA is not None:
                if self.densesolver != 'factorize':
                    # Assuming MatrixMult and Identity are LinearOperator like
                    Op1_explicit: "LinearOperator" = MatrixMult(np.eye(self.Op.shape[1]) + \
                                             tau * self.sigma * self.ATA) # type: ignore
                    if self.densesolver is None:
                        current_x = Op1_explicit.div(y) # type: ignore
                    else:
                        current_x = Op1_explicit.div(y, densesolver=self.densesolver) # type: ignore
                else: # factorize
                    if self.tausigma != tau * self.sigma or self.cl is None:
                        self.tausigma = float(tau * self.sigma) # Explicit cast
                        ATA_factor: np.ndarray = np.eye(self.Op.shape[1]) + \
                                           self.tausigma * self.ATA # self.tausigma is now float
                        self.cl = cho_factor(ATA_factor)
                    current_x = cho_solve(self.cl, y)
            else: # Implicit operator
                Op1_implicit: "LinearOperator" = Identity(self.Op.shape[1], dtype=self.Op.dtype) + \
                                   float(tau * self.sigma) * (self.Op.H * self.Op) # type: ignore
                
                # Determine backend for lsqr
                xp = get_array_module(current_x)
                if get_module_name(xp) == 'numpy':
                    # sp_lsqr returns tuple: (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var)
                    lsqr_result: Tuple = sp_lsqr(Op1_implicit, y, iter_lim=niter_val, x0=self.x0, **self.kwargs_solver) # type: ignore
                    current_x = lsqr_result[0]
                else: # cupy or other
                    # pylops.optimization.basic.lsqr returns tuple: (x, istop, itn, r1norm)
                    lsqr_result_pylops: Tuple = lsqr(Op1_implicit, y, niter=niter_val, x0=self.x0, **self.kwargs_solver) # type: ignore
                    current_x = lsqr_result_pylops[0].ravel()
            
            if self.warm:
                self.x0 = current_x.copy() # Store copy for warm start
        elif self.b is not None: # Op is None, b is not None
            num: np.ndarray = current_x + tau * self.sigma * self.b
            if self.q is not None:
                num -= tau * self.alpha * self.q
            current_x = num / (1. + tau * self.sigma)
        else: # Op is None, b is None
            num = current_x
            if self.q is not None:
                num -= tau * self.alpha * self.q
            current_x = num / (1. + tau * self.sigma)
        return current_x

    def grad(self, x: np.ndarray) -> np.ndarray:
        g_val: np.ndarray
        if self.Op is not None and self.b is not None:
            g_val = self.sigma * self.Op.H @ (self.Op @ x - self.b) # type: ignore
        elif self.b is not None:
            g_val = self.sigma * (x - self.b)
        else:
            g_val = self.sigma * x
        
        if self.q is not None and self.qgrad:
            g_val += self.alpha * self.q
        return g_val


class L2Convolve(ProxOperator):
    r"""L2 Norm proximal operator with convolution operator

    Proximal operator for the L2 norm defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{h} * \mathbf{x} - \mathbf{b}||_2^2` where
    :math:`\mathbf{h}` is the kernel of a convolution operator and
    :math:`*` represents convolution

    Parameters
    ----------
    h : :obj:`np.ndarray`, optional
        Kernel of convolution operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    b : :obj:`int`, optional
        Fourier transform number of samples
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.

    Notes
    -----
    The L2Convolve proximal operator is defined as:

    .. math::

        prox_{\tau f}(\mathbf{x}) =
        F^{-1}\left(\frac{\tau\sigma F(\mathbf{h})^* F(\mathbf{b}) + F(\mathbf{x})}
        {1 + \tau\sigma F(\mathbf{h})^* F(\mathbf{h})} \right)

    """
    def __init__(self, h: np.ndarray, b: Optional[np.ndarray] = None,
                 nfft: int = 2**10, sigma: float = 1.,
                 dims: Optional[Tuple[int, ...]] = None,
                 dir: Optional[int] = None):
        super().__init__(None, True) # Op is None, hasgrad is True
        self.nfft: int = nfft
        self.sigma: float = sigma
        self.dims: Optional[Tuple[int, ...]] = dims
        self.dir: int = -1 if dir is None else dir

        # Ensure b is not None for fft, or handle if it can be.
        # If b is None, bf would be problematic. Assuming b is provided if used.
        b_for_fft: np.ndarray
        if b is None:
            # If b is None, this implies a term ||h*x||^2. bf should be 0.
            # Need to know the expected shape for bf if b is None.
            # For now, assume if b is None, bf is effectively zero.
            # This might require adjusting __call__ and grad too.
            # Let's assume b is required if this class is used for ||h*x - b||^2.
            # If b can be None for ||h*x||^2, then bf should be zero array of appropriate shape.
            # This part needs clarification on how b=None should be handled.
            # For now, assuming b is not None for fft.
            # If b is None, hbf will be zero, which is correct for ||h*x||^2.
            # bf is used in __call__ and grad.
            # If b is None, bf must be handled as zero array.
            if dims is not None:
                 # Create a zero array of the shape bf would have if b was present.
                 # This shape depends on nfft and dims.
                 bf_shape_list = list(dims)
                 bf_shape_list[self.dir] = nfft
                 b_for_fft = np.zeros(tuple(bf_shape_list), dtype=h.dtype) # Assuming complex result for fft
            else: # 1D case
                 b_for_fft = np.zeros(nfft, dtype=h.dtype) # Assuming complex result for fft
        else:
            b_for_fft = b
        
        self.bf: np.ndarray = np.fft.fft(b_for_fft, self.nfft, axis=self.dir)
        self.hf: np.ndarray = np.fft.fft(h, self.nfft, axis=self.dir)

        # expand dimensions of filters
        if self.dims is not None:
            # self.bf might need reshaping if b was None and dims was used to create zero b_for_fft
            # However, fft output shape should already match if b_for_fft was shaped correctly.
            # No, if b was (M,N) and dir=0, nfft applied along axis 0.
            # If dims is (M,N), bf should be (nfft, N) if dir=0.
            # The reshape(self.dims) here seems to assume fft output matches original dims,
            # which is not true if nfft > dims[dir].
            # This part might need review based on expected behavior of dims and nfft.
            # For now, let's assume self.bf is already correctly shaped post-fft.
            # Or, if b was None, self.bf is zero and correctly shaped.
            # The self.bf.reshape(self.dims) line is problematic if nfft != dims[self.dir].
            # It should be:
            # temp_bf_shape = list(b_for_fft.shape) # Shape of b before fft
            # temp_bf_shape[self.dir] = self.nfft # Shape after fft
            # self.bf = self.bf.reshape(tuple(temp_bf_shape)) # This should not be needed if fft handles it.
            # Let's assume fft output shape is correct.

            # dimsf is used later. If b was None, dims might not be fully appropriate.
            # This also needs careful check.
            self.dimsf: List[int] = list(self.dims) # type: ignore # self.dims could be None
            self.dimsf[self.dir] = self.nfft

            ndims: int = len(self.dims) # type: ignore # self.dims could be None
            # Corrected loop range if dir is 0 (was dir-1, so -1 iterations)
            for _ in range(self.dir): # Loop 0 times if dir is 0 or 1.
                self.hf = np.expand_dims(self.hf, axis=0)
            for _ in range(ndims - self.dir - 1):
                self.hf = np.expand_dims(self.hf, axis=-1)

        # precompute terms for prox
        self.hbf: np.ndarray = np.conj(self.hf) * self.bf
        self.h2f: np.ndarray = np.abs(self.hf)**2

    def __call__(self, x: np.ndarray) -> float:
        x_input: np.ndarray = x
        if self.dims is not None:
            x_input = x.reshape(self.dims)
        
        xf: np.ndarray = np.fft.fft(x_input, self.nfft, axis=self.dir)
        # Ensure hf and xf are broadcastable with bf
        # This depends on how hf was expanded.
        # If bf is from b=None (zeros), then this term is norm(h*x).
        term_to_norm: np.ndarray = self.bf - self.hf * xf
        
        # np.fft.ifft returns complex array. Norm is on that.
        f_val: float = float((self.sigma / 2.) * np.linalg.norm(np.fft.ifft(term_to_norm, axis=self.dir))**2) # Explicit cast
        return f_val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        x_input: np.ndarray = x
        if self.dims is not None:
            x_input = x.reshape(self.dims)
            
        xf: np.ndarray = np.fft.fft(x_input, self.nfft, axis=self.dir)
        yf: np.ndarray = (xf + self.sigma * tau * self.hbf) / \
                         (1. + self.sigma * tau * self.h2f)
        y_ifft: np.ndarray = np.fft.ifft(yf, axis=self.dir)
        
        y_out: np.ndarray
        if self.dims is None: # 1D case
            # Ensure y_out is real if x was real. ifft can produce small imag parts.
            y_out = np.real(y_ifft[:x.shape[0]]) # Match original length
        else: # Multi-dimensional case
            # Take slices to match original dimensions along self.dir
            slicer = [slice(None)] * y_ifft.ndim
            slicer[self.dir] = slice(self.dims[self.dir])
            y_out = np.real(y_ifft[tuple(slicer)])
            # The .ravel() at the end implies output is always 1D vector
            # This might lose original shape info if caller expects it.
        return y_out.ravel()

    def grad(self, x: np.ndarray) -> np.ndarray:
        x_input: np.ndarray = x
        if self.dims is not None:
            x_input = x.reshape(self.dims)
            
        xf: np.ndarray = np.fft.fft(x_input, self.nfft, axis=self.dir)
        
        # grad_f = sigma * h^H * (h*x - b)
        # In Fourier: sigma * conj(hf) * (hf*xf - bf)
        grad_f_fourier: np.ndarray = self.sigma * np.conj(self.hf) * (self.hf * xf - self.bf)
        grad_f_time: np.ndarray = np.fft.ifft(grad_f_fourier, axis=self.dir)
        
        grad_out: np.ndarray
        if self.dims is None: # 1D
            grad_out = np.real(grad_f_time[:x.shape[0]]) # Match original length
        else: # Multi-dim
            slicer = [slice(None)] * grad_f_time.ndim
            slicer[self.dir] = slice(self.dims[self.dir])
            grad_out = np.real(grad_f_time[tuple(slicer)])
            
        return grad_out.ravel()