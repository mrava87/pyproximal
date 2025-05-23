import numpy as np
from typing import Union, Callable, Any # Any for Optional Op in super().__init__

from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.L0 import L0BallProj, L01BallProj # Ensure correct import path
from pyproximal.proximal.L1 import _current_sigma # Assuming _current_sigma is appropriately typed in L1.py


def _hardthreshold(x: np.ndarray, thresh: float) -> np.ndarray:
    r"""Hard thresholding.

    Applies hard thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_0`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., "Computing the proximity
       operator of the Lp norm with 0 < p < 1",
       IET Signal Processing, 10, 2016.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    x1: np.ndarray = x.copy()
    x1[np.abs(x1) <= thresh] = 0 # Apply thresholding on the copy
    return x1


class L0(ProxOperator):
    r""":math:`L_0` norm proximal operator.

    Proximal operator of the :math:`\ell_0` norm:
    :math:`\sigma\|\mathbf{x}\|_0 = \text{count}(x_i \ne 0)`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`list` or :obj:`np.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L0 norm. This can be a constant number, a list
        of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``sigma`` to be used.

    Notes
    -----
    The :math:`\ell_0` proximal operator is defined as:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_0}(\mathbf{x}) =
        \operatorname{hard}(\mathbf{x}, \tau \sigma) =
        \begin{cases}
        x_i, & x_i < -\tau \sigma \\
        0, & -\tau\sigma \leq x_i \leq \tau\sigma \\
        x_i,  & x_i > \tau\sigma\\
        \end{cases}

    where :math:`\operatorname{hard}` is the so-called called *hard thresholding*.

    """
    def __init__(self, sigma: Union[float, list, np.ndarray, Callable[[int], Union[float, list, np.ndarray]]] = 1.):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.sigma: Union[float, list, np.ndarray, Callable[[int], Union[float, list, np.ndarray]]] = sigma
        self.count: int = 0

    def __call__(self, x: np.ndarray) -> int: # L0 norm returns an int (count)
        # _current_sigma is assumed to return float or np.ndarray compatible with > op
        current_sigma_val = _current_sigma(self.sigma, self.count)
        return int(np.sum(np.abs(x) > current_sigma_val))

    # Defining _increment_count as a static method or a free function if it doesn't need 'self'
    # For simplicity as a nested function for now, or could be a static method.
    # However, to be a decorator for an instance method, it needs to take 'self'.
    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'L0', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore # Decorators can be tricky for static checkers
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        current_sigma_val = _current_sigma(self.sigma, self.count)
        # Ensure tau * current_sigma_val is float for _hardthreshold
        threshold: float
        if isinstance(current_sigma_val, (np.ndarray, list)):
            # This case implies sigma can be an array, which might be an issue
            # if _hardthreshold expects a scalar threshold. Assuming _current_sigma
            # and _hardthreshold are designed to handle this.
            # For now, let's assume _current_sigma here resolves to a float,
            # or _hardthreshold can handle array thresholds element-wise.
            # If sigma is per-element, _hardthreshold would need to be element-wise.
            # The current _hardthreshold takes a scalar thresh.
            # This suggests sigma should resolve to a scalar here.
            # If sigma can be array for element-wise thresholding, _hardthreshold needs change.
            # Assuming _current_sigma provides a float or scalar np.ndarray for now.
            if isinstance(current_sigma_val, np.ndarray) and current_sigma_val.size == 1:
                threshold = float(current_sigma_val.item()) * tau
            elif isinstance(current_sigma_val, float):
                threshold = current_sigma_val * tau
            else:
                # This case would be problematic if _hardthreshold expects scalar thresh
                # and current_sigma_val is a list/array.
                # For now, proceeding with assumption it resolves to scalar or _hardthreshold is robust.
                # This might require a re-evaluation of _current_sigma's return type or _hardthreshold's design.
                # To avoid error, let's assume it should be a float, maybe average or first element if array.
                # This part is ambiguous without knowing the exact design intent of array-sigma.
                # For safety, let's raise an error or use a placeholder if it's not float.
                if isinstance(current_sigma_val, (list, np.ndarray)):
                     raise TypeError("Array/list sigma for L0 prox needs element-wise hardthreshold or scalar sigma.")
                threshold = float(current_sigma_val) * tau # Should be float
        else: # float
            threshold = current_sigma_val * tau
        
        x_prox: np.ndarray = _hardthreshold(x, threshold)
        return x_prox


class L0Ball(ProxOperator):
    r""":math:`L_0` ball proximal operator.

    Proximal operator of the L0 ball: :math:`L0_{r} =
    \{ \mathbf{x}: ||\mathbf{x}||_0 \leq r \}`.

    Parameters
    ----------
    radius : :obj:`int` or :obj:`func`, optional
        Radius. This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has been
        invoked before and returns a scalar ``radius`` to be used.

    Notes
    -----
    As the L0 ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.L0BallProj` for details.

    """
    def __init__(self, radius: Union[int, Callable[[int], int]]):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.radius_arg: Union[int, Callable[[int], int]] = radius # Store original arg
        # Initialize L0BallProj with initial radius
        initial_radius: int = radius if isinstance(radius, int) else radius(0)
        self.ball: L0BallProj = L0BallProj(initial_radius)
        self.count: int = 0

    def __call__(self, x: np.ndarray, tol: float = 1e-4) -> bool:
        # Assuming _current_sigma can handle int or Callable[[int], int] for radius
        # Cast to float first, then to int, to handle potential numpy scalar types from _current_sigma
        current_radius_val = _current_sigma(self.radius_arg, self.count)
        if isinstance(current_radius_val, (list, np.ndarray)):
            # This case should ideally not happen if radius_arg is Union[int, Callable[[int], int]]
            # and the callable returns int. Raise error or take first element.
            raise TypeError("Radius function returned a list/array, expected scalar int/float.")
        current_radius: int = int(float(current_radius_val))
        # L0 norm is the count of non-zero elements
        return bool(np.count_nonzero(np.abs(x) > tol) <= current_radius)


    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'L0Ball', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        current_radius_val = _current_sigma(self.radius_arg, self.count)
        if isinstance(current_radius_val, (list, np.ndarray)):
            raise TypeError("Radius function returned a list/array, expected scalar int/float.")
        current_radius: int = int(float(current_radius_val))
        self.ball.radius = current_radius # Update radius in L0BallProj instance
        y: np.ndarray = self.ball(x)
        return y


class L01Ball(ProxOperator):
    r""":math:`L_{0,1}` ball proximal operator.

    Proximal operator of the :math:`L_{0,1}` ball: :math:`L_{0,1}^{r} =
    \{ \mathbf{x}: \text{count}([||\mathbf{x}_1||_1, ||\mathbf{x}_2||_1, ...,
    ||\mathbf{x}_1||_1] \ne 0) \leq r \}`

    Parameters
    ----------
    ndim : :obj:`int`
        Number of dimensions :math:`N_{dim}`. Used to reshape the input array
        in a matrix of size :math:`N_{dim} \times N'_{x}` where
        :math:`N'_x = \frac{N_x}{N_{dim}}`. Note that the input
        vector ``x`` should be created by stacking vectors from different
        dimensions.
    radius : :obj:`int` or :obj:`func`, optional
        Radius. This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has been
        invoked before and returns a scalar ``radius`` to be used.

    Notes
    -----
    As the L0 ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.L01BallProj` for details.

    """
    def __init__(self, ndim: int, radius: Union[int, Callable[[int], int]]):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.ndim: int = ndim
        self.radius_arg: Union[int, Callable[[int], int]] = radius # Store original arg
        # Initialize L01BallProj with initial radius
        initial_radius: int = radius if isinstance(radius, int) else radius(0)
        self.ball: L01BallProj = L01BallProj(initial_radius)
        self.count: int = 0

    def __call__(self, x: np.ndarray, tol: float = 1e-4) -> bool:
        # Assuming x is 1D and can be reshaped
        x_reshaped: np.ndarray = x.reshape(self.ndim, len(x) // self.ndim)
        current_radius_val = _current_sigma(self.radius_arg, self.count)
        if isinstance(current_radius_val, (list, np.ndarray)):
            raise TypeError("Radius function returned a list/array, expected scalar int/float.")
        current_radius: int = int(float(current_radius_val))
        # L0 norm of L1 norms of columns
        l1_norms_of_cols: np.ndarray = np.linalg.norm(x_reshaped, ord=1, axis=0)
        # Count columns whose L1 norm is > tol
        return bool(np.count_nonzero(l1_norms_of_cols > tol) <= current_radius)

    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'L01Ball', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        # Assuming x is 1D and needs reshaping for L01BallProj
        x_reshaped: np.ndarray = x.reshape(self.ndim, len(x) // self.ndim)
        current_radius_val = _current_sigma(self.radius_arg, self.count)
        if isinstance(current_radius_val, (list, np.ndarray)):
            raise TypeError("Radius function returned a list/array, expected scalar int/float.")
        current_radius: int = int(float(current_radius_val))
        self.ball.radius = current_radius # Update radius in L01BallProj instance
        y: np.ndarray = self.ball(x_reshaped)
        return y.ravel()