import numpy as np
from typing import Callable, Optional, Any, Union
from pylops.LinearOperator import LinearOperator # type: ignore[import-untyped]


def _check_tau(func: Callable[..., Any]) -> Callable[..., Any]: # More generic Callable
    """Check that tau>0

    This utility function is used to decorate every prox and dualprox method
    to check that tau is positive before performing any computation

    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if np.any(args[2] <= 0):
            raise ValueError('tau must be positive')
        return func(*args, **kwargs)
    return wrapper


class ProxOperator:
    r"""Common interface for proximal operators of a function.

    This class defines the overarching structure of any proximal operator. It
    contains two main methods, ``prox`` and ``dualprox`` which are both
    implemented by means of the Moreau decomposition assuming explicit
    knowledge of the other method. For this reason any proximal operators that
    subclasses the ``ProxOperator`` class needs at least one of these two
    methods to be implemented directly.

    Moreover, the method ``grad`` is also defined to compute the gradient of
    the Moreau envelope of the function. This function is only called if the
    user does not provide a gradient function when creating the proximal operator.
    The variable ``hasgrad`` is used to indicate if the function has a gradient
    or not (and thus if the ``grad`` method computes the gradient of the actual
    function or of its Moreau envelope).

    .. note:: End users of PyProximal should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyProximal. Find more details regarding implementation of
      new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator used by the Proximal operator
    hasgrad : :obj:`bool`, optional
        Flag to indicate if the function is differentiable, i.e., has a
        uniquely defined gradient (``True``) or not (``False``).
    sigmame : :obj:`float`, optional
        Relaxation parameter of the Moreau envelope (when ``sigmame`` tends to infinity
        the gradient of the Moreau envelope tends to the gradient of the function itself).
        Refer to the docstring of the ``grad`` method for more details.

    Notes
    -----
    The proximal operator of a function ``f`` is defined as:

    .. math::

        prox_{\tau f} (\mathbf{x}) = \argmin_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2

    """
    def __init__(self, Op: Optional[LinearOperator] = None, hasgrad: bool = False, sigmame: float = 1.) -> None:
        self.Op: Optional[LinearOperator] = Op
        self.hasgrad: bool = hasgrad
        self.sigmame: float = sigmame

    def __call__(self, x: np.ndarray) -> float: # Changed Any to float
        """Apply the operator (functional evaluation).
        Subclasses should implement this. Returns the value of the function.
        """
        # This base implementation is a placeholder for type checking.
        # Specific ProxOperator subclasses should provide a meaningful implementation
        # if they are intended to be callable for functional evaluation, returning a float.
        # For example, an indicator function might return 0.0 or np.inf.
        # A norm might return its computed value.
        raise NotImplementedError("This ProxOperator's __call__ method (functional evaluation) "
                                  "must be implemented by subclasses to return a float.")

    @_check_tau
    def _prox_moreau(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        """Proximal operator applied to a vector via Moreau decomposition

        """
        p: np.ndarray = x - tau * self.proxdual(x / tau, 1. / tau, **kwargs)
        return p

    @_check_tau
    def _proxdual_moreau(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        """Dual proximal operator applied to a vector via Moreau decomposition

        """
        pdual: np.ndarray = x - tau * self.prox(x / tau, 1. / tau, **kwargs)
        return pdual

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        """Proximal operator applied to a vector

        The proximal operator can always be computed given its dual
        proximal operator using the Moreau decomposition as defined in
        :func:`pyproximal.moreau`. For this reason we can easily create a common
        method for all proximal operators that can be evaluated provided the
        dual proximal is implemented.

        However, direct implementations are generally available. This can
        be done by simply implementing ``prox`` for a specific proximal
        operator, which will overwrite the general method.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        """
        return self._prox_moreau(x, tau, **kwargs)

    @_check_tau
    def proxdual(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        """Dual proximal operator applied to a vector

        The dual of a proximal operator can always be computed given its
        proximal operator using the Moreau decomposition as defined in
        :func:`pyproximal.moreau`. For this reason we can easily create a common
        method for all dual proximal operators that can be evaluated provided
        the proximal is implemented.

        However, since the dual of a proximal operator of a function is
        equivalent to the proximal operator of the conjugate function, smarter
        and faster implementation may be available in special cases. This can
        be done by simply implementing ``proxdual`` for a specific proximal
        operator, which will overwrite the general method.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        """
        return self._proxdual_moreau(x, tau, **kwargs)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of the Moreau envelope of the function.

        This method is only called if the user does not provide a gradient
        because the function is not differentiable. In this case, the gradient
        of the Moreau envelope of the function is computed instead:

        .. math::

            \nabla_\mathbf{x} M_{\sigma f) = 
            \frac{1}{sigma} (\mathbf{x} - \prox_{\sigma f}(\mathbf{x}))

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        
        Returns
        -------
        g : :obj:`np.ndarray`
            Gradient vector

        """
        g: np.ndarray = (x - self.prox(x, self.sigmame)) / self.sigmame
        return g
    
    def affine_addition(self, v: np.ndarray) -> '_SumOperator':
        """Affine addition

        Adds the dot-product of vector ``v`` and vector ``x`` (which is passed
        to ``dual`` or ``proxdual``) to the current function.

        This method can also be accessed via the ``+`` operator.

        Parameters
        ----------
        v : :obj:`np.ndarray`
            Vector

        Notes
        -----
        The proximal operator of a function :math:`g=f(\mathbf{x}) +
        \mathbf{v}^T \mathbf{x}` is defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) =
            prox_{\tau f} (\mathbf{x} - \tau \mathbf{v})

        """
        if isinstance(v, np.ndarray):
            return _SumOperator(self, v)
        else:
            raise NotImplementedError('v must be of type numpy.ndarray')

    def postcomposition(self, sigma: float) -> '_PostcompositionOperator':
        r"""Postcomposition

        Multiplies a scalar ``sigma`` to the current function.

        This method can also be accessed via the ``*`` operator.

        Parameters
        ----------
        sigma : :obj:`float`
            Scalar

        Notes
        -----
        The proximal operator of a function :math:`g= \sigma f(\mathbf{x})` is
        defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) =
            prox_{\sigma \tau f} (\mathbf{x})

        """
        if isinstance(sigma, float):
            return _PostcompositionOperator(self, sigma)
        else:
            raise NotImplementedError('sigma must be of type float')

    def precomposition(self, a: float, b: Union[float, np.ndarray]) -> '_PrecompositionOperator':
        r"""Precomposition

        Multiplies and add scalars ``a`` and ``b`` to ``x`` when evaluating
        the proximal function

        Parameters
        ----------
        a : :obj:`float`
            Multiplicative scalar
        b : :obj:`float` or obj:`np.ndarray`
            Additive scalar (or vector)

        Notes
        -----
        The proximal operator of a function :math:`g= f(a \mathbf{x} + b)` is
        defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) = \frac{1}{a} (
            prox_{a^2 \tau f} (a \mathbf{x} + b) - b)

        """
        if isinstance(a, float) and isinstance(b, (float, np.ndarray)):
            return _PrecompositionOperator(self, a, b)
        else:
            raise NotImplementedError('a must be of type float and b '
                                      'must be of type float or '
                                      'numpy.ndarray')

    def chain(self, g: 'ProxOperator') -> '_ChainOperator':
        r"""Chain

        Chains two proximal operators. This must be used with care only when
        aware that the combination of two proximal operators can be simply
        obtained by chaining them

        Parameters
        ----------
        g : :obj:`pyproximal.proximal.ProxOperator`
            Rigth operator

        Notes
        -----
        The proximal operator of the chain of two operators is defined as:

        .. math::

            prox_{\tau f g} (\mathbf{x}) = prox_{\tau g}(prox_{\tau f g}(x))

        """
        return _ChainOperator(self, g)

    def __add__(self, v: np.ndarray) -> '_SumOperator':
        return self.affine_addition(v)

    def __sub__(self, v: np.ndarray) -> '_SumOperator':
        return self.__add__(-v)

    def __rmul__(self, sigma: Union[float, int, 'ProxOperator']) -> Union['_PostcompositionOperator', '_ChainOperator']:
        if isinstance(sigma, (int, float)):
            return self.postcomposition(float(sigma))
        else:
            return self.chain(sigma)

    #__rmul__ = __mul__

    def _adjoint(self) -> '_AdjointOperator':
        """Adjoint operator - swaps prox and proxdual"""
        return _AdjointOperator(self)

    H = property(_adjoint)


class _AdjointOperator(ProxOperator):
    def __init__(self, f: ProxOperator) -> None:
        self.f: ProxOperator = f
        super().__init__(None, True if f.hasgrad else False)

    def __call__(self, x: np.ndarray) -> float: # Return type consistent with base
        # self.f is a ProxOperator, its __call__ should return float.
        return self.f(x)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        return self.f.proxdual(x, tau, **kwargs)

    @_check_tau
    def proxdual(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        return self.f.prox(x, tau, **kwargs)


class _SumOperator(ProxOperator):
    def __init__(self, f: ProxOperator, v: np.ndarray) -> None:
        #if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(v, np.ndarray):
            raise ValueError('Second input must be a numpy array')
        self.f: ProxOperator = f
        self.v: np.ndarray = v
        super().__init__(None, True if f.hasgrad else False)

    def __call__(self, x: np.ndarray) -> float: # Return type consistent with base
        # self.f(x) returns float, np.dot(self.v, x) returns float or compatible.
        val: float = self.f(x) + np.dot(self.v, x)
        return val

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        return self.f.prox(x - tau * self.v, tau)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.f.grad(x) + self.v


class _ChainOperator(ProxOperator):
    def __init__(self, f: ProxOperator, g: ProxOperator) -> None:
        #if not isinstance(f, ProxOperator) or not isinstance(g, ProxOperator):
        #    raise ValueError('Inputs must be a ProxOperator')
        self.f: ProxOperator = f
        self.g: ProxOperator = g
        super().__init__(None, True if f.hasgrad else False)

    def __call__(self, x: np.ndarray) -> float: # Return type consistent with base
        # If a chain operator is to be callable for evaluation,
        # it implies f(g(input_to_g)).
        # This would require g's __call__ to return an np.ndarray suitable for f's __call__.
        # This is generally not true; __call__ returns float.
        # Thus, a generic chain's __call__ value is ill-defined without more structure.
        # Raising NotImplementedError via super() is appropriate.
        return super().__call__(x)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        # Assuming self.f and self.g are ProxOperator instances
        # and their prox methods are correctly typed.
        # The original logic: self.g.prox(self.f.prox(x, tau), tau)
        # This implies f.prox is applied first, then g.prox.
        # However, chain rule for prox is usually prox_f(prox_g(x)) if it's f(g(x)).
        # Or if it's (f+g)(x) and they operate on different parts, it's different.
        # The note says: prox_fg(x) = prox_g(prox_f(x)) - this seems to be prox_{f circ g}
        # The implementation prox_g(prox_f(x, tau), tau) seems to apply f then g.
        # Let's assume the implementation is what's intended.
        intermediate_x = self.f.prox(x, tau, **kwargs) # Pass kwargs
        return self.g.prox(intermediate_x, tau, **kwargs) # Pass kwargs

    def grad(self, x: np.ndarray) -> np.ndarray: # Changed from None to np.ndarray
        # The gradient of a chain f(g(x)) is f'(g(x)) * g'(x).
        # This is complex for general proximal operators.
        # If grad is not well-defined or always available for a chain,
        # raising NotImplementedError is appropriate.
        # The base class expects np.ndarray.
        # If this operator *can* have a gradient (e.g., if f and g are differentiable),
        # it should be computed. Otherwise, this breaks the Liskov Substitution Principle.
        # For now, to satisfy mypy, we must return np.ndarray or ensure base allows Optional.
        # Raising an error is safer if it's not generally computable.
        # However, the original code had 'pass', implying it might not always be used
        # or subclasses are expected to override if they support grad.
        # The error 'Return type "None" ... incompatible with ... "ndarray"'
        # means this must return an ndarray.
        # If a general gradient for a chain cannot be provided, this is problematic.
        # A placeholder that matches type:
        # return np.zeros_like(x) # Or handle more meaningfully if possible
        raise NotImplementedError("Gradient for a generic chain of operators is not implemented.")


class _PostcompositionOperator(ProxOperator):
    def __init__(self, f: ProxOperator, sigma: float) -> None:
        #if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(sigma, float):
            raise ValueError('Second input must be a float')
        self.f: ProxOperator = f
        self.sigma: float = sigma
        super().__init__(None, True if f.hasgrad else False)

    def __call__(self, x: np.ndarray) -> float: # Return type consistent with base
        return self.sigma * self.f(x)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        return self.f.prox(x, self.sigma * tau)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.sigma * self.f.grad(x)


class _PrecompositionOperator(ProxOperator):
    def __init__(self, f: ProxOperator, a: float, b: Union[float, np.ndarray]) -> None:
        #if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(a, float):
            raise ValueError('Second input must be a float')
        if not isinstance(b, (float, np.ndarray)):
            raise ValueError('Second input must be a float')
        self.f: ProxOperator = f
        self.a: float = a
        self.b: Union[float, np.ndarray] = b
        super().__init__(None, True if f.hasgrad else False)

    def __call__(self, x: np.ndarray) -> float: # Return type consistent with base
        # self.f is called with an np.ndarray, so it should work.
        return self.f(self.a * x + self.b)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float, **kwargs: Any) -> np.ndarray:
        return (self.f.prox(self.a * x + self.b, (self.a ** 2) * tau) -
                self.b) / self.a

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.a * self.f.grad(self.a * x + self.b)