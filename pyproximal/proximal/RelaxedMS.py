import numpy as np
from typing import Union, Callable, Any, List # Added List for SigmaType

from pyproximal.ProxOperator import _check_tau, ProxOperator
# Assuming _current_sigma is already typed in L1.py, if not, it needs to be.
# For clarity, let's define SigmaType and KappaType similar to how it might be in L1.py
SigmaType = Union[float, List[Any], np.ndarray, Callable[[int], Union[float, List[Any], np.ndarray]]]
KappaType = Union[float, List[Any], np.ndarray, Callable[[int], Union[float, List[Any], np.ndarray]]]
ResolvedSigmaType = Union[float, List[Any], np.ndarray]
ResolvedKappaType = Union[float, List[Any], np.ndarray]

# Assuming _current_sigma is imported and correctly typed elsewhere, or define a local version if needed.
# For this file, we need _current_sigma and a similar _current_kappa
# Re-using _current_sigma logic for _current_kappa for now.
from pyproximal.proximal.L1 import _current_sigma


def _l2(x: np.ndarray, alpha: float) -> np.ndarray:
    r"""Scaling operation.

    Applies the proximal of ``alpha||y - x||_2^2`` which is essentially a scaling operation.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    alpha : :obj:`float`
        Scaling parameter

    Returns
    -------
    y : :obj:`numpy.ndarray`
        Scaled vector

    """
    y: np.ndarray = 1 / (1 + 2 * alpha) * x
    return y


def _current_kappa(kappa: KappaType, count: int) -> ResolvedKappaType:
    if not callable(kappa):
        return kappa
    else:
        # If kappa is callable, it's Callable[[int], Union[float, list, np.ndarray]]
        # So, its return type matches ResolvedKappaType.
        return kappa(count)


class RelaxedMumfordShah(ProxOperator):
    r"""Relaxed Mumford-Shah norm proximal operator.

    Proximal operator of the relaxed Mumford-Shah norm:
    :math:`\text{rMS}(x) = \min (\alpha\Vert x\Vert_2^2, \kappa)`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`list` or :obj:`numpy.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L2 norm that controls the smoothness of the solutuon.
        This can be a constant number, a list of values (for multidimensional inputs, acting
        on the second dimension) or a function that is called passing a counter which keeps
        track of how many times the ``prox`` method has been invoked before and returns a
        scalar (or a list of) ``sigma`` to be used.
    kappa : :obj:`float` or :obj:`list` or :obj:`numpy.ndarray` or :obj:`func`, optional
        Constant value in the rMS norm which essentially controls when the norm allows a jump. This can be a
        constant number, a list of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``kappa`` to be used.

    Notes
    -----
    The :math:`rMS` proximal operator is defined as [1]_:

    .. math::
        \text{prox}_{\tau \text{rMS}}(x) =
        \begin{cases}
        \frac{1}{1+2\tau\alpha}x & \text{ if } & \vert x\vert \leq \sqrt{\frac{\kappa}{\alpha}(1 + 2\tau\alpha)} \\
        \kappa & \text{ else }
        \end{cases}.

    .. [1] Strekalovskiy, E., and D. Cremers, 2014, Real-time minimization of the piecewise smooth
            Mumford-Shah functional: European Conference on Computer Vision, 127–141.

    """
    def __init__(self, sigma: SigmaType = 1., kappa: KappaType = 1.):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.sigma_arg: SigmaType = sigma # Store original arg
        self.kappa_arg: KappaType = kappa # Store original arg
        self.count: int = 0

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar
        # Assuming _current_sigma and _current_kappa resolve to float or compatible ndarray.
        # For np.minimum and norm, scalar sigma and kappa are expected here.
        current_sigma_val = _current_sigma(self.sigma_arg, self.count)
        current_kappa_val = _current_kappa(self.kappa_arg, self.count)

        # Ensure they are float for the formula
        if not isinstance(current_sigma_val, (float, int)):
            # Handle array/list case: maybe take first element, or mean, or error.
            # For now, assume it should resolve to scalar for this __call__ context.
            # This might need refinement based on how list/array sigma/kappa are intended.
            raise TypeError("sigma in RelaxedMumfordShah __call__ must resolve to a scalar.")
        if not isinstance(current_kappa_val, (float, int)):
            raise TypeError("kappa in RelaxedMumfordShah __call__ must resolve to a scalar.")

        # np.linalg.norm(x)**2 is ||x||_2^2
        # The function is min(sigma * ||x||_2^2, kappa)
        return float(np.minimum(float(current_sigma_val) * np.linalg.norm(x)**2, float(current_kappa_val)))

    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'RelaxedMumfordShah', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # Resolve sigma and kappa for current iteration
        current_sigma_val = _current_sigma(self.sigma_arg, self.count)
        current_kappa_val = _current_kappa(self.kappa_arg, self.count)

        # Ensure they are float or ndarray compatible with element-wise operations.
        # The formula implies element-wise comparison for |x_i|.
        # If sigma/kappa are lists/arrays, they should apply element-wise or broadcast.
        # For now, assuming they resolve to types that are compatible with x for np.where.
        # If sigma/kappa are scalar, they broadcast. If arrays, must match x shape or be broadcastable.
        
        # For safety, ensure they are float if expected to be scalar for formula.
        # The formula seems to be element-wise: |x_i| <= sqrt(kappa_i/sigma_i * (...))
        # This implies sigma and kappa could be arrays. _l2 expects scalar alpha.
        # This needs careful handling if sigma/kappa are arrays.
        
        # Let's assume for now that _current_sigma and _current_kappa provide
        # either a scalar, or an ndarray that is broadcastable with x.
        # The _l2 function takes scalar alpha. So tau * current_sigma must be scalar for _l2.
        # This means current_sigma must resolve to a scalar for this prox.

        if not isinstance(current_sigma_val, (float, int)):
            # This would require element-wise _l2 or a loop if sigma is array.
            # For now, assume scalar sigma for _l2.
            raise TypeError("sigma for RelaxedMumfordShah prox must resolve to a scalar for _l2.")
        if not isinstance(current_kappa_val, (float, int, np.ndarray)): # kappa can be array for condition
             raise TypeError("kappa for RelaxedMumfordShah prox must resolve to scalar or ndarray.")
        
        sigma_resolved: float = float(current_sigma_val)
        # kappa_resolved can be float or ndarray for the condition
        kappa_resolved: Union[float, np.ndarray] = current_kappa_val


        # Condition for np.where: |x| <= sqrt(kappa/sigma * (1 + 2*tau*sigma))
        # If kappa_resolved is an array, this comparison is element-wise.
        # Ensure no division by zero if sigma_resolved is zero.
        if sigma_resolved == 0:
            # If sigma is 0, rMS(x) = min(0, kappa). If kappa >=0, rMS(x)=0. prox is x.
            # If kappa < 0 (not typical for penalty constant), rMS(x)=kappa. prox is x.
            # The condition becomes |x| <= infinity (if kappa>0) or undefined.
            # If sigma=0, the problem is ill-defined for the formula.
            # Typically sigma > 0. Let's assume sigma > 0 based on typical use.
            # If sigma is zero, the _l2 part is just x. The condition becomes problematic.
            # For now, assume sigma > 0.
            if sigma_resolved <= 0: # Strict check
                 raise ValueError("sigma must be positive for RelaxedMumfordShah prox calculation.")

        condition_threshold_sq: Union[float, np.ndarray] = \
            (kappa_resolved / sigma_resolved) * (1 + 2 * tau * sigma_resolved)
        
        # Ensure threshold is non-negative for sqrt
        condition_threshold_sq = np.maximum(0, condition_threshold_sq) # Element-wise if array
        condition_threshold: Union[float, np.ndarray] = np.sqrt(condition_threshold_sq)

        # np.where(condition, x_if_true, x_if_false)
        # x_if_true is _l2(x, tau * sigma_resolved)
        # x_if_false is x (original value, as per formula, though formula shows kappa - this is prox_f(x)=x for indicator)
        # The formula given is prox_f(x) = x if |x| > threshold, not kappa.
        # The value kappa in the rMS definition is for the function value, not its prox directly.
        # The prox result should be x or a scaled version of x.
        # The note's formula seems to imply the result is kappa if condition is false,
        # which would mean prox_{\tau f}(x) = \kappa, which is unusual for prox.
        # Proximal operator maps a vector to a vector.
        # Let's re-check the paper or common implementations for this prox.
        # Strekalovskiy 2014 paper, equation (10) is prox_{gamma*phi_alpha,beta}(p)
        # phi_alpha,beta(u) = min(alpha*u^2, beta)
        # prox_gamma*phi(p) = p / (1+2*gamma*alpha) if |p| <= sqrt(beta/alpha * (1+2*gamma*alpha))
        #                  = p                         if |p| >  sqrt(beta/alpha * (1+2*gamma*alpha))
        # This matches _l2(x, tau*sigma) for the first case, and x for the second.
        # So, the original np.where(..., _l2(...), x) is correct based on this.
        
        x_prox: np.ndarray = np.where(np.abs(x) <= condition_threshold,
                                      _l2(x, tau * sigma_resolved),
                                      x)
        return x_prox
