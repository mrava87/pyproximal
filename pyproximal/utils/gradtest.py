import numpy as np
from typing import Optional, Union, Any # Added Optional, Union
from pyproximal.ProxOperator import ProxOperator # For Op type
from pyproximal.utils.bilinear import BilinearOperator # For Op type in bilinear test

from pylops.utils.backend import get_module, to_numpy # type: ignore


def gradtest_proximal(Op: ProxOperator, n: int, x: Optional[np.ndarray] = None,
                      dtype: str = "float64", delta: float = 1e-6,
                      rtol: float = 1e-6, atol: float = 1e-21,
                      complexflag: bool = False, raiseerror: bool = True,
                      verb: bool = False, backend: str = "numpy") -> bool:
    r"""Gradient test for Proximal operator.

    Compute the gradient of ``Op`` using both the provided method and a
    numerical approximation with a perturbation ``delta`` applied to a
    single, randomly selected parameter of the input vector.

    Parameters
    ----------
    Op : :obj:`pyproximal.Proximal`
        Proximal operator to test.
    n : :obj:`int`
        Size of input vector
    x : :obj:`numpy.ndarray`, optional
        Input vector (if ``None``, randomly drawn from a
        Normal distribution)
    dtype : :obj:`str`, optional
        Dtype of vector ``x`` to generate (only used when ``x=None``)
    delta : :obj:`float`, optional
        Perturbation
    rtol : :obj:`float`, optional
        Relative gradtest tolerance
    atol : :obj:`float`, optional
        Absolute gradtest tolerance
    complexflag : :obj:`bool`, optional
        Generate random vectors with real (``False``) or
        complex (``True``) entries
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    passed : :obj:`bool`
        Passed flag.

    Raises
    ------
    AssertionError
        If grad-test is not verified within chosen tolerances.

    Notes
    -----
    A gradient-test is mathematical tool used in the development of numerical
    nonliner operators.

    More specifically, a correct implementation of the gradient for
    a nonlinear operator should verify the following *equality*
    within a numerical tolerance:

    .. math::
        \frac{\partial Op(\mathbf{x})}{\partial \mathbf{x}} =
        \frac{Op(\mathbf{x}+\delta \mathbf{x})-Op(\mathbf{x})}{\delta \mathbf{x}}

    """
    ncp = get_module(backend) # ncp can be numpy or cupy module

    current_x: np.ndarray
    # get random vectors for x and y
    if x is None:
        current_x = ncp.random.normal(0., 1., n).astype(dtype) # Use ncp for array creation
        if complexflag:
            current_x = current_x + 1j * ncp.random.normal(0., 1., n).astype(dtype)
    else:
        current_x = x

    # compute function
    f_val: Any = Op(current_x) # Op.__call__ can return various types, ProxOperator.__call__ is not strictly float

    # compute gradient
    g_val: np.ndarray = Op.grad(current_x)

    # choose location of perturbation, whether to act on x or y and on real or imag part
    iqx: int = np.random.randint(0, n) # iqx must be int
    r_or_i: int = np.random.randint(0, 2) # r_or_i must be int

    delta1: Union[float, complex]
    if r_or_i == 0: # Real perturbation
        delta1 = delta
    else: # Imaginary perturbation (only makes sense if current_x can be complex)
        delta1 = delta * 1j

    # Perturb x
    # Create a copy to avoid modifying the input array if it's passed multiple times
    x_perturbed: np.ndarray = current_x.copy()
    x_perturbed[iqx] = x_perturbed[iqx] + delta1
    
    # extract gradient value to test
    grad_analytic_component: Union[float, complex] = g_val[iqx]

    # compute new function at perturbed location
    fdelta: Any = Op(x_perturbed)

    # evaluate if gradient test passed
    # Ensure fdelta and f_val are scalar (float or complex)
    f_val_scalar: Union[float, complex]
    fdelta_scalar: Union[float, complex]

    if not isinstance(f_val, (float, int, complex)):
        raise TypeError(f"Op call returned non-scalar type {type(f_val)} for f_val in gradtest_proximal.")
    f_val_scalar = f_val
    if not isinstance(fdelta, (float, int, complex)):
        raise TypeError(f"Op call returned non-scalar type {type(fdelta)} for fdelta in gradtest_proximal.")
    fdelta_scalar = fdelta

    # Numerical gradient calculation
    # grad_numerical_estimate corresponds to (f(x+pert) - f(x)) / pert_magnitude
    # If perturbation is complex (delta1 = 1j*delta), we are checking relation to grad_analytic_component.imag
    # If perturbation is real (delta1 = delta), we are checking relation to grad_analytic_component.real
    
    grad_numerical_estimate: float # This will store the scalar value to compare
    
    if r_or_i == 0: # Real perturbation (delta1 is float delta)
        # For f: C -> C or f: C -> R
        # We test d f_real / d x_real or d f_imag / d x_real
        # The comparison is with grad_analytic_component.real
        # So, numerical estimate should be real part of ( (f(x+h) - f(x)) / h )
        # If f is real-valued, (fdelta_scalar - f_val_scalar) is real.
        grad_numerical_estimate = (fdelta_scalar - f_val_scalar).real / delta
    else: # Imaginary perturbation (delta1 is complex 1j*delta)
        # We test d f_real / d x_imag or d f_imag / d x_imag
        # The comparison is with grad_analytic_component.imag
        # Numerical estimate for dG/d(Im(z)) is often Im((G(z+ih)-G(z))/h) for G:C->C
        # Or for f:C->R, (f(z+ih)-f(z))/h is often used to check Wirtinger derivative component.
        # Let's use (f(z+ih)-f(z))/h and take its imaginary part if analytic is imag part.
        # Or real part if analytic is real part.
        # Original code takes: (fdelta - f) / np.abs(delta) then compares to .imag part of analytic grad.
        # This implies (fdelta-f)/delta should be real if f is R-valued.
        # If f is C-valued, (fdelta-f)/delta is complex. Then its imag part is compared.
        # This is effectively ( (fdelta-f)/delta ).imag
        grad_numerical_estimate = ((fdelta_scalar - f_val_scalar) / delta).imag # (f(x+j*delta)-f(x))/delta -> take imag part

    grad_analytic_part_to_compare: float = grad_analytic_component.real if r_or_i == 0 else grad_analytic_component.imag
    grad_diff: float = grad_numerical_estimate - grad_analytic_part_to_compare
    passed: bool = bool(np.isclose(grad_diff, 0, rtol=rtol, atol=atol))

    # verbosity or error raising
    if (not passed and raiseerror) or verb:
        passed_status: str = "passed" if passed else "failed"
        # Ensure grad_analytic_component is float for printing if it was complex
        msg: str = f"Grad test {passed_status}, Analytic={grad_analytic_part} - Numeric={grad_numeric}"
        if not passed and raiseerror:
            raise AssertionError(msg)
        else:
            print(msg)

    return passed


def gradtest_bilinear(Op: BilinearOperator, delta: float = 1e-6,
                      rtol: float = 1e-6, atol: float = 1e-21,
                      complexflag: bool = False, raiseerror: bool = True,
                      verb: bool = False, backend: str = "numpy") -> bool:
    r"""Gradient test for Bilinear operator.

    Compute the gradient of ``Op`` using both the provided method and a
    numerical approximation with a perturbation ``delta`` applied to a
    single, randomly selected parameter of either the ``x`` or ``y``
    vectors.

    Parameters
    ----------
    Op : :obj:`pyproximal.utils.BilinearOperator`
        Bilinear operator to test.
    delta : :obj:`float`, optional
        Perturbation
    rtol : :obj:`float`, optional
        Relative gradtest tolerance
    atol : :obj:`float`, optional
        Absolute gradtest tolerance
    complexflag : :obj:`bool`, optional
        Generate random vectors with real (``False``) or
        complex (``True``) entries
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    passed : :obj:`bool`
        Passed flag.

    Raises
    ------
    AssertionError
        If grad-test is not verified within chosen tolerances.

    Notes
    -----
    A gradient-test is mathematical tool used in the development of numerical
    bilinear operators.

    More specifically, a correct implementation of the gradient for
    a bilinear operator should verify the following *equalities*
    within a numerical tolerance:

    .. math::
        \frac{\partial Op(\mathbf{x})}{\partial \mathbf{x}} =
        \frac{Op(\mathbf{x}+\delta \mathbf{x}, \mathbf{y})-
        Op(\mathbf{x})}{\delta \mathbf{x}, \mathbf{y}}

    and

    .. math::
        \frac{\partial Op(\mathbf{x}, \mathbf{y})}{\partial \mathbf{y}} =
        \frac{Op(\mathbf{x}, \mathbf{y}+\delta \mathbf{y})-
        Op(\mathbf{x}, \mathbf{y})}{\delta \mathbf{y}}

    """
    ncp = get_module(backend)

    nx = Op.sizex
    ny = Op.sizey

    # extract x and y from Op
    x, y = Op.x.ravel(), Op.y.ravel()

    # compute function at x and y
    f = Op(x, y)

    # compute gradients at x and y
    gx = Op.gradx(x)
    gy = Op.grady(y)

    # choose location of perturbation, whether to act on x or y and on real or imag part
    iqx, iqy = np.random.randint(0, nx), np.random.randint(0, ny)
    x_or_y = np.random.randint(0, 2)

    delta1 = delta
    if complexflag:
        r_or_i = np.random.randint(0, 2)
        if r_or_i == 1:
            delta1 = delta * 1j

    # extract gradient value to test
    if x_or_y == 0:
        x[iqx] = x[iqx] + delta1
        grad = gx[iqx]
    else:
        y[iqy] = y[iqy] + delta1
        grad = gy[iqy]

    # compute new function at perturbed location
    fdelta = Op(x, y)

    # evaluate if gradient test passed
    # f and fdelta are results of Op(x,y) which returns Any.
    # Assume they are scalar (float or complex) for this test.
    f_val_bilinear_sc: Union[float, complex]
    fdelta_bilinear_sc: Union[float, complex]

    if not isinstance(f, (float, int, complex)):
        raise TypeError(f"Op call returned non-scalar type {type(f)} for f_val in gradtest_bilinear.")
    f_val_bilinear_sc = f
    if not isinstance(fdelta, (float, int, complex)):
        raise TypeError(f"Op call returned non-scalar type {type(fdelta)} for fdelta in gradtest_bilinear.")
    fdelta_bilinear_sc = fdelta

    # grad is a component of gx or gy, so it's a scalar (float or complex)
    grad_analytic_bilinear_comp: Union[float, complex] = grad
    
    grad_numeric_bilinear: float
    
    # Perturbation delta1 can be float or complex.
    # If delta1 is real (r_or_i == 0 for complexflag, or complexflag is False)
    # If delta1 is imag (r_or_i == 1 for complexflag)
    if not complexflag or r_or_i == 0: # Real perturbation or not complex case
        # We compare with grad_analytic_bilinear_comp.real
        # Numerical gradient is ( f(perturbed) - f_orig ).real / delta
        grad_numeric_bilinear = (fdelta_bilinear_sc - f_val_bilinear_sc).real / delta
        analytic_part_to_compare = grad_analytic_bilinear_comp.real
    else: # Complex perturbation (imaginary part: delta1 = 1j * delta)
        # We compare with grad_analytic_bilinear_comp.imag
        # Numerical gradient is ( f(perturbed) - f_orig ).imag / delta
        grad_numeric_bilinear = (fdelta_bilinear_sc - f_val_bilinear_sc).imag / delta
        analytic_part_to_compare = grad_analytic_bilinear_comp.imag
        
    grad_diff_bilinear: float = grad_numeric_bilinear - analytic_part_to_compare
    passed: bool = bool(np.isclose(grad_diff_bilinear, 0, rtol=rtol, atol=atol))


    # verbosity or error raising
    if (not passed and raiseerror) or verb:
        passed_status = "passed" if passed else "failed"
        msg = f"Grad test {passed_status}, Analytic={grad.real if r_or_i == 0 else grad.imag} - " \
              f"Numeric={grad_delta}"
        if not passed and raiseerror:
            raise AssertionError(msg)
        else:
            print(msg)

    return passed
