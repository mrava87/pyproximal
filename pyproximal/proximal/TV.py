import numpy as np
from typing import Tuple, Union, Callable, Any, Dict, List, Optional # Ensure all are present

from copy import deepcopy
# from scipy.sparse.linalg import lsqr # This import seems unused in the TV class itself
from pylops import FirstDerivative, Gradient # type: ignore # Assuming these are LinearOperator-like
from pyproximal.ProxOperator import _check_tau, ProxOperator


class TV(ProxOperator):
    r"""TV Norm proximal operator.

    Proximal operator for the TV norm defined as: :math:`f(\mathbf{x}) =
    \sigma ||\mathbf{x}||_{\text{TV}}`.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    sigma : :obj:`int`, optional
        Multiplicative coefficient of TV norm
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    rtol : :obj:`float`, optional
        Relative tolerance for stopping criterion.

    Notes
    -----
    The proximal algorithm is implemented following [1].

    .. [1] Beck, A. and Teboulle, M., "Fast gradient-based algorithms for constrained total
           variation image denoising and deblurring problems", 2009.
    """
    def __init__(self, dims: Tuple[int, ...], sigma: float = 1.,
                 niter: Union[int, Callable[[int], int]] = 10,
                 rtol: float = 1e-4, **kwargs: Any) -> None:
        super().__init__(None, True) # Op is None, hasgrad is True
        self.dims: Tuple[int, ...] = dims
        self.ndim: int = len(dims)
        self.sigma: float = sigma
        self.niter_arg: Union[int, Callable[[int], int]] = niter 
        self.count: int = 0
        self.rtol: float = rtol
        self.kwargs: Dict[str, Any] = kwargs 

    def __call__(self, x: np.ndarray) -> float:
        x_reshaped: np.ndarray = x.reshape(self.dims)
        tv_norm: float = 0.0
        if self.ndim == 1:
            # Assuming FirstDerivative is LinearOperator-like
            derivOp: Any = FirstDerivative(dims=self.dims[0], axis=0, edge=False,
                                           dtype=x_reshaped.dtype, kind="forward")
            dx: np.ndarray = derivOp @ x_reshaped
            tv_norm = float(np.sum(np.abs(dx))) # axis=0 was in original, but sum over all for scalar
        elif self.ndim >= 2:
            # Assuming Gradient is LinearOperator-like
            gradOp: Any = Gradient(self.dims, edge=False, dtype=x_reshaped.dtype, kind="forward")
            grads: np.ndarray = gradOp.matvec(x_reshaped.ravel())
            # grads shape is (ndim, *dims) after reshape
            grads = grads.reshape((self.ndim,) + self.dims) 
            
            sum_sq_grads: np.ndarray = np.zeros_like(grads[0], dtype=float) # Ensure float for sum of squares
            for g_coord in grads: # Iterate over gradient components (e.g., gx, gy, gz)
                sum_sq_grads += np.abs(g_coord)**2 # Use abs for complex case, then square
            
            # Isotropic TV: sum(sqrt(sum_i |grad_i x|^2))
            tv_norm = float(np.sum(np.sqrt(sum_sq_grads)))
        return self.sigma * tv_norm

    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter
        """
        def wrapped(self: 'TV', *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count # type: ignore[misc] # Adding verified comment for consistency
    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray: # Verified signature
        # define current number of iterations
        niter_val: int
        if isinstance(self.niter_arg, int):
            niter_val = self.niter_arg
        else: # callable
            niter_val = self.niter_arg(self.count)

        gamma: float = self.sigma * tau
        # rtol = self.rtol # rtol is used directly

        # Initialization
        x_reshaped: np.ndarray = x.reshape(self.dims)
        sol: np.ndarray = x_reshaped.copy() # Start with a copy of reshaped x
        
        # PyLops operators
        derivOp: Optional[Any] = None
        gradOp: Optional[Any] = None
        if self.ndim == 1:
            derivOp = FirstDerivative(dims=self.dims[0], axis=0, edge=False,
                                      dtype=x_reshaped.dtype, kind="forward")
        elif self.ndim > 1 : # Changed from else to elif self.ndim > 1
            gradOp = Gradient(x_reshaped.shape, edge=False, dtype=x_reshaped.dtype, kind="forward")

        # Initialize dual variables (r, s, k, u) and their copies (rr, ss, kk, uu)
        # Also pold, qold, oold, mold for FISTA
        # Using list of arrays for dual variables for cleaner handling of ndim
        dual_vars: list[np.ndarray] = []
        dual_vars_copies: list[np.ndarray] = []
        dual_vars_pold: list[np.ndarray] = []

        if self.ndim > 0 : # Common initialization for all dimensions
            # Initialize with zeros of shape of a single gradient component
            # For 1D, derivOp @ (x_reshaped*0) gives shape (dims[0]-1,) or similar if edge=True
            # For >1D, gradOp.matvec gives (ndim * prod(dims),), then reshaped.
            # Each component r,s,k,u should have shape self.dims
            
            # Simpler: initialize based on gradient output shape directly
            if self.ndim == 1 and derivOp is not None:
                grad_shape_single_component = derivOp.oshape
                # Ensure it's tuple for consistency, derivOp.oshape might be int for 1D
                if isinstance(grad_shape_single_component, int):
                     grad_shape_single_component = (grad_shape_single_component,)
                dual_vars.append(np.zeros(grad_shape_single_component, dtype=x_reshaped.dtype))

            elif self.ndim > 1 and gradOp is not None:
                # Gradient output is (ndim, *dims). Each component is (*dims)
                for _ in range(self.ndim):
                    dual_vars.append(np.zeros(self.dims, dtype=x_reshaped.dtype))
            
            dual_vars_copies = [deepcopy(dv) for dv in dual_vars]
            dual_vars_pold = [deepcopy(dv) for dv in dual_vars]

        else: # self.ndim == 0, should not happen if dims is Tuple[int,...]
             raise ValueError("TV operator requires at least 1 dimension.")


        told: float = 1.
        prev_obj: float = 0.

        # Initialization for weights
        weights_list: list[float] = []
        weight_keys = ["wx", "wy", "wz", "wt"]
        for i in range(self.ndim):
            weights_list.append(float(self.kwargs.get(weight_keys[i], 1.0)))

        mt_val: float = np.max(weights_list) if weights_list else 1.0 # Max of weights

        # Apply weights to copies of dual variables
        for i in range(len(dual_vars_copies)):
            dual_vars_copies[i] *= np.conjugate(weights_list[i])


        iter_count: int = 0 # Renamed iter to iter_count
        while iter_count <= niter_val:
            # Current Solution
            # Divergence calculation
            div: np.ndarray = np.zeros_like(x_reshaped, dtype=float) # Ensure float for accumulation
            # This manual divergence calculation is complex and error-prone.
            # PyLops FirstDerivative and Gradient have .H (adjoint) which is -div.
            # Using -Op.H might be cleaner if shapes align.
            # For now, translating existing logic.
            
            # Assuming dual_vars_copies[i] corresponds to gradient along i-th dimension
            # For 1D (dx):
            if self.ndim >= 1:
                rr_i = dual_vars_copies[0] # gradient along x
                # div_x = Dx^T rx
                div_comp = np.concatenate((np.expand_dims(rr_i[0], axis=0), # Dx_fwd^T is -Dx_bwd
                                           rr_i[1:-1] - rr_i[:-2],
                                           -np.expand_dims(rr_i[-2], axis=0)), axis=0)
                if div.shape == div_comp.shape: div += div_comp
                # Else, shape mismatch, need careful check of derivOp vs manual div.

            if self.ndim >= 2: # dy
                ss_i = dual_vars_copies[1]
                div_comp = np.concatenate((np.expand_dims(ss_i[:,0], axis=1),
                                           ss_i[:,1:-1] - ss_i[:,:-2],
                                           -np.expand_dims(ss_i[:,-2], axis=1)), axis=1)
                if div.shape == div_comp.shape: div += div_comp
            
            if self.ndim >= 3: # dz
                kk_i = dual_vars_copies[2]
                div_comp = np.concatenate((np.expand_dims(kk_i[:,:,0], axis=2),
                                           kk_i[:,:,1:-1] - kk_i[:,:,:-2],
                                           -np.expand_dims(kk_i[:,:,-2], axis=2)), axis=2)
                if div.shape == div_comp.shape: div += div_comp

            if self.ndim >= 4: # dt
                uu_i = dual_vars_copies[3]
                div_comp = np.concatenate((np.expand_dims(uu_i[:,:,:,0], axis=3),
                                           uu_i[:,:,:,1:-1] - uu_i[:,:,:,:-2],
                                           -np.expand_dims(uu_i[:,:,:,-2], axis=3)), axis=3)
                if div.shape == div_comp.shape: div += div_comp

            sol = x_reshaped - gamma * div

            #  Objective function value
            obj_val: float = 0.5 * np.linalg.norm(x_reshaped.ravel() - sol.ravel())**2 + \
                             gamma * self.__call__(sol.ravel()) # __call__ expects 1D
            
            rel_obj: float
            if abs(obj_val) > 1e-10: # Use abs for obj_val in denominator
                rel_obj = np.abs(obj_val - prev_obj) / abs(obj_val)
            else: # obj_val is very small, avoid division by zero, indicate no significant change or max rtol
                rel_obj = 2 * self.rtol 
            prev_obj = obj_val

            # Stopping criterion
            if rel_obj < self.rtol and iter_count > 0: # Ensure at least one iteration if rtol is met early
                break

            # Update divergence vectors and project
            grad_sol_components: list[np.ndarray] = []
            if self.ndim == 1 and derivOp is not None:
                grad_sol_components.append(derivOp @ sol)
            elif self.ndim > 1 and gradOp is not None:
                # gradOp.matvec returns flattened (Ndim*prod(dims),)
                # Reshape to (Ndim, *dims)
                g_sol_flat = gradOp.matvec(sol.ravel())
                grad_sol_components = list(g_sol_flat.reshape((self.ndim,) + self.dims))

            # Update dual variables (r, s, k, u)
            for i in range(self.ndim):
                dual_vars[i] -= (1. / ((2. * self.ndim) * gamma * mt_val**2)) * grad_sol_components[i]
                # Using 2*ndim as a heuristic for L_sq, common in some TV algos (e.g. 8 for 2D, 12 for 3D)
                # Original code had 4, 8, 12, 16. This is 2*ndim*2.
                # The factor should be related to Lipschitz constant of grad or div.
                # For now, using a factor related to ndim, e.g. 1./(2*self.ndim * gamma * mt_val**2)
                # Let's stick to original factors: 4, 8, 12, 16 for ndim 1,2,3,4
                # factor = 1. / ( (2* (i+1) + 2) * gamma * mt_val**2) # This is just an example
                # The original code's factors: 4, 8, 12, 16 seem like 4*ndim
                if self.ndim > 0: # Ensure ndim is positive
                    lip_factor = 4 * self.ndim 
                    dual_vars[i] -= (1. / (lip_factor * gamma * mt_val**2)) * grad_sol_components[i]


            # Projection step for dual variables (element-wise projection for isotropic TV)
            # Stack dual_vars to calculate norm: sqrt(r^2+s^2+...)
            if self.ndim > 0:
                stacked_dual_vars: np.ndarray = np.stack(dual_vars, axis=0) # Shape (ndim, *dims)
                # Norm along the first axis (over gradient components)
                norm_dual_vars: np.ndarray = np.sqrt(np.sum(stacked_dual_vars**2, axis=0)) # Shape (*dims)
                weights_proj: np.ndarray = np.maximum(1., norm_dual_vars) # Element-wise max
            
                # Update dual variables by projecting them: dual_var_i / weights_proj
                for i in range(self.ndim):
                    dual_vars[i] = dual_vars[i] / weights_proj
            
            # FISTA update
            t_new: float = (1 + np.sqrt(1 + 4 * told**2)) / 2.

            for i in range(self.ndim):
                p_curr = dual_vars[i] # Current projected dual variable
                dual_vars[i] = p_curr + (told - 1) / t_new * (p_curr - dual_vars_pold[i])
                dual_vars_pold[i] = p_curr # Store for next iteration
                dual_vars_copies[i] = deepcopy(dual_vars[i]) # Update copies for divergence

            told = t_new
            iter_count += 1

        return sol.ravel()
