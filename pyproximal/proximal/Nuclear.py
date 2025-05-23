import numpy as np
from typing import Tuple, Union, Any # Any for Optional Op in super().__init__

from pylops.optimization.cls_sparsity import _softthreshold # Assuming this is correctly typed
from pyproximal.ProxOperator import _check_tau, ProxOperator
from pyproximal.projection.Nuclear import NuclearBallProj # Ensure correct import path


class Nuclear(ProxOperator):
    r"""Nuclear norm proximal operator.

    The nuclear norm is defined as
    :math:`\sigma\|\mathbf{X}\|_* = \sigma \sum_i \lambda_i` where :math:`\mathbf{X}`
    is a matrix of size :math:`M \times N` and :math:`\lambda_i` is the *i*:th
    singular value of :math:`\mathbf{X}`, where :math:`i=1,\ldots, \min(M, N)`.

    The *weighted* nuclear norm, with the positive weight vector :math:`\boldsymbol\sigma`, is
    defined as

    .. math::

         \|\mathbf{X}|\|_{{\boldsymbol\sigma},*} = \sum_i \sigma_i\lambda_i(\mathbf{X}) .

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of matrix :math:`\mathbf{X}`
    sigma : :obj:`float` or :obj:`numpy.ndarray`, optional
        Multiplicative coefficient of the nuclear norm penalty. If ``sigma`` is a float
        the same penalty is applied for all singular values. If instead ``sigma`` is an
        array the weight ``sigma[i]`` will be applied to the *i*:th singular value.
        This is often referred to as the *weighted nuclear norm*.

    Notes
    -----
    The nuclear norm proximal operator is:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_*}(\mathbf{X}) =
        \mathbf{U} \diag \{ \prox_{\tau \sigma \|\cdot\|_1}(\boldsymbol\lambda) \} \mathbf{V}^H

    where :math:`\mathbf{U}`, :math:`\boldsymbol\lambda`, and
    :math:`\mathbf{V}` define the SVD of :math:`X`.

    The weighted nuclear norm is convex if the sequence :math:`\{\sigma_i\}_i` is
    non-ascending, but is in general non-convex; however, when the weights are
    non-descending it can be shown that applying the soft-thresholding operator on the
    singular values still yields a fixed point (w. r. t. a specific algorithm), see
    [1]_ for details.

    .. [1] Gu et al. "Weighted Nuclear Norm Minimization with Application to Image
        Denoising", In the IEEE Conference on Computer Vision and Pattern Recognition,
        2862-2869, 2014.

    """

    def __init__(self, dim: Tuple[int, ...], sigma: Union[float, np.ndarray] = 1.):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.dim: Tuple[int, ...] = dim
        self.sigma: Union[float, np.ndarray] = sigma

    def __call__(self, x: np.ndarray) -> float: # Returns a scalar
        X: np.ndarray = x.reshape(self.dim)
        # Nuclear norm is sum of singular values. Singular values are sqrt of eigenvalues of X.T @ X.
        # np.linalg.eigvalsh returns eigenvalues in ascending order.
        eigs: np.ndarray = np.linalg.eigvalsh(X.T @ X)
        eigs[eigs < 0] = 0  # Ensure all eigenvalues are positive (or zero)
        singular_values: np.ndarray = np.sqrt(eigs)
        
        # If self.sigma is an array, it's weighted nuclear norm.
        # Typically, weights are applied to singular values in their standard (descending) order.
        # eigvalsh gives ascending, so np.flip is needed if sigma corresponds to descending order.
        # Assuming standard definition where sigma applies to sorted singular values.
        # The original code uses np.flip(self.sigma), implying self.sigma is ordered for descending singular values.
        # Or, if sigma is scalar, it's just sigma * sum(singular_values).
        
        # For weighted norm, ensure sigma aligns with singular_values.
        # Singular values from sqrt(eigvalsh) are ascending. If sigma is for descending, flip one of them.
        # Let's assume sigma is provided in an order that matches singular_values after sorting (e.g. ascending).
        # If self.sigma is an array, its length should match number of singular values.
        # Number of singular values is min(self.dim).
        
        # Original code: np.sum(np.flip(self.sigma) * np.sqrt(eigs))
        # This implies self.sigma is ordered for descending singular values, and eigs (so sqrt(eigs)) are ascending.
        # To match this, we sort singular_values descending if sigma is array.
        # Or, if sigma is scalar, just sum singular values.
        
        current_sigma: Union[float, np.ndarray] = self.sigma
        
        if isinstance(current_sigma, np.ndarray):
            # singular_values are already sorted in ascending order from np.linalg.eigvalsh
            # If current_sigma is for descending singular values, it should be flipped.
            num_svals: int = len(singular_values)
            sigma_adj: np.ndarray
            if len(current_sigma) > num_svals:
                sigma_adj = current_sigma[:num_svals]
            elif len(current_sigma) < num_svals:
                # If sigma is shorter, pad with its last value (or handle error/warning)
                # This matches a common convention for weighted norms if weights run out.
                # Or, more simply, assume it will broadcast if correctly shaped (e.g. (1,)) or error.
                # For explicit sum, ensure lengths match or broadcasting is intended.
                # The original code didn't explicitly handle this padding for the __call__.
                # For now, let's assume if current_sigma is an array, its length is appropriate
                # or it should be truncated. Truncation is safer.
                sigma_adj = current_sigma[:min(len(current_sigma), num_svals)]
                # If sigma_adj is now shorter than singular_values, the product will broadcast
                # or error depending on exact shapes. To be safe, match lengths if possible.
                # This part is tricky. The original sum(np.flip(sigma_adj)*singular_values) implies lengths match.
                # Let's stick to simple truncation for sigma_adj if it's longer.
                # If it's shorter, the user should ensure it's compatible or it's a scalar.
                # A common way is to use the first len(sigma_adj) singular values.
                # For now, just truncate sigma_adj if it's longer.
                # The `np.flip` is applied to `sigma_adj` to match ascending `singular_values`.
                if len(sigma_adj) < num_svals and len(sigma_adj) > 0 : # if sigma is shorter and not empty
                    # This case needs clarification: what if len(sigma_adj) < num_svals?
                    # Using only the first len(sigma_adj) singular values? Or error?
                    # For now, let's assume this means we operate on the first len(sigma_adj) values.
                    # This is not what np.flip and then sum would do if lengths mismatch without broadcasting.
                    # Let's assume that if sigma is an array, it's meant to be the same size as singular_values
                    # or will be truncated/broadcasted correctly by numpy's rules.
                    # The flip implies sigma is ordered for descending singular values.
                    # So we flip it to match ascending singular_values.
                     pass # sigma_adj is set
                elif len(sigma_adj) == 0 and num_svals > 0: # Empty sigma array for non-empty singular values
                    return 0.0 # Or raise error
            else: # Lengths are compatible or sigma_adj is scalar-like from previous logic.
                sigma_adj = current_sigma

            # Ensure sigma_adj is a numpy array for np.flip
            if not isinstance(sigma_adj, np.ndarray): # Should not happen if current_sigma was ndarray
                sigma_adj = np.array(sigma_adj)

            # The sum involves element-wise product. If sigma_adj is shorter than singular_values
            # after all adjustments, numpy will attempt broadcasting or raise an error.
            # To be safe, ensure they are compatible for element-wise product.
            # A robust way: if len(sigma_adj) < len(singular_values), take only first len(sigma_adj) singular_values
            # or pad sigma_adj.
            # Given the flip, it's likely they are expected to be of same length.
            min_len = min(len(sigma_adj), len(singular_values))
            result = np.sum(np.flip(sigma_adj[:min_len]) * singular_values[:min_len])
            # Add remaining terms if singular_values is longer (unweighted sum, or error, or use last sigma)
            # This part of logic is complex if sigma can be arbitrarily sized array.
            # Assuming for now that if sigma is array, it's meant to be compatible.
            # The original code was: return float(np.sum(np.flip(sigma_adj) * singular_values))
            # This implies sigma_adj and singular_values will broadcast or have same shape.
            # If sigma_adj comes from self.sigma[:num_svals], it should be fine.
            return float(np.sum(np.flip(sigma_adj) * singular_values))

        else: # current_sigma is float
            return float(current_sigma * np.sum(singular_values))


    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        X: np.ndarray = x.reshape(self.dim)
        U: np.ndarray
        S_vec: np.ndarray # Singular values vector
        Vh: np.ndarray
        U, S_vec, Vh = np.linalg.svd(X, full_matrices=False)
        
        # Determine sigma for soft-thresholding: scalar or array
        current_sigma: Union[float, np.ndarray]
        if np.isscalar(self.sigma):
            current_sigma = float(self.sigma)
        else: # self.sigma is np.ndarray
            # Ensure sigma is correctly sized for the number of singular values
            if len(self.sigma) >= S_vec.size:
                current_sigma = self.sigma[:S_vec.size]
            else:
                # If sigma array is shorter than S_vec, how to handle?
                # Option 1: Pad sigma (e.g., with its last value or zeros)
                # Option 2: Error or warning
                # Option 3: Use only the provided part (will fail in _softthreshold if shapes mismatch)
                # For now, assume it's either scalar or correctly sized (or larger, then truncated).
                # This behavior should be clarified by docs or examples if sigma can be shorter.
                # To match original: sigma = self.sigma if np.isscalar(self.sigma) else self.sigma[:S.size]
                # This implies truncation if self.sigma is longer, or direct use if shorter (potential broadcast error).
                # Let's ensure it's correctly sliced or scalar.
                current_sigma = self.sigma # Keep as is, _softthreshold might handle broadcasting if needed
                                          # or it assumes sigma is scalar or S_vec sized.
                                          # _softthreshold typically takes scalar threshold.
                                          # If sigma is array, tau*sigma is element-wise.
        
        # _softthreshold expects a scalar threshold, or if x is array and thresh is array,
        # it should be element-wise.
        # Here, S_vec is 1D array of singular values.
        # tau * current_sigma will be scalar if current_sigma is scalar,
        # or array if current_sigma is array (element-wise multiplication).
        S_thresholded: np.ndarray = _softthreshold(S_vec, tau * current_sigma) # type: ignore
        
        X_prox: np.ndarray = np.dot(U * S_thresholded, Vh) # U * S_thresholded is like U @ diag(S_thresholded)
        return X_prox.ravel()


class NuclearBall(ProxOperator):
    r"""Nuclear ball proximal operator.

    Proximal operator of the Nuclear ball: :math:`N_{r} =
    \{ \mathbf{X}: \|\mathbf{X}\|_* \leq r \}`.

    Parameters
    ----------
    dims : :obj:`tuple`
        Dimensions of input matrix
    radius : :obj:`float`
        Radius
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by :func:`scipy.optimize.bisect`
    xtol : :obj:`float`, optional
        Absolute tolerance of :func:`scipy.optimize.bisect`

    Notes
    -----
    As the Nuclear ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.NuclearBallProj` for details.

    """
    def __init__(self, dims: Tuple[int, ...], radius: float,
                 maxiter: int = 100, xtol: float = 1e-5):
        super().__init__(None, False) # Op is None, hasgrad is False
        self.dims: Tuple[int, ...] = dims
        self.radius: float = radius
        self.maxiter: int = maxiter
        self.xtol: float = xtol
        # NuclearBallProj constructor takes n (min of dims), radius, maxiter, xtol
        self.ball: NuclearBallProj = NuclearBallProj(min(self.dims), self.radius,
                                                     self.maxiter, self.xtol)

    def __call__(self, x: np.ndarray, tol: float = 1e-5) -> bool:
        # Check if nuclear norm of X (reshaped x) is within radius (plus tolerance)
        X: np.ndarray = x.reshape(self.dims)
        return bool(np.linalg.norm(X, ord='nuc') - self.radius < tol)

    @_check_tau
    def prox(self, x: np.ndarray, tau: float) -> np.ndarray:
        # tau is not used in projection for indicator function
        # Reshape x to matrix, apply projection, then flatten back
        X_reshaped: np.ndarray = x.reshape(self.dims)
        y_projected: np.ndarray = self.ball(X_reshaped)
        return y_projected.ravel()
