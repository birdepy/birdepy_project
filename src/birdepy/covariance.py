"""Observed-information covariance calculations for estimation interfaces."""

import numpy as np

from birdepy.utility import Hessian


def from_hessian(log_likelihood, estimate, bounds, require_positive_definite=False):
    """Compute covariance from a bounded finite-difference Hessian.

    ``require_positive_definite`` is used for continuous-time likelihoods,
    where a flat or nonconcave likelihood has no Gaussian covariance.  The
    default keeps the historical discrete/EM behavior, including its
    one-parameter array shape.
    """
    hessian = Hessian(log_likelihood, estimate, bounds)
    information = -np.atleast_2d(hessian)
    if not np.all(np.isfinite(information)):
        raise ValueError("Nonfinite observed information.")
    if require_positive_definite:
        information = (information + information.T) / 2
        np.linalg.cholesky(information)
        return np.linalg.inv(information)
    if np.asarray(estimate).size == 1:
        return np.array([-1 / hessian])
    return np.linalg.inv(information)
