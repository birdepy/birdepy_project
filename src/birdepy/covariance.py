"""Observed-information covariance calculations for estimation interfaces."""

import numpy as np

from birdepy.utility import Hessian

UNAVAILABLE = "Covariance matrix could not be determined."


class CovarianceError(ValueError):
    """Raised for invalid observed information or covariance values."""


def from_hessian(log_likelihood, estimate, bounds):
    """Return a finite, symmetric positive-definite covariance matrix.

    The bounded finite-difference Hessian supplied by :func:`utility.Hessian`
    is converted to observed information.  Flat, non-finite, or nonconcave
    likelihoods raise ``ValueError`` or ``numpy.linalg.LinAlgError``.  A
    successful result always has shape ``(n, n)``, including when ``n == 1``.
    """
    hessian = Hessian(log_likelihood, estimate, bounds)
    information = -np.atleast_2d(hessian)
    if not np.all(np.isfinite(information)):
        raise CovarianceError("Nonfinite observed information.")
    information = information / 2 + information.T / 2
    np.linalg.cholesky(information)
    covariance = np.linalg.inv(information)
    if not np.all(np.isfinite(covariance)):
        raise CovarianceError("Nonfinite covariance.")
    return covariance


def safe_from_hessian(log_likelihood, estimate, bounds):
    """Return covariance or :data:`UNAVAILABLE` for numerical failures only."""
    try:
        return from_hessian(log_likelihood, estimate, bounds)
    except (CovarianceError, np.linalg.LinAlgError, FloatingPointError):
        return UNAVAILABLE
