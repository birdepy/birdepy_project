"""Serializable result objects returned by BirDePy public APIs."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeResult


class EstimationOutput(OptimizeResult):
    """The mapping-like result returned by :func:`birdepy.estimate`.

    This module-level subclass deliberately retains ``OptimizeResult``'s
    attribute and dictionary behaviour while making estimation results safe to
    serialize with :mod:`pickle`.
    """


@dataclass
class ForecastResult:
    """Numerical output of :func:`birdepy.compute_forecast`.

    ``central`` is the curve used by the legacy forecast plot: the estimated
    mean when there are no interval samples and the middle requested quantile
    otherwise. ``mean`` is the arithmetic mean of samples when samples are
    present. Fluid-method curves approximate the mean deterministically;
    simulation-based mean curves are Monte Carlo estimates. Quantiles have
    shape ``(len(times), len(percentiles))`` and samples, when present, have
    shape ``(k, len(times))``.
    """

    times: np.ndarray
    central: np.ndarray
    mean: np.ndarray
    quantiles: np.ndarray | None
    percentiles: np.ndarray
    samples: np.ndarray | None
    interval: str
    method: str
    metadata: dict = field(default_factory=dict)
