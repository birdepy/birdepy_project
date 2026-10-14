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


@dataclass(eq=False)
class ForecastResult:
    """Numerical output of :func:`birdepy.compute_forecast`.

    ``central`` is the curve used by the legacy forecast plot: the estimated
    mean when there are no interval samples and the sample median
    otherwise. ``mean`` is the arithmetic mean of samples when samples are
    present. Fluid-method curves approximate the mean deterministically;
    simulation-based mean curves are Monte Carlo estimates. Quantiles have
    shape ``(len(times), len(percentiles))`` and samples, when present, have
    shape ``(k, len(times))``. Metadata records the actual ``n_samples`` and
    ``n_trajectories`` used (zero trajectories for fluid forecasts). Equality
    uses object identity; compare individual arrays with NumPy when needed.
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

    def __post_init__(self):
        """Reject inconsistent array shapes at the result boundary."""
        if np.ndim(self.times) != 1 or np.ndim(self.percentiles) != 1:
            raise ValueError("times and percentiles must be one-dimensional.")
        size = len(self.times)
        if np.shape(self.central) != (size,) or np.shape(self.mean) != (size,):
            raise ValueError("central and mean must have one value per time.")
        if self.quantiles is not None and np.shape(self.quantiles) != (size, len(self.percentiles)):
            raise ValueError("quantiles must have shape (times, percentiles).")
        if self.samples is not None and (np.ndim(self.samples) != 2 or self.samples.shape[1] != size):
            raise ValueError("samples must have shape (samples, times).")
