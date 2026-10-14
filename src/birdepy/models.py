"""Specifications for the built-in CPU birth-and-death models.

The GPU implementation intentionally keeps its compiled kernels separate.  A
``ModelSpec`` describes the capabilities of the registered CPU built-ins.
Custom rate callables retain their existing interface-specific handling.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

_CAPABILITIES = ("birth_rate", "death_rate", "drift_derivative", "fixed_points")


class UnknownModelError(ValueError):
    """Raised when a model name has no registered CPU specification."""


class ModelCapabilityError(ValueError):
    """Raised when a requested calculation is unsupported by a model."""


@dataclass(frozen=True)
class ModelSpec:
    """Rate functions and optional fluid-model capabilities for one model."""

    name: str
    birth_rate: Callable
    death_rate: Callable
    drift_derivative: Optional[Callable] = None
    fixed_points: Optional[Callable] = None

    def require(self, capability):
        """Return a supported capability or raise a descriptive error."""
        if capability not in _CAPABILITIES:
            raise ValueError("Unknown model capability: {!r}.".format(capability))
        value = getattr(self, capability)
        if value is None:
            raise ModelCapabilityError(
                "Model {!r} does not support {!r}.".format(self.name, capability)
            )
        return value


def _verhulst_birth(z, p):
    return ((p[2] * z) <= 1) * p[0] * (1 - p[2] * z) * z


def _verhulst_death(z, p):
    return p[1] * (1 + p[3] * z) * z


def _ricker_birth(z, p):
    return p[0] * np.exp(-((p[2] * z) ** p[3])) * z


def _linear_death(z, p):
    return p[1] * z


def _moran_birth(z, p):
    return (z <= p[4]) * (
        (p[4] - z)
        * (p[0] * z * (1 - p[2]) / p[4] + p[1] * (p[4] - z) * p[3] / p[4])
        / p[4]
    )


def _moran_death(z, p):
    return (z <= p[4]) * (
        z * (p[1] * (p[4] - z) * (1 - p[3]) / p[4] + p[0] * z * p[2] / p[4]) / p[4]
    )


def _zero_rate(z, p):
    """Return zero with the shape of the requested state values."""
    return 0 * z


_MODEL_SPECS = (
    ModelSpec(
        "Verhulst", _verhulst_birth, _verhulst_death,
        lambda z, p: p[0] - p[1] - 2 * z * (p[0] * p[2] + p[1] * p[3]),
        lambda p: [0, (p[0] - p[1]) / (p[2] * p[0] + p[1] * p[3])],
    ),
    ModelSpec(
        "Ricker", _ricker_birth, _linear_death,
        lambda z, p: p[0] * np.exp(-((p[2] * z) ** p[3]))
        * (1 - p[3] * (p[2] * z) ** p[3]) - p[1],
        lambda p: [0, ((np.log(p[0] / p[1])) ** (1 / p[3])) / p[2]],
    ),
    ModelSpec(
        "Hassell", lambda z, p: p[0] * z / (1 + p[2] * z) ** p[3], _linear_death,
        lambda z, p: p[0] * (1 + p[2] * z * (1 - p[3]))
        / (1 + p[2] * z) ** (p[3] + 1) - p[1],
        lambda p: [0, ((p[0] / p[1]) ** (1 / p[3]) - 1) / p[2]],
    ),
    ModelSpec(
        "MS-S", lambda z, p: p[0] * z / (1 + (p[2] * z) ** p[3]), _linear_death,
        lambda z, p: p[0] * (1 + (p[2] * z) ** p[3] * (1 - p[3]))
        / (1 + (p[2] * z) ** p[3]) ** 2 - p[1],
        lambda p: [0, ((p[0] / p[1] - 1) ** (1 / p[3])) / p[2]],
    ),
    ModelSpec(
        "Moran", _moran_birth, _moran_death,
        lambda z, p: (p[0] * (p[4] * (1 - p[2]) - 2 * z)
        + p[1] * (2 * z - p[4] * (1 + p[3]))) / p[4] ** 2,
        lambda p: [
            p[4] * (p[0] * (1 - p[2]) - p[1] * (1 + p[3]) + np.sqrt(
                (p[2] - 1) ** 2 * p[0] ** 2
                + 2 * p[0] * p[1] * (p[2] + p[3] + p[2] * p[3] - 1)
                + p[1] ** 2 * (p[3] - 1) ** 2
            )) / (2 * (p[0] - p[1])),
            p[4] * (p[0] * (1 - p[2]) - p[1] * (1 + p[3]) - np.sqrt(
                (p[2] - 1) ** 2 * p[0] ** 2
                + 2 * p[0] * p[1] * (p[2] + p[3] + p[2] * p[3] - 1)
                + p[1] ** 2 * (p[3] - 1) ** 2
            )) / (2 * (p[0] - p[1])),
        ],
    ),
    ModelSpec(
        "pure-birth", lambda z, p: p[0] * z, _zero_rate,
        lambda z, p: p[0] * (z - z + 1), lambda p: [np.nan],
    ),
    ModelSpec(
        "pure-death", _zero_rate, lambda z, p: p[0] * z,
        lambda z, p: -p[0] * (z - z + 1), lambda p: [np.nan],
    ),
    ModelSpec(
        "Poisson", lambda z, p: p[0], _zero_rate,
        lambda z, p: 0 * (z - z + 1), lambda p: [np.nan],
    ),
    ModelSpec("linear", lambda z, p: p[0] * z, _linear_death,
                        lambda z, p: (p[0] - p[1]) * (z - z + 1), lambda p: [0]),
    ModelSpec(
        "linear-migration", lambda z, p: p[0] * z + p[2], _linear_death,
        lambda z, p: (p[0] - p[1]) * (z - z + 1),
        lambda p: [-p[2] / (p[0] - p[1])],
    ),
    ModelSpec(
        "M/M/1", lambda z, p: p[0], lambda z, p: (z > 0) * p[1],
        lambda z, p: 0 * (z - z + 1), lambda p: [np.nan],
    ),
    ModelSpec(
        "M/M/inf", lambda z, p: p[0], _linear_death,
        lambda z, p: -p[1] * (z - z + 1), lambda p: [p[0] / p[1]],
    ),
    ModelSpec("loss-system", lambda z, p: (z < p[2]) * p[0],
              _linear_death, fixed_points=lambda p: [np.nan]),
)

MODEL_REGISTRY = {spec.name: spec for spec in _MODEL_SPECS}


def get_model(model):
    """Return the registered built-in model specification."""
    try:
        return MODEL_REGISTRY[model]
    except KeyError as error:
        raise UnknownModelError("Unknown model: {!r}.".format(model)) from error
