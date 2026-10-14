"""Specifications for the built-in CPU birth-and-death models.

The GPU implementation intentionally keeps its compiled kernels separate.  A
``ModelSpec`` describes the capabilities available through the CPU interfaces;
custom models provide rates only unless a caller supplies additional functions.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


Rate = Callable[[object, object], object]
FixedPoints = Callable[[object], object]


class UnknownModelError(ValueError):
    """Raised when a model name has no registered CPU specification."""


class ModelCapabilityError(ValueError):
    """Raised when a requested calculation is unsupported by a model."""


@dataclass(frozen=True)
class ModelSpec:
    """Rate functions and optional fluid-model capabilities for one model."""

    name: str
    birth_rate: Rate
    death_rate: Rate
    drift_derivative: Optional[Rate] = None
    fixed_points: Optional[FixedPoints] = None

    def require(self, capability):
        """Return a supported capability or raise a descriptive error."""
        value = getattr(self, capability, None)
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


def _constant(value):
    return lambda z, p: value


MODEL_REGISTRY = {
    "Verhulst": ModelSpec(
        "Verhulst", _verhulst_birth, _verhulst_death,
        lambda z, p: p[0] - p[1] - 2 * z * (p[0] * p[2] + p[1] * p[3]),
        lambda p: [0, (p[0] - p[1]) / (p[2] * p[0] + p[1] * p[3])],
    ),
    "Ricker": ModelSpec(
        "Ricker", _ricker_birth, _linear_death,
        lambda z, p: p[0] * np.exp(-((p[2] * z) ** p[3]))
        * (1 - p[3] * (p[2] * z) ** p[3]) - p[1],
        lambda p: [0, ((np.log(p[0] / p[1])) ** (1 / p[3])) / p[2]],
    ),
    "Hassell": ModelSpec(
        "Hassell", lambda z, p: p[0] * z / (1 + p[2] * z) ** p[3], _linear_death,
        lambda z, p: p[0] * (1 + p[2] * z * (1 - p[3]))
        / (1 + p[2] * z) ** (p[3] + 1) - p[1],
        lambda p: [0, ((p[0] / p[1]) ** (1 / p[3]) - 1) / p[2]],
    ),
    "MS-S": ModelSpec(
        "MS-S", lambda z, p: p[0] * z / (1 + (p[2] * z) ** p[3]), _linear_death,
        lambda z, p: p[0] * (1 + (p[2] * z) ** p[3] * (1 - p[3]))
        / (1 + (p[2] * z) ** p[3]) ** 2 - p[1],
        lambda p: [0, ((p[0] / p[1] - 1) ** (1 / p[3])) / p[2]],
    ),
    "Moran": ModelSpec(
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
    "pure-birth": ModelSpec(
        "pure-birth", lambda z, p: p[0] * z, _constant(0),
        lambda z, p: p[0] * (z - z + 1), lambda p: [np.nan],
    ),
    "pure-death": ModelSpec(
        "pure-death", _constant(0), lambda z, p: p[0] * z,
        lambda z, p: -p[0] * (z - z + 1), lambda p: [np.nan],
    ),
    "Poisson": ModelSpec(
        "Poisson", lambda z, p: p[0], _constant(0),
        lambda z, p: 0 * (z - z + 1), lambda p: [np.nan],
    ),
    "linear": ModelSpec("linear", lambda z, p: p[0] * z, _linear_death,
                        lambda z, p: (p[0] - p[1]) * (z - z + 1), lambda p: [0]),
    "linear-migration": ModelSpec(
        "linear-migration", lambda z, p: p[0] * z + p[2], _linear_death,
        lambda z, p: (p[0] - p[1]) * (z - z + 1),
        lambda p: [-p[2] / (p[0] - p[1])],
    ),
    "M/M/1": ModelSpec(
        "M/M/1", lambda z, p: p[0], lambda z, p: (z > 0) * p[1],
        lambda z, p: 0 * (z - z + 1), lambda p: [np.nan],
    ),
    "M/M/inf": ModelSpec(
        "M/M/inf", lambda z, p: p[0], _linear_death,
        lambda z, p: -p[1] * (z - z + 1), lambda p: [p[0] / p[1]],
    ),
    "loss-system": ModelSpec("loss-system", lambda z, p: (z < p[2]) * p[0],
                             _linear_death, None, lambda p: [np.nan]),
}


def get_model(model):
    """Return the registered built-in model specification."""
    try:
        return MODEL_REGISTRY[model]
    except KeyError as error:
        raise UnknownModelError("Unknown model: {!r}.".format(model)) from error


def resolve_model(model, b_rate=None, d_rate=None):
    """Resolve a built-in model or explicit custom CPU rate functions."""
    if model != "custom":
        return get_model(model)
    if not callable(b_rate) or not callable(d_rate):
        raise ModelCapabilityError(
            "Custom models require callable 'b_rate' and 'd_rate' functions."
        )
    return ModelSpec("custom", b_rate, d_rate)
