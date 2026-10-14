"""Parity and capability checks for the CPU model registry."""

import numpy as np
import pytest

from birdepy import models
import birdepy.utility as ut


PARAMETERS = {
    "Verhulst": [.8, .4, .02, .01],
    "Ricker": [.8, .4, .02, 2],
    "Hassell": [.8, .4, .02, 2],
    "MS-S": [.8, .4, .02, 2],
    "Moran": [2, 3, .1, .2, 100],
    "pure-birth": [.8],
    "pure-death": [.8],
    "Poisson": [.8],
    "linear": [.8, .4],
    "linear-migration": [.8, .4, .2],
    "M/M/1": [.8, .4],
    "M/M/inf": [.8, .4],
    "loss-system": [.8, .4, 5],
}


def _legacy_rates(model, z, p):
    """Independent copy of the pre-registry rate equations."""
    if model == "Verhulst":
        return ((p[2] * z) <= 1) * p[0] * (1 - p[2] * z) * z, p[1] * (1 + p[3] * z) * z
    if model == "Ricker":
        return p[0] * np.exp(-((p[2] * z) ** p[3])) * z, p[1] * z
    if model == "Hassell":
        return p[0] * z / (1 + p[2] * z) ** p[3], p[1] * z
    if model == "MS-S":
        return p[0] * z / (1 + (p[2] * z) ** p[3]), p[1] * z
    if model == "Moran":
        return (
            (z <= p[4]) * (p[4] - z)
            * (p[0] * z * (1 - p[2]) / p[4] + p[1] * (p[4] - z) * p[3] / p[4]) / p[4],
            (z <= p[4]) * z
            * (p[1] * (p[4] - z) * (1 - p[3]) / p[4] + p[0] * z * p[2] / p[4]) / p[4],
        )
    if model == "pure-birth":
        return p[0] * z, 0
    if model == "pure-death":
        return 0, p[0] * z
    if model == "Poisson":
        return p[0], 0
    if model == "linear":
        return p[0] * z, p[1] * z
    if model == "linear-migration":
        return p[0] * z + p[2], p[1] * z
    if model == "M/M/1":
        return p[0], (z > 0) * p[1]
    if model == "M/M/inf":
        return p[0], p[1] * z
    if model == "loss-system":
        return (z < p[2]) * p[0], p[1] * z
    raise AssertionError("missing reference equation")


def _legacy_drift_derivative(model, z, p):
    if model == "Verhulst":
        return p[0] - p[1] - 2 * z * (p[0] * p[2] + p[1] * p[3])
    if model == "Ricker":
        return p[0] * np.exp(-((p[2] * z) ** p[3])) * (1 - p[3] * (p[2] * z) ** p[3]) - p[1]
    if model == "Hassell":
        return p[0] * (1 + p[2] * z * (1 - p[3])) / (1 + p[2] * z) ** (p[3] + 1) - p[1]
    if model == "MS-S":
        return p[0] * (1 + (p[2] * z) ** p[3] * (1 - p[3])) / (1 + (p[2] * z) ** p[3]) ** 2 - p[1]
    if model == "Moran":
        return (p[0] * (p[4] * (1 - p[2]) - 2 * z) + p[1] * (2 * z - p[4] * (1 + p[3]))) / p[4] ** 2
    if model in {"linear", "linear-migration"}:
        return (p[0] - p[1]) * (z - z + 1)
    if model in {"M/M/1", "Poisson"}:
        return 0 * (z - z + 1)
    if model == "M/M/inf":
        return -p[1] * (z - z + 1)
    if model == "pure-birth":
        return p[0] * (z - z + 1)
    if model == "pure-death":
        return -p[0] * (z - z + 1)
    raise AssertionError("missing reference derivative")


def _legacy_fixed_points(model, p):
    if model == "Verhulst":
        return [0, (p[0] - p[1]) / (p[2] * p[0] + p[1] * p[3])]
    if model == "Ricker":
        return [0, ((np.log(p[0] / p[1])) ** (1 / p[3])) / p[2]]
    if model == "Hassell":
        return [0, ((p[0] / p[1]) ** (1 / p[3]) - 1) / p[2]]
    if model == "MS-S":
        return [0, ((p[0] / p[1] - 1) ** (1 / p[3])) / p[2]]
    if model == "Moran":
        discriminant = np.sqrt(
            (p[2] - 1) ** 2 * p[0] ** 2
            + 2 * p[0] * p[1] * (p[2] + p[3] + p[2] * p[3] - 1)
            + p[1] ** 2 * (p[3] - 1) ** 2
        )
        numerator = p[0] * (1 - p[2]) - p[1] * (1 + p[3])
        return [p[4] * (numerator + discriminant) / (2 * (p[0] - p[1])),
                p[4] * (numerator - discriminant) / (2 * (p[0] - p[1]))]
    if model == "linear":
        return [0]
    if model == "linear-migration":
        return [-p[2] / (p[0] - p[1])]
    if model == "M/M/inf":
        return [p[0] / p[1]]
    return [np.nan]


@pytest.mark.parametrize("model", sorted(PARAMETERS))
@pytest.mark.parametrize("state", [0, 1, np.array([0, 1, 5, 100, 101])])
def test_registry_rates_match_pre_registry_equations_at_scalar_vector_and_boundaries(model, state):
    p = PARAMETERS[model]
    expected_birth, expected_death = _legacy_rates(model, state, p)
    spec = models.get_model(model)
    np.testing.assert_allclose(spec.birth_rate(state, p), expected_birth)
    np.testing.assert_allclose(spec.death_rate(state, p), expected_death)
    np.testing.assert_allclose(ut.higher_birth(model)(state, p), expected_birth)
    np.testing.assert_allclose(ut.higher_death(model)(state, p), expected_death)


@pytest.mark.parametrize("model", sorted(PARAMETERS))
def test_registry_exposes_all_builtin_rate_capabilities(model):
    spec = models.get_model(model)
    assert spec.name == model
    assert callable(spec.birth_rate)
    assert callable(spec.death_rate)


@pytest.mark.parametrize("model", sorted(set(PARAMETERS) - {"loss-system"}))
@pytest.mark.parametrize("state", [0, np.array([0, 1, 100])])
def test_registry_derivatives_match_pre_registry_equations(model, state):
    p = PARAMETERS[model]
    expected = _legacy_drift_derivative(model, state, p)
    actual = models.get_model(model).require("drift_derivative")(state, p)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(ut.higher_h_fun(model)(state, p), expected)


@pytest.mark.parametrize("model", sorted(PARAMETERS))
def test_registry_fixed_points_match_pre_registry_equations(model):
    p = PARAMETERS[model]
    expected = _legacy_fixed_points(model, p)
    actual = models.get_model(model).require("fixed_points")(p)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(ut.higher_zf_bld(model)(p), expected, equal_nan=True)


def test_registry_preserves_scalar_constant_rate_shapes():
    assert isinstance(models.get_model("pure-death").birth_rate(0, [.8]), int)
    assert isinstance(models.get_model("Poisson").death_rate(np.array([0, 1]), [.8]), int)
    assert models.get_model("Poisson").drift_derivative(np.array([0, 1]), [.8]).shape == (2,)
    assert models.get_model("M/M/1").drift_derivative(np.array([0, 1]), [.8, .4]).shape == (2,)


def test_registry_exposes_known_fluid_capabilities_and_rejects_missing_ones():
    linear = models.get_model("linear")
    np.testing.assert_allclose(linear.require("drift_derivative")(np.array([0, 2]), [.8, .4]), [.4, .4])
    assert linear.require("fixed_points")([.8, .4]) == [0]
    with pytest.raises(models.ModelCapabilityError, match="loss-system.*drift_derivative"):
        models.get_model("loss-system").require("drift_derivative")
    with pytest.raises(TypeError, match="Unknown model"):
        ut.higher_h_fun("loss-system")


def test_custom_model_resolution_requires_rates_and_has_explicit_capabilities():
    birth = lambda z, p: p[0] + z
    death = lambda z, p: p[1] * z
    custom = models.resolve_model("custom", birth, death)
    assert custom.birth_rate(2, [3, 4]) == 5
    assert custom.death_rate(2, [3, 4]) == 8
    with pytest.raises(models.ModelCapabilityError, match="drift_derivative"):
        custom.require("drift_derivative")
    with pytest.raises(models.ModelCapabilityError, match="require callable"):
        models.resolve_model("custom", b_rate=birth)


def test_compatibility_adapters_preserve_unknown_model_errors():
    with pytest.raises(TypeError, match="Argument 'model' has an unknown value"):
        ut.higher_birth("not-a-model")
    with pytest.raises(TypeError, match="Unknown model"):
        ut.higher_zf_bld("not-a-model")
