"""Behavioral checks for the CPU model registry."""

import numpy as np
import pytest

import birdepy as bd
from birdepy import models
import birdepy.utility as ut
from birdepy.probability_ilt import lam_product, mu_product


PARAMETERS = {
    "Verhulst": [.8, .4, .02, .01], "Ricker": [.8, .4, .02, 2],
    "Hassell": [.8, .4, .02, 2], "MS-S": [.8, .4, .02, 2],
    "Moran": [2, 3, .1, .2, 100], "pure-birth": [.8],
    "pure-death": [.8], "Poisson": [.8], "linear": [.8, .4],
    "linear-migration": [.8, .4, .2], "M/M/1": [.8, .4],
    "M/M/inf": [.8, .4], "loss-system": [.8, .4, 5],
}


def test_registry_names_are_spec_names_and_adapters_resolve_them():
    assert set(models.MODEL_REGISTRY) == set(PARAMETERS)
    assert {spec.name for spec in models.MODEL_REGISTRY.values()} == set(PARAMETERS)
    for name, param in PARAMETERS.items():
        for state in (0, np.array([0, 1, 5])):
            assert np.shape(ut.higher_birth(name)(state, param)) == np.shape(
                models.get_model(name).birth_rate(state, param)
            )
            assert np.shape(ut.higher_death(name)(state, param)) == np.shape(
                models.get_model(name).death_rate(state, param)
            )


def test_rate_boundaries_have_known_values_and_zero_rates_broadcast():
    assert ut.higher_birth("Verhulst")(50, PARAMETERS["Verhulst"]) == 0
    assert ut.higher_birth("Verhulst")(51, PARAMETERS["Verhulst"]) == 0
    assert ut.higher_birth("Moran")(100, PARAMETERS["Moran"]) == 0
    assert ut.higher_death("Moran")(101, PARAMETERS["Moran"]) == 0
    assert ut.higher_birth("loss-system")(4, PARAMETERS["loss-system"]) == .8
    assert ut.higher_birth("loss-system")(5, PARAMETERS["loss-system"]) == 0
    states = np.array([0, 1, 2])
    for name, rate in (("pure-birth", "death_rate"),
                       ("pure-death", "birth_rate"), ("Poisson", "death_rate")):
        value = getattr(models.get_model(name), rate)(states, PARAMETERS[name])
        np.testing.assert_array_equal(value, np.zeros_like(states))


@pytest.mark.parametrize("model", sorted(set(PARAMETERS) - {"loss-system", "M/M/1"}))
def test_registered_drift_derivatives_match_rate_finite_differences(model):
    param, state, step = PARAMETERS[model], 3., 1e-5
    spec = models.get_model(model)
    expected = ((spec.birth_rate(state + step, param) - spec.death_rate(state + step, param))
                - (spec.birth_rate(state - step, param) - spec.death_rate(state - step, param))) / (2 * step)
    assert spec.require("drift_derivative")(state, param) == pytest.approx(expected)


@pytest.mark.parametrize("model", sorted(PARAMETERS))
def test_finite_fixed_points_balance_the_registered_rates(model):
    spec, param = models.get_model(model), PARAMETERS[model]
    for point in spec.require("fixed_points")(param):
        if np.isfinite(point):
            assert spec.birth_rate(point, param) == pytest.approx(spec.death_rate(point, param))


def test_capability_errors_distinguish_unknown_names_from_unsupported_ones():
    with pytest.raises(ValueError, match="Unknown model capability"):
        models.get_model("linear").require("drift_derivatve")
    with pytest.raises(models.ModelCapabilityError, match="loss-system.*drift_derivative"):
        models.get_model("loss-system").require("drift_derivative")
    with pytest.raises(TypeError, match="Unknown model"):
        ut.higher_h_fun("loss-system")
    with pytest.raises(TypeError, match="Argument 'model' has an unknown value"):
        ut.higher_birth("not-a-model")


@pytest.mark.parametrize("model,param,state", [
    ("pure-birth", [.5], 5), ("Poisson", [.8], 3),
])
def test_ilt_diagonal_matches_expm_for_constant_zero_death_rates(model, param, state):
    ilt = bd.probability(state, state, 1., param, model=model, method="ilt")[0, 0]
    expm = bd.probability(state, state, 1., param, model=model, method="expm")[0, 0]
    assert ilt == pytest.approx(expm, rel=3e-3)


def test_ilt_rate_products_broadcast_constant_custom_rates_and_keep_empty_product():
    constant = lambda z, p: p[0]
    assert lam_product(2, 5, [3], constant) == 27
    assert mu_product(5, 2, [3], constant) == 27
    assert lam_product(2, 2, [3], constant) == 1
    assert mu_product(2, 2, [3], constant) == 1
