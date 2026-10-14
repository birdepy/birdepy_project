"""Regression tests for optimizer setup and shared observation validation."""

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, OptimizeResult

import birdepy as bd
from birdepy.configuration import optimizer_options
import birdepy.utility as ut
from birdepy.utility import data_sort


def test_optimizer_defaults_are_solver_specific_and_do_not_mutate_user_options():
    supplied = {"maxiter": 7, "likelihood": "expm"}
    de_options = optimizer_options(supplied, "differential-evolution")
    lbfgsb_options = optimizer_options(supplied, "L-BFGS-B")

    assert supplied == {"maxiter": 7, "likelihood": "expm"}
    assert de_options["maxiter"] == 7
    assert de_options["popsize"] == 15
    assert "maxcor" not in de_options
    assert lbfgsb_options["maxiter"] == 7
    assert lbfgsb_options["maxcor"] == 10
    assert "popsize" not in lbfgsb_options
    assert lbfgsb_options["likelihood"] == "expm"


def test_optimizer_dispatch_uses_solver_defaults_without_mutating_options(monkeypatch):
    de_call, slsqp_call = {}, {}

    def fake_de(*args, **kwargs):
        de_call.update(kwargs)
        return OptimizeResult(x=np.array([1.]))

    def fake_minimize(*args, **kwargs):
        slsqp_call.update(kwargs)
        return OptimizeResult(x=np.array([1.]))

    monkeypatch.setattr(ut, "differential_evolution", fake_de)
    monkeypatch.setattr(ut, "minimize", fake_minimize)
    de_options = {"polish": False}
    slsqp_options = {}
    ut.minimize_(lambda p: p[0] ** 2, [1.], [[0., 2.]], (),
                 "differential-evolution", de_options)
    ut.minimize_(lambda p: p[0] ** 2, [1.], [[0., 2.]], (), "SLSQP", slsqp_options)

    assert de_call["maxiter"] == 1000
    assert de_call["seed"] is None
    assert slsqp_call["options"] == {
        "ftol": 1e-06, "eps": 1.4901161193847656e-08,
        "disp": False, "maxiter": 100, "finite_diff_rel_step": None,
    }
    assert de_options == {"polish": False}
    assert slsqp_options == {}


def test_differential_evolution_adapts_legacy_constraints_and_preserves_polish_input(monkeypatch):
    calls = {}

    def fake_de(*args, **kwargs):
        calls["de"] = kwargs
        return OptimizeResult(x=np.array([.75]))

    def fake_minimize(*args, **kwargs):
        calls["polish"] = kwargs
        return OptimizeResult(x=np.array([.75]))

    constraints = (
        {"type": "ineq", "fun": lambda x, offset: x[0] - offset, "args": (.5,)},
        {"type": "eq", "fun": lambda x: x[0] - .75},
    )
    monkeypatch.setattr(ut, "differential_evolution", fake_de)
    monkeypatch.setattr(ut, "minimize", fake_minimize)
    ut.minimize_(lambda p: p[0] ** 2, [1.], [[0., 2.]], constraints,
                 "differential-evolution", {"polish": True})

    adapted = calls["de"]["constraints"]
    assert len(adapted) == 2
    assert adapted[0].fun([.6]) == pytest.approx(.1)
    assert adapted[1].fun([.75]) == pytest.approx(0)
    assert calls["polish"]["constraints"] is constraints


def test_differential_evolution_honors_dict_list_and_scipy_constraints():
    objective = lambda x: (x[0] - .1) ** 2
    result = ut.minimize_(objective, [.2], [[0., 1.]],
                          [{"type": "ineq", "fun": lambda x: x[0] - .5}],
                          "differential-evolution",
                          {"seed": 42, "popsize": 8, "maxiter": 40, "tol": 1e-9})
    assert result.x[0] >= .5 - 1e-7
    scipy_result = ut.minimize_(objective, [.2], [[0., 1.]],
                                LinearConstraint([[1.]], [.5], [np.inf]),
                                "differential-evolution",
                                {"seed": 42, "popsize": 8, "maxiter": 40, "tol": 1e-9})
    assert scipy_result.x[0] >= .5 - 1e-7


@pytest.mark.parametrize("constraint,error", [
    ({"type": "other", "fun": lambda x: x[0]}, "type"),
    ({"type": "ineq"}, "callable"),
    ("not-a-constraint", "constraint"),
])
def test_differential_evolution_rejects_malformed_constraints(constraint, error):
    with pytest.raises((TypeError, ValueError), match=error):
        ut.minimize_(lambda p: p[0] ** 2, [1.], [[0., 2.]], constraint,
                     "differential-evolution", {"polish": False})


def test_lower_case_documented_optimizer_name_is_normalized():
    result = bd.estimate([0, 1], [2, 1], [.5], [[.01, 2]], model="pure-death",
                         se_type="none", opt_method="l-bfgs-b")
    assert result.success


def test_invalid_optimizer_type_is_rejected_before_estimation():
    with pytest.raises(ValueError, match="opt_method"):
        bd.estimate([0, 1], [2, 1], [.5], [[.01, 2]], model="pure-death",
                    se_type="none", opt_method=None)


@pytest.mark.parametrize(
    "times,populations,expected_count",
    [
        ((0, 1, 2), (2, 1, 0), 2),
        ([np.array([0, 1]), np.array([3, 4, 5])],
         [np.array([2, 1]), np.array([4, 5, 4])], 3),
        (np.array([(0, 1), (2, 3, 4)], dtype=object),
         np.array([(2, 1), (4, 5, 4)], dtype=object), 3),
    ],
)
def test_discrete_observations_accept_tuple_array_and_ragged_paths(times, populations, expected_count):
    transitions = data_sort(times, populations)
    assert sum(transitions.values()) == expected_count


def test_discrete_observations_preserve_range_paths():
    transitions = data_sort(range(3), range(2, 5))
    assert sum(transitions.values()) == 2


@pytest.mark.parametrize(
    "times,populations,error",
    [
        ([0, np.nan], [1, 1], "finite"),
        ([1, 0], [1, 1], "nondecreasing"),
        ([0, 1], [-1, 1], "nonnegative integers"),
        ([0, 1], [1, 1.5], "nonnegative integers"),
        ([0, 1], [1, 1e30], "integer range"),
        ([0, 0], [1, 2], "positive elapsed"),
        ([0, 1], [1, 3], "one event"),
    ],
)
def test_continuous_observation_validation_rejects_invalid_paths(times, populations, error):
    with pytest.raises(ValueError, match=error):
        bd.estimate(times, populations, [.5], [[.01, 2]], model="pure-death",
                    scheme="continuous", se_type="none")


def test_translated_clocks_and_event_free_continuous_endpoint_are_equivalent():
    args = dict(p0=[.5], p_bounds=[[.01, 2]], model="pure-death",
                scheme="continuous", se_type="none")
    baseline = bd.estimate([0, 1, 3], [2, 1, 1], **args)
    translated = bd.estimate([-10, -9, -7], [2, 1, 1], **args)
    np.testing.assert_allclose(translated.p, baseline.p)
    assert translated.val == pytest.approx(baseline.val)
