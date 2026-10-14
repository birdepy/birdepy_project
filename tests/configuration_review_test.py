"""Boundary and cross-framework checks prompted by the architecture review."""

import numpy as np
import pytest

import birdepy as bd
from birdepy.observations import normalize_observations
from birdepy.interface_abc import estimate_prior_pdf, param_sampler


@pytest.mark.parametrize('values', [
    np.array([2**53 + 1, 2**53], dtype=np.int64),
    np.array([2**31 + 1, 2**31], dtype=np.uint64),
    np.array([np.iinfo(np.int64).max, np.iinfo(np.int64).max - 1]),
])
def test_integer_population_precision_and_unsigned_deaths(values):
    _, populations = normalize_observations([0, 1], values, 'continuous')
    np.testing.assert_array_equal(populations[0], values)
    assert populations[0].dtype == np.int64


@pytest.mark.parametrize('value', [np.uint64(2**63), float(2**63), 1e30])
def test_out_of_int64_population_range_is_rejected_before_casting(value):
    with pytest.raises(ValueError, match='integer range'):
        normalize_observations([0, 1], np.array([0, value], dtype=type(value)), 'discrete')


@pytest.mark.parametrize('framework', ['abc', 'em'])
def test_unsupported_equalities_fail_before_fitting(framework):
    with pytest.raises(TypeError, match='inequality'):
        bd.estimate([0, 1], [2, 1], [.5], [[.01, 2]], model='pure-death',
                    framework=framework, con={'type': 'eq', 'fun': lambda p: p[0] - .5})


def test_abc_vector_constraints_forward_arguments_in_prior_and_proposals():
    bounds = np.array([[0., 2.]])
    con = {'type': 'ineq', 'fun': lambda p, offset: np.array([p[0] + offset, 3 - p[0]]),
           'args': (1.,)}
    rng = np.random.default_rng(42)
    assert estimate_prior_pdf(bounds, con, rng) == pytest.approx(.5)
    proposal = param_sampler([], 1, [], None, bounds, con, rng)
    assert 0 <= proposal[0] <= 2


def test_em_qn1_accepts_vector_constraints_with_arguments():
    con = {'type': 'ineq', 'fun': lambda p, bound: np.array([p[0] - bound, 2 - p[0]]),
           'args': (.01,)}
    result = bd.estimate([0, 1, 2], [3, 2, 1], [.5], [[.01, 2]], model='pure-death',
                         framework='em', accelerator='qn1', max_it=2, con=con)
    assert .01 <= result.p[0] <= 2


def test_invalid_scheme_wins_over_malformed_data():
    with pytest.raises(ValueError, match='scheme'):
        normalize_observations(None, None, 'typo')


def test_de_empty_constraints_use_unconstrained_polishing_and_preserve_options(monkeypatch):
    from scipy.optimize import OptimizeResult
    import birdepy.utility as ut
    calls = []
    monkeypatch.setattr(ut, 'differential_evolution', lambda *args, **kw: OptimizeResult(x=np.array([.5])))
    def minimize(*args, **kwargs):
        calls.append(kwargs)
        return OptimizeResult(x=np.array([.5]))
    monkeypatch.setattr(ut, 'minimize', minimize)
    for constraints in ([], ()):
        ut.minimize_(lambda p: p[0]**2, [.5], [[0, 1]], constraints,
                     'differential-evolution', {})
    assert [call['method'] for call in calls] == ['L-BFGS-B', 'L-BFGS-B']


def test_de_constraint_jacobian_receives_arguments():
    from birdepy.configuration import differential_evolution_constraints
    original = {'type': 'INEQ', 'fun': lambda p, scale: scale * p[0],
                'jac': lambda p, scale: np.array([scale]), 'args': (3.,)}
    adapted, polish = differential_evolution_constraints(original)
    assert adapted.fun([2.]) == 6.
    np.testing.assert_array_equal(adapted.jac([2.]), [3.])
    assert polish is original
    with pytest.raises(TypeError, match='con'):
        differential_evolution_constraints(np.array([1, 2]))


@pytest.mark.parametrize('method', ['SLSQP', 'L-BFGS-B'])
def test_minimizers_forward_explicit_controls_without_mutating_input(monkeypatch, method):
    from scipy.optimize import OptimizeResult
    import birdepy.utility as ut
    from types import MappingProxyType
    calls = []
    callback = lambda p: None
    controls = MappingProxyType({'callback': callback, 'maxiter': 7, 'disp': True,
                                 'workers': 2, 'likelihood': 'expm'})
    def minimize(*args, **kwargs):
        calls.append(kwargs)
        return OptimizeResult(x=np.array([.5]))
    monkeypatch.setattr(ut, 'minimize', minimize)
    ut.minimize_(lambda p: p[0]**2, [.5], [[0, 1]], (), method, controls)
    assert calls[0]['callback'] is callback
    assert calls[0]['options']['maxiter'] == 7
    assert calls[0]['options']['disp'] is True
    assert calls[0]['options']['workers'] == 2
    assert 'likelihood' not in calls[0]['options']
    assert set(controls) == {'callback', 'maxiter', 'disp', 'workers', 'likelihood'}
