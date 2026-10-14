"""Check numerical contracts against analytic results, not saved implementation output."""

import numpy as np
import pytest

import birdepy as bd
import birdepy.utility as ut
from birdepy.interface_dnm import bld_ll_fun
from birdepy.simulate import beta1, beta2


def test_rounding_intervals_preserves_observations_and_likelihood():
    data = ut.data_sort([0, .1, .2, .3], [1, 2, 1, 2])
    grouped = ut.data_sort_2(data)
    assert grouped == {.1: {(1, 2): 2, (2, 1): 1}}
    expected = sum(n * np.log(bd.probability(i, j, t, [1, 1], model='linear',
                                            z_trunc=[0, 10])[0, 0])
                   for (i, j, t), n in data.items())
    actual = bld_ll_fun(data, 'expm', 'linear', [0, 10], {})([1, 1])
    assert actual == pytest.approx(expected)


def test_moran_rates_have_the_expected_mutation_and_selection_terms():
    p = [2, 3, .1, .2, 100]
    assert ut.higher_death('Moran')(40, p) == pytest.approx(.608)
    # With no mutation and equal fitness, upward and downward rates agree.
    z = np.arange(101)
    neutral = [1, 1, 0, 0, 100]
    expected = z * (100 - z) / 100 ** 2
    np.testing.assert_allclose(ut.higher_birth('Moran')(z, neutral), expected)
    np.testing.assert_allclose(ut.higher_death('Moran')(z, neutral), expected)


@pytest.mark.parametrize('model,param', [
    ('Verhulst', [.8, .4, .02, .01]),
    ('Ricker', [.8, .4, .02, 2]),
    ('Hassell', [.8, .4, .02, 2]),
    ('MS-S', [.8, .4, .02, 2]),
    ('Moran', [2, 3, .1, .2, 100]),
])
def test_drift_derivative_matches_rates(model, param):
    z = np.array([1., 10., 20.])
    step = 1e-4
    birth, death = ut.higher_birth(model), ut.higher_death(model)
    expected = ((birth(z + step, param) - death(z + step, param)) -
                (birth(z - step, param) - death(z - step, param))) / (2 * step)
    np.testing.assert_allclose(ut.higher_h_fun(model)(z, param), expected,
                               rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('method', ['ea', 'ma', 'gwa'])
def test_critical_linear_simulation_preserves_mean(method):
    sample = bd.simulate.discrete([1, 1], 'linear', 100, [0, .1],
                                  method=method, tau=.1, k=20000, seed=1234)[:, 1]
    # The exact mean is 100 and variance is 2 * 100 * .1.
    assert abs(sample.mean() - 100) < 6 * np.sqrt(20 / len(sample))


@pytest.mark.parametrize('rate,time', [(1., .1), (1e-10, 1.), (1000., 1.)])
def test_gwa_probabilities_include_pure_birth_and_death_limits(rate, time):
    change = -np.expm1(-rate * time)
    assert beta1(rate, 0, time) == 0
    assert beta2(rate, 0, time) == pytest.approx(change)
    assert beta1(0, rate, time) == pytest.approx(change)
    assert beta2(0, rate, time) == 0


def test_gwa_pure_birth_distribution():
    sample = bd.simulate.discrete([.7], 'pure-birth', 4, [0, .2],
                                  method='gwa', tau=.2, k=10000, seed=1234)[:, 1]
    growth = np.exp(.7 * .2)
    assert np.all(sample >= 4)
    assert abs(sample.mean() - 4 * growth) < 6 * np.sqrt(4 * growth * (growth - 1) / len(sample))


def test_gwa_can_leave_zero_when_immigration_is_possible():
    sample = bd.simulate.discrete([3], 'Poisson', 0, [0, 1],
                                  method='gwa', tau=1, k=5000, seed=1234)[:, 1]
    assert abs(sample.mean() - 3) < 6 * np.sqrt(3 / len(sample))
    assert bd.simulate.discrete([1, 1], 'linear', 0, [0, 1], method='gwa') == [0, 0]


@pytest.mark.parametrize('tau', [0, -1, np.inf, np.nan])
def test_approximate_simulation_rejects_invalid_step(tau):
    with pytest.raises(ValueError, match='tau'):
        bd.simulate.discrete([1, 1], 'linear', 1, [0, 1], method='ea', tau=tau)


@pytest.mark.parametrize('start', [0., .1, .5])
def test_continuous_likelihood_does_not_accept_zero_rate_for_an_observed_birth(start):
    result = bd.estimate([0, 10], [1, 2], [start], [[0, 1]], model='pure-birth',
                         scheme='continuous')
    assert result.success
    assert result.p[0] == pytest.approx(.1, abs=1e-6)
    assert result.val == pytest.approx(np.log(.1) - 1)
    assert result.cov.shape == (1, 1)
    assert result.se[0] == pytest.approx(.1, rel=1e-4)


def test_continuous_likelihood_allows_zero_rate_without_events():
    result = bd.estimate([0, 10], [1, 1], [.5], [[0, 1]], model='pure-birth',
                         scheme='continuous')
    assert result.p[0] == pytest.approx(0)
    assert result.val == pytest.approx(0)
    assert isinstance(result.cov, str)
    assert isinstance(result.se, str)


def test_impossible_continuous_data_cannot_report_success():
    result = bd.estimate([0, 1], [1, 2], [.5], [[0, 1]], model='pure-death',
                         scheme='continuous', se_type='none')
    assert not result.success
    assert result.val == -np.inf
    assert isinstance(result.cov, str)


def test_constant_observations_can_be_estimated_with_default_truncation():
    result = bd.estimate([0, 1, 2], [3, 3, 3], [.5], [[0, 2]],
                         model='pure-death', se_type='none')
    assert result.success
    assert result.p[0] == pytest.approx(0)


def test_uniformization_controls_tail_error_at_large_rate_times_time():
    states = np.arange(21)
    reference = bd.probability(states, states, 10, [2], model='pure-death',
                               z_trunc=[0, 20], method='expm')
    actual = bd.probability(states, states, 10, [2], model='pure-death',
                            z_trunc=[0, 20], method='uniform')
    np.testing.assert_allclose(actual, reference, atol=2e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=2e-12)
    with pytest.warns(RuntimeWarning, match='cutoff'):
        truncated = bd.probability(10, 0, 10, [2], model='pure-death',
                                   z_trunc=[0, 20], method='uniform', k=150)
    assert truncated[0, 0] < 1e-40


@pytest.mark.parametrize('method', ['expm', 'uniform', 'Erlang'])
def test_matrix_methods_support_zero_time_and_zero_generator(method):
    states = np.arange(3)
    actual = bd.probability(states, states, [0, 1], [0, 0], model='linear',
                            z_trunc=[0, 2], method=method)
    np.testing.assert_allclose(actual, np.array([np.eye(3), np.eye(3)]))
    assert bd.probability(0, 0, 1, [0, 0], model='linear', z_trunc=[0, 0],
                          method=method)[0, 0] == 1


def test_probability_default_model_and_explicit_options_take_precedence():
    p = [.8, .4, .01, .001]
    assert bd.probability(20, 25, 1, p)[0, 0] == pytest.approx(.08189476)
    actual = bd.probability(10, 0, 1, [1], model='pure-death',
                            z_trunc=[0, 12], options={'z_trunc': [0, 5]})
    assert actual[0, 0] == pytest.approx((1 - np.exp(-1)) ** 10)
