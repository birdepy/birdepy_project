"""Regressions for shared numerical paths and public API consistency."""

import numpy as np
import pytest
from scipy.stats import binom, nbinom

import birdepy as bd
from birdepy.utility_probability import p_lin


@pytest.mark.parametrize('lam,mu,z0,time', [(4., .1, 5, 10.), (400., 0., 2, .1)])
def test_supercritical_gwa_has_representable_geometric_success_probability(lam, mu, z0, time):
    sample = bd.simulate.discrete([lam, mu], 'linear', z0, [0, time],
                                  method='gwa', tau=time, k=4000, seed=42)[:, 1]
    growth = np.exp((lam - mu) * time)
    mean = z0 * growth
    variance = z0 * (lam + mu) / (lam - mu) * growth * (growth - 1)
    assert np.all(sample > 0)
    assert abs(sample.mean() - mean) < 6 * np.sqrt(variance / len(sample))


def test_subcritical_gwa_retains_small_survival_probability():
    z0 = 10 ** 18
    sample = bd.simulate.discrete([400.], 'pure-death', z0, [0, .1],
                                  method='gwa', tau=.1, k=3000, seed=42)[:, 1]
    mean = z0 * np.exp(-40)
    assert abs(sample.mean() - mean) < 6 * np.sqrt(mean / len(sample))


@pytest.mark.parametrize('time', [0., .1, 1., 40.])
def test_gwa_probabilities_match_pure_birth_distribution(time):
    states = np.arange(1, 10)
    actual = bd.probability(2, states, time, [1.], model='pure-birth', method='gwa')[0]
    expected = nbinom.pmf(states - 2, 2, np.exp(-time))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_gwa_probability_evaluates_each_requested_time():
    times = np.array([.1, 1., 40.])
    actual = bd.probability(2, 3, times, [1.], model='pure-birth', method='gwa')[:, 0, 0]
    expected = nbinom.pmf(1, 2, np.exp(-times))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_gwa_probabilities_match_pure_death_distribution():
    states = np.arange(7)
    actual = bd.probability(5, states, 1, [1.], model='pure-death', method='gwa')[0]
    np.testing.assert_allclose(actual, binom.pmf(states, 5, np.exp(-1)), rtol=1e-12)


def test_linear_probability_retains_high_precision_fallback():
    expected = nbinom.pmf(500, 1000, np.exp(-.4))
    assert p_lin(1000, 1500, 1., 0., .4) == pytest.approx(expected, rel=1e-10)


def test_extinction_probability_retains_tiny_per_ancestor_survival():
    z0 = 10 ** 18
    # The binomial zero count agrees with this Poisson limit to better than
    # machine precision: n*p**2/2 < 1e-17.
    expected = np.exp(-z0 * np.exp(-40))
    assert p_lin(z0, 0, 0., 400., .1) == pytest.approx(expected, rel=1e-12)


def test_saddlepoint_boundary_uses_shared_pure_birth_probabilities():
    actual = bd.probability(1, 1, 40, [1.], model='pure-birth', method='gwasa')[0, 0]
    assert actual == pytest.approx(np.exp(-40), rel=1e-12, abs=0)


def test_estimation_merges_options_without_mutating_the_nested_dictionary():
    args = dict(t_data=[0, 1, 2, 3, 4], p_data=[4, 5, 4, 6, 5], p0=[.5, .4],
                p_bounds=[[.01, 2], [.01, 2]], model='linear', se_type='none')
    nested = {'likelihood': 'expm', 'z_trunc': [0, 8]}
    expected = bd.estimate(**args, z_trunc=[0, 6], likelihood='expm')
    actual = bd.estimate(**args, z_trunc=[0, 6], options=nested)
    np.testing.assert_allclose(actual.p, expected.p)
    assert actual.val == pytest.approx(expected.val)
    assert nested == {'likelihood': 'expm', 'z_trunc': [0, 8]}


def test_explicit_abc_particle_count_overrides_nested_options():
    result = bd.estimate([0, 1], [1, 0], [1.], [[0, 2]], model='pure-death',
                         framework='abc', method='exact', k=30,
                         options={'k': 8, 'eps_abc': 0}, seed=42)
    assert result.samples[0].shape == (30, 1)
    assert result.sample_weights[0].shape == (30,)


@pytest.mark.parametrize('container', ['list_of_arrays', 'tuple_of_tuples', 'matrix', 'single_array'])
def test_continuous_path_containers_match_analytic_mle(container):
    times = [[0, .3, .8, 1.], [0, .2, 1.]]
    pops = [[3, 2, 1, 1], [2, 1, 1]]
    count, exposure = 3, 3.3
    if container == 'list_of_arrays':
        times, pops = list(map(np.array, times)), list(map(np.array, pops))
    elif container == 'tuple_of_tuples':
        times, pops = tuple(map(tuple, times)), tuple(map(tuple, pops))
    elif container == 'matrix':
        times, pops = np.array([times[0]] * 2), np.array([pops[0]] * 2)
        count, exposure = 4, 4.2
    else:
        times, pops = np.array(times[0]), np.array(pops[0])
        count, exposure = 2, 2.1
    result = bd.estimate(times, pops, [1.], [[.01, 4]], model='linear',
                         scheme='continuous', known_p=[0], idx_known_p=[0])
    assert result.success
    assert result.p[0] == pytest.approx(count / exposure, rel=1e-5)
    assert result.cov[0, 0] == pytest.approx(count / exposure ** 2, rel=1e-4)
    assert result.sample_weights == result.samples


def test_continuous_bootstrap_accepts_array_paths():
    times, pops = [[0, .3, .8, 1.], [0, .2, 1.]], [[3, 2, 1, 1], [2, 1, 1]]
    args = dict(p0=[1.], p_bounds=[[.01, 4]], model='linear', scheme='continuous',
                known_p=[0], idx_known_p=[0], se_type='simulated', num_samples=8, seed=42)
    lists = bd.estimate(times, pops, **args)
    arrays = bd.estimate(list(map(np.array, times)), list(map(np.array, pops)), **args)
    np.testing.assert_allclose(arrays.cov, lists.cov)
    assert arrays.cov[0, 0] > 0


@pytest.mark.parametrize('framework', ['dnm', 'em', 'lse'])
def test_non_abc_results_expose_sample_weights_sentinel(framework):
    result = bd.estimate([0, 1, 2], [3, 4, 3], [.5, .4], [[.01, 2], [.01, 2]],
                         model='linear', framework=framework, se_type='none',
                         max_it=2, accelerator='none', z_trunc=[0, 8])
    assert isinstance(result.sample_weights, str)
    assert result.sample_weights == result.samples


def test_abc_stat_validation_retains_public_exception_type():
    with pytest.raises(TypeError, match='stat'):
        bd.estimate([0, 1], [1, 0], [1.], [[0, 2]], model='pure-death',
                    framework='abc', stat='mode')


@pytest.mark.parametrize('tolerance', [1e-6, 1e-12, 1e-15])
def test_scaled_uniformization_respects_total_tail_budget(tolerance):
    actual = bd.probability(10, np.arange(11), 1000, [2.], model='pure-death',
                            method='uniform', z_trunc=[0, 10], uniform_tol=tolerance)[0]
    # Extinction is effectively certain at this horizon. Allow roundoff beyond
    # the Poisson truncation budget, which is not a total floating-point bound.
    assert abs(actual.sum() - 1) < tolerance + 5e-12
    assert abs(actual[0] - 1) < tolerance + 5e-12
    assert np.all(actual >= 0)


def test_scaled_uniformization_agrees_with_expm_for_large_rate_time_product():
    args = dict(z0=[10, 50], zt=np.arange(151), t=100,
                param=[.75, .25, .02, 1], model='Ricker', z_trunc=[0, 150])
    actual = bd.probability(**args, method='uniform')
    reference = bd.probability(**args, method='expm')
    np.testing.assert_allclose(actual, reference, atol=5e-12, rtol=1e-10)
