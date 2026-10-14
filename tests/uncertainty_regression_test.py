import numpy as np
import pytest
from scipy.stats import multivariate_normal

import birdepy as bd
from birdepy.interface_abc import _importance_weights, _proposal_covariance


def test_explicit_dynamic_abc_threshold_matches_the_default():
    args = dict(model='pure-death', framework='abc', method='exact',
                k=50, max_its=2, seed=42)
    default = bd.estimate([0, 1], [1, 0], [1.], [[0, 2]], **args)
    explicit = bd.estimate([0, 1], [1, 0], [1.], [[0, 2]], eps_abc='dynamic', **args)
    np.testing.assert_allclose(explicit.p, default.p)
    np.testing.assert_allclose(explicit.cov, default.cov)


def test_abc_repeated_populations_match_an_analytic_posterior():
    # One death from an initial population of one: likelihood 1-exp(-theta).
    # The prior is uniform on [0, 2], so its posterior moments are analytic.
    result = bd.estimate([0, 1], [1, 0], [1.], [[0, 2]], model='pure-death',
                         framework='abc', method='exact', k=1000,
                         eps_abc=[0, 0, 0], eps_change=-1, max_its=3,
                         seed=1234, display=False)
    mean = (1 + 3 * np.exp(-2)) / (1 + np.exp(-2))
    second_moment = (2 / 3 + 10 * np.exp(-2)) / (1 + np.exp(-2))
    assert result.p[0] == pytest.approx(mean, abs=.06)
    assert result.cov[0, 0] == pytest.approx(second_moment - mean ** 2, abs=.05)
    assert len(result.sample_weights) == len(result.samples) == 3
    for samples, weights in zip(result.samples, result.sample_weights):
        assert weights.shape == (len(samples),)
        assert weights.sum() == pytest.approx(1)
        assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(result.p, np.average(result.samples[-1], axis=0,
                                                    weights=result.sample_weights[-1]))


def test_abc_weights_use_the_full_proposal_covariance():
    previous = np.array([[-1., 0.], [1., 2.]])
    prior_weights = np.array([.25, .75])
    cov = np.array([[1., .6], [.6, 2.]])
    bounds = np.array([[-3, 3], [-3, 3]])
    sample = np.array([[0., 0.], [1., 1.], [-.5, 2.]])
    kernel_cov = _proposal_covariance(cov, bounds)
    density = sum(w * multivariate_normal(mean=m, cov=kernel_cov).pdf(sample)
                  for w, m in zip(prior_weights, previous))
    expected = (1 / density) / np.sum(1 / density)
    np.testing.assert_allclose(_importance_weights(sample, previous, prior_weights,
                                                    cov, bounds), expected)


def test_em_bootstrap_refits_simulated_data():
    args = dict(model='linear', framework='em', accelerator='none',
                p0=[.5, .4], p_bounds=[[.01, 2], [.01, 2]], z_trunc=[0, 15],
                max_it=5, display=False, se_type='simulated', num_samples=6)
    first = bd.estimate([0, 1, 2, 3, 4], [4, 5, 4, 6, 5], seed=42, **args)
    repeated = bd.estimate([0, 1, 2, 3, 4], [4, 5, 4, 6, 5], seed=42, **args)
    assert np.all(np.diag(first.cov) > 0)
    np.testing.assert_allclose(first.cov, repeated.cov)
    np.testing.assert_allclose(first.se, np.sqrt(np.diag(first.cov)))


@pytest.mark.parametrize('multiple_paths', [False, True])
def test_continuous_bootstrap_with_known_parameters(multiple_paths):
    times, pops = [0, .3, .8, 1.], [3, 2, 1, 1]
    if multiple_paths:
        times, pops = [times, times], [pops, pops]
    args = dict(model='linear', scheme='continuous', p0=[1.], p_bounds=[[.01, 4]],
                known_p=[0], idx_known_p=[0], se_type='simulated', num_samples=10)
    result = bd.estimate(times, pops, seed=42, **args)
    repeated = bd.estimate(times, pops, seed=42, **args)
    assert result.cov.shape == (1, 1)
    assert result.cov[0, 0] > 0
    np.testing.assert_allclose(result.cov, repeated.cov)


def test_custom_model_bootstrap_retains_rate_functions():
    result = bd.estimate([0, 1, 2], [3, 2, 1], [1.], [[.01, 4]], model='custom',
                         b_rate=lambda z, p: 0 * z, d_rate=lambda z, p: p[0] * z,
                         se_type='simulated', num_samples=6, seed=42)
    assert result.cov[0, 0] > 0
