"""Numerical failure and shape contracts for observed-information covariance."""

import warnings

import numpy as np
import pytest

import birdepy as bd
from birdepy import covariance
import birdepy.interface_dnm as dnm
import birdepy.utility as ut


def test_covariance_is_a_square_matrix_for_one_parameter_likelihood():
    cov = covariance.from_hessian(lambda p: -(p[0] - 2) ** 2 / 2, np.array([2.]), [[0, 4]])
    assert cov.shape == (1, 1)
    assert cov[0, 0] == pytest.approx(1)


def test_safe_covariance_returns_unavailable_for_flat_or_indefinite_information():
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        flat = covariance.safe_from_hessian(lambda p: 1., np.array([1.]), [[0, 2]])
    indefinite = covariance.safe_from_hessian(
        lambda p: (p[0] ** 2 - p[1] ** 2) / 2,
        np.array([0., 0.]), [[-1, 1], [-1, 1]],
    )
    assert flat == covariance.UNAVAILABLE
    assert indefinite == covariance.UNAVAILABLE


def test_safe_covariance_handles_strict_floating_point_overflow_and_underflow():
    with np.errstate(all="raise"):
        result = covariance.safe_from_hessian(
            lambda p: -1e-308 * p[0] ** 2 / 2, np.array([1.]), [[0, 2]]
        )
    assert result == covariance.UNAVAILABLE


def test_safe_covariance_does_not_hide_likelihood_programming_errors():
    def broken_likelihood(p):
        raise RuntimeError("broken rate function")

    with pytest.raises(RuntimeError, match="broken rate function"):
        covariance.safe_from_hessian(broken_likelihood, np.array([1.]), [[0, 2]])


def test_discrete_estimation_can_skip_unused_covariance(monkeypatch):
    monkeypatch.setattr(dnm, "safe_from_hessian", lambda *args: pytest.fail("called"))
    data = {(1, 1, 1.): 1}
    opt, cov = dnm.discrete_est_dnm(
        data, "expm", "pure-death", [0, 5], [], [], np.array([.5]), [[0, 2]],
        (), "L-BFGS-B", ut.add_options({}), compute_cov=False,
    )
    assert opt.x.shape == (1,)
    assert cov == covariance.UNAVAILABLE


def test_bootstrap_refits_skip_observed_information(monkeypatch):
    calls = []
    original = dnm.safe_from_hessian

    def count(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(dnm, "safe_from_hessian", count)
    bd.estimate([0, 1, 2], [3, 2, 1], [1.], [[.01, 3]], model="pure-death",
                seed=12, se_type="simulated", num_samples=3, z_trunc=[0, 6])
    assert len(calls) == 1


def test_correlated_quadratic_covariance_matches_analytic_inverse():
    information = np.array([[2., 1.], [1., 3.]])
    actual = covariance.from_hessian(lambda p: -.5 * p @ information @ p,
                                     np.array([0., 0.]), [[-1, 1], [-1, 1]])
    np.testing.assert_allclose(actual, [[.6, -.2], [-.2, .4]], rtol=1e-5)


@pytest.mark.parametrize('hessian', [np.array([[np.nan]]), np.array([[-np.inf]])])
def test_nonfinite_information_returns_sentinel(monkeypatch, hessian):
    monkeypatch.setattr(covariance, 'Hessian', lambda *args: hessian)
    assert covariance.safe_from_hessian(None, None, None) == covariance.UNAVAILABLE


@pytest.mark.parametrize('kwargs', [
    {'scheme': 'continuous'}, {'framework': 'em', 'accelerator': 'none', 'max_it': 2},
])
def test_other_bootstrap_paths_skip_unused_covariance(monkeypatch, kwargs):
    calls = []
    original = covariance.Hessian
    def counted(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(covariance, 'Hessian', counted)
    bd.estimate([0, 1, 2], [3, 2, 1], [.5], [[.01, 3]], model='pure-death',
                seed=12, se_type='simulated', num_samples=2, z_trunc=[0, 6], **kwargs)
    assert len(calls) == 1
