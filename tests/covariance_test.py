"""Numerical and interface contracts for the shared covariance stage."""

import numpy as np
import pytest

import birdepy as bd
import birdepy.covariance as covariance


@pytest.mark.parametrize("information", [np.array([[4.]]), np.array([[4., 1.], [1., 3.]])])
def test_covariance_matches_analytic_quadratic_information(information):
    estimate = np.zeros(information.shape[0])
    actual = covariance.from_hessian(
        lambda x: -0.5 * x @ information @ x, estimate,
        [[-2, 2]] * len(estimate), require_positive_definite=True)
    assert actual.shape == information.shape
    np.testing.assert_allclose(actual, np.linalg.inv(information), rtol=1e-6)


@pytest.mark.parametrize("likelihood", [lambda x: 0.0, lambda x: x[0] ** 2])
def test_continuous_covariance_rejects_flat_or_nonconcave_likelihood(likelihood):
    with pytest.raises(np.linalg.LinAlgError):
        covariance.from_hessian(likelihood, np.array([0.]), [[-2, 2]],
                                require_positive_definite=True)


def test_covariance_does_not_hide_programmer_errors():
    def broken_likelihood(x):
        raise RuntimeError("broken custom likelihood")
    with pytest.raises(RuntimeError, match="broken custom likelihood"):
        covariance.from_hessian(broken_likelihood, np.array([0.]), [[-2, 2]])


@pytest.mark.parametrize("framework", ["dnm", "em"])
def test_single_parameter_fit_computes_covariance_once_without_printing(framework, monkeypatch, capsys):
    original = covariance.Hessian
    calls = []
    def tracked_hessian(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(covariance, "Hessian", tracked_hessian)
    result = bd.estimate(
        [0, 1, 2], [3, 2, 1], [.5], [[.01, 2]], model="pure-death",
        framework=framework, accelerator="none", max_it=3, display=False)
    assert len(calls) == 1
    assert capsys.readouterr().out == ""
    assert result.cov.shape == (1, 1)
    assert result.cov[0, 0] > 0
    assert np.isfinite(result.se[0])
