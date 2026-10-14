import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint

import birdepy as bd
from birdepy.forecast_compute import parameter_sampler


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_estimation_output_round_trips_through_pickle():
    result = bd.estimate([0, 1], [1, 0], [1.], [[0, 2]], model='pure-death',
                         se_type='none')
    restored = pickle.loads(pickle.dumps(result))
    assert isinstance(restored, bd.EstimationOutput)
    assert restored.p == result.p
    assert restored['model'] == result.model


def test_compute_forecast_is_numerical_and_has_known_linear_mean():
    with pytest.warns(RuntimeWarning, match='cov'):
        result = bd.compute_forecast('linear', 10, [0, 1, 2], [.6, .4])
    assert plt.get_fignums() == []
    np.testing.assert_allclose(result.central, 10 * np.exp(.2 * np.arange(3)),
                               rtol=1e-3)
    np.testing.assert_allclose(result.mean, result.central)
    assert result.quantiles is None


def test_compute_forecast_preserves_seeded_intervals_and_single_time():
    first = bd.compute_forecast('linear', 10, [0, 1], [.6, .4],
                                interval='prediction', method='gwa', k=12, seed=42)
    repeated = bd.compute_forecast('linear', 10, [0, 1], [.6, .4],
                                   interval='prediction', method='gwa', k=12, seed=42)
    np.testing.assert_array_equal(first.samples, repeated.samples)
    np.testing.assert_array_equal(first.quantiles, repeated.quantiles)
    with pytest.warns(RuntimeWarning):
        single = bd.compute_forecast('linear', 10, [100], [.6, .4])
    np.testing.assert_array_equal(single.central, [10])


def test_plot_forecast_uses_the_supplied_axis():
    with pytest.warns(RuntimeWarning):
        result = bd.compute_forecast('linear', 10, [0, 1], [.6, .4])
    fig, ax = plt.subplots()
    assert bd.plot_forecast(result, ax=ax, xticks=np.array([0, 1])) is ax
    assert fig.axes == [ax]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), result.central)


def test_infeasible_parameter_sampling_fails_explicitly():
    with pytest.raises(RuntimeError, match='within 2 attempts'):
        bd.compute_forecast('linear', 10, [0, 1], [.6, .4], cov=[[0, 0], [0, 0]],
                            p_bounds=[[0, .1], [0, .1]], k=1, max_rejections=2)


def test_parameter_sampling_accepts_open_ended_bounds():
    result = bd.compute_forecast('linear', 10, [0, 1], [.6, .4],
                                 cov=[[0, 0], [0, 0]],
                                 p_bounds=[[0, np.inf], [-np.inf, np.inf]],
                                 k=1, seed=42)
    assert result.samples.shape == (1, 2)


def test_parameter_sampler_accepts_single_vector_scipy_constraint():
    proposal = parameter_sampler(
        np.array([.6, .4]), np.zeros((2, 2)), None, (), (),
        LinearConstraint([[1, 0], [0, 1]], [.5, .3], [.7, .5]),
        np.random.default_rng(42))
    np.testing.assert_array_equal(proposal, [.6, .4])


def test_parameter_sampler_honours_mixed_vector_and_equality_constraints():
    constraints = [
        {'type': 'eq', 'fun': lambda p, target: np.array([p[0] - target]),
         'args': (.6,)},
        NonlinearConstraint(lambda p: np.array([p[0] + p[1], p[1]]),
                            [.9, .3], [1.1, .5]),
    ]
    proposal = parameter_sampler(
        np.array([.6, .4]), np.zeros((2, 2)), None, (), (), constraints,
        np.random.default_rng(42))
    np.testing.assert_array_equal(proposal, [.6, .4])


def test_parameter_sampler_rejects_nonfinite_constraint_values():
    with pytest.raises(RuntimeError, match='within 1 attempts'):
        parameter_sampler(np.array([.6]), np.zeros((1, 1)), None, (), (),
                          {'type': 'ineq', 'fun': lambda p: np.nan},
                          np.random.default_rng(42), max_rejections=1)
