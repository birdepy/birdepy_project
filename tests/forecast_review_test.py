"""Independent regression checks for forecast review findings."""

import copy
import pickle
import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pytest
from scipy.optimize import LinearConstraint

import birdepy as bd
from birdepy.forecast_compute import parameter_sampler


@pytest.fixture(autouse=True)
def close_figures():
    plt.close('all')
    yield
    plt.close('all')


def result(**kwargs):
    return bd.compute_forecast('linear', 10, [0, 1, 2], [.6, .4],
                               interval='prediction', k=20, seed=123, **kwargs)


@pytest.mark.parametrize('constraint', [
    {'fun': lambda p: 1}, {'type': 'ineq', 'fun': lambda p: 1, 'args': 1},
    LinearConstraint([[1, 0]], [1], [1]), np.array([object()], dtype=object),
])
def test_malformed_or_equality_constraints_leave_random_state_untouched(constraint):
    rng = np.random.default_rng(123)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises((ValueError, TypeError)):
        bd.compute_forecast('linear', 10, [0, 1], [.6, .4], cov=np.eye(2),
                            con=constraint, k=1, seed=rng)
    assert before == rng.bit_generator.state


def test_scalar_parameters_none_bounds_and_infinite_constraint_values():
    con = np.array([{'type': 'INEQ', 'fun': lambda p, bound: np.array([np.inf, p[0] - bound]),
                     'args': (.1,)}], dtype=object)
    actual = parameter_sampler(.6, 0., [(None, None)], (), (), con, np.random.default_rng(5))
    np.testing.assert_array_equal(actual, [.6])


def test_constraint_generator_is_not_consumed_and_ignored():
    constraints = ({'type': 'ineq', 'fun': lambda p: -1} for _ in range(1))
    with pytest.raises(RuntimeError, match='within 2 attempts'):
        parameter_sampler([.6], [[.01]], None, (), (), constraints,
                          np.random.default_rng(5), max_rejections=2)


def test_zero_covariance_infeasibility_fails_without_random_draws():
    rng = np.random.default_rng(5)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(RuntimeError, match='Zero covariance'):
        parameter_sampler([.6], [[0]], [[0, .1]], (), (), (), rng)
    assert before == rng.bit_generator.state


def test_sampling_failure_reports_completed_samples():
    class ControlledRng:
        def __init__(self):
            self.calls = 0
        def multivariate_normal(self, *args):
            self.calls += 1
            return np.array([.5]) if self.calls == 1 else np.array([-1.])
    import birdepy.forecast_compute as compute
    from unittest.mock import patch
    with patch.object(compute.np.random, 'default_rng', return_value=ControlledRng()):
        with pytest.raises(RuntimeError, match='after 1 of 3 samples.*within 2 attempts'):
            bd.compute_forecast('pure-death', 10, [0, 1], [.5], cov=[[.1]],
                                p_bounds=[[0, 2]], k=3, max_rejections=2)


def test_custom_bands_have_correct_vertices_labels_and_explicit_overrides():
    forecast = result(percentiles=[5, 25, 75, 95])
    ax = bd.plot_forecast(forecast)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [r'$90\%$', r'$50\%$']
    for band, (lo, hi) in zip(ax.collections, [(0, 3), (1, 2)]):
        vertices = band.get_paths()[0].vertices
        for index, time in enumerate(forecast.times):
            values = vertices[vertices[:, 0] == time, 1]
            assert np.min(values) == forecast.quantiles[index, lo]
            assert np.max(values) == forecast.quantiles[index, hi]
    custom = bd.plot_forecast(forecast, labels=['outer', 'inner'])
    assert [text.get_text() for text in custom.get_legend().get_texts()] == ['outer', 'inner']
    with pytest.raises(ValueError, match='one label'):
        bd.plot_forecast(forecast, labels=['missing'])


def test_legacy_band_colors_labels_and_wrapper_custom_percentiles():
    ax = bd.plot_forecast(result())
    for index, band in enumerate(ax.collections):
        np.testing.assert_array_equal(band.get_facecolor()[0], cm.Purples((index + 1) / 4))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [r'$95\%$', r'$80\%$', r'$50\%$']
    bd.forecast('linear', 10, [0, 1], [.6, .4], interval='prediction', k=10,
                percentiles=[5, 95], seed=2, labels=['custom'])
    assert plt.gca().get_legend().get_texts()[0].get_text() == 'custom'


def test_subfigure_axis_and_prediction_ylabel_render():
    fig = plt.figure(layout='constrained')
    ax = fig.subfigures(1, 1).subplots()
    assert bd.plot_forecast(result(), ax=ax) is ax
    assert ax.get_ylabel() == '$Z(t)$'
    fig.canvas.draw()


def test_pickle_preserves_all_forecast_arrays_and_rejects_bad_shapes():
    forecast = result()
    restored = pickle.loads(pickle.dumps(forecast))
    for name in ('times', 'central', 'mean', 'percentiles', 'quantiles', 'samples'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(forecast, name))
    assert restored.metadata == forecast.metadata
    assert restored != forecast
    with pytest.raises(ValueError, match='central'):
        replace(forecast, central=np.ones(1))
    with pytest.raises(ValueError, match='quantiles'):
        replace(forecast, quantiles=np.ones((1, 1)))


def test_warning_attribution_and_actual_sample_metadata():
    for function in (bd.compute_forecast, bd.forecast):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            function('linear', 10, [0, 1], [.6, .4])
        relevant = [item for item in recorded if 'cov' in str(item.message)]
        assert len(relevant) == 1
        assert relevant[0].filename == __file__
    with pytest.warns(RuntimeWarning):
        fluid = bd.compute_forecast('linear', 10, [0, 1], [.6, .4])
    assert fluid.metadata['n_samples'] == fluid.metadata['n_trajectories'] == 0
    sampled = bd.compute_forecast('pure-death', 10, [0, 1], [.5], cov=[[0]],
                                  method='exact', n=3, k=2, seed=123)
    assert sampled.metadata['n_samples'] == 2
    assert sampled.metadata['n_trajectories'] == 6
    assert result().metadata['n_trajectories'] == 20
