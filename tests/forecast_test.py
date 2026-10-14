import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import birdepy as bd


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_default_forecast_draws_the_known_linear_mean_without_covariance():
    with pytest.warns(RuntimeWarning, match='cov'):
        assert bd.forecast('linear', 10, [0, 1, 2], [.6, .4], xticks=[0, 2]) == 0
    ax = plt.gcf().axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 10 * np.exp(.2 * np.arange(3)),
                               rtol=1e-3)
    np.testing.assert_array_equal(ax.get_xticks(), [0, 2])


def test_forecast_without_covariance_keeps_known_parameters():
    with pytest.warns(RuntimeWarning):
        bd.forecast('linear', 10, [0, 1], [.6], known_p=[.4], idx_known_p=[1])
    np.testing.assert_allclose(plt.gcf().axes[0].lines[0].get_ydata(), [10, 10 * np.exp(.2)],
                               rtol=1e-3)
    bd.forecast('linear', 10, [0, 1], [.6], known_p=[.4], idx_known_p=[1],
                interval='prediction', k=20, seed=42)
    assert len(plt.gcf().axes[0].lines[0].get_ydata()) == 2


def test_forecast_rejects_a_fluid_prediction_interval():
    with pytest.raises(TypeError, match='prediction'):
        bd.forecast('linear', 10, [0, 1], [.6, .4], interval='prediction', method='fm')


def test_forecast_with_a_single_time_returns_initial_population():
    with pytest.warns(RuntimeWarning):
        bd.forecast('linear', 10, [100], [.6, .4])
    np.testing.assert_array_equal(plt.gcf().axes[0].lines[0].get_ydata(), [10])
