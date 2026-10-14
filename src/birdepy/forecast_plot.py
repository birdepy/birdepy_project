"""Matplotlib rendering for numerical BirDePy forecast results."""

import matplotlib.pyplot as plt
from matplotlib import cm

from birdepy.forecast_compute import DEFAULT_PERCENTILES

DEFAULT_LABELS = (r'$95\%$', r'$80\%$', r'$50\%$')
DEFAULT_COLORMAP = cm.Purples


def plot_forecast(result, ax=None, labels=None,
                  colormap=DEFAULT_COLORMAP, xlabel='t', ylabel='default', xticks='default',
                  rotation=45):
    """Plot a :class:`~birdepy.results.ForecastResult` on an optional axis.

    Returns the axis used for rendering. Labels default to each band's
    percentile mass. Custom endpoints are paired from outside to inside around
    the median; the default percentile grid retains the legacy three bands.
    An explicit label sequence must have one label per band. A supplied axis
    retains its figure's layout policy.
    """
    pairs = []
    if result.quantiles is not None:
        if tuple(result.percentiles) == DEFAULT_PERCENTILES:
            pairs = [(1, -2), (2, -3), (3, -4)]
        else:
            lows = [i for i, p in enumerate(result.percentiles) if p < 50]
            highs = [i for i, p in enumerate(result.percentiles) if p > 50]
            pairs = list(zip(lows, reversed(highs)))
        if labels is None:
            labels = [f'${result.percentiles[high] - result.percentiles[low]:g}\\%$'
                      for low, high in pairs]
        elif isinstance(labels, str) or len(labels) != len(pairs):
            raise ValueError("labels must contain one label per percentile band.")
    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    times = result.times
    if isinstance(xticks, str):
        xticks = times
    if ylabel == 'default':
        ylabel = r'$\mathbb{E} Z(t)$' if result.interval == 'confidence' else '$Z(t)$'
    ax.plot(times, result.central, color='k')
    for i, (low, high) in enumerate(pairs):
        ax.fill_between(times, result.quantiles[:, low], result.quantiles[:, high],
                        color=colormap((i + 1) / (len(pairs) + 1)), label=labels[i])
    if pairs:
        ax.legend(loc="upper left")
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{t}' for t in xticks], rotation=rotation)
    ax.set_ylabel(ylabel, fontsize=14)
    if created_axis:
        ax.figure.tight_layout()
    return ax
