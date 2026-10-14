"""Matplotlib rendering for numerical BirDePy forecast results."""

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_forecast(result, ax=None, labels=(r'$95\%$', r'$80\%$', r'$50\%$'),
                  colormap=cm.Purples, xlabel='t', ylabel='default', xticks='default',
                  rotation=45):
    """Plot a :class:`~birdepy.results.ForecastResult` on an optional axis.

    Returns the axis used for rendering. No numerical calculation occurs here.
    """
    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    times = result.times
    if isinstance(xticks, str):
        xticks = times
    if ylabel == 'default':
        ylabel = r'$\mathbb{E} Z(t)$' if result.interval == 'confidence' else '$Z(t)$'

    if result.quantiles is None:
        ax.plot(times, result.central, color='k')
    else:
        m = len(result.percentiles)
        half = int((m - 1) // 2)
        ax.plot(times, result.central, color='k')
        legacy = tuple(result.percentiles) == (0, 2.5, 10, 25, 50, 75, 90, 97.5, 100)
        if legacy:
            pairs = [(i + 1, -(i + 2)) for i in range(half - 1)]
        else:
            lows = [i for i, p in enumerate(result.percentiles) if p < 50]
            highs = [i for i, p in enumerate(result.percentiles) if p > 50]
            pairs = list(zip(lows, reversed(highs)))
        for i, (low, high) in enumerate(pairs):
            label = labels[i] if legacy and i < len(labels) else (
                f'${result.percentiles[high] - result.percentiles[low]:g}\\%$')
            ax.fill_between(times, result.quantiles[:, low], result.quantiles[:, high],
                            color=colormap((i + 1) / (half if legacy else max(1, len(pairs))), label=label)
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
