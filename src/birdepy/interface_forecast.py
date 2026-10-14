"""Legacy plotting wrapper for the numerical forecasting API."""

from birdepy.forecast_compute import (DEFAULT_PERCENTILES, compute_forecast,
                                      mean_curve, parameter_sampler)
from birdepy.forecast_plot import DEFAULT_COLORMAP, DEFAULT_LABELS, plot_forecast

# Keep historical helper import paths explicit for downstream callers.
__all__ = ["forecast", "mean_curve", "parameter_sampler"]


def forecast(model, z0, times, param, cov=None, interval='confidence', method=None,
             percentiles=DEFAULT_PERCENTILES,
             labels=DEFAULT_LABELS,
             p_bounds=None, con=(), known_p=(), idx_known_p=(),
             k=10 ** 3, n=10 ** 3, seed=None, colormap=DEFAULT_COLORMAP,
             xlabel='t', ylabel='default', xticks='default', rotation=45,
             display=False, export=False, **options):
    """Simulation based forecasting for continuous-time birth-and-death processes.
    Produces a plot of the likely range of mean population sizes subject to parameter uncertainty
    (confidence intervals) or the likely range of population sizes subject to parameter
    uncertainty and model stochasticity (prediction intervals).

    Parameters
    ----------
    model : string, optional
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes>`). Should be one of:

            - 'Verhulst' (default)
            - 'Ricker'
            - 'Hassell'
            - 'MS-S'
            - 'Moran'
            - 'pure-birth'
            - 'pure-death'
            - 'Poisson'
            - 'linear'
            - 'linear-migration'
            - 'M/M/1'
            - 'M/M/inf'
            - 'loss-system'
            - 'custom'

         If set to 'custom', then kwargs `b_rate` and `d_rate` must also be
         specified. See :ref:`here <Custom Models>` for more information.

    z0: int or callable
        The population for each sample path at the time of the first element
        of the argument of `times`.
        If it is a callable it should be a function that has no arguments and
        returns an int:

         ``z0() -> int``

    times : array_like
        Times to provide a forecast for.

    param : array_like
        The parameters governing the evolution of the birth-and-death
        process to be forecast.
        Array of real elements of size (m,), where ‘m’ is the number of
        parameters.
        When `cov` is provided this is taken to be a mean value.

    cov : array_like, optional
        The parameters are assumed to follow a truncated normal distribution
        with this covariance. If this is specified, then p_bounds should also be
        specified to avoid unwanted parameters.

    interval : string, optional
        Type of forecast. Should be one of 'confidence' (default) or
        'prediction'. Confidence intervals show the likely range of mean future
        population values, reflecting parameter uncertainty. Prediction interals
        show the likely range of future population values, incorporating
        parameter uncertainty and model stochasticity.

    method : string, optional
        Method used to generate samples. For confidence intervals samples are
        trajectories of future expected values. For prediction intervals
        samples are trajectories of future population values. Should be one of:

            - 'fm' (default for confidence intervals)
            - 'exact'
            - 'ea'
            - 'ma'
            - 'gwa' (default for prediction intervals)

    percentiles : list, optional
        List of percentiles to split the data into.

    labels : list, optional
        List of strings containing labels for each percentile split.

    p_bounds : list
        Bounds on parameters. Should be specified as a sequence of
        ``(min, max)`` pairs for each unknown parameter. See :ref:`here <Parameter Constraints>`.

    con : {Constraint, dict} or List of {Constraint, dict}, optional
        Inequality constraints on unknown parameters, applied when sampling
        with ``cov``. Dictionaries support vector-valued functions and ``args``;
        SciPy linear and nonlinear inequalities are also supported. Exact
        equalities are unsupported: use ``known_p`` or reparameterize the model.
        See :ref:`here <Parameter Constraints>` for more information.

    known_p : array_like, optional
        List of known parameter values. For built in models these must be in
        their canonical order as given (:ref:`here <Birth-and-death Processes>`). If this
        argument is given, then argument `idx_known_p` must also be specified.
        See :ref:`here <Known Parameters>` for more information.

    idx_known_p : array_like, optional
        List of indices of known parameters (as given in argument 'known_p').
        For built in models indices must correspond to canonical order as given
        :ref:`here <Birth-and-death Processes>`. If this argument is given, then argument
        `known_p` must also be specified. See :ref:`here <Known Parameters>`
        for more information.

    k : int, optional
        Number of samples used to generate forecast. For confidence intervals
        each sample corresponds to an estimate of the mean for a sampled
        parameter value. For prediction intervals each sample corresponds to
        a trajectory of population size for a sampled parameter value.

    n : int, optional
        Number of samples used to estimate each sample of a mean for confidence
        interval samples. Only applicable when method is 'exact', 'ea', 'ma'
        or 'gwa'.

    seed : int, Generator, optional
        If *seed* is not specified the random numbers are generated according
        to *np.random.default_rng()*. If *seed* is an *int*, random numbers are
        generated according to *np.random.default_rng(seed)*. If seed is a
        *Generator*, then that object is used. See
        :ref:`here <Reproducibility>` for more information.

    colormap : matplotlib.colors.LinearSegmentedColormap, optional
        Colors used for plot.

    xlabel : str, optional
        Label for x axis of plot.

    ylabel : str, optional
        Label for y axis of plot.

    xticks : array_like, optional
        Locations of x ticks.

    rotation : int, optional
        Rotation of x tick labels.

    display : bool, optional
        If True, then progress updates are provided.

    export : str, optional
        File name for export of the figure to a tex file.

    max_rejections : int, optional
        Maximum proposals per uncertain parameter sample (default 10000).
        Raises ``RuntimeError`` if bounds and constraints cannot be satisfied
        within this limit. Pass this control as a keyword argument.

    Examples
    --------
    First simulate some sample paths using : :`birdepy.simulate.discrete()`: ::

        import birdepy as bd
        t_data = list(range(101))
        p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
                                      survival=True, seed=2021)

    Then, using the simulated data, estimate the parameters: ::

        est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                          model='Ricker', idx_known_p=[3], known_p=[1])

    Then, use the estimated parameters and covariances to generate a forecast: ::

        future_t = list(range(101,151,1))
        bd.forecast('Ricker', p_data[-1], future_t, est.p, cov=est.cov,
                    p_bounds=[[0,1], [0,1], [0, 0.1]], idx_known_p=[3], known_p=[1],
                    interval='prediction')


    Notes
    -----
    This function creates a plot and returns ``0`` for compatibility.
    Use :func:`birdepy.compute_forecast` to obtain numerical results without
    creating a figure, and :func:`birdepy.plot_forecast` to render them on an
    optional supplied axis. For example::

        import matplotlib.pyplot as plt
        result = bd.compute_forecast(
            "linear", 10, [0, 1, 2], [.6, .4],
            interval="prediction", k=100, seed=123)
        fig, ax = plt.subplots()
        bd.plot_forecast(result, ax=ax)

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    :func:`birdepy.gpu_functions.probability()`  :func:`birdepy.gpu_functions.discrete()`

    References
    ----------
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    result = compute_forecast(
        model, z0, times, param, cov=cov, interval=interval, method=method,
        percentiles=percentiles, p_bounds=p_bounds, con=con, known_p=known_p,
        idx_known_p=idx_known_p, k=k, n=n, seed=seed, display=display,
        _warning_stacklevel=3, **options)
    plot_forecast(result, labels=None if labels is DEFAULT_LABELS else labels, colormap=colormap, xlabel=xlabel,
                  ylabel=ylabel, xticks=xticks, rotation=rotation)
    if isinstance(export, str):
        import tikzplotlib
        tikzplotlib.save(export + ".tex")
    return 0
