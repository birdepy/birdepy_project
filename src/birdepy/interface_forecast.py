import numpy as np
from birdepy import simulate
import birdepy.utility as ut
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from scipy.integrate import solve_ivp


def parameter_sampler(param, cov, p_bounds, known_p, idx_known_p, con, rng):
    while True:
        # First obtain proposal satisfying bounds:
        while True:
            param_prop = rng.multivariate_normal(param, cov)
            cond = True
            for bd_idx, bd in enumerate(p_bounds):
                if param_prop[bd_idx] > bd[1] or \
                        param_prop[bd_idx] < bd[0]:
                    cond = False
            if not cond:
                continue
            break
        # Then check proposal also satisfies constraints:
        if type(con) == dict:
            # If there is one constraint specified
            if con['fun'](param_prop) < 0:
                continue
        else:
            # If there is strictly more than one or 0 constraints specified
            cond = True
            for c in con:
                if c['fun'](param_prop) < 0:
                    cond = False
            if not cond:
                continue
        break
    param_prop = ut.p_bld(np.array(param_prop), idx_known_p, known_p)
    return param_prop


def mean_curve(param, b_rate, d_rate, method, z0, solver_methods,
               shift_times, n, rng):
    if method == 'fm':
        if callable(z0):
            z0_ = z0()
        else:
            z0_ = z0
        for solver_method in solver_methods:
            fluid_path = solve_ivp(
                lambda t, z: b_rate(z, param) - d_rate(z, param),
                [0, shift_times[-1]],
                [z0_],
                t_eval=shift_times,
                method=solver_method)
            if fluid_path.success:
                forecasted_mean = fluid_path.y[0]
                break
    elif method in ['exact', 'ea', 'ma', 'gwa']:
        forecasted_mean = np.zeros(len(shift_times))
        for idx in range(n):
            forecasted_mean += simulate.discrete(param, 'custom', z0,
                                                 shift_times,
                                                 b_rate=b_rate,
                                                 d_rate=d_rate, k=1,
                                                 method=method,
                                                 seed=rng)
        forecasted_mean = np.divide(forecasted_mean, n)
    return forecasted_mean


def forecast(model, z0, times, param, cov=None, interval='confidence', method=None,
             percentiles=(0, 2.5, 10, 25, 50, 75, 90, 97.5, 100),
             labels=('$95\%$', '$80\%$', '$50\%$'),
             p_bounds=None, con=(), known_p=(), idx_known_p=(),
             k=10 ** 3, n=10 ** 3, seed=None, colormap=cm.Purples,
             xlabel='t', ylabel='default', xticks='default',
             rotation=45, display=False, export=False, **options):
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
        Constraints definition for parameters.
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

    Examples
    --------
    First simulate some sample paths using :func:`birdepy.simulate.discrete()`:

    >>> import birdepy as bd
    >>> t_data = [t for t in range(101)]
    >>> p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
    ...                               survival=True, seed=2021)

    Then, using the simulated data, estimate the parameters:

    >>> est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
    ...                   model='Ricker', idx_known_p=[3], known_p=[1])

    Then, use the estimated parameters and covariances to generate a forecast:

    >>> future_t = [t for t in range(101,151,1)]
    >>> bd.forecast('Ricker', p_data[-1], future_t, est.p, cov=est.cov,
    ...             p_bounds=[[0,1], [0,1], [0, 0.1]], idx_known_p=[3], known_p=[1],
    ...             interval='prediction')


    Notes
    -----
    This function creates a plot but does not return anything.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    If `model` is 'Poisson', then the true dnm is immediately returned (i.e.,
    the total number of events during the observation periods divided by the
    total observation time).

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    :func:`birdepy.gpu_functions.probability()`  :func:`birdepy.gpu_functions.discrete()`

    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    times = np.array(times)

    if type(xticks) == str:
        xticks = times

    if type(xticks) is list:
        xticks = times
    else:
        xticks = np.array(xticks)

    if ylabel == 'default':
        if interval == 'confidence':
            ylabel = '$\mathbb{E} Z(t)$'
        else:
            ylabel = '$Z(t)'

    if interval == 'confidence' and method is None:
        method = 'fm'
    elif interval == 'prediction' and method is None:
        method = 'gwa'
    elif interval == 'prediction' and method == 'fm':
        TypeError("Argument of `method` equal 'fm' not possible when argument "
                  "of `interval` equals 'prediction'.")

    shift_times = times - times[0]

    if type(cov) == float or type(cov) == int:
        cov = [[cov]]

    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    #

    solver_methods = ['RK45', 'Radau', 'RK23', 'BDF', 'DOP853']

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    if interval == 'confidence':
        if cov is None:
            warnings.warn("Confidence intervals show the likely range of the mean "
                          "future population level given the uncertainty of the "
                          "parameter values, however since argument `cov` has value "
                          "None no uncertainty in parameter values has been specified.",
                          category=RuntimeWarning)
            forecast_ = mean_curve(param, b_rate, d_rate, method, z0,
                                   solver_methods, shift_times, n, rng)
            ax1.plot(times, forecast_, color='k')
        else:
            samples = np.zeros((k, times.shape[0]))
            for idx in range(k):
                param_prop = parameter_sampler(param, cov, p_bounds, known_p,
                                               idx_known_p, con, rng)
                samples[idx, :] = mean_curve(param_prop, b_rate, d_rate,
                                             method, z0, solver_methods,
                                             shift_times, n, rng)
                if display:
                    print(f"Forecast is ", 100 * (idx + 1) / k, f"% complete.")
    elif interval == 'prediction':
        samples = np.zeros((k, times.shape[0]))
        if cov is None:
            for idx in range(k):
                samples[idx, :] = simulate.discrete(param, 'custom', z0,
                                                    shift_times,
                                                    b_rate=b_rate,
                                                    d_rate=d_rate, k=1,
                                                    method=method,
                                                    seed=rng)
                if display:
                    print(f"Forecast is ", 100 * (idx + 1) / k, f"% complete.")
        else:
            for idx in range(k):
                param_prop = parameter_sampler(param, cov, p_bounds, known_p,
                                               idx_known_p, con, rng)
                samples[idx, :] = simulate.discrete(param_prop, 'custom', z0,
                                                    shift_times,
                                                    b_rate=b_rate,
                                                    d_rate=d_rate, k=1,
                                                    method=method,
                                                    seed=rng)
                if display:
                    print(f"Forecast is ", 100 * (idx + 1) / k, f"% complete.")
    else:
        raise TypeError("Argument 'interval' has an unknown value.")

    m = len(percentiles)
    SDist = np.zeros((times.shape[0], m))
    for i in range(m):
        for t in range(times.shape[0]):
            SDist[t, i] = np.percentile(samples[:, t], percentiles[i])
    half = int(np.floor((m - 1) / 2))
    fig.canvas.draw()
    ax1.plot(times, SDist[:, half], color='k')
    for i in range(half-1):
        ax1.fill_between(times, SDist[:, i+1], SDist[:, -(i + 2)], color=colormap((i+1) / half),
                         label=labels[i])
    ax1.tick_params(labelsize=11.5)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_xticks(xticks)
    ax1.legend(loc="upper left")
    labels = [f'{t}' for t in xticks]
    ax1.set_xticklabels(labels, rotation=rotation)
    ax1.set_ylabel(ylabel, fontsize=14)
    fig.tight_layout()
    if isinstance(export, str):
        import tikzplotlib
        tikzplotlib.save(export + ".tex")
    return 0
