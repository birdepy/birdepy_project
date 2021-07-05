import numpy as np
from birdepy import simulate
import birdepy.utility as ut
import matplotlib.pyplot as plt
from matplotlib import cm


def forecast(z0, times, param, model,
             percentiles=[0, 0.01, 0.2, 2.3, 15.9, 50, 84.1, 97.7, 99.8, 99.9, 100],
             cov=None, p_bounds=None, con=(), known_p=(), idx_known_p=(),
             sim_method='gwa', k=10 ** 3, seed=None, colormap=cm.Purples,
             xlabel='Time', ylabel='Forecast Population',
             rotation=45, display=False, **options):
    """Parameter estimation for (continuously or discretely observed)
    continuous-time birth-and-death processes.

    Parameters
    ----------
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

    model : string, optional
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes>`). Should be one of:

            - 'Verhulst 1' (default)
            - 'Verhulst 2 (SIS)'
            - 'Ricker 1'
            - 'Ricker 2'
            - 'Beverton-Holt'
            - 'Hassell'
            - 'M-SS'
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

    percentiles : list, optional
        List of percentiles to split data into for displaying. The default values
        split the data at 1, 2, and 3 standard deviations from the mean in both
        directions.

    cov : array_like, optional
        The parameters are assumed to follow a truncated normal distribution
        with this covariance. If this is specified, then p_bounds should also be
        specified to avoid unwanted parameters. Incorporating uncertainty into
        parameters means the function will provide prediction intervals instead
        or confidence intervals (prediction intervals are typically
        substantially wider).

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

    sim_method : string, optional
        Simulation algorithm used to generate samples (see
        :ref:`here<Simulation Algorithms>`). Should be one of:

            - 'exact' (default)
            - 'ea'
            - 'ma'
            - 'gwa'

    k : int, optional
        Number of sample paths used to generate forecast.

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

    rotation : int, optional
        Rotation of x tick labels.

    display : bool, optional
        If True, then progress updates are provided.

    Examples
    --------
    First simulate some sample paths of a Verhulst 2 (SIS) model using
    :func:`birdepy.simulate.discrete()`:

    >>> import birdepy as bd
    ... t_data = [t for t in range(100)]
    ... p_data = bd.simulate.discrete([0.75, 0.25, 50], model='Verhulst 2 (SIS)', z0=10,
    ...                               times=t_data, k=1, survival=True, seed=2021)

    Then, using the simulated data, estimate the parameters:

    >>> est = bd.estimate(t_data, p_data, [0.5, 0.5], [[0, 1], [0, 1]], model='Verhulst 2 (SIS)',
    ...                   known_p=[50], idx_known_p=[2])

    Then, use the estimated parameters and covariances to generate a forecast:

    >>> bd.forecast(86, [t for t in range(1999, 2021, 1)], est.p, model='Ricker',display=True)

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

    :func:`birdepy.gpu_functions.probability_gpu()`  :func:`birdepy.gpu_functions.discrete_gpu()`

    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    times = np.array(times)

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

    if cov is None:
        param = ut.p_bld(np.array(param), idx_known_p, known_p)
        sampled_data = np.array(simulate.discrete(param, 'custom', z0, times-times[0],
                                                  b_rate=b_rate, d_rate=d_rate,
                                                  k=k, method=sim_method,
                                                  seed=rng))
    else:
        sampled_data = np.zeros((k, times.shape[0]))
        for idx in range(k):
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
            sampled_data[idx, :] = np.array(simulate.discrete(param_prop, 'custom', z0, times-times[0],
                                                              b_rate=b_rate, d_rate=d_rate,
                                                              k=1, method=sim_method,
                                                              seed=rng))
            if display:
                print(f"Forecast is ", 100 * (idx + 1) / k, f"% complete.")

    n = len(percentiles)

    SDist = np.zeros((times.shape[0], n))
    for i in range(n):
        for t in range(times.shape[0]):
            SDist[t, i] = np.percentile(sampled_data[:, t], percentiles[i])

    half = int((n - 1) / 2)

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8, 4))
    fig.canvas.draw()
    ax1.plot(np.arange(0, times.shape[0], 1), SDist[:, half], color='k')
    for i in range(half):
        ax1.fill_between(np.arange(0, times.shape[0], 1), SDist[:, i], SDist[:, -(i + 1)],
                         color=colormap(i / half))
    ax1.tick_params(labelsize=11.5)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_xticks(times-times[0])
    labels = [f'{t}' for t in times]
    ax1.set_xticklabels(labels, rotation=rotation)
    ax1.set_ylabel(ylabel, fontsize=14)
    fig.tight_layout()

    return 0
