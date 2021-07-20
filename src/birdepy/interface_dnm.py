import numpy as np
import birdepy.utility as ut
from birdepy.interface_probability import probability
import warnings


def bld_ll_fun(data, likelihood, model, z_trunc, options):
    if likelihood in ['da', 'gwa', 'gwasa', 'ilt', 'oua']:
        def ll_fun(param):
            ll = 0
            for i in data:
                pr = probability(i[0], i[1], i[2], param, method=likelihood,
                                 model=model, z_trunc=z_trunc,
                                 options=options)[0][0]
                if np.isnan(pr):
                    warnings.warn("Computation of a transition probability has failed and been "
                                  "replaced by 1e-100. The results may be unreliable. ",
                                  category=RuntimeWarning)
                    ll += data[i] * np.log(1e-100)
                else:
                    ll += data[i] * np.log(np.maximum(1e-100, pr))
            return ll
    elif likelihood in ['Erlang', 'expm', 'uniform']:
        sorted_data = ut.data_sort_2(data)
        z_min, z_max = z_trunc

        def ll_fun(param):
            ll = 0
            for t in sorted_data:
                p_mat = probability(np.arange(z_min, z_max + 1, 1),
                                    np.arange(z_min, z_max+1, 1), t, param,
                                    method=likelihood, model=model,
                                    z_trunc=z_trunc, options=options)
                for i in sorted_data[t]:
                    ll += sorted_data[t][i] * np.log(
                        np.maximum(1e-100, p_mat[i[0] - z_min, i[1] - z_min]))
            return ll

    else:
        raise TypeError("Argument likelihood has an unknown value.")

    return ll_fun


def discrete_est_dnm(data, likelihood, model, z_trunc, idx_known_p, known_p,
                     p0, p_bounds, con, opt_method, options):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using direct numerical maximization of
    approximate likelihood functions.
    See :ref:`here <Direct Numerical Maximization>` for more information.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'dnm'::

        birdepy.estimate(t_data, p_data, p0, p_bounds, framework='dnm', likelihood='expm', z_trunc=())

    The parameters associated with this framework (listed below) can be
    accessed using kwargs in :func:`birdepy.estimate()`. See documentation of
    :func:`birdepy.estimate` for the main arguments.

    Parameters
    ----------
    likelihood : string, optional
        Likelihood approximation method. Should be one of
        (alphabetical ordering):

            - 'da' (see :ref:`here <birdepy.probability(method='da')>`)
            - 'Erlang' (see :ref:`here <birdepy.probability(method='Erlang')>`)
            - 'expm'  (default) (see :ref:`here <birdepy.probability(method='expm')>`)
            - 'gwa' (see :ref:`here <birdepy.probability(method='gwa')>`)
            - 'gwasa' (see :ref:`here <birdepy.probability(method='gwasa')>`)
            - 'ilt' (see :ref:`here <birdepy.probability(method='ilt')>`)
            - 'oua' (see :ref:`here <birdepy.probability(method='oua')>`)
            - 'uniform' (see :ref:`here <birdepy.probability(method='uniform')>`)
        The links point to the documentation of the relevant `method` in
        :func:`birdepy.probability`. The arguments associated with each of
        these methods may be used as a kwarg in :func:`birdepy.estimate()`
        when `likelihood` is set to use the method.

    z_trunc : array_like, optional
        Truncation thresholds, i.e., minimum and maximum states of process
        considered. Array of real elements of size (2,) by default
        ``z_trunc=[z_min, z_max]`` where
        ``z_min=max(0, min(p_data) - 2*z_range)``
        and ``z_max=max(p_data) + 2*z_range`` where
        ``z_range=max(p_data)-min(p_data)``. Only applicable to `likelihood`
        methods 'Erlang', 'expm' and 'uniform'.

    Examples
    --------
    Simulate a sample path and estimate the parameters using the various
    likelihood approximation methods.

    The constraint ``con={'type': 'ineq', 'fun': lambda p: p[0]-p[1]}`` ensures
    that p[0] > p[1] (i.e., rate of spread greater than recovery rate).

    >>> import birdepy as bd
    >>> t_data = [t for t in range(100)]
    >>> p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
    ...                               survival=True, seed=2021)
    >>> for likelihood in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'ilt', 'oua', 'uniform']:
    ...     likelihood = 'gwasa'
    >>>     est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[1e-6,1], [1e-6,1], [1e-6, 0.1]],
    ...                       framework='dnm', model='Ricker', idx_known_p=[3], known_p=[1],
    ...                       likelihood=likelihood)
    >>>     print('dnm estimate using', likelihood, 'is', est.p, ', with standard errors',
    ...           est.se, 'computed in ', est.compute_time, 'seconds.')
    dnm estimate using da is [0.742967758293063, 0.21582082476775125, 0.022598554340938885] , with standard errors [0.16633436 0.03430501 0.00436483] computed in  12.29400086402893 seconds.
    dnm estimate using Erlang is [0.7478158070214841, 0.2150341741826537, 0.022748822721356705] , with standard errors [0.16991822 0.0345762  0.00435054] computed in  0.9690003395080566 seconds.
    dnm estimate using expm is [0.7477255598292176, 0.2150476994316206, 0.022745305129350565] , with standard errors [0.16904095 0.03443197 0.00433563] computed in  1.6919987201690674 seconds.
    dnm estimate using gwa is [0.6600230500097711, 0.16728663936008945, 0.02512248420514078] , with standard errors [0.14248815 0.02447161 0.00488879] computed in  37.52255415916443 seconds.
    dnm estimate using gwasa is [0.6604981297820195, 0.16924607541398484, 0.02492054535741541] , with standard errors [0.14244908 0.02485465 0.00488222] computed in  0.8699958324432373 seconds.
    dnm estimate using ilt is [0.7466254648849691, 0.21415145383850764, 0.022794996238547492] , with standard errors [0.10187377 0.03137803        nan] computed in  1185.0924031734467 seconds.
    dnm estimate using oua is [0.5000083585920406, 0.5, 0.05] , with standard errors [       nan        nan 0.01961143] computed in  3.466001272201538 seconds.
    dnm estimate using uniform is [0.7477293759434092, 0.215047068344254, 0.022745437226772615] , with standard errors [0.16900378 0.03443071 0.00433504] computed in  3.275972366333008 seconds.

    How well methods perform varies from case to case. In this instance most methods perform well,
    while some throw errors but return useful output regardless, and some fail altogether.

    Notes
    -----
    Estimation algorithms and models are also described in [1]. If you use this
    function for published work, then please cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.
    """

    pre_ll_fun = bld_ll_fun(data, likelihood, model, z_trunc, options)

    def error_fun(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return -pre_ll_fun(param)

    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll_fun(param)

    opt = ut.minimize_(error_fun, p0, p_bounds, con, opt_method, options)

    if p0.size > 1:
        try:
            cov = np.linalg.inv(-ut.Hessian(ll, opt.x, p_bounds))
        except:
            cov = 'Covariance matrix could not be determined.'
    else:
        try:
            cov = -1/ut.Hessian(ll, opt.x, p_bounds)
        except:
            cov = 'Covariance matrix could not be determined.'

    return opt, cov


def continuous_est_dnm(t_data, p_data, p0, b_rate, d_rate, p_bounds, con,
                       known_p, idx_known_p, opt_method, options):
    """Parameter estimation for continuously  observed continuous-time
    birth-and-death processes using direct numerical maximization of
    the likelihood.
    See :ref:`here <Continuously Observed Data>` for more information.

    To use this function call :func:`birdepy.estimate` with `scheme` set to
    'continuous'::

        birdepy.estimate(t_data, p_data, p0, p_bounds, scheme='continuous')


    Examples
    --------
    Simulate a continuous sample path and estimate the parameters.

    Import BirDePy:

    >>> import birdepy as bd

    Simulate some synthetic data:

    >>> t_data, p_data = bd.simulate.continuous([0.75, 0.25, 0.02, 1], 'Ricker', 10,
    ...                                         100, survival=True, seed=2021)

    Estimate:

    >>> est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
    ...                   model='Ricker', idx_known_p=[3], known_p=[1], scheme='continuous')
    >>> print(est.p)
    [0.7603171062895576, 0.2514810854871476, 0.020294342655751033]

    Notes
    -----
    Estimation algorithms and models are also described in [1]. If you use this
    function for published work, then please cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    See also
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    --------


    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.
    """

    if type(t_data[0]) != list:
        t_data = [t_data]
    if type(p_data[0]) != list:
        p_data = [p_data]

    holding_times = []
    for idx in range(len(t_data)):
        holding_times.append(np.array(t_data[idx][1:]) -
                             np.array(t_data[idx][0:-1]))

    flattened_p_data = [item for sublist in p_data for item in sublist]

    states_visited = np.arange(np.amin(flattened_p_data),
                               np.amax(flattened_p_data) + 1, 1)
    holding_t_data = np.zeros(len(states_visited))
    downward_jump_counts = np.zeros(len(states_visited))
    upward_jump_counts = np.zeros(len(states_visited))

    for idx1 in range(len(t_data)):
        state_record_temp = np.array(p_data[idx1])
        holding_times_temp = np.array(holding_times[idx1])
        for idx2, state in enumerate(states_visited):
            holding_t_data[idx2] += \
                np.sum(holding_times_temp[state_record_temp[:-1] == state])
            downward_jump_counts[idx2] += \
                np.sum(np.less(state_record_temp[1:], state_record_temp[0:-1])
                       & (state_record_temp[0:-1] == state))
            upward_jump_counts[idx2] += \
                np.sum(np.greater(state_record_temp[1:], state_record_temp[0:-1])
                       & (state_record_temp[0:-1] == state))


    def pre_ll(param):
        ll_ = 0
        for idx_, state_ in enumerate(states_visited):
            if b_rate(state_, param) > 0:
                ll_ += np.log(b_rate(state_, param)) * upward_jump_counts[idx_]
            if d_rate(state_, param) > 0:
                ll_ += np.log(d_rate(state_, param)) * downward_jump_counts[idx_]
            ll_ -= (b_rate(state_, param) + d_rate(state_, param)) * holding_t_data[idx_]
        return ll_

    def error_fun(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return -pre_ll(param)

    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    opt = ut.minimize_(error_fun, p0, p_bounds, con, opt_method, options)

    if p0.size > 1:
        try:
            cov = np.linalg.inv(-ut.Hessian(ll, opt.x, p_bounds))
        except:
            cov = 'Covariance matrix could not be determined.'
    else:
        try:
            cov = -1/ut.Hessian(ll, opt.x, p_bounds)
        except:
            cov = 'Covariance matrix could not be determined.'

    return opt, cov
