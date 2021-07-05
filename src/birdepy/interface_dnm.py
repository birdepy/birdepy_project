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
    ... t_data = [t for t in range(100)]
    ... p_data = bd.simulate.discrete([0.75, 0.25, 50], model='Verhulst 2 (SIS)', z0=10,
    ...                               times=t_data, k=1, survival=True, seed=2021)
    ...     for likelihood in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'ilt', 'oua', 'uniform']:
    ...         est = bd.estimate(t_data, p_data, [0.51, 0.5], [[1e-6, 1], [1e-6, 1]],
    ...                           framework='dnm', likelihood=likelihood, model='Verhulst 2 (SIS)',
    ...                           known_p=[50], idx_known_p=[2],
    ...                           con={'type': 'ineq', 'fun': lambda p: p[0]-p[1]})
    ...         print('dnm estimate using', likelihood, 'is', est.p, ', with standard errors', est.se,
    ...               'computed in ', est.compute_time, 'seconds.')
    dnm estimate using da is [0.7884566542732246, 0.2628678940919104] , with standard errors [0.13912062 0.04737368] computed in  6.304335355758667 seconds.
    dnm estimate using Erlang is [0.7983016993133849, 0.2628201261897523] , with standard errors [0.14214373 0.04809272] computed in  0.3658630847930908 seconds.
    dnm estimate using expm is [0.7962965268028664, 0.262163461352853] , with standard errors [0.14114384 0.04776676] computed in  0.4570772647857666 seconds.
    dnm estimate using gwa is [0.5375673483197848, 0.17656810406126427] , with standard errors [0.06734435 0.02312377] computed in  20.887160778045654 seconds.
    dnm estimate using gwasa is [0.54391956118784, 0.17875443423303075] , with standard errors [0.06836115 0.02347495] computed in  0.712468147277832 seconds.
    dnm estimate using ilt is [0.8004878587607526, 0.26356570622759573] , with standard errors [0.00023999 0.00023996] computed in  1226.9041030406952 seconds.
    dnm estimate using oua is [0.653866482752309, 0.2267362102279563] , with standard errors [0.09957281 0.03645399] computed in  3.288478136062622 seconds.
    dnm estimate using uniform is [0.7962959505977425, 0.262163262841564] , with standard errors [0.14114499 0.04776725] computed in  1.3137309551239014 seconds.

    Using different parameters, we see different algorithms perform better/
    worse:

    >>> p_data = bd.simulate.discrete([0.075, 0.025, 50], model='Verhulst 2 (SIS)', z0=10,
    ...                               times=t_data, k=1, survival=True,
    ...                               seed=2021)
    ... for likelihood in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'ilt', 'oua',
    ...                    'uniform']:
    ...     est = bd.estimate(t_data, p_data, [0.51, 0.5], [[1e-6, 1], [1e-6, 1]],
    ...                       framework='dnm', likelihood=likelihood, model='Verhulst 2 (SIS)',
    ...                       known_p=[50], idx_known_p=[2],
    ...                       con={'type': 'ineq', 'fun': lambda p: p[0]-p[1]})
    ...     print('dnm estimate using', likelihood, 'is',
    ...           est.p, ', with standard errors', est.se,
    ...           'computed in ', est.compute_time, 'seconds.')
    dnm estimate using da is [0.06505654898375403, 0.022302239704218848] , with standard errors [0.00930282 0.00446215] computed in  6.565070152282715 seconds.
    /home/bp/anaconda3/envs/testing/lib/python3.8/site-packages/scipy/optimize/optimize.py:282: RuntimeWarning: Values in x were outside bounds during a minimize step, clipping to bounds
      warnings.warn("Values in x were outside bounds during a "
    dnm estimate using Erlang is [0.06458375273876914, 0.02193397249882732] , with standard errors [0.01048676 0.00450661] computed in  0.89235520362854 seconds.
    dnm estimate using expm is [0.06449716075484259, 0.021894088479430286] , with standard errors [0.01045434 0.00449354] computed in  0.8028414249420166 seconds.
    dnm estimate using gwa is [0.06295369690363607, 0.02129198225320385] , with standard errors [0.01007199 0.00432168] computed in  25.937010288238525 seconds.
    /home/bp/Dropbox/Brendan/Active Projects/BranchingProcessEstimation/NumericsSimulations/birdepy_project/birdepy/probability_gwasa.py:277: RuntimeWarning: Probability not in [0, 1] computed, some output has been replaced by a default value.  Results may be unreliable.
      warnings.warn("Probability not in [0, 1] computed, "
    dnm estimate using gwasa is [0.060933808369354664, 0.020348140612595354] , with standard errors [0.01042864 0.00450257] computed in  1.134232759475708 seconds.
    dnm estimate using ilt is [0.06450625629765833, 0.021898206532375137] , with standard errors [0.01045783 0.00449486] computed in  833.8152034282684 seconds.
    dnm estimate using oua is [0.05129604751115264, 0.02146969246135162] , with standard errors [0.00699228 0.00415237] computed in  1.780461311340332 seconds.
    dnm estimate using uniform is [0.06449715998400529, 0.021894085243895373] , with standard errors [0.01045434 0.00449354] computed in  2.339176654815674 seconds.

    This time warnings are given for methods 'da' and 'gwa', most likely due to numerical
    overflow, however accurate estimates are still provided.

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

    Simulate:

    >>> import birdepy as bd
    ... t_data, p_data = bd.simulate.continuous([0.75, 0.25, 50],'Verhulst 2 (SIS)',
    ...                                          10, 100, seed=2021)

    Estimate:

    >>> bd.estimate(t_data, p_data, [0.5, 0.5, 100], [[1e-6, 10], [1e-6, 10], [1,200]],
    ...             model ='Verhulst 2 (SIS)', scheme='continuous')
    ... print(est.p)
    [0.8468206412942119, 0.24924760035405574, 47.52248709677161]

    Notes
    -----
    Estimation algorithms and models are also described in [1]. If you use this
    function for published work, then please cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    See also
    :func:`birdepy.estimate()`
    :func:`birdepy.simulate.discrete()`
    :func:`birdepy.simulate.continuous()`
    :func:`birdepy.probability()`
    :func:`birdepy.forecast()`
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
