import numpy as np
import warnings
from birdepy.interface_abc import discrete_est_abc
from birdepy.interface_dnm import discrete_est_dnm
from birdepy.interface_dnm import continuous_est_dnm
from birdepy.interface_lse import discrete_est_lse
from birdepy.interface_em import discrete_est_em
import birdepy.utility as ut
import time
from scipy.optimize import OptimizeResult
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy import simulate


def estimate(t_data, p_data, p0, p_bounds, framework='dnm', model='Verhulst',
             scheme='discrete', con=(), known_p=(), idx_known_p=(),
             se_type='asymptotic', seed=None, ci_plot=False, export=False, display=False,
             **options):
    """Parameter estimation for (continuously or discretely observed)
    continuous-time birth-and-death processes.

    Parameters
    ----------
    t_data : array_like
        Observation times of birth-and-death process. If one trajectory is
        observed, then this is a list. If multiple trajectories are observed,
        then this is a list of lists where each list corresponds to a
        trajectory.

    p_data : array_like
        Observed populations of birth-and-death process at times in argument
        `t_data`. If one trajectory is observed, then this is a list. If
        multiple trajectories are observed, then this is a list of lists
        where each list corresponds to a trajectory.

    p0 : array_like
        Initial parameter guess. Array of real elements of size (n,), where n
        is the number of unknown parameters.

    p_bounds : list
        Bounds on parameters. Should be specified as a sequence of
        ``(min, max)`` pairs for each unknown parameter. See :ref:`here <Parameter Constraints>`.

    framework : string, optional
        Parameter estimation framework. Should be one of:

            - 'abc' (see :ref:`here <birdepy.estimate(framework='abc')>`)
            - 'dnm' (default) (see :ref:`here <birdepy.estimate(framework='dnm')>`)
            - 'em' (see :ref:`here <birdepy.estimate(framework='em')>`)
            - 'lse' (see :ref:`here <birdepy.estimate(framework='lse')>`)

        Additional kwargs are available for each framework.

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

    scheme : string, optional
        Observation scheme. Should be one of:

            - 'discrete' (default)
            - 'continuous' (see :ref:`here <birdepy.estimate(scheme='continuous')>`)

        If set to 'continuous', then it is assumed that the population is
        observed continuously with jumps occuring at times in `t_data` into
        corresponding states in `p_data`.

    con : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition for parameters (only for kwarg `opt_method`
        equal to 'differential-evolution', 'COBYLA', 'SLSQP' or
        'trust-constr').   See :ref:`here <Parameter Constraints>` for more
        information.

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

    se_type: string, optional
        Should be one of: 'none' (default), 'simulated', or 'asymptotic'.
        See :ref:`here <Confidence Regions>` for more information.

    display : bool, optional
        If True, then progress updates are provided for some methods.

    ci_plot : bool, optional
        Enables confidence region plotting for 2d parameter estimates.

    export : str, optional
        File name for export of confidence region figure to a LaTeX file.

    Returns
    -------
    res : EstimationOutput
        The estimation output represented as an :func:`EstimationOutput`
        object. Important attributes are: `p` the parameter estimate, `se` the
        standard error estimate, `cov` the estimated covariance of the
        assumed distribution of the parameter estimate, `val` is the log-likelihood
        for 'framework' `dnm' and 'em', squared error for 'framework' `lse',
        `capacity` is the estimated carrying capacity.

    Examples
    --------
    Example 1: Simulate a discretely observed sample path and estimate the parameters using the
    alternative frameworks.
    First simulate some sample paths of a Verhulst 2 (SIS) model using
    :func:`birdepy.simulate.discrete()`:

    >>> import birdepy as bd
    >>> t_data = [t for t in range(100)]
    >>> p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
    ...                               survival=True, seed=2021)

    Then, using the simulated data, estimate the parameters using 'dnm', 'lse'
    and 'em' as the argument of `framework`:

    >>> est_dnm = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
    ...                       framework='dnm', model='Ricker', idx_known_p=[3], known_p=[1])
    >>> est_em = bd.estimate(t_data, p_data, [1, 0.5, 0.05], [[0,1], [0,1], [1e-6,0.1]],
    ...                       framework='em', model='Ricker', idx_known_p=[3], known_p=[1])
    >>> est_abc = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
    ...                       framework='abc', model='Ricker', idx_known_p=[3], known_p=[1])
    >>> est_lse = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
    ...                       framework='lse', model='Ricker', idx_known_p=[3], known_p=[1], se_type='simulated')
    >>> print('abc estimate:', est_abc.p, ', abc standard errors:', est_abc.se)
    >>> print('dnm estimate:', est_dnm.p, ', dnm standard errors:', est_dnm.se)
    >>> print('lse estimate:', est_lse.p, ', lse standard errors:', est_lse.se)
    >>> print('em estimate:', est_em.p, ', em standard errors:', est_em.se)
    abc estimate: [0.5237212840549004, 0.15633742500248485, 0.04781193037194212] , abc standard errors: [0.26827164 0.13484149 0.02876892]
    dnm estimate: [0.7477212189824904, 0.2150484536334751, 0.022745124483227304] , dnm standard errors: [0.16904225 0.03443199 0.00433567]
    em estimate: [0.7375802511179848, 0.19413965548145604, 0.024402343633644553] , em standard errors: [0.15742852 0.02917437 0.00429763]
    lse estimate: [0.7941741586214265, 0.2767698457541133, 0.01935636627568731] , lse standard errors: [0.1568291  0.19470746 0.01243208]

    Alternatively, we may be interested in continuously observed data.

    Example 2: Simulate a continuous sample path and estimate the parameters.

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
    tic = time.time()

    if 'options' in options.keys():
        options = options['options']

    p0 = np.array(p0)
    known_p = np.array(known_p)
    idx_known_p = np.array(idx_known_p)

    options = ut.add_options(options)

    if 'seed' is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    data = ut.data_sort(t_data, p_data)

    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    if model == 'Poisson':
        warnings.warn("Since argument 'model' is set to 'Poisson', "
                      "argument 'method' has been overridden and the true "
                      "dnm will be returned. ",
                      category=RuntimeWarning)
        total_count = 0
        total_time = 0
        for idx1 in range(len(t_data)):
            total_count += p_data[idx1][-1] - p_data[idx1][0]
            total_time += t_data[idx1][-1] - t_data[idx1][0]
        return total_count / total_time

    if 'z_trunc' in options.keys():
        z_trunc = options['z_trunc']
    else:
        z_obs_min = np.inf
        z_obs_max = 0
        for idx in range(len(p_data)):
            z_obs_min = np.minimum(z_obs_min, np.amin(p_data[idx]))
            z_obs_max = np.maximum(z_obs_max, np.amax(p_data[idx]))
        obs_range = z_obs_max - z_obs_min
        z_trunc = [int(np.maximum(0, z_obs_min - 2 * obs_range)),
                   int(z_obs_max + 2 * obs_range)]
        options['z_trunc'] = z_trunc

    class EstimationOutput(OptimizeResult):
        pass

    if scheme == 'continuous':
        if 'opt_method' in options.keys():
            opt_method = options['opt_method']
        else:
            opt_method = 'L-BFGS-B'

        if framework != 'dnm':
            raise TypeError("Argument `framework` must be 'dnm' when "
                            "argument `scheme` is 'continuous'.")

        opt, cov = continuous_est_dnm(t_data, p_data, p0, b_rate, d_rate,
                                      p_bounds, con, known_p, idx_known_p,
                                      opt_method, options)
        p_est = opt.x
        message = opt.message
        success = opt.success
        err = opt.fun
        iterations = 'Not applicable for `mle`. '
        accelerator = 'Not applicable for `mle`. '
        samples = 'Not applicable for `mle`. '
        method = 'mle'

    elif scheme == 'discrete':
        if framework == 'abc':
            sorted_data = ut.data_sort_2(data)
            if 'eps0' in options.keys():
                eps0 = options['eps0']
            else:
                eps0 = 10
                options['eps0'] = eps0
            if 'k' in options.keys():
                k = options['k']
            else:
                k = 100
                options['k'] = k
            if 'its' in options.keys():
                its = options['its']
            else:
                its = 2
                options['its'] = its
            if 'method' in options.keys():
                method = options['method']
            else:
                method = 'gwa'
                options['method'] = method
            if 'tau' in options.keys():
                tau = options['tau']
            else:
                tau = min(min(sorted_data.keys()) / 10, 0.1)
                options['tau'] = tau
            if 'distance' in options.keys():
                distance = options['distance']
            else:
                distance = None
                options['distance'] = distance
            if 'stat' in options.keys():
                stat = options['stat']
            else:
                stat = 'median'
                options['stat'] = distance
            if 'c' in options.keys():
                c = options['c']
            else:
                c = 2
                options['c'] = c

            pre_p_est, cov, samples = \
                discrete_est_abc(sorted_data, eps0, distance, stat,
                                 k, its, c, method, b_rate, d_rate, idx_known_p,
                                 known_p, p_bounds, con, tau, rng, display)

            if its > 1:
                p_est = pre_p_est[-1]
            else:
                p_est = pre_p_est
            message = 'Not applicable for `abc`.'
            success = 'Not applicable for `abc`.'
            iterations = 'Not applicable for `abc`. '
            err = 'Not applicable for `abc`.'

        elif framework == 'dnm':
            if 'likelihood' in options.keys():
                likelihood = options['likelihood']
            else:
                likelihood = 'expm'
                options['likelihood'] = 'expm'
            if 'opt_method' in options.keys():
                opt_method = options['opt_method']
            else:
                if con == ():
                    opt_method = 'L-BFGS-B'
                else:
                    opt_method = 'SLSQP'
                options['opt_method'] = opt_method

            opt, cov = discrete_est_dnm(data, likelihood, model,
                                        z_trunc, idx_known_p, known_p, p0,
                                        p_bounds, con, opt_method, options)
            p_est = opt.x
            message = opt.message
            success = opt.success
            err = -opt.fun
            iterations = 'Not applicable for `dnm`. '
            method = 'Not applicable for `dnm`. '
            samples = 'Not applicable for `dnm`. '

        elif framework == 'em':
            if 'accelerator' in options.keys():
                accelerator = options['accelerator']
            else:
                accelerator = 'Lange'
            if 'opt_method' in options.keys():
                opt_method = options['opt_method']
            elif accelerator == 'none':
                opt_method = 'SLSQP'
            else:
                opt_method = 'SLSQP'
            if 'technique' in options.keys():
                technique = options['technique']
            else:
                technique = 'expm'
            if 'likelihood' in options.keys():
                likelihood = options['likelihood']
            elif technique == 'expm':
                likelihood = 'expm'
            elif technique == 'ilt':
                likelihood = 'ilt'
            else:
                likelihood = 'expm'
            if 'max_it' in options.keys():
                max_it = options['max_it']
            else:
                max_it = 25
            if 'i_tol' in options.keys():
                i_tol = options['i_tol']
            else:
                i_tol = 1e-3
            if 'j_tol' in options.keys():
                j_tol = options['j_tol']
            else:
                j_tol = 1e-2
            if 'h_tol' in options.keys():
                h_tol = options['h_tol']
            else:
                h_tol = 1e-2

            if accelerator == 'test_aug':
                # This can be used to return the augmented data for
                # parameter values p0 using bd.discrete_est, which may be useful
                # for performance evaluation or debugging.
                sorted_data = ut.data_sort_2(data)

                aug_data = f_fun_bld(sorted_data, p0, b_rate, d_rate, likelihood,
                                     technique, idx_known_p, known_p, model,
                                     z_trunc, j_tol, h_tol, options)[1]

                return aug_data

            p_est, cov, ll, iterations = discrete_est_em(
                data, p0, technique, accelerator, likelihood, p_bounds, con,
                known_p, idx_known_p, model, b_rate, d_rate, z_trunc, max_it,
                i_tol, j_tol, h_tol, display, opt_method, options)

            message = 'Not applicable for `em`.'
            success = 'Not applicable for `em`.'
            err = ll
            method = accelerator
            samples = 'Not applicable for `em`. '

        elif framework == 'lse':
            if 'opt_method' in options.keys():
                opt_method = options['opt_method']
            else:
                opt_method = 'L-BFGS-B'
                options['opt_method'] = opt_method
            if 'squares' in options.keys():
                squares = options['squares']
            else:
                squares = 'fm'
                options['squares'] = squares

            opt = discrete_est_lse(data, squares, model, b_rate, d_rate, z_trunc,
                                   idx_known_p, known_p, p0, p_bounds,
                                   con, opt_method, options)

            p_est = opt.x
            message = opt.message
            success = opt.success
            err = opt.fun
            cov = 'Not applicable for `lse`.'
            iterations = 'Not applicable for `lse`. '
            method = squares
            samples = 'Not applicable for `lse`. '

        else:
            raise ValueError("Argument `framework` has an unknown value. Should "
                             "be one of: 'dnm', 'lse', 'em'.")
    else:
        raise ValueError("Argument `scheme` has an unknown value. Should "
                         "be one of: 'discrete' or 'continuous'.")

    if 'xlabel' in options.keys():
        xlabel = options['xlabel']
    else:
        xlabel = "$\\theta_1$"
    if 'ylabel' in options.keys():
        ylabel = options['ylabel']
    else:
        ylabel = "$\\theta_2$"

    # Compute confidence regions and standard errors
    if se_type == 'asymptotic':
        if framework == 'lse':
            se = "Asymptotic confidence intervals are not available for " \
                 "`framework' 'lse'. Set argument `se_type` to 'none'" \
                 "or 'simulated'."
        else:
            # For frameworks 'abc', 'dnm' and 'em', use cov value computed above
            try:
                se = np.sqrt(np.diag(cov))
            except:
                se = 'Error computing standard errors. Covariance matrix may have' \
                     'negative diagonal entries.'
            if ci_plot:
                ut.confidence_region(mean=p_est, cov=cov, obs=None, se_type=se_type, xlabel=xlabel,
                                     ylabel=ylabel, export=export)
    elif se_type == 'simulated':
        if 'num_samples' in options.keys():
            num_samples = options['num_samples']
        else:
            num_samples = 100

        param = ut.p_bld(p_est, idx_known_p, known_p)

        bootstrap_samples = np.zeros((num_samples, p0.size))
        for idx in range(num_samples):
            if scheme == 'continuous':
                if type(t_data[0]) == list:
                    t_data_temp = []
                    p_data_temp = []
                    for idx in t_data:
                        times, pops = simulate.continuous(p_est,
                                                          model,
                                                          p_data[idx][0],
                                                          model,
                                                          t_data[idx][-1],
                                                          seed=rng)
                        t_data_temp.append(times)
                        p_data_temp.append(pops)
                else:
                    t_data_temp, p_data_temp = simulate.continuous(
                        p_est, model, p_data[idx][0], t_data[idx][-1])
                bootstrap_samples[idx, :] = continuous_est_dnm(
                    t_data_temp, p_data_temp, p0, b_rate, d_rate, p_bounds, con,
                    known_p, idx_known_p, opt_method, options)
            else:  # observation scheme is discrete
                temp_data = {}
                for i in data:
                    for _ in range(data[i]):
                        new_zt = simulate.discrete(param, model,
                                                   i[0], [0, i[2]], k=1,
                                                   seed=rng)[1]
                        new_point = (i[0], new_zt, i[2])
                        if new_point in temp_data:
                            temp_data[new_point] += 1
                        else:
                            temp_data[new_point] = 1
                if framework == 'dnm':
                    bootstrap_samples[idx, :] = discrete_est_dnm(
                        temp_data, likelihood, model, z_trunc, idx_known_p,
                        known_p, p_est, p_bounds, con, opt_method, options)[0].x
                elif framework == 'lse':
                    bootstrap_samples[idx, :] = discrete_est_lse(
                        temp_data, squares, model, b_rate, d_rate, z_trunc,
                        idx_known_p, known_p, p_est, p_bounds, con, opt_method,
                        options).x
                else:  # framework is 'em'
                    bootstrap_samples[idx, :] = discrete_est_em(
                        data, p_est, technique, accelerator, likelihood, p_bounds,
                        con, known_p, idx_known_p, model, b_rate, d_rate,
                        z_trunc, max_it, i_tol, j_tol, h_tol, display,
                        opt_method, options)[0]
            if display:
                print('Boostrap confidence region progress:',
                      100 * (idx + 1) / num_samples, '%')
            cov = np.cov(bootstrap_samples, rowvar=False)
        try:
            se = np.sqrt(np.diag(cov))
        except:
            se = 'Error computing standard errors. Covariance matrix may have ' \
                 'negative diagonal entries.'
        if ci_plot:
            try:
                ut.confidence_region(mean=p_est, cov=cov, se_type=se_type,
                                     obs=bootstrap_samples, xlabel=xlabel,
                                     ylabel=ylabel, export=export)
            except:
                warnings.warn("Error plotting confidence regions. Estimated "
                              "covariance matrix may have negative diagonal "
                              "entries.'",
                              category=RuntimeWarning)
    elif se_type == 'none':
        se = 'Not requested (see argument `se_type`).'
    else:
        raise TypeError("Argument `se_type` has an unknown value. Possible "
                        "options are 'none', 'asymptotic' and 'simulated' ")

    
    if model != 'custom':
        capacity_finder = ut.higher_zf_bld(model)
        capacity = capacity_finder(ut.p_bld(np.array(p_est), idx_known_p, known_p))
    else:
        capacity = "Functionality not available for custom models."

    return EstimationOutput(p=list(p_est), capacity=capacity, val=err, cov=cov,
                            se=list(se), compute_time=time.time() - tic,
                            framework=framework, message=message,
                            success=success, iterations=iterations,
                            method=method, p0=list(p0),
                            scheme=scheme, samples=samples)
