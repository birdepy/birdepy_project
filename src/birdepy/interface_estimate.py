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
    t_data : list
        Observation times of birth-and-death process. If one trajectory is
        observed, then this is a list. If multiple trajectories are observed,
        then this is a list of lists where each list corresponds to a
        trajectory.

    p_data : list
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
    First simulate some sample paths of a Ricker model using
    :func:`birdepy.simulate.discrete()`: ::

        import birdepy as bd
        t_data = list(range(100))
        p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
                                      survival=True, seed=2021)

    Then, using the simulated data, estimate the parameters using 'dnm', 'lse'
    and 'em' as the argument of `framework`: ::

        est_dnm = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                              framework='dnm', model='Ricker', idx_known_p=[3], known_p=[1])
        est_em = bd.estimate(t_data, p_data, [1, 0.5, 0.05], [[0,1], [0,1], [1e-6,0.1]],
                              framework='em', model='Ricker', idx_known_p=[3], known_p=[1])
        est_abc = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                              framework='abc', model='Ricker', idx_known_p=[3], known_p=[1])
        est_lse = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                              framework='lse', model='Ricker', idx_known_p=[3], known_p=[1], se_type='simulated')
        print(f'dnm estimate: {est_dnm.p}, dnm standard errors: {est_dnm.se}')
        print(f'lse estimate: {est_lse.p}, lse standard errors: {est_lse.se}')
        print(f'abc estimate: {est_abc.p}, abc standard errors: {est_abc.se}')
        print(f'em estimate: {est_em.p}, em standard errors: {est_em.se}')

    Outputs: ::

        dnm estimate: [0.7477212189824904, 0.2150484536334751, 0.022745124483227304] , dnm standard errors: [0.16904225 0.03443199 0.00433567]
        em estimate: [0.7375802511179848, 0.19413965548145604, 0.024402343633644553] , em standard errors: [0.15742852 0.02917437 0.00429763]
        abc estimate: [0.6318632898413052, 0.02074882329749562, 0.06580340596326038], abc standard errors: [0.22865334, 0.0148124, 0.0129306]
        lse estimate: [0.7941741586214265, 0.2767698457541133, 0.01935636627568731] , lse standard errors: [0.1568291  0.19470746 0.01243208]

    Alternatively, we may be interested in continuously observed data.

    Example 2: Simulate a continuous sample path and estimate the parameters.

    Simulate some synthetic data: ::

        t_data, p_data = bd.simulate.continuous([0.75, 0.25, 0.02, 1], 'Ricker', 10,
                                                100, survival=True, seed=2021)

    Estimate: ::

        est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                          model='Ricker', idx_known_p=[3], known_p=[1], scheme='continuous')
        print(est.p)

    Outputs: ::

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
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    # Start a timer so that the total time to run the function can be returned
    tic = time.time()

    # Key word arguments can be passed as a dictionary which is itself the value
    # of a key word argument with key 'options', in this case the dictionary
    # must be extracted:
    if 'options' in options.keys():
        options = options['options']

    # Convert the arguments of 'p0', 'known_p' and 'idx_known_p' into numpy arrays
    p0 = np.array(p0)
    known_p = np.array(known_p)
    idx_known_p = np.array(idx_known_p)

    # Augment the options dictionary with items used by the optimization routines later
    options = ut.add_options(options)

    # Set the random number generator
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    # Define the model-dependent birth and death rate functions
    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    # Create object to return results in
    class EstimationOutput(OptimizeResult):
        pass

    # Determine maximum and minimum population sizes (or "truncation" points) of process
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

    # Continuously and discretely observed data each have different parameter estimation methods.
    # This control structure diverts to the continuous observation estimation dnm module for
    # continuous data or into another control structure that then diverts between the 4
    # ('abc', 'dnm', 'em' and 'lse) modules for discrete observation parameter estimation.
    if scheme == 'continuous':
        # Extract scheme specific key word arguments the dictionary 'options'
        # method.
        if 'opt_method' in options.keys():
            opt_method = options['opt_method']
        else:
            opt_method = 'L-BFGS-B'

        if framework != 'dnm':
            raise TypeError("Argument `framework` must be 'dnm' when "
                            "argument `scheme` is 'continuous'.")

        # Obtain estimate and associated covariance matrix
        opt, cov = continuous_est_dnm(t_data, p_data, p0, b_rate, d_rate,
                                      p_bounds, con, known_p, idx_known_p,
                                      opt_method, options)

        # Prepare items for output
        p_est = opt.x
        message = opt.message
        success = opt.success
        err = opt.fun
        iterations = 'Not applicable for `mle`. '
        accelerator = 'Not applicable for `mle`. '
        samples = 'Not applicable for `mle`. '
        method = 'mle'

    elif scheme == 'discrete':

        # Sort the data into a more efficient format
        data = ut.data_sort(t_data, p_data)

        # Each of the frameworks for obtaining parameter estimates are contained in their own
        # module interface_* where * depends on the framework. This control structure determines
        # which module to use depending on the value of 'framework'.  Framework specific key word
        # arguments are extracted from the dictionary 'options' and then used in the module
        # associated with the framework.
        if framework == 'abc':
            if 'k' in options.keys():
                k = options['k']
            else:
                k = 100
                options['k'] = k
            if 'eps_abc' in options.keys():
                eps_abc = options['eps_abc']
                if np.isscalar(eps_abc):
                    eps_abc = [eps_abc]
            else:
                eps_abc = 'dynamic'
                options['eps_abc'] = eps_abc
            if 'max_its' in options.keys():
                max_its = options['max_its']
            else:
                if eps_abc == 'dynamic':
                    max_its = 3
                else:
                    max_its = len(eps_abc)
                options['max_its'] = max_its
            if max_its != len(eps_abc) and eps_abc != 'dynamic':
                raise ValueError("Length of argument `abc_eps` "
                                 "does not match argument `its`.")
            if 'max_q' in options.keys():
                max_q = options['max_q']
            else:
                max_q = 0.99
                options['max_q'] = max_q
            if 'method' in options.keys():
                method = options['method']
            else:
                method = 'gwa'
                options['method'] = method
            if 'tau' in options.keys():
                tau = options['tau']
            else:
                tau = np.inf
                for i in data:
                    if i[2] < tau:
                        tau = i[2]
                tau = min(tau/10, 0.1)
                options['tau'] = tau
            if 'distance' in options.keys():
                distance = options['distance']
            else:
                distance = None
                options['distance'] = distance
            if 'stat' in options.keys():
                stat = options['stat']
            else:
                stat = 'mean'
                options['stat'] = stat
            if stat not in ['mean', 'median']:
                raise TypeError("Argument of 'stat' has an unknown value.")
            if 'gam' in options.keys():
                gam = options['gam']
            else:
                gam = 5
                options['gam'] = gam
            if 'eps_change' in options.keys():
                eps_change = options['eps_change']
            else:
                eps_change = 5
                options['eps_change'] = eps_change
            # Obtain estimate, associated covariance matrix, and accepted samples
            p_est, cov, samples = \
                discrete_est_abc(data, eps_abc, distance, stat, k, gam, max_its,
                                 max_q, eps_change, method, b_rate, d_rate,
                                 idx_known_p, known_p, p_bounds, con, tau, rng,
                                 display)

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

            # Obtain estimate and associated covariance matrix
            opt, cov = discrete_est_dnm(data, likelihood, model,
                                        z_trunc, idx_known_p, known_p, p0,
                                        p_bounds, con, opt_method, options)

            # Prepare items for output
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

            # Obtain estimate, associated covariance matrix and per-iteration estimates
            p_est, cov, ll, iterations = discrete_est_em(
                data, p0, technique, accelerator, likelihood, p_bounds, con,
                known_p, idx_known_p, model, b_rate, d_rate, z_trunc, max_it,
                i_tol, j_tol, h_tol, display, opt_method, options)

            # Prepare items for output
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

            # Obtain estimate
            opt = discrete_est_lse(data, squares, model, b_rate, d_rate, z_trunc,
                                   idx_known_p, known_p, p0, p_bounds,
                                   con, opt_method, options)

            # Prepare items for output
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
                             "be one of: 'abc', 'dnm', 'lse', or 'em'.")
    else:
        raise ValueError("Argument `scheme` has an unknown value. Should "
                         "be one of: 'discrete' or 'continuous'.")

    # Prepare for plotting
    if 'xlabel' in options.keys():
        xlabel = options['xlabel']
    else:
        xlabel = "$\\theta_1$"
    if 'ylabel' in options.keys():
        ylabel = options['ylabel']
    else:
        ylabel = "$\\theta_2$"

    # Compute confidence regions and standard errors, and plot confidence regions.
    # This control structure diverts between the 'asymptotic' and 'simulated'
    # approaches to doing these tasks.
    if se_type == 'asymptotic':
        if framework == 'lse':
            se = "Asymptotic confidence intervals are not available for " \
                 "`framework' 'lse'. Set argument `se_type` to 'none'" \
                 "or 'simulated'."
        else:
            # For frameworks 'abc', 'dnm' and 'em', use cov value computed above
            try:
                if cov.shape[0] == 1:
                    se = list(np.sqrt(np.diag(cov))[0])
                else:
                    se = list(np.sqrt(np.diag(cov)))
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

        # Augment estimate of unknown parameters with known parameters
        param = ut.p_bld(p_est, idx_known_p, known_p)

        # This section generates synthetic data samples according to the estimate and then
        # performs the estimation using the synthetic data to obtain a collection of estimates.
        # The control structures are mostly to ensure the correct estimation procedure is used on
        # the synthetic data.
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
            # This control structure ensures standard errors are returned in the correct format
            if cov.shape[0] == 1:
                se = list(np.sqrt(np.diag(cov))[0])
            else:
                se = list(np.sqrt(np.diag(cov)))
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

    # Determine capacity attribute of the output
    if model != 'custom':
        # Build a function that returns fixed points of fluid model
        capacity_finder = ut.higher_zf_bld(model)
        # Use function to find fixed points in terms of estimate of unknown parameters augmented
        # with known parameters
        capacity = np.ceil(capacity_finder(ut.p_bld(np.array(p_est), idx_known_p, known_p)))
    else:
        capacity = "Functionality not available for custom models."

    return EstimationOutput(p=list(p_est), capacity=capacity, val=err, cov=cov,
                            se=se, compute_time=time.time() - tic,
                            framework=framework, message=message,
                            success=success, iterations=iterations,
                            method=method, p0=list(p0),
                            scheme=scheme, samples=samples, model=model)
