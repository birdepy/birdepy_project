import numpy as np
import birdepy.utility as ut


def discrete(param, model, z0, times, k=1, method='exact', tau=0.1,
             survival=False, seed=None, display=False, **options):
    """Simulation of continuous-time birth-and-death processes at discrete
    observation times.

    Parameters
    ----------
    param : array_like
        The parameters governing the evolution of the birth-and-death
        process to be simulated.
        Array of real elements of size (m,), where ‘m’ is the number of
        parameters.

    model : string
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes>`). Should be one of:

            - 'Verhulst' (default)
            - 'Ricker'
            - 'Hassell'
            - 'MS-S'
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
        The initial population size for each sample path.
        If it is a callable it should be a function that has no arguments and
        returns an int:

         ``z0() -> int``

    times : array_like
        The times at which the simulated birth-and-death is observed.
        Array of real elements of size (n,), where ‘n’ is the number of
        observation times.

    k : int, optional
        The number of sample paths to be simulated.

    method : string, optional
        Simulation algorithm used to generate samples (see
        :ref:`here<Simulation Algorithms>`). Should be one of:

            - 'exact' (default)
            - 'ea'
            - 'ma'
            - 'gwa'

    tau : scalar, optional
        Time between samples for the approximation methods 'ea', 'ma' and 'gwa'.

    survival : bool, optional
        If set to True, then the simulated sample paths are conditioned to
        have a positive population size at the final observation time.  Since This
        can greatly increase computation time.

    seed : int, Generator, optional
        If *seed* is not specified the random numbers are generated according
        to *np.random.default_rng()*. If *seed* is an *int*, random numbers are
        generated according to *np.random.default_rng(seed)*. If seed is a
        *Generator*, then that object is used. See
        :ref:`here <Reproducibility>` for more information.

    display : bool, optional
        If set to True, then a progress indicator is printed as the simulation
        is performed.

    Return
    -------
    out : array_like
        If k=1 a list containing sampled population size observations at
        `times`, generated according to `model`.
        Or if k>1, a numpy.ndarray containing k sample paths, each
        contained in a row of the array.

    Examples
    --------
    Simulating a unit rate Poisson process with observations at times
    [0, 1, 2, 3, 4, 5]:

    >>> import birdepy as bd
    >>> bd.simulate.discrete(1, 'Poisson', 0, times=[0, 1, 3, 4, 5])
    [0, 1, 3, 5, 5]

    Notes
    -----
    If you use this function for published work, then please cite [1].

    Sample paths are generated using a discrete-event simulation algorithm.
    See, for example, Algorithm 5.8 in [2].

    For a text book treatment on the theory of birth-and-death processes
    see [3].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    :func:`birdepy.gpu_functions.probability()`  :func:`birdepy.gpu_functions.discrete()`

    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Kroese, D.P., Taimre, T. and Botev, Z.I. (2013) Handbook of Monte
     Carlo methods. John Wiley & Sons.

    .. [3] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    param = np.array(param)

    if len(param.shape) == 0:
        # In case param is specified as a scalar
        param = np.array([param])
    elif len(param.shape) > 1:
        raise TypeError("Argument `param` has an unsupported shape.")

    times = np.array(times)

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    if method == 'exact':
        return discrete_exact(param, z0, times, k, survival, rng, display, b_rate, d_rate)
    elif method == 'ea':
        return discrete_ea(param, z0, times, k, survival, rng, display, b_rate, d_rate, tau)
    elif method == 'ma':
        return discrete_ma(param, z0, times, k, survival, rng, display, b_rate, d_rate, tau)
    elif method == 'gwa':
        return discrete_gwa(param, z0, times, k, survival, rng, display, b_rate, d_rate, tau)
    else:
        raise Exception(f"Argument `method` has unknown value '{method}'.")


def discrete_exact(param, z0, times, k, survival, rng, display, b_rate, d_rate):
    sample = np.empty((k, times.shape[0]), dtype=np.int64)
    for sample_path in range(k):
        while True:
            if callable(z0):
                pop = z0()
            else:
                pop = z0
            rate = b_rate(pop, param) + d_rate(pop, param)
            if rate == 0:
                next_event_time = np.inf
            else:
                next_event_time = rng.exponential(1 / rate)
            for idx, next_observation_time in enumerate(times):
                while next_event_time <= next_observation_time:
                    if (rng.uniform(0, 1) * rate) <= b_rate(pop, param):
                        pop += 1
                    else:
                        pop -= 1
                    rate = b_rate(pop, param) + d_rate(pop, param)
                    if rate > 0:
                        next_event_time += rng.exponential(1 / rate)
                    else:
                        next_event_time = np.inf
                    # print('event time:', next_event_time)
                sample[sample_path, idx] = pop
                # print('observation time:', next_observation_time)
            if (survival and sample[sample_path, -1] > 0) or not survival:
                # This conditions on survival of the population for the
                # monitoring period
                break
        if display:
            print(100 * (sample_path + 1) / k, '% Complete')
    if k == 1:
        out = sample[0].tolist()
    else:
        out = sample
    return out


def discrete_ea(param, z0, times, k, survival, rng, display, b_rate, d_rate, tau):
    sample = np.empty((k, times.shape[0]), dtype=np.int64)
    for sample_path in range(k):
        while True:
            if callable(z0):
                pop = z0()
            else:
                pop = z0
            # if ((b_rate(pop, param) + d_rate(pop, param))) == 0:
            #     next_event_time = np.inf
            # else:
            #     next_event_time = tau
            next_event_time = tau
            for idx, next_observation_time in enumerate(times):
                while next_event_time <= next_observation_time:
                    pop += rng.poisson(b_rate(pop, param) * tau)
                    pop -= rng.poisson(d_rate(pop, param) * tau)
                    if pop < 0:
                        pop = 0
                    # if ((b_rate(pop, param) + d_rate(pop, param))) > 0:
                    #     next_event_time += tau
                    # else:
                    #     next_event_time = np.inf
                    next_event_time += tau
                sample[sample_path, idx] = pop
                # print('observation time:', next_observation_time)
            if (survival and sample[sample_path, -1] > 0) or not survival:
                # This conditions on survival of the population for the
                # monitoring period
                break
        if display:
            print(100 * (sample_path + 1) / k, '% Complete')
    if k == 1:
        out = sample[0].tolist()
    else:
        out = sample
    return out


def discrete_ma(param, z0, times, k, survival, rng, display, b_rate, d_rate, tau):
    sample = np.empty((k, times.shape[0]), dtype=np.int64)
    for sample_path in range(k):
        while True:
            if callable(z0):
                pop = z0()
            else:
                pop = z0
            # if rate_fun(pop) == 0:
            #     next_event_time = np.inf
            # else:
            #     next_event_time = tau
            next_event_time = tau
            for idx, next_observation_time in enumerate(times):
                while next_event_time <= next_observation_time:
                    pop += rng.poisson(
                        b_rate((pop + 0.5 * tau * (b_rate(pop, param) -
                                                   d_rate(pop, param))), param) * tau)
                    pop -= rng.poisson(
                        d_rate((pop + 0.5 * tau * (b_rate(pop, param) -
                                                   d_rate(pop, param))), param) * tau)
                    if pop < 0:
                        pop = 0
                    # if rate_fun(pop) > 0:
                    #     next_event_time += tau
                    # else:
                    #     next_event_time = np.inf
                    next_event_time += tau
                sample[sample_path, idx] = pop
                # print('observation time:', next_observation_time)
            if (survival and sample[sample_path, -1] > 0) or not survival:
                # This conditions on survival of the population for the
                # monitoring period
                break
        if display:
            print(100 * (sample_path + 1) / k, '% Complete')
    if k == 1:
        out = sample[0].tolist()
    else:
        out = sample
    return out


def discrete_gwa(param, z0, times, k, survival, rng, display, b_rate, d_rate, tau):
    sample = np.empty((k, times.shape[0]), dtype=np.int64)
    for sample_path in range(k):
        while True:
            if callable(z0):
                pop = z0()
            else:
                pop = z0
            next_event_time = tau
            for idx, next_observation_time in enumerate(times):
                while next_event_time <= next_observation_time:
                    lam = b_rate(pop, param) / pop if pop > 0 else 0
                    mu = d_rate(pop, param) / pop if pop > 0 else 0
                    p = 1 - beta1(lam, mu, tau)
                    # if p < 0 or p > 1 or p is np.isnan(p):
                    #     print('p:', p)
                    #     print('lam:', lam)
                    #     print('mu:', mu)
                    #     print('tau:', tau)
                    number_survivors = rng.binomial(pop, p)
                    if number_survivors > 0:
                        pop = number_survivors + \
                              rng.negative_binomial(number_survivors,
                                                    1-beta2(lam, mu, tau))
                    else:
                        pop = 0
                    # pop = 0
                    # for i in range(number_survivors):
                    #     pop += rng.geometric(1-beta2(lam, mu, tau))
                    next_event_time += tau
                sample[sample_path, idx] = pop
                # print('observation time:', next_observation_time)
            if (survival and sample[sample_path, -1] > 0) or not survival:
                # This conditions on survival of the population for the
                # monitoring period
                break
        if display:
            print(100 * (sample_path + 1) / k, '% Complete')
    if k == 1:
        out = sample[0].tolist()
    else:
        out = sample
    return out


def beta1(lam, mu, t):
    if lam == mu:
        return lam * t / (1 + lam * t)
    else:
        return mu * (np.exp((lam - mu) * t) - 1) / (lam * np.exp((lam - mu) * t) - mu)


def beta2(lam, mu, t):
    if lam == mu:
        return lam * t / (1 + lam * t)
    else:
        return lam * beta1(lam, mu, t) / mu


def continuous(param, model, z0, t_max, k=1, survival=False, seed=None,
               **options):
    """Simulation of continuous-time birth-and-death processes at birth and
    death event times.

    Parameters
    ----------
    param : array_like
        The parameters governing the evolution of the birth-and-death
        process to be simulated.
        Array of real elements of size (n,), where ‘n’ is the number of
        param.

    model : string
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes>`). Should be one of:

            - 'Verhulst'
            - 'Ricker'
            - 'Beverton-Holt'
            - 'Hassell'
            - 'MS-S'
            - 'pure-birth'
            - 'pure-death'
            - 'Poisson' (default)
            - 'linear'
            - 'linear-migration'
            - 'M/M/1'
            - 'M/M/inf'
            - 'loss-system'
            - 'custom'
         If set to 'custom', then kwargs `b_rate` and `d_rate` must also be
         specified. See :ref:`here <Custom Models>` for more information.

    z0: int or callable
        The initial population size for each sample path.
        If it is a callable it should be a function that has no arguments and
        returns an int:

         ``z0() -> int``

    t_max : scalar
        The simulation horizon. All events up to and including this time are
        included in the output.

    k : int, optional
        The number of sample paths to be simulated.

    survival : bool, optional
        If set to True, then the simulated sample paths are conditioned to
        have a positive population size at the final observation time. This
        can greatly increase computation time.

    seed : int, numpy.random._generator.Generator, optional
        If *seed* is not specified the random numbers are generated according
        to *np.random.default_rng()*. If *seed* is an *int*, random numbers are
        generated according to *np.random.default_rng(seed)*. If seed is a
        *Generator*, then that object is used. See
        :ref:`here <Reproducibility>` for more information.


    Returns
    -------
    jump_times : list
        If n=1 a list containing jump times, generated according to `model`
        or according to a birth-and-death process evolving according to
        `b_rate` and `d_rate`.
        Or if n>= 1, a list of lists where each list corresponds to the jump
        times from one sample path.

    pop_sizes : list
        If k=1 a list containing population sizes at the corresponding jump
        times, generated according to `model`.
        Or if k>1, a list of lists where each list corresponds to the
        population sizes corresponding to jump times from one sample path.

    Examples
    --------
    Simulating a unit rate Poisson process up to a t_max of 5:

    >>> import birdepy as bd
    >>> jump_times, pop_sizes = bd.simulate.continuous(1,'Poisson', 0, t_max=5)
    >>> print(jump_times)
    >>> print(pop_sizes)
    [0, 0.0664050052043501, 0.48462937097695785, 2.2065719224651157]
    [0, 1, 2, 3]

    Notes
    -----
    If you use this function for published work, then please cite [1].

    Sample paths are generated using a discrete-event simulation algorithm.
    See, for example, Algorithm 5.8 in [2].

    For a text book treatment on the theory of birth-and-death processes
    see [3].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Kroese, D.P., Taimre, T. and Botev, Z.I. (2013) Handbook of Monte
     Carlo methods. John Wiley & Sons.

    .. [3] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """

    param = np.array(param)

    if len(param.shape) == 0:
        # In case param is specified as a scalar
        param = np.array([param])
    elif len(param.shape) > 1:
        raise TypeError("Argument `param` has an unsupported shape.")

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    pop_sizes = []
    jump_times = []
    for sample_path in range(k):
        while True:
            if callable(z0):
                pop = z0()
            else:
                pop = z0
            rate = b_rate(pop, param) + d_rate(pop, param)
            if rate == 0:
                next_event_time = np.inf
            else:
                next_event_time = rng.exponential(1 / rate)

            _pop_sample = [pop]
            _jump_times = [0]
            while next_event_time <= t_max:
                _jump_times.append(next_event_time)
                if (rng.uniform(0, 1) * rate) <= b_rate(pop, param):
                    pop += 1
                else:
                    pop -= 1
                rate = b_rate(pop, param) + d_rate(pop, param)
                if rate > 0:
                    next_event_time += rng.exponential(1 / rate)
                else:
                    next_event_time = np.inf
                _pop_sample.append(pop)
            if (survival and pop > 0) or not survival:
                # This conditions on survival of the population for the
                # monitoring period
                break
        pop_sizes.append(_pop_sample)
        jump_times.append(_jump_times)
    if k == 1:
        jump_times = jump_times[0]
        pop_sizes = pop_sizes[0]
    return jump_times, pop_sizes
