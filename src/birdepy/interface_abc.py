import numpy as np
import birdepy.utility as ut
import birdepy.simulate as simulate
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt


def estimate_prior_pdf(p_bounds, con, rng):
    """
    Uses Monte Carlo samples to provide an estimate of the density function
    for a uniform distribution with support defined by 'p_bounds' and 'con'.

    This function is used by :func:`discrete_est_abc()`.
    """
    p_bounds = np.array(p_bounds)
    successes = 0
    for idx in range(10 ** 3):
        param_prop = (p_bounds[:, 1] - p_bounds[:, 0]) * \
                     rng.uniform(size=p_bounds.shape[0]) + \
                     p_bounds[:, 0]
        success = True
        if type(con) == dict:
            # If there is one constraint specified
            if con['fun'](param_prop) < 0:
                success = False
        # If there is strictly more than one or exactly 0 constraints specified
        else:
            for c in con:
                if c['fun'](param_prop) < 0:
                    success = False
                    break
        successes += success
    return 1 / ((successes / 10 ** 3) * np.prod(p_bounds[:, 1] - p_bounds[:, 0]))


def param_sampler(previous_sample, k, weights, cov, p_bounds, con, rng):
    cond2 = True
    while cond2:
        cond2 = False
        # If there are no previous samples, sample a proposal satisfying the bounds
        # using a uniform distribution. Otherwise sample a proposal according to the
        # mechanism described in [3].
        if len(previous_sample) == 0:
            param_prop = (p_bounds[:, 1] - p_bounds[:, 0]) * \
                         rng.uniform(size=p_bounds.shape[0]) + p_bounds[:, 0]
        else:
            cond1 = True
            while cond1:
                cond1 = False
                index_mean = rng.choice(list(range(k)), p=np.divide(weights, np.sum(weights)))
                mean = previous_sample[index_mean, :]
                if p_bounds.shape[0] > 1:
                    param_prop = rng.multivariate_normal(mean, 2 * cov)
                else:
                    param_prop = rng.normal(mean, 2 * cov)
                for bnd_idx, bnd in enumerate(p_bounds):
                    if param_prop[bnd_idx] > bnd[1] or param_prop[bnd_idx] < bnd[0]:
                        cond1 = True
        # Then check proposal also satisfies constraints:
        if type(con) == dict:
            # If there is one constraint specified
            if con['fun'](param_prop) < 0:
                cond2 = True
        else:
            # If there is strictly more than one or 0 constraints specified
            for c in con:
                if c['fun'](param_prop) < 0:
                    cond2 = True
                    break  # Leave for loop on first failure
    return param_prop


def simulated_data_maker(data, param_prop, idx_known_p, known_p, b_rate,
                         d_rate, method, tau, rng):
    """
    Generates a simulated version of 'data' according to 'param_prop'.
    """
    param_full = ut.p_bld(param_prop, idx_known_p, known_p)
    data_pairs = np.empty((sum(data.values()), 2))
    lower_index = 0
    for i in data:
        # Here a transition is used where z0 = i[0], zt = i[1], t = i[2] and
        # the number of times the transition is observed is data[i]
        sample = simulate.discrete(param_full, model='custom',
                                   b_rate=b_rate, d_rate=d_rate, z0=i[0],
                                   times=[0, i[2]], k=data[i], method=method,
                                   tau=tau, seed=rng)
        if data[i] == 1:
            data_pairs[lower_index, :] = [i[1], sample[1]]
        else:
            data_pairs[lower_index:(lower_index + data[i]), 0] = i[1]
            data_pairs[lower_index:(lower_index + data[i]), 1] = sample[:, 1]
        lower_index += data[i]
    return data_pairs


def basic_abc(data, previous_sample, weights, cov, eps_abc, distance, k,
              method, b_rate, d_rate, idx_known_p, known_p, p_bounds, con, tau,
              rng, iteration, display):
    """
    Performs a single iteration of the basic ABC algorithm.
    """
    param_samples = np.empty((k, p_bounds.shape[0]))
    dist_samples = np.empty(k)
    for idx in range(k):
        while True:
            param_prop = param_sampler(previous_sample, k, weights, cov, p_bounds,
                                       con, rng)
            data_pairs = simulated_data_maker(data, param_prop, idx_known_p,
                                              known_p, b_rate, d_rate, method,
                                              tau, rng)
            dist = distance(data_pairs[:, 0], data_pairs[:, 1])
            if dist <= eps_abc:
                param_samples[idx, :] = param_prop
                dist_samples[idx] = dist
                break
        if display:
            print(f"Iteration {iteration} is {100 * (idx + 1) / k}% complete.")
    return param_samples, dist_samples


def discrete_est_abc(data, eps_abc, distance, stat, k, gam, max_its, max_q,
                     eps_change, method, b_rate, d_rate, idx_known_p, known_p,
                     p_bounds, con, tau, seed, display):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using approximate Bayesian computation.
    See :ref:`here <Approximate Bayesian Computation>` for more information.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'abc'::

        bd.estimate(t_data, p_data, p0, p_bounds, framework='abc', eps_abc='dynamic', k=100,
                    max_its=3, max_q=0.99, eps_change=5, gam=5, method='gwa', tau=None, seed=None,
                    distance=None, stat='mean', display=False)

    The parameters associated with this framework (listed below) can be
    accessed using kwargs in :func:`birdepy.estimate()`. See documentation of
    :func:`birdepy.estimate` for the main arguments.

    Parameters
    ----------
    eps_abc : list, str, optional
        Threshold distance betweeen simulated data and observed data for accepting
        parameter proposals. If set to 'dynamic' (default), then the procedure
        described in [3] is used. Otherwise `eps_abc` must be a list which
        specifies epsilon for each iteration.

    k : int, optional
        Number of successful parameter samples used to obtain estimate.

    max_its : int, optional
        Maximum number of iterations of algorithm.

    max_q : scalar, optional
        Tolerance threshold for stopping algorithm (see Equation 2.5 in [3]).
        Is only checked after at least two iterations have occurred.

    eps_change : scalar, optional
        An iteration is only performed if the percentage decrease in 'eps_abc'
        compared to the previous iteration is greater than this value.

    gam : int, optional
        If `eps_abc` is set to 'dynamic', then k*gam samples are initially
        sampled and the distance between the data and the k-th largest of these
        samples is used as the first value of epsilon

    method : string, optional
        Simulation algorithm used to generate samples (see
        :ref:`here<Simulation Algorithms>`). Should be one of:

            - 'exact'
            - 'ea'
            - 'ma'
            - 'gwa' (default)

    tau : scalar, optional
        Time between samples for the approximation methods 'ea', 'ma' and 'gwa'.
        Has default value ``min(x/10, 0.1)`` where 'x' is the smallest
        inter-observation time.

    seed : int, Generator, optional
        If `seed` is not specified the random numbers are generated according
        to :func:`np.random.default_rng()`. If `seed` is an 'int;, random numbers are
        generated according to :func:`np.random.default_rng(seed)`. If `seed` is a
        'Generator', then that object is used. See
        :ref:`here <Reproducibility>` for more information.

    distance : callable, optional
        Computes the distance between simulated data and observed data. Default
        value is :func:`scipy.spatial.distance.euclidean`.

    stat : string, optional
        Determines which statistic is used to summarize the posterior distribution.
        Should be one of: 'mean' or 'median'. 

    Examples
    --------
    Simulate a sample path and estimate the parameters using ABC: ::

        import birdepy as bd
        t_data = list(range(100))
        p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
                                      survival=True, seed=2021)

    Assume that the death rate and population size are known, then estimate the rate of spread: ::

        est = bd.estimate(t_data, p_data, [0.5], [[0,1]], framework='abc',
                          model='Ricker', idx_known_p=[1, 2, 3],
                          known_p=[0.25, 0.02, 1], display=True, seed=2021)
        print(f"abc estimate is {est.p}, with standard errors {est.se},
              computed in {est.compute_time} seconds.")

    Outputs: ::

        abc estimate is [0.72434934654868] , with standard errors [0.0520378066100896] computed in  122.48563146591187 seconds.

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
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    .. [3] Simola, U., Cisewski-Kehe, J., Gutmann, M. U., & Corander, J. (2021).
     Adaptive approximate Bayesian computation tolerance selection. Bayesian analysis,
     16(2), 397-423.
    """
    if distance is None:
        distance = euclidean

    prior_pdf = estimate_prior_pdf(p_bounds, con, seed)

    known_p = np.array(known_p)
    p_bounds = np.array(p_bounds)

    list_of_samples = []
    param_sample = []
    weights = [1 / k] * k
    cov = []
    mean = []
    previous_mean = []
    previous_cov = []
    iteration = 0
    q = 0

    current_eps = np.inf
    while True:
        iteration += 1
        if iteration > max_its:
            break
        previous_eps = current_eps
        # First set the value of epsilon (the cut off for acceptance of parameter proposals)
        if eps_abc == 'dynamic':
            if iteration == 1:
                # param_sample = np.empty((gam * k, p_bounds.shape[0]))
                dist_samples = np.empty(gam * k)
                for idx in range(gam * k):
                    # The first argument of param_sample is [] to ensure a uniform dist is used
                    param_prop = param_sampler([], k, weights, cov, p_bounds,
                                               con, seed)
                    data_pairs = simulated_data_maker(data, param_prop, idx_known_p,
                                                      known_p, b_rate, d_rate, method,
                                                      tau, seed)
                    dist = distance(data_pairs[:, 0], data_pairs[:, 1])
                    # param_sample[idx, :] = param_prop
                    dist_samples[idx] = dist
                indices_of_smallest_distances = np.argpartition(dist_samples, k)
                dist_samples = dist_samples[indices_of_smallest_distances[:k]]
                # previous_sample = param_sample[indices_of_smallest_distances[:k], :]
                current_eps = np.amax(dist_samples)
            elif iteration == 2:
                if p_bounds.shape[0] > 1:
                    opt = minimize(lambda p: -multivariate_normal.pdf(p, mean, cov) / prior_pdf,
                                   x0=mean, method='SLSQP', bounds=p_bounds,
                                   constraints=con)
                else:
                    opt = minimize(
                        lambda p: -norm.pdf(p, loc=mean[0], scale=np.sqrt(cov)) / prior_pdf,
                        mean[0], method='SLSQP', bounds=p_bounds)
                q = -1 / opt.fun
                current_eps = np.quantile(dist_samples, q)
            else:
                if p_bounds.shape[0] > 1:
                    opt = minimize(
                        lambda p: -multivariate_normal.pdf(p, mean, cov) /
                                  max(1e-30,
                                      multivariate_normal.pdf(p, previous_mean,
                                                              previous_cov)),
                        x0=mean, method='SLSQP', bounds=p_bounds,
                        constraints=con)
                else:
                    opt = minimize(lambda p: -norm.pdf(p, loc=mean[0], scale=np.sqrt(cov)) /
                                             max(1e-30, norm.pdf(p, loc=previous_mean[0],
                                                                 scale=np.sqrt(
                                                                     previous_cov))),
                                   mean[0], method='SLSQP', bounds=p_bounds)
                q = -1 / opt.fun
                current_eps = np.quantile(dist_samples, q)
        else:
            current_eps = eps_abc[iteration - 1]
        if ((q > max_q and iteration >= 3) or
                (100 * (previous_eps / current_eps - 1) < eps_change)):
            break
        # Now run the basic ABC algorithm for this iteration
        previous_sample = param_sample
        param_sample, dist_samples = \
            basic_abc(data, previous_sample, weights, cov, current_eps, distance,
                      k, method, b_rate, d_rate, idx_known_p, known_p, p_bounds,
                      con, tau, seed, iteration, display)
        list_of_samples.append(param_sample)
        # create a temporary array (since the new values depend on the old ones)
        previous_weights = weights
        previous_cov = cov
        previous_mean = mean
        # Determine weights for next iteration
        if iteration == 1:
            weights = [1 / k] * k
            if p_bounds.shape[0] > 1:
                cov = np.cov(param_sample, rowvar=False, aweights=previous_weights)
            else:
                cov = np.var(np.multiply(param_sample.T, previous_weights))
        else:
            if p_bounds.shape[0] > 1:
                cov = np.cov(param_sample, rowvar=False, aweights=previous_weights)
                sd = np.sqrt(np.diag(cov))
                weights = np.empty(k)
                for i in range(k):
                    temp = 0
                    for j in range(k):
                        temp += previous_weights[j] * \
                                multivariate_normal.pdf(
                                    np.divide(param_sample[i, :] - previous_sample[j, :], sd),
                                    cov=np.eye(p_bounds.shape[0]))
                    weights[i] = prior_pdf / temp
            else:
                cov = np.var(np.multiply(param_sample.T, previous_weights))
                sd = np.sqrt(cov)
                weights = np.empty(k)
                for i in range(k):
                    temp = 0
                    for j in range(k):
                        temp += previous_weights[j] * \
                                norm.pdf((param_sample[i] -
                                          previous_sample[j]) / sd)
                    weights[i] = prior_pdf / temp
        mean = np.mean(param_sample, axis=0)
    if stat == 'mean':
        est = np.mean(param_sample, axis=0)
    else:
        est = np.median(param_sample, axis=0)
    if p_bounds.shape[0] > 1:
        cov = np.cov(param_sample, rowvar=False)
    else:
        cov = np.array([np.var(param_sample)])
    return est, cov, list_of_samples
