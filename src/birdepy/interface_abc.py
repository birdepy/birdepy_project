import numpy as np
import birdepy.utility as ut
import birdepy.simulate as simulate
from scipy import optimize
from sklearn.mixture import GaussianMixture

def default_distance(eps, data_pairs):
    data_pairs = np.array(data_pairs)
    dist = np.sqrt(np.sum(np.square(data_pairs[:, 0] -
                                    data_pairs[:, 1]))
                   ) - eps
    return dist


def determine_prior(p_bounds, rng, con):
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
        # If there is strictly more than one or 0 constraints specified
        else:
            for c in con:
                if c['fun'](param_prop) < 0:
                    success = False
        successes += success
    return 1 / ((successes / 10 ** 3) * np.prod(p_bounds[:, 1] - p_bounds[:, 0]))


def basic_abc(sorted_data, gm, eps, distance, stat, it, k, method, b_rate,
              d_rate, idx_known_p, known_p, p_bounds, con, tau, rng,
              display):
    param_samples = np.empty((k, p_bounds.shape[0]))
    list_of_data_pairs = []
    for idx in range(k):
        while True:
            # Propose parameters satisfying bounds and constraints:
            while True:
                # First obtain proposal satisfying bounds:
                if gm == 0:
                    param_prop = (p_bounds[:, 1] - p_bounds[:, 0]) * \
                                 rng.uniform(size=p_bounds.shape[0]) + \
                                 p_bounds[:, 0]
                else:
                    while True:
                        gm.random_state = rng.integers(2 ** 32 - 2)
                        param_prop = gm.sample(p_bounds.shape[0])[0][0]
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
            param_full = ut.p_bld(param_prop, idx_known_p, known_p)
            data_pairs = []
            for t in sorted_data:
                for zz in sorted_data[t]:
                    z0 = zz[0]
                    zt = zz[1]
                    for _ in range(sorted_data[t][zz]):
                        sample = simulate.discrete(param_full, model='custom',
                                                   b_rate=b_rate, d_rate=d_rate,
                                                   z0=z0, times=[0, t], k=1,
                                                   method=method, tau=tau,
                                                   seed=rng)
                        data_pairs.append([zt, sample[1]])
            if distance(eps, data_pairs) <= 0:
                param_samples[idx, :] = param_prop
                list_of_data_pairs.append(data_pairs)
                break
        if display:
            print(f"Iteration ", it, f" is ", 100 * (idx + 1) / k, f"% complete.")
    if stat == 'mean':
        return np.mean(param_samples, 0), param_samples, list_of_data_pairs
    elif stat == 'median':
        return np.median(param_samples, 0), param_samples, list_of_data_pairs
    else:
        raise TypeError("Argument of 'stat' has an unknown value.")


def discrete_est_abc(sorted_data, eps0, distance, stat, k, its, c, method,
                     b_rate, d_rate, idx_known_p, known_p, p_bounds, con, tau,
                     rng, display):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using approximate Bayesian computation.
    See :ref:`here <Approximate Bayesian Computation>` for more information.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'abc'::

        birdepy.estimate(t_data, p_data, p0, p_bounds, framework='abc', eps0=10, k=100, its=2,
                         method='gwa', tau=None, seed=None, distance=None, stat='median', c=2)

    The parameters associated with this framework (listed below) can be
    accessed using kwargs in :func:`birdepy.estimate()`. See documentation of
    :func:`birdepy.estimate` for the main arguments.

    Parameters
    ----------
    eps0 : scalar, optional
        Iteration 1 maximum distance between sampled data and observed data
        for a parameter sample to be included in estimate.

    k : int, optional
        Number of successful parameter samples used to obtain estimate.

    its : int, optional
        Number of iterations of algorithm.

    method : string, optional
        Simulation algorithm used to generate samples (see
        :ref:`here<Simulation Algorithms>`). Should be one of:

            - 'exact'
            - 'ea'
            - 'ma'
            - 'gwa' (default)

    tau : scalar, optional
        Time between samples for the approximation methods 'ea', 'ma' and 'gwa'.
        Has default value ``min(min(sorted_data.keys()) / 10, 0.1)`` where
        'sorted_data' is the output of utility function :func:`bd.utility.data_sort2`.

    seed : int, Generator, optional
        If *seed* is not specified the random numbers are generated according
        to *np.random.default_rng()*. If *seed* is an *int*, random numbers are
        generated according to *np.random.default_rng(seed)*. If seed is a
        *Generator*, then that object is used. See
        :ref:`here <Reproducibility>` for more information.

    distance : callable, optional
        Computes the distance between simulated data and observed data.

    stat : string, optional
        Determines whcih statistic is used to summarize the posterior distribution.
         Should be one of: 'mean' or 'median'.

    c : int, optional
        Number of mixture components in the mixed multivariate normal which is
        used as a posterior distribution when updating epsilon over iterations.

    Examples
    --------
    Simulate a sample path and estimate the parameters using ABC.

    >>> import birdepy as bd
    >>> t_data = [t for t in range(100)]
    >>> p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
    ...                               survival=True, seed=2021)

    Assume that the death rate and population size are known, then estimate the rate of spread:

    >>> est = bd.estimate(t_data, p_data, [0.5], [[0,1]], framework='abc',
    ...                   model='Ricker', idx_known_p=[1, 2, 3],
    ...                   known_p=[0.25, 0.02, 1], display=True, its=2, seed=2021)
    >>> print('abc estimate is', est.p, ', with standard errors', est.se,
    ...       'computed in ', est.compute_time, 'seconds.')
    abc estimate is [0.7445234348233319] , with standard errors [[0.06708738]] computed in  47.6836621761322 seconds.

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
    if distance is None:
        distance = default_distance

    known_p = np.array(known_p)
    p_bounds = np.array(p_bounds)

    if its == 1:
        est, sample, list_of_data_pairs = \
            basic_abc(sorted_data, 0, eps0, distance, stat, 1, k, method,
                      b_rate, d_rate, idx_known_p, known_p, p_bounds, con,
                      tau, rng, display)
        if p_bounds.shape[0] > 1:
            cov = np.cov(sample, rowvar=False)
        else:
            cov = np.array([[np.var(sample)]])
        return est, cov, sample
    else:
        estimates = []
        list_of_samples = []
        gm = 0
        prior = determine_prior(p_bounds, rng, con)
        current_epsilon = eps0
        for it in range(1, its + 1):
            est, sample, list_of_data_pairs = \
                basic_abc(sorted_data, gm, current_epsilon, distance, stat, it,
                          k, method, b_rate, d_rate, idx_known_p, known_p,
                          p_bounds, con, tau, rng, display)
            estimates.append(est)
            list_of_samples.append(sample)
            gm = GaussianMixture(n_components=c,
                                 random_state=rng.integers(10 ** 6)).fit(sample)
            # Normalise probabilities returned by GaussianMixture to reduce
            # errors:
            gm.weights_ = np.divide(gm.weights_, np.sum(gm.weights_))
            # Update epsilon:
            if it < (its + 1):
                def shifted_eff_sample_size(epsilon):
                    out = 0
                    for idx, data_pairs in enumerate(list_of_data_pairs):
                        d = distance(epsilon, data_pairs)
                        out += (prior * (d <= 0) /
                                gm.score_samples(np.array(
                                   list_of_samples[it-1]
                                   [idx, :]).reshape(1, -1))[0]) ** 2
                    return out * 0.5 * k - 1

                current_epsilon = optimize.root_scalar(shifted_eff_sample_size,
                                                       bracket=[0, current_epsilon],
                                                       method='brentq').root
        if p_bounds.shape[0] > 1:
            cov = np.cov(np.array(sample), rowvar=False)
        else:
            cov = np.array([[np.var(sample)]])
        return estimates, cov, list_of_samples
