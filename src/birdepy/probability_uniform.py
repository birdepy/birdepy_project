import numpy as np
import warnings
import birdepy.utility as ut
from scipy.stats import poisson


def _poisson_cutoff(mean, log_tolerance):
    """Smallest cutoff meeting the tail bound, including tiny tail budgets."""
    lower, upper = -1, max(1, int(np.ceil(mean)))
    while poisson.logsf(upper, mean) > log_tolerance:
        lower, upper = upper, 2 * upper
    while upper - lower > 1:
        midpoint = (lower + upper) // 2
        if poisson.logsf(midpoint, mean) > log_tolerance:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def p_mat_bld_uniform(q_mat, t, _k, num_states, tol=1e-12):
    """
    Builds and returns the discrete time Markov chain utilized by
    the uniformization method.
    """
    m = np.amax(np.absolute(np.diag(q_mat)))
    t = float(np.asarray(t).reshape(-1)[0])
    if not np.isfinite(t) or t < 0:
        raise ValueError("Elapsed time must be finite and nonnegative.")
    if not 0 < tol < 1:
        raise ValueError("The uniformization tolerance must lie between zero and one.")
    if _k is not None and (not isinstance(_k, (int, np.integer)) or _k < 0):
        raise ValueError("Argument `k` must be a nonnegative integer or None.")
    if m == 0 or t == 0:
        return np.eye(num_states)
    mean = m * t
    if not np.isfinite(mean):
        raise ValueError("The uniformization rate times elapsed time must be finite.")
    squarings = 0
    if _k is None:
        # Uniformize over a shorter interval, then use the semigroup property.
        # A per-interval tail of tol / 2**s bounds the total omitted mass by tol.
        if mean > 16:
            squarings = int(np.ceil(np.log2(mean / 16)))
            mean = np.ldexp(mean, -squarings)
        _k = _poisson_cutoff(mean, np.log(tol) - squarings * np.log(2))
    elif poisson.sf(_k, mean) > tol:
        warnings.warn("The uniformization cutoff omits significant probability mass; "
                      "increase k or use k=None for automatic truncation.",
                      RuntimeWarning)
    a_mat = np.divide(q_mat, m) + np.eye(num_states)
    w = np.eye(num_states)
    poisson_terms = poisson.pmf(range(_k + 1), mean)
    p_mat = np.multiply(w, poisson_terms[0])
    for idx in np.arange(1, _k + 1, 1):
        w = np.matmul(w, a_mat)
        p_mat += np.multiply(w, poisson_terms[idx])
    for _ in range(squarings):
        p_mat = np.matmul(p_mat, p_mat)

    return p_mat


def probability_uniform(z0, zt, t, param, b_rate, d_rate, z_trunc, k, tol=1e-12):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *uniformization* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'uniform'::

        birdepy.probability(z0, zt, t, param, method='uniform', k=None, z_trunc=())

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------
    z_trunc : array_like, optional
        Truncation thresholds, i.e., minimum and maximum states of process
        considered. Array of real elements of size (2,) by default
        ``z_trunc=[z_min, z_max]`` where ``z_min=max(0, min(z0, zt) - 100)``
        and ``z_max=max(z0, zt) + 100``.

    k : int or None, optional
        Largest Poisson-series index to include. By default (None), choose
        the cutoff to bound the omitted Poisson tail by `uniform_tol`, using
        shorter intervals and repeated squaring when the rate-time product
        is large. Each interval receives a share of the total tail budget.
        An explicit cutoff is respected, with a warning if it is insufficient.

    uniform_tol : float, optional
        Upper bound on the omitted Poisson tail (default 1e-12). This does
        not control the separate error from state-space truncation.

    Examples
    --------
    Approximate transition probability for a Verhulst model using uniformization: ::

        import birdepy as bd
        bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='uniform')[0][0]

    Outputs: ::

        0.002741422482539626

    Notes
    -----
    Estimation algorithms and models are also described in [1]. If you use this
    function for published work, then please cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    For more information on this method see [3] and [4].

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

    .. [3] Grassman, K. W. Transient solutions in Markovian queueing systems.
     Computers & Operations Research, 4(1):47-53, 1977.

    .. [4] van Dijk N.M., van Brummelen, S.P.J and Boucherie, R.J.
     Uniformization: Basics, extensions and applications. Performance
     Evaluation, 118:8-32, 2018.

    """
    # Extract the min and max population sizes considered
    z_min, z_max = z_trunc

    # Count how many states are considered
    num_states = int(z_max - z_min + 1)

    # If more than one time is requested it is easiest to divert into a different code block
    if t.size == 1:
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        p_mat = p_mat_bld_uniform(q_mat, t, k, num_states, tol)
        output = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                              np.array(zt - z_min, dtype=np.int32))]
    else:
        # Initialize an output array to be filled in as we loop over times
        output = np.zeros((t.size, z0.size, zt.size))
        # Determine the transition rate matrix
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        for idx in range(t.size):
            # The transition probabilities follow from uniformization
            p_mat = p_mat_bld_uniform(q_mat, t[idx], k, num_states, tol)
            # Fill in the output according to requested initial and final states (although many more
            # probabilities are actually computed)
            output[idx, :, :] = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                                             np.array(zt - z_min, dtype=np.int32))]
    return output
