import numpy as np
import birdepy.utility as ut
from scipy.stats import poisson


def p_mat_bld_uniform(q_mat, t, _k, num_states):
    m = np.amax(np.absolute(np.diag(q_mat)))
    a_mat = np.divide(q_mat, m) + np.eye(num_states)
    w = np.eye(num_states)
    poisson_terms = poisson.pmf(range(_k + 1), m * t)
    p_mat = np.multiply(w, poisson_terms[0])
    for idx in np.arange(1, _k + 1, 1):
        w = np.matmul(w, a_mat)
        p_mat += np.multiply(w, poisson_terms[idx])

    return p_mat


def probability_uniform(z0, zt, t, param, b_rate, d_rate, z_trunc, k):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *uniformization* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'uniform'::

        birdepy.probability(z0, zt, t, param, method='uniform', k=1000, eps=0.01, cut_meth=None,
                            z_trunc=())

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

    k : int, optional
        Number of terms to include in approximation to probability. If `eps` 
        is not None, then this is determined dynamically. 


    Examples
    --------
    >>> import birdepy as bd
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='uniform')[0][0]
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
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    .. [3] Grassman, K. W. Transient solutions in Markovian queueing systems.
     Computers & Operations Research, 4(1):47-53, 1977.

    .. [4] van Dijk N.M., van Brummelen, S.P.J and Boucherie, R.J.
     Uniformization: Basics, extensions and applications. Performance
     Evaluation, 118:8-32, 2018.

    """
    z_min, z_max = z_trunc

    num_states = int(z_max - z_min + 1)

    if t.size == 1:
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        p_mat = p_mat_bld_uniform(q_mat, t, k, num_states)
        output = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                              np.array(zt - z_min, dtype=np.int32))]
    else:
        output = np.zeros((t.size, z0.size, zt.size))
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        for idx in range(t.size):
            p_mat = p_mat_bld_uniform(q_mat, t[idx], k, num_states)
            output[idx, :, :] = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                                             np.array(zt - z_min, dtype=np.int32))]
    return output
