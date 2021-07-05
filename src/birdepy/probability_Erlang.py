import numpy as np
import birdepy.utility as ut
from scipy.optimize import root_scalar


def p_mat_bld_Erlang(q_mat, t, cut_meth, eps, _k, num_states):
    if cut_meth == 'Chernoff':
        m = np.max(np.absolute(np.diag(q_mat)))
        _k = root_scalar(lambda k: eps * m + np.exp(k * eps / t) *
                                   (1 - eps / t) ** k + np.exp(-k * eps / t) *
                                   (1 + eps / t) ** k,
                         _k,
                         _k + 10)
    r_matrix = (_k / t) * np.linalg.inv((_k / t) * np.eye(num_states) - q_mat)
    return np.linalg.matrix_power(r_matrix, _k)


def probability_Erlang(z0, zt, t, param, b_rate, d_rate, z_trunc, cut_meth,
                       eps, k):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *Erlangization* method.

    To use this function call ``birdepy.probability`` with `method` set to
    'Erlang'::

        birdepy.probability(z0, zt, t, param, method='Erlang', z_trunc=(), k=150, eps=1e-2,
                            cut_meth=None)

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------
    z_trunc : array_like, optional
        Truncation thresholds, i.e., minimum and maximum states of process
        considered. Array of real elements of size (2,) by default
        ``z_trunc=[z_min, z_max]`` where ``z_min=max(0, min(z0, zt) - 100)``
        and ``z_max=max(z0, zt) + 100``

    k : int, optional
        Number of terms to include in approximation to probability.
        Can be set to be updated dynamically to ensure error is
        bounded by argument `eps` by setting argument 'cut_method' to
        'Chernoff'.

    eps : scalar, optional
        Error bound when argument 'cut_meth' is set to 'Chernoff'.

    cut_meth : string, optional
        If set to 'Chernoff', ensures error of nsures error of
        probability approximation are bounded by argument 'eps' by
        dynamically choosing argument 'k'.

    Examples
    --------
    >>> import birdepy as bd
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 100], model='Verhulst 2 (SIS)', method='Erlang')[0][0]
    0.02773268796308342
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 40], model='Verhulst 2 (SIS)', method='Erlang')[0][0]
    0.0016455223175386804

    Notes
    -----
    Estimation algorithms and models are also described in [1]. If you use this
    function for published work, then please cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    For a method related to this one see [3].

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

    .. [3] Asmussen, S., Avram, F. and Usabel, M. Erlangian approximations for
     finite-horizon ruin probabilities. ASTIN Bulletin: The Journal of the IAA,
     32(2):267-281, 2002.

    """

    z_min, z_max = z_trunc

    num_states = z_max - z_min + 1

    if t.size == 1:
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        p_mat = p_mat_bld_Erlang(q_mat, t, cut_meth, eps, k, num_states)
        output = p_mat[np.ix_(z0 - z_min, zt - z_min)]
    else:
        output = np.zeros((t.size, z0.size, zt.size))
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        for idx in range(t.size):
            p_mat = p_mat_bld_Erlang(q_mat, t[idx], cut_meth, eps,
                                     k, num_states)
            output[idx, :, :] = p_mat[np.ix_(z0 - z_min, zt - z_min)]
    return output
