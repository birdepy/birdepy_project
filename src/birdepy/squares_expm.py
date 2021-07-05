import numpy as np
from scipy.linalg import expm
import birdepy.utility as ut


def sq_bld(data, b_rate, d_rate, z_trunc):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using the *matrix exponential least lse
    estimation* method.

    To use this function call ``birdepy.estimate`` with `method` set to
    `mexplse`::

        birdepy.estimate(t_data, p_data, method='mexplse', model='Verhulst 1',
                         b_rate=None, d_rate=None, p_size=None, known_p=[],
                         idx_known_p=[], p0=None, opt_method='L-BFGS-B',
                         p_bounds=None, con=None, seed=None, z_trunc=None)

    See documentation of ``birdepy.estimate`` (see :ref:`here <Estimation>`)
    or use ``help(birdepy.estimate)`` for the rest of the arguments.

    Parameters
    ----------
    z_trunc : array_like, optional
        Truncation thresholds, i.e., minimum and maximum states of process
        considered. Array of real elements of size (2,) by default
        ``z_trunc=[z_min, z_max]`` where
        ``z_min=max(0, z_obs_min - obs_range)`` and
        ``z_max=z_obs_max + obs_range`` with ``obs_range`` equal to the
        absolute difference between the highest and lowest observed
        populations.

    Examples
    --------
    >>> import birdepy as bd
    >>> z0 = 19,
    zt = 27
    t = 1.0
    N = 100
    gamma = 0.5
    nu = 0.3
    p = [gamma, nu, N]
    print(bd.probability(z0, zt, t, p, model='Verhulst 2 (SIS)', method='da', k=2)[0][0])
    0.02937874214086395

    See also
    --------
    birdepy.estimate
    birdepy.forecast

    References
    ----------
    .. [1]

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    z_min, z_max = z_trunc

    sorted_data = ut.data_sort_2(data)

    def error_fun(param):
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        err = 0
        for t in sorted_data:
            p_mat = expm(np.multiply(q_mat, t))
            for i in sorted_data[t]:
                expected_pop = np.dot(np.arange(z_min, z_max + 1, 1),
                                      p_mat[i[0] - z_min, :])
                err += sorted_data[t][i] * np.square(expected_pop - i[1])
        return err
    return lambda p_prop: error_fun(p_prop)

