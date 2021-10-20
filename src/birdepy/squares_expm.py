import numpy as np
from scipy.linalg import expm
import birdepy.utility as ut


def sq_bld(data, b_rate, d_rate, z_trunc):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using the *matrix exponential least lse
    estimation* method.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'lse' and `squares` set to 'expm': ::

        bd.estimate(t_data, p_data, p0, p_bounds, framework='lse', squares='expm', z_trunc=())

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

    """
    # Extract the min and max population sizes considered
    z_min, z_max = z_trunc

    # Sort the data into a more efficient format
    sorted_data = ut.data_sort_2(data)

    def error_fun(param):
        # Determine the transition rate matrix
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        err = 0
        for t in sorted_data:
            # The transition probabilities follow from taking a matrix exponential
            p_mat = expm(np.multiply(q_mat, t))
            for i in sorted_data[t]:
                expected_pop = np.dot(np.arange(z_min, z_max + 1, 1),
                                      p_mat[i[0] - z_min, :])
                err += sorted_data[t][i] * np.square(expected_pop - i[1])
        return err
    return lambda p_prop: error_fun(p_prop)

