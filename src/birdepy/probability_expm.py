import numpy as np
from scipy.linalg import expm
import birdepy.utility as ut


def probability_expm(z0, zt, t, param, b_rate, d_rate, z_trunc):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *matrix exponential* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'expm': ::

        birdepy.probability(z0, zt, t, param, method='expm', z_trunc=())

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

    Examples
    --------
    Approximate transition probability for a Verhulst model using a matrix exponential: ::

        import birdepy as bd
        bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='expm')[0][0]

    Outputs: ::

        0.0027414224836612463

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

    # If more than one time is requested it is easiest to divert into a different code block
    if t.size == 1:
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        p_mat = expm(np.multiply(q_mat, t))
        output = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                              np.array(zt - z_min, dtype=np.int32))]
    else:
        # Initialize an output array to be filled in as we loop over times
        output = np.zeros((t.size, z0.size, zt.size))
        # Determine the transition rate matrix
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        for idx in range(t.size):
            # The transition probabilities follow from taking a matrix exponential
            p_mat = expm(np.multiply(q_mat, t[idx]))
            # Fill in the output according to requested initial and final states (although many more
            # probabilities are actually computed)
            output[idx, :, :] = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                                             np.array(zt - z_min, dtype=np.int32))]
    return output
