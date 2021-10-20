import numpy as np
import birdepy.utility as ut


def probability_Erlang(z0, zt, t, param, b_rate, d_rate, z_trunc, k):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *Erlangization* method.

    To use this function call ``birdepy.probability`` with `method` set to
    'Erlang'::

        bd.probability(z0, zt, t, param, method='Erlang', z_trunc=(), k=1502)

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

    Examples
    --------
    Approximate transition probability for a Verhulst model using Erlangization: ::

        import birdepy as bd
        bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='Erlang')[0][0]

    Outputs: ::

        0.002731464736623327

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
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    .. [3] Asmussen, S., Avram, F. and Usabel, M. Erlangian approximations for
     finite-horizon ruin probabilities. ASTIN Bulletin: The Journal of the IAA,
     32(2):267-281, 2002.

    """

    # Extract the min and max population sizes considered
    z_min, z_max = z_trunc

    # Count how many states are considered
    num_states = int(z_max - z_min + 1)

    # If more than one time is requested it is easiest to divert into a different code block
    if t.size == 1:
        # Determine the transition rate matrix
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        # Determine the associated 'R' matrix of discrete time transitions at Erlang distributed
        # epochs
        r_mat = (k / t) * np.linalg.inv((k / t) * np.eye(num_states) - q_mat)
        # Take k Erlang distributed epochs so the considered time converges to the requested t
        # with low variance
        p_mat = np.linalg.matrix_power(r_mat, k)
        # Fill in the output according to requested initial and final states (although many more
        # probabilities are actually computed)
        output = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                              np.array(zt - z_min, dtype=np.int32))]
    else:
        # Initialize an output array to be filled in as we loop over times
        output = np.zeros((t.size, z0.size, zt.size))
        # Determine the transition rate matrix
        q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)
        for idx in range(t.size):
            # Determine the associated 'R' matrix of discrete time transitions at Erlang distributed
            # epochs
            r_mat = (k / t[idx]) * np.linalg.inv((k / t[idx]) * np.eye(num_states) - q_mat)
            # Take k Erlang distributed epochs so the considered time converges to the requested t
            # with low variance
            p_mat = np.linalg.matrix_power(r_mat, k)
            # Fill in the output according to requested initial and final states (although many more
            # probabilities are actually computed)
            output[idx, :, :] = p_mat[np.ix_(np.array(z0 - z_min, dtype=np.int32),
                                             np.array(zt - z_min, dtype=np.int32))]
    return output
