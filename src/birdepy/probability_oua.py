import numpy as np
from scipy.stats import norm


def probability_oua(z0, zt, t, param, b_rate, d_rate, h_fun, zf_bld):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *Ornstein-Uhlenbeck approximation* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'oua'::

        birdepy.probability(z0, zt, t, param, method='oua')

    This function does not have any arguments which are not already described
    by the documentation of :func:`birdepy.probability`

    Examples
    --------
    >>> import birdepy as bd
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='oua')[0][0]
    0.0018882966813798246

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

    .. [3] Ross, J.V., Taimre, T. and Pollett, P.K. On parameter estimation
     in population models. Theoretical Population Biology, 70(4):498-510, 2006.

    """

    zfs = np.array(zf_bld(param))

    zfs = zfs[~np.isnan(zfs)]

    h_vals = []
    for point in zfs:
        h_vals.append(h_fun(point, param))

    # print('h vals', h_vals)

    idx_min_h_val = np.argmin(h_vals)

    zf = zfs[idx_min_h_val]
    h = h_vals[idx_min_h_val]

    if t.size == 1:
        output = np.zeros((z0.size, zt.size))
    else:
        output = np.zeros((t.size, z0.size, zt.size))

    for idx1, _z0 in enumerate(z0):
        m = zf + np.multiply(np.exp(np.multiply(h, t)), (_z0 - zf))

        s2 = np.multiply(
            np.divide(np.add(b_rate(zf, param), d_rate(zf, param)),
                      2 * h),
            np.exp(np.multiply(2 * h, t)) - 1)

        for idx2 in range(zt.size):
            if t.size == 1:
                output[idx1, idx2] = norm.pdf(zt[idx2], loc=m[0],
                                              scale=np.sqrt(max(s2[0],
                                                                1e-30)))
            else:
                for idx3 in range(t.size):
                    output[idx3, idx1, idx2] = \
                        norm.pdf(zt[idx2], loc=m[idx3],
                                 scale=np.sqrt(max(s2[idx3], 1e-30)))

    return output
