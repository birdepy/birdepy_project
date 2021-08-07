import numpy as np
import birdepy.utility_probability as ut
import warnings


def probability_gwa(z0, zt, t, param, b_rate, d_rate, anchor):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *Galton-Watson approximation* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'gwa'::

        birdepy.probability(z0, zt, t, param, method='gwa', anchor='midpoint')

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------
    anchor : string, optional
        Determines which state z is used to determine the linear approximation.
        Should be one of: 'initial' (z0 is used), 'midpoint' (default, 0.5*(z0+zt) is used)
        or 'terminal' (zt is used).

    Examples
    --------
    >>> import birdepy as bd
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='gwa')[0][0]
    0.00227651766770292

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

    if t.size == 1:
        output = np.zeros((z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            for idx2, _zt in enumerate(zt):
                if anchor == 'midpoint':
                    midpoint = 0.5 * (_z0 + _zt)
                    lam = b_rate(midpoint, param) / midpoint
                    mu = d_rate(midpoint, param) / midpoint
                elif anchor == 'initial':
                    lam = b_rate(_z0, param) / _z0
                    mu = d_rate(_z0, param) / _z0
                elif anchor == 'terminal':
                    lam = b_rate(_zt, param) / _zt
                    mu = d_rate(_zt, param) / _zt
                pr = ut.p_lin(_z0, _zt, lam, mu, t[0])
                if not 0 <= pr <= 1.0:
                    warnings.warn("Probability not in [0, 1] computed, "
                                  "some output has been replaced by a "
                                  "default value. "
                                  " Results may be unreliable.",
                                  category=RuntimeWarning)
                    if pr < 0:
                        pr = 0.0
                    else:
                        pr = 1.0
                output[idx1, idx2] = pr
    else:
        output = np.zeros((t.size, z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            for idx2, _zt in enumerate(zt):
                for idx3, _t in enumerate(t):
                    if anchor == 'midpoint':
                        midpoint = 0.5 * (_z0 + _zt)
                        lam = b_rate(midpoint, param) / midpoint
                        mu = d_rate(midpoint, param) / midpoint
                    elif anchor == 'initial':
                        lam = b_rate(_z0, param) / _z0
                        mu = d_rate(_z0, param) / _z0
                    elif anchor == 'terminal':
                        lam = b_rate(_zt, param) / _zt
                        mu = d_rate(_zt, param) / _zt
                    pr = ut.p_lin(_z0, _zt, lam, mu, t[0])
                    if not 0 <= pr <= 1.0:
                        warnings.warn("Probability not in [0, 1] computed, "
                                      "some output has been replaced by a "
                                      "default value. "
                                      " Results may be unreliable.",
                                      category=RuntimeWarning)
                        if pr < 0:
                            pr = 0.0
                        else:
                            pr = 1.0
                    output[idx3, idx1, idx2] = pr

    return output
