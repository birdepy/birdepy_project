import numpy as np
import birdepy.utility_probability as ut
import warnings


def probability_gwa(z0, zt, t, param, b_rate, d_rate):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *Galton-Watson approximation* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'gwa'::

        birdepy.probability(z0, zt, t, param, method='gwa')

    This function does not have any arguments which are not already described
    by the documentation of :func:`birdepy.probability`

    Examples
    --------
    >>> import birdepy as bd
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 100], model='Verhulst 2 (SIS)', method='gwa')[0][0]
    0.030788446607032095
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 40], model='Verhulst 2 (SIS)', method='gwa')[0][0]
    0.004363221877757165

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
                pr = ut.p_lin(_z0, _zt, b_rate(_z0, param) / _z0,
                              d_rate(_z0, param) / _z0, t[0])
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
                    pr = ut.p_lin(_z0, _zt, b_rate(_z0, param) / _z0,
                                  d_rate(_z0, param) / _z0, _t)
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
