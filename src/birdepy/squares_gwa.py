import numpy as np


def sq_bld(data, b_rate, d_rate):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using the *Galton-Watson approximation least
    lse estimation* method.

    To use this function call ``birdepy.estimate`` with `method` set to
    `gwalse`::

        birdepy.estimate(t_data, p_data, method='gwalse', model='Verhulst 1',
                         b_rate=None, d_rate=None, p_size=None, known_p=[],
                         idx_known_p=[], p0=None, opt_method='L-BFGS-B',
                         p_bounds=None, con=None, seed=None)

    This function does not have any arguments which are not already described
    by the documentation of ``birdepy.estimate`` (see :ref:`here <Estimation>`)
    or use ``help(birdepy.estimate)``.

    Examples
    --------


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
    def error_fun(param):
        err = 0
        for i in data:
            z0 = i[0]
            zt = i[1]
            expected_pop = z0 * np.exp(
                (b_rate(z0, param) / z0 - d_rate(z0, param) / z0) * i[2])
            err += np.square(expected_pop - zt)
        return err

    return lambda p_prop: error_fun(p_prop)
