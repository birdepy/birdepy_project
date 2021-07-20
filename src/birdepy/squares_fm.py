import numpy as np
from scipy.integrate import solve_ivp
import warnings


def sq_bld(data, b_rate, d_rate):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using the *fluid model least lse estimation*
    method.

    To use this function call ``birdepy.estimate`` with `method` set to
    `fmlse`::

        birdepy.estimate(t_data, p_data, method='fmlse', model='Verhulst 1',
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
    solver_methods = ['RK45', 'Radau', 'RK23', 'BDF', 'DOP853']

    def error_fun(param):
        err = 0
        for i in data:
            for meth in solver_methods:
                fluid_path = solve_ivp(
                    lambda t, z: b_rate(z, param) - d_rate(z, param),
                    [0, i[2]+1e-100],
                    [i[0]],
                    t_eval=[i[2]],
                    method=meth)
                if fluid_path.success:
                    err += np.square(fluid_path.y[0][0] - i[1])
                    break
            if not fluid_path.success:
                err += 1
                warnings.warn(
                    "Failed to find a solution to an ordinary "
                    "differential equation, some output has been replaced"
                    " by a default value. ",
                    category=RuntimeWarning)
        return err

    return lambda param_prop: error_fun(param_prop)
