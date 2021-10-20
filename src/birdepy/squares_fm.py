import numpy as np
from scipy.integrate import solve_ivp
import warnings


def sq_bld(data, b_rate, d_rate):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using the *fluid model least lse estimation*
    method.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'lse' and `squares` set to 'fm': ::

        bd.estimate(t_data, p_data, p0, p_bounds, framework='lse', squares='fm')

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
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    # These are methods used to solve differential equations as used by scipy.integrate.solve_ivp()
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
