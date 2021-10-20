import numpy as np
from scipy import integrate as int
from scipy.stats import norm
import warnings
import birdepy.utility as ut


def probability_da(z0, zt, t, param, b_rate, d_rate, h_fun, k):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *diffusion approximation* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'da'::

        bd.probability(z0, zt, t, param, method='da', k=10)

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------
    k : int, optional
        ``2**k + 1`` equally spaced samples between observation times are used
        to compute integrals.

    Examples
    --------
    Approximate transition probability for a Verhulst model using diffusion approximation: ::

        import birdepy as bd
        bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='da')[0][0])

    Outputs: ::

        0.002266101391343583

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

    .. [3] Ross, J.V., Pagendam, D.E. and Pollett, P.K. On parameter estimation
     in population models II: Multi-dimensional processes and transient
     dynamics. Theoretical Population Biology, 75(2-3):123-132, 2009.

    """
    # Determine a partition for numerical integration
    t_linspace = np.union1d(np.linspace(0, t[-1], t.size * (2 ** k + 1)), t)

    # These are methods used to solve differential equations as used by scipy.integrate.solve_ivp()
    solver_methods = ['RK45', 'Radau', 'RK23', 'BDF', 'DOP853']

    # Output format depends on whether multiple times are requested
    if t.size == 1:
        output = np.zeros((z0.size, zt.size))
    else:
        output = np.zeros((t.size, z0.size, zt.size))

    # A distribution over all possible final states is determined for each requested initial state
    for idx1, _z0 in enumerate(z0):
        # Determine the mean of the distribution
        for meth in solver_methods:
            fluid_path = int.solve_ivp(lambda _, z:
                                       b_rate(z, param) - d_rate(z, param),
                                       [t_linspace[0],
                                        t_linspace[-1] + 1e-100],
                                       [_z0],
                                       t_eval=t_linspace,
                                       method=meth)
            if fluid_path.success:
                break

        fluid_path = np.array(fluid_path.y[0])

        # The next few steps determine the covariance. See Section 2.3 of reference [1] for info
        m_function = np.exp(ut.trap_int(h_fun(fluid_path, param),
                                        t_linspace))

        integrand = np.divide(np.add(b_rate(fluid_path, param),
                                     d_rate(fluid_path, param)),
                              np.square(m_function))

        integral = ut.trap_int(integrand, t_linspace)

        indices = np.where(np.in1d(t_linspace, t))[0]

        m = fluid_path[indices]

        s2 = np.multiply(np.square(m_function[indices]),
                         integral[indices])

        # Given the mean and covariance, use a normal distribution pdf to determine probabilities
        # for requested terminal states
        for idx2 in range(zt.size):
            if t.size == 1:
                output[idx1, idx2] = norm.pdf(zt[idx2], loc=m[0],
                                              scale=np.sqrt(s2[0]))
            else:
                for idx3 in range(t.size):
                    output[idx3, idx1, idx2] = \
                        norm.pdf(zt[idx2], loc=m[idx3], scale=np.sqrt(s2[idx3]))
    # Numerical overflow is possible (particularly for stiff odes when determining 'fluid_path'),
    # which result in nan values. Provide a warning about this.
    if np.isnan(output).any():
        warnings.warn("Failed to compute some probabilities, some values "
                      "have been returned as nan. ",
                      category=RuntimeWarning)
    return output
