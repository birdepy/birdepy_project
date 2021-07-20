import birdepy.squares_expm as _squares_expm
import birdepy.squares_fm as _squares_fm
import birdepy.squares_gwa as _squares_gwa
import birdepy.utility as ut


def discrete_est_lse(data, squares, model, b_rate, d_rate, z_trunc,
                     idx_known_p, known_p, p0, p_bounds, con, opt_method,
                     options):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using least squares estimation.
    See :ref:`here <Least Squares Estimation>` for more information.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'lse'::

        birdepy.estimate(t_data, p_data, p0, p_bounds, framework='lse', squares='fm', z_trunc=())

    The parameters associated with this framework (listed below) can be
    accessed using kwargs in :func:`birdepy.estimate()`. See documentation of
    :func:`birdepy.estimate` for the main arguments.

    Parameters
    ----------
    squares : string, optional
        Squared error from the expected value approximation method. Should be
        one of (alphabetical ordering):

            - 'expm'
            - 'fm' (default)
            - 'gwa'

    z_trunc : array_like, optional
        Truncation thresholds, i.e., minimum and maximum states of process
        considered. Array of real elements of size (2,) by default
        ``z_trunc=[z_min, z_max]`` where
        ``z_min=max(0, min(p_data) - 2*z_range)``
        and ``z_max=max(p_data) + 2*z_range`` where
        ``z_range=max(p_data)-min(p_data)``. Only applicable to `squares`
        method 'expm'.

    Examples
    --------
    Simulate a sample path and estimate the parameters using the various
    likelihood approximation methods.

    The constraint ``con={'type': 'ineq', 'fun': lambda p: p[0]-p[1]}`` ensures
    that p[0] > p[1] (i.e., rate of spread greater than recovery rate).

    >>> import birdepy as bd
    >>> t_data = [t for t in range(100)]
    >>> p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
    ...                               survival=True, seed=2021)
    >>> for squares in ['expm', 'fm', 'gwa']:
    >>>     est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[1e-6,1], [1e-6,1], [1e-6, 0.1]],
    ...                       framework='lse', model='Ricker', idx_known_p=[3], known_p=[1],
    ...                       squares=squares)
    >>>     print('lse estimate using', squares, 'is', est.p, 'computed in ', est.compute_time, 'seconds.')
    lse estimate using expm is [0.7879591925854611, 0.26289368236374644, 0.02000871498805996] computed in  1.2579967975616455 seconds.
    lse estimate using fm is [0.7941603523732229, 0.2766621569715867, 0.019363240909074483] computed in  5.483000755310059 seconds.
    lse estimate using gwa is [0.7024952317382023, 0.20563650045779058, 0.022598851311981704] computed in  0.09800028800964355 seconds.

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

    if model != 'custom':
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    if squares == 'fm':
        sq = _squares_fm.sq_bld(data, b_rate, d_rate)

    elif squares == 'expm':
        sq = _squares_expm.sq_bld(data, b_rate, d_rate, z_trunc)

    elif squares == 'gwa':
        sq = _squares_gwa.sq_bld(data, b_rate, d_rate)

    else:
        raise TypeError("Argument squares has an unknown value.")

    def error_fun(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return sq(param)

    return ut.minimize_(error_fun, p0, p_bounds, con, opt_method, options)
