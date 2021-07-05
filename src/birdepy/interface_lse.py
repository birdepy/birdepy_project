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
    ... t_data = [t for t in range(100)]
    ... p_data = bd.simulate.discrete([0.75, 0.25, 50], model='Verhulst 2 (SIS)', z0=10,
    ...                               times=t_data, k=1, survival=True,
    ...                               seed=2021)
    ... for squares in ['expm', 'fm', 'gwa']:
    ...     est = bd.estimate(t_data, p_data, [0.51, 0.5], [[1e-6, 1], [1e-6, 1]],
    ...                       framework='lse', squares=squares, model='Verhulst 2 (SIS)',
    ...                       known_p=[50], idx_known_p=[2],
    ...                       con={'type': 'ineq', 'fun': lambda p: p[0]-p[1]})
    ...     print('LSE estimate using', squares, 'is', est.p, ', computed in ', est.compute_time,
    ...           'seconds.')
    LSE estimate using expm is [0.9518853583017443, 0.31399108634106315] , computed in  0.5210022926330566 seconds.
    LSE estimate using fm is [0.8523239928832193, 0.2796039852281925] , computed in  1.1931242942810059 seconds.
    LSE estimate using gwa is [0.7084033634201875, 0.23080799483220513] , computed in  0.021984338760375977 seconds.

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
