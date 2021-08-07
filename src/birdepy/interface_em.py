import numpy as np
import birdepy.em_none as em_none
import birdepy.em_Lange as em_Lange
import birdepy.em_qn1 as em_qn1
import birdepy.em_qn2 as em_qn2
import birdepy.em_cg as em_cg
import birdepy.utility as ut
from birdepy.interface_dnm import bld_ll_fun


def discrete_est_em(data, p0, technique, accelerator, likelihood,
                    p_bounds, con, known_p, idx_known_p, model, b_rate, d_rate,
                    z_trunc, max_it, i_tol, j_tol, h_tol, display, opt_method,
                    options):
    """Parameter estimation for discretely observed continuous-time
    birth-and-death processes using the expectation maximization algorithm.
    See :ref:`here <Expectation Maximization>` for more information.

    To use this function call :func:`birdepy.estimate` with `framework` set to
    'em'::

        birdepy.estimate(t_data, p_data, p0, p_bounds, framework='em', technique='expm',
                         accelerator='none', likelihood='expm', laplace_method='cme-talbot',
                         lentz_eps=1e-6, max_it=25, i_tol=1e-2, j_tol=1e-1, h_tol=1e-2,
                         z_trunc=())

    The parameters associated with this framework (listed below) can be
    accessed using kwargs in :func:`birdepy.estimate()`. See documentation of
    :func:`birdepy.estimate` for the main arguments.

    Parameters
    ----------
    technique : string, optional
        Expectation step technique. Should be one of
        (alphabetical ordering):

            - 'expm' (default)
            - 'ilt'
            - 'num'

        See :ref:`here <Expectation Step Techniques>` for more information.

    accelerator : string, optional
        EM accelerator method. Should be one of
        (alphabetical ordering):

            - 'cg' (see [3])
            - 'none' (see [4])
            - 'Lange' (default) (see [5])
            - 'qn1' (see [6])
            - 'qn2' (see [6])

        See :ref:`here <Acceleration>` for more information.

    likelihood : string, optional
        Likelihood approximation method. Should be one of
        (alphabetical ordering):

            - 'da' (default) (see :ref:`here <birdepy.probability(method='da')>`)
            - 'Erlang' (default) (see :ref:`here <birdepy.probability(method='Erlang')>`)
            - 'expm' (see :ref:`here <birdepy.probability(method='expm')>`)
            - 'gwa' (see :ref:`here <birdepy.probability(method='gwa')>`)
            - 'gwasa' (see :ref:`here <birdepy.probability(method='gwasa')>`)
            - 'ilt' (see :ref:`here <birdepy.probability(method='ilt')>`)
            - 'oua' (see :ref:`here <birdepy.probability(method='oua')>`)
            - 'uniform' (see :ref:`here <birdepy.probability(method='uniform')>`)

        The links point to the documentation of the relevant `method` in
        :func:`birdepy.probability`. The arguments associated with each of
        these methods may be used as a kwarg in :func:`birdepy.estimate()`
        when `likelihood` is set to use the method.

    laplace_method : string, optional
        Numerical inverse Laplace transform algorithm to use. Should be one of:
        'cme-talbot' (default), 'cme', 'euler', 'gaver', 'talbot', 'stehfest',
        'dehoog', 'cme-mp' or 'gwr'.

    lentz_eps : scalar, optional
        Termination threshold for Lentz algorithm computation of Laplace
        domain functions.

    max_it : int, optional
        Maximum number of iterations of the algorithm to perform.

    i_tol : scalar, optional
        Algorithm terminates when ``sum(abs(p(i) - p(i-1)) < i_tol``
        where `p(i)` and `p(i-1)` are estimates corresponding to iteration `i`
        and `i-1`.

    j_tol : scalar, optional
        States with expected number of upward transitions or expected number
        of downward transitions greater than `j_tol` are included in E steps.

    h_tol : scalar, optional
        States with expected holding time greater than `h_tol` are included in
        E steps.

    z_trunc : array_like, optional
        Truncation thresholds, i.e., minimum and maximum states of process
        considered. Array of real elements of size (2,) by default
        ``z_trunc=[z_min, z_max]`` where
        ``z_min=max(0, min(p_data) - 2*z_range)``
        and ``z_max=max(p_data) + 2*z_range`` where
        ``z_range=max(p_data)-min(p_data)``.

    Examples
    --------
    Use :func:`birdepy.simulate.discrete` to simulate a sample path of the 'Verhulst 2 (SIS)'
    model:

    >>> import birdepy as bd
    >>> t_data = [t for t in range(100)]
    >>> p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
    ...                               survival=True, seed=2021)

    Estimate the parameter values from the simulated data using the 'em'
    `framework` with various `technique` and `accelerator` approaches.

    The constraint ``con={'type': 'ineq', 'fun': lambda p: p[0]-p[1]}`` ensures
    that p[0] > p[1] (i.e., rate of spread greater than recovery rate).

    for technique in ['expm', 'ilt', 'num']:
        for accelerator in ['cg', 'none', 'Lange', 'qn1', 'qn2']:
            tic = time.time()
            est = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[1e-6,1], [1e-6,1], [1e-6, 0.1]],
                              framework='em', technique=technique, accelerator=accelerator,
                              model='Ricker', idx_known_p=[3], known_p=[1])

    A ``RuntimeWarning`` associated with SciPy's :func:`minimize` function  may
    appear, this can be ignored.

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

    .. [3] Jamshidian, M. and  Jennrich, R.I. Conjugate gradient acceleration
     of the EM algorithm. Journal of the American Statistical Association,
     88(421):221-228, 1993.

    .. [4] Dempster, A.P., Laird, N.M. and Rubin, D.B. Maximum likelihood from
     incomplete data via the EM algorithm. Journal of the Royal Statistical
     Society: Series B (Methodological), 39(1):1-22, 1977.

    .. [5] Lange, K. A quasi-Newton acceleration of the EM algorithm.
     Statistica Sinica, 1-18, 1995.

    .. [6] Jamshidian, M. and  Jennrich, R.I. Acceleration of the EM algorithm
      by using quasi-Newton methods. Journal of the Royal Statistical
      Society: Series B (Methodological), 59(3):569-587, 1997.

    """
    sorted_data = ut.data_sort_2(data)
    if accelerator == 'none':
        p_est, iterations = em_none.discrete_est_em_none(
            sorted_data, p0, likelihood, technique, known_p, idx_known_p,
            model, b_rate, d_rate, z_trunc, p_bounds, con, max_it, i_tol,
            j_tol, h_tol, display, opt_method, options)

    elif accelerator == 'cg':
        p_est, iterations = em_cg.discrete_est_em_cg(
            data, sorted_data, p0, likelihood, technique, known_p,
            idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
            con, max_it, i_tol, j_tol, h_tol, display, opt_method, options)

    elif accelerator == 'Lange':
        p_est, iterations = em_Lange.discrete_est_em_Lange(
            data, sorted_data, p0, likelihood, technique, known_p,
            idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds, con,
            max_it, i_tol, j_tol, h_tol, display, opt_method, options)

    elif accelerator == 'qn1':
        p_est, iterations = em_qn1.discrete_est_em_qn1(
            sorted_data, p0, likelihood, technique, known_p,
            idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
            con, max_it, i_tol, j_tol, h_tol, display, opt_method, options)

    elif accelerator == 'qn2':
        p_est, iterations = em_qn2.discrete_est_em_qn2(
            data, sorted_data, p0, likelihood, technique, known_p,
            idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds, con,
            max_it, i_tol, j_tol, h_tol, display, opt_method, options)

    else:
        raise ValueError("Argument 'accelerator' has an unknown value. Should "
                         "be one of: 'none', 'cg', 'Lange', 'qn1' or 'qn2'.")

    pre_ll_fun = bld_ll_fun(data, likelihood, model, z_trunc, options)

    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll_fun(param)

    for idx in range(len(p0)):
        p_est[idx] = min(p_bounds[idx][1],
                         max(p_bounds[idx][0], p_est[idx]))

    if p0.size > 1:
        try:
            cov = np.linalg.inv(-ut.Hessian(ll, p_est, p_bounds))
        except:
            cov = 'Covariance matrix could not be determined.'
    else:
        try:
            print(np.array([-1/ut.Hessian(ll, p_est, p_bounds)]))
            cov = np.array([-1/ut.Hessian(ll, p_est, p_bounds)])
        except:
            cov = 'Covariance matrix could not be determined.'

    ll = ll(np.array(p_est))

    return p_est, cov, ll, iterations
