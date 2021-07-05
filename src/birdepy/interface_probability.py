import numpy as np
import mpmath as mp
import birdepy.probability_expm as probability_expm
import birdepy.probability_uniform as probability_uniform
import birdepy.probability_Erlang as probability_Erlang
import birdepy.probability_gwa as probability_gwa
import birdepy.probability_gwasa as probability_gwasa
import birdepy.probability_oua as probability_oua
import birdepy.probability_da as probability_da
import birdepy.probability_ilt as probability_ilt
import birdepy.probability_sim as probability_sim
import birdepy.utility as ut


def probability(z0, zt, t, param, model='Verhulst 1', method='expm', **options):
    """Transition probabilities for continuous-time birth-and-death processes.

    Parameters
    ----------
    z0 : array_like
        States of birth-and-death process at time 0.

    zt : array_like
        States of birth-and-death process at time(s) `t`.

    t : array_like
        Elapsed time(s) (if has size greater than 1, then must be increasing).

    param : array_like
        The parameters governing the evolution of the birth-and-death
        process.
        Array of real elements of size (n,), where ‘n’ is the number of
        parameters.

    model : string, optional
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes
        >`). Should be one of:

            - 'Verhulst 1' (default)
            - 'Verhulst 2 (SIS)'
            - 'Ricker'
            - 'Beverton-Holt'
            - 'Hassell'
            - 'M-SS'
            - 'Moran'
            - 'pure-birth'
            - 'pure-death'
            - 'Poisson'
            - 'linear'
            - 'linear-migration'
            - 'M-SS'
            - 'M/M/inf'
            - 'loss-system'
            - 'custom'

        If set to 'custom', then arguments `b_rate` and `d_rate` must also be
        specified. See :ref:`here <Custom Models>` for more information.

    method : string, optional
        Transition probability approximation method. Should be one of
        (alphabetical ordering):

            - 'da' (see :ref:`here <birdepy.probability(method='da')>`)
            - 'Erlang'(see :ref:`here <birdepy.probability(method='Erlang')>`)
            - 'expm'  (default) (see :ref:`here <birdepy.probability(method='expm')>`)
            - 'gwa' (see :ref:`here <birdepy.probability(method='gwa')>`)
            - 'gwasa' (see :ref:`here <birdepy.probability(method='gwasa')>`)
            - 'ilt' (see :ref:`here <birdepy.probability(method='ilt')>`)
            - 'oua' (see :ref:`here <birdepy.probability(method='oua')>`)
            - 'sim (see  :ref:`here <birdepy.probability(method='sim')>`)
            - 'uniform' (see :ref:`here <birdepy.probability(method='uniform')>`)

    options : dict, optional
        A dictionary of method specific options.

    Return
    -------
    transition_probability : ndarray
        An array of transition probabilities. If t has size bigger than 1,
        then the first coordinate corresponds to `t`, the second coordinate
        corresponds to `z0` and the third coordinate corresponds to `zt`.
        Otherwise the first coordinate corresponds to `z0` and the second
        coordinate corresponds to `zt`

    Examples
    --------
    >>> for method in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'ilt', 'oua', 'uniform']:
    ...     print(method, 'approximation:',
    ...     probability(19, 27, 1.0, [0.5, 0.3, 100], model='Verhulst 2 (SIS)', method=method)[0][0])
    da approximation: 0.02937423095216112
    Erlang approximation: 0.02773268796308342
    expm approximation: 0.02784949584578541
    gwa approximation: 0.030788446607032095
    gwasa approximation: 0.030731643943282434
    ilt approximation: 0.027836532672791882
    oua approximation: 0.057511543975184994
    uniform approximation: 0.027849495845785503

    >>> for method in ['da', 'Erlang', 'expm', 'gwa', 'gwasa', 'ilt', 'oua', 'uniform']:
    ...     print(method, 'approximation:',
    ...     probability(19, 27, 1.0, [0.5, 0.3, 40], model='Verhulst 2 (SIS)', method=method)[0][0])
    da approximation: 0.0015028016489724214
    Erlang approximation: 0.0016455223175386804
    expm approximation: 0.001646125809214328
    gwa approximation: 0.004363221877757165
    gwasa approximation: 0.004354457417019735
    ilt approximation: 0.0016460890596731762
    oua approximation: 0.0014074199062890974
    uniform approximation: 0.0016461258092143286

    Notes
    -----
    Methods for computing transition probabilities and models are also
    described in [1]. If you use this function for published work, then please
    cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    :func:`birdepy.gpu_functions.probability_gpu()`  :func:`birdepy.gpu_functions.discrete_gpu()`

    References
    ----------
    .. [1] Hautphenne, S. and Patch, B. BirDePy: Parameter estimation for
     population-size-dependent birth-and-death processes in Python. ArXiV, 2021.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    if 'options' in options.keys():
        options = options['options']

    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)

    if np.isscalar(z0):
        z0 = np.array([z0])
    else:
        z0 = np.array(z0)
    if np.isscalar(zt):
        zt = np.array([zt])
    else:
        zt = np.array(zt)
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = np.array(t)

    if method == 'da':
        if 'k' in options.keys():
            k = options['k']
        else:
            k = 10
        if 'h_fun' in options.keys():
            h_fun = options['h_fun']
        else:
            h_fun = ut.higher_h_fun(model)
        return probability_da.probability_da(z0, zt, t, param, b_rate, d_rate,
                                             h_fun, k)

    elif method == 'Erlang':
        if 'z_trunc' in options.keys():
            z_trunc = options['z_trunc']
        else:
            z_vals = np.hstack((z0, zt))
            z_trunc = [np.maximum(0, np.amin(z_vals) - 100),
                       np.amax(z_vals) + 100]
        if 'k' in options.keys():
            k = options['k']
        else:
            k = 150
        if 'eps' in options.keys():
            eps = options['eps']
        else:
            eps = 1e-2
        if 'cut_meth' in options.keys():
            cut_meth = options['cut_meth']
        else:
            cut_meth = None
        return probability_Erlang.probability_Erlang(z0, zt, t, param, b_rate,
                                                     d_rate, z_trunc, cut_meth,
                                                     eps, k)

    elif method == 'expm':
        if 'z_trunc' in options.keys():
            z_trunc = options['z_trunc']
        else:
            z_vals = np.hstack((z0, zt))
            z_trunc = [np.maximum(0, np.amin(z_vals) - 100),
                       np.amax(z_vals) + 100]
        return probability_expm.probability_expm(z0, zt, t, param, b_rate,
                                                 d_rate,
                                                 z_trunc)

    elif method == 'gwa':
        return probability_gwa.probability_gwa(z0, zt, t, param, b_rate,
                                               d_rate)

    elif method == 'gwasa':
        return probability_gwasa.probability_gwasa(z0, zt, t, param, b_rate,
                                                   d_rate)

    elif method == 'ilt':
        if 'laplace_method' in options.keys():
            laplace_method = options['laplace_method']
        else:
            laplace_method = 'cme-talbot'
        if 'lentz_eps' in options.keys():
            lentz_eps = options['lentz_eps']
        else:
            lentz_eps = 1e-6
        if 'precision' in options.keys():
            mp.dps = options['precision']
        else:
            mp.dps = 1000
        if 'k' in options.keys():
            k = options['k']
        else:
            k = 25

        return probability_ilt.probability_ilt(z0, zt, t, param, b_rate,
                                               d_rate,
                                               lentz_eps, laplace_method, k)

    elif method == 'oua':
        if 'h_fun' in options.keys():
            h_fun = options['h_fun']
        else:
            h_fun = ut.higher_h_fun(model)
        if 'zf_bld' in options.keys():
            zf_bld = options['zf_bld']
        else:
            zf_bld = ut.higher_zf_bld(model)
        return probability_oua.probability_oua(z0, zt, t, param, b_rate,
                                               d_rate,
                                               h_fun, zf_bld)

    elif method == 'sim':
        if 'k' in options.keys():
            k = options['k']
        else:
            k = 10**6
        if 'seed' in options.keys():
            seed = options['seed']
        else:
            seed = None
        if 'sim_method' in options.keys():
            sim_method = options['sim_method']
        else:
            sim_method = 'exact'
        return probability_sim.probability_sim(z0, zt, t, param, b_rate,
                                               d_rate, k, sim_method, seed)

    elif method == 'uniform':
        if 'z_trunc' in options.keys():
            z_trunc = options['z_trunc']
        else:
            z_vals = np.hstack((z0, zt))
            z_trunc = [np.maximum(0, np.amin(z_vals) - 100),
                       np.amax(z_vals) + 100]
        if 'k' in options.keys():
            k = options['k']
        else:
            k = 150
        if 'eps' in options.keys():
            eps = options['eps']
        else:
            eps = 1e-2
        if 'cut_meth' in options.keys():
            cut_meth = options['cut_meth']
        else:
            cut_meth = None
        return probability_uniform.probability_uniform(z0, zt, t, param,
                                                       b_rate,
                                                       d_rate, z_trunc,
                                                       cut_meth,
                                                       eps, k)
    else:
        raise TypeError("Specified 'method' for computing "
                        "probabilities is unknown. Should be one of 'da', "
                        "'Erlang', 'gwa', 'gwasa', 'ilt', 'expm', 'oua', 'sim', "
                        "or 'uniform'. ")
