import numpy as np
from birdepy import simulate
from collections import Counter


def probability_sim(z0, zt, t, param, b_rate, d_rate, k, sim_method, seed):
    """Transition probabilities for continuous-time birth-and-death processes
    using crude Monte Carlo simulation.

    To use this function call :func:`birdepy.probability` with `method` set to
    'sim'::

        birdepy.probability(z0, zt, t, param, method='sim', k=10**5, seed=None)

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------

    k: int, optional
        Number of samples used to generate each probability estimate.
        The total number of samples used will be z0.size * k.

    seed : int, Generator, optional
        If *seed* is not specified the random numbers are generated according
        to *np.random.default_rng()*. If *seed* is an *int*, random numbers are
        generated according to *np.random.default_rng(seed)*. If seed is a
        *Generator*, then that object is used. See
        :ref:`here <Reproducibility>` for more information.

    sim_method : string, optional
        Simulation algorithm used to generate samples (see
        :ref:`here<Simulation Algorithms>`). Should be one of:

            - 'exact' (default)
            - 'ea'
            - 'ma'
            - 'gwa'

    Examples
    --------
    >>> import birdepy as bd
    >>> bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='sim', k=10**5,
    ...                seed=2021)[0][0]
    0.00294

    Notes
    -----
    Estimation algorithms and models are also described in [1]. If you use this
    function for published work, then please cite this paper.

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    For more details on this method see [3].

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

    .. [3] Crawford, F.W. and Suchard, M.A., 2012. Transition probabilities
     for general birth-death processes with applications in ecology, genetics,
     and evolution. Journal of Mathematical Biology, 65(3), pp.553-580.

    """

    if t.size == 1:
        output = np.zeros((z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            counts = Counter(simulate.discrete(param, 'custom', _z0, [0, t[0]],
                                               b_rate=b_rate, d_rate=d_rate,
                                               k=k, seed=seed,
                                               method=sim_method)[:, 1])
            for idx2, _zt in enumerate(zt):
                output[idx1, idx2] = counts[_zt]/k
    else:
        output = np.zeros((t.size, z0.size, zt.size))
        for idx3, _t in enumerate(t):
            for idx1, _z0 in enumerate(z0):
                counts = Counter(simulate.discrete(param, 'custom', _z0,
                                                   [0, _t], b_rate=b_rate,
                                                   d_rate=d_rate, k=k,
                                                   seed=seed,
                                                   method=sim_method)[:, 1])
                for idx2, _zt in enumerate(zt):
                    output[idx3, idx1, idx2] = counts[_zt] / k
    return output
