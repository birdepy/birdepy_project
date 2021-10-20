import numpy as np
import mpmath as mp
import birdepy.utility as ut


def continued_fraction_terms_00_mp(idx, s, param, b_rate, d_rate):
    """
    Returns a partial numerator and denominator for the Laplace transform of the
    transition probability from 0 to 0 of a population-size-dependent
    birth-and-death process (as defined as $a_n$ and $b_n$ below Equation (9) in
    reference [1]).

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    if idx == 0:
        return mp.mpf('0.0'), mp.mpf('0.0')
    elif idx == 1:
        return mp.mpf('1.0'), mp.mpmathify(s + b_rate(0, param))
    else:
        return mp.fmul(-b_rate(idx - 2, param), d_rate(idx - 1, param)), \
               mp.fsum((s, b_rate(idx - 1, param), d_rate(idx - 1, param)))


def big_b_mp(i, j, s, param, b_rate, d_rate):
    """
    Returns the denominator of the 'j', 'i', and 'i+1' convergent (approximant)
    of the Laplace transform of the transition probability from 0 to 0 of a
    population-size-dependent birth-and-death process (as defined as $B_k$ in
    Lemma 1 of reference [1]).

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    # This function assumes i >= j and returns [B_j, B_i, B_{i+1}]
    if i == 1:  # return B0, B1, B2
        B_0 = mp.mpf('1.0')
        B_1 = mp.fadd(mp.mpmathify(s), mp.mpmathify(b_rate(0, param)))
        B_2 = mp.fsub(
            mp.fmul(mp.fadd(s + mp.mpmathify(b_rate(1, param)),
                            mp.mpmathify(d_rate(1, param))),
                    B_1),
            mp.fprod((mp.mpmathify(b_rate(0, param)),
                      mp.mpmathify(d_rate(1, param)),
                      B_0))
        )
        if j == 0:
            return B_0, B_1, B_2
        elif j == 1:
            return B_1, B_1, B_2
    else:
        cv = [1,
              mp.fadd(s, mp.mpmathify(b_rate(0, param)))]  # [B0, B1]
        if j == 0:
            B_j = cv[0]
        elif j == 1:
            B_j = cv[1]
        for terms in np.arange(2, i + 2, 1):
            cv = [cv[1],
                  mp.fsub(
                      mp.fmul(
                          mp.fsum((s,
                                   mp.mpmathify(
                                       b_rate(terms - 1, param)),
                                   mp.mpmathify(
                                       d_rate(terms - 1, param)))
                                  ),
                          cv[1]),
                      mp.fprod((mp.mpmathify(b_rate(terms - 2, param)),
                                mp.mpmathify(d_rate(terms - 1, param)),
                                cv[0])))
                  ]
            if terms == j:
                B_j = cv[1]
        return B_j, cv[0], cv[1]


def mu_product(i, j, param, d_rate):
    """
    Returns a product of the death rates from 'j+1' to 'i' as used in
    Theorem 1 of reference [1].

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    return np.prod(d_rate(np.arange(j + 1, i + 1, 1), param))


def lam_product(i, j, param, b_rate):
    """
    Returns a product of the birth rates from 'i' to 'j-1' as used in
    Theorem 1 of reference [1].

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    return np.prod(b_rate(np.arange(i, j, 1), param))


def mod_lentz_00_mp(fun, eps):
    """
    Computes part of the Laplace transform of the transition probability from
    0 to 0 of a population-size-dependent birth-and-death process (as defined
    by Equation (15) in reference [1]) in high precision using the modified
    Lentz algorithm.

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    (a0, b0) = fun(0)
    f = mp.mpmathify(b0)
    if f == 0:
        f = mp.mpf('1e-30')
    C = f
    D = 0
    error = 1 + eps
    idx = 0
    while error > eps:
        idx += 1
        (a, b) = fun(idx)
        D = mp.fadd(b, mp.fmul(a, D))
        if D == 0:
            D = mp.mpf('1e-30')
        C = mp.fadd(b, mp.fdiv(a, C))
        if C == 0:
            C = mp.mpf('1e-30')
        D = mp.fdiv(1.0, D)
        delta = mp.fmul(C, D)
        f = mp.fmul(f, delta)
        error = mp.fabs(mp.fsub(delta, 1.0))
    return f


def mod_lentz_ij_mp(s, i, j, param, b_rate, d_rate, eps):
    """
    Computes part of the Laplace transform of the transition probability from
    i to j of a population-size-dependent birth-and-death process (as defined
    by Equation (15) in reference [1]) in high precision using the modified
    Lentz algorithm.

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    B_j, B_i, B_ip1 = big_b_mp(i, j, s, param, b_rate, d_rate)
    f = mp.mpf('1e-30')
    C = f
    a1, b1 = mp.mpmathify(B_j), mp.mpmathify(B_ip1)
    D = b1
    if D == 0:
        D = mp.mpf('1e-30')
    C = mp.fadd(b1, mp.fdiv(a1, C))
    if C == 0:
        C = mp.mpf('1e-30')
    D = mp.fdiv(mp.mpf('1.0'), D)
    delta = mp.fmul(C, D)
    f = mp.fmul(f, delta)
    a2, b2 = mp.fprod((B_i,
                       -mp.mpmathify(b_rate(i, param)),
                       mp.mpmathify(d_rate(i + 1, param)))), \
             mp.fsum((s,
                      mp.mpmathify(b_rate(i + 1, param)),
                      mp.mpmathify(d_rate(i + 1, param))))
    D = mp.fadd(b2, mp.fmul(a2, D))
    if D == 0:
        D = mp.mpf('1e-30')
    C = mp.fadd(b2, mp.fdiv(a2, C))
    if C == 0:
        C = mp.mpf('1e-30')
    D = mp.fdiv(mp.mpf('1.0'), D)
    delta = mp.fmul(C, D)
    f = mp.fmul(f, delta)
    error = mp.fabs(mp.fsub(delta, mp.mpf('1.0')))
    idx = 3
    while error > eps:  # & (idx < 1000):
        (a, b) = mp.fmul(-mp.mpmathify(b_rate(i + idx - 2, param)),
                         mp.mpmathify(d_rate(i + idx - 1, param))), \
                 mp.fsum((s,
                          mp.mpmathify(b_rate(i + idx - 1, param)),
                          mp.mpmathify(d_rate(i + idx - 1, param))))
        D = mp.fadd(b, mp.fmul(a, D))
        if D == 0:
            D = mp.mpf('1e-30')
        C = mp.fadd(b, mp.fdiv(a, C))
        if C == 0:
            C = mp.mpf('1e-30')
        D = mp.fdiv(mp.mpf('1.0'), D)
        delta = mp.fmul(C, D)
        f = mp.fmul(f, delta)
        error = mp.fabs(mp.fsub(delta, mp.mpf('1.0')))
        idx += 1
    return f


def laplace_p(s, i, j, param, b_rate, d_rate, eps):
    """
    Computes the Laplace transform of the transition probability from
    'i' to 'j' of a population-size-dependent birth-and-death process (as defined
    by Equation (15) in reference [1]) in high precision using the modified
    Lentz algorithm.

    [1] Crawford, F. W., & Suchard, M. A. (2012). Transition probabilities for
    general birth–death processes with applications in ecology, genetics, and
    evolution. Journal of mathematical biology, 65(3), 553-580.
    """
    if i == 0 and j == 0:
        return mod_lentz_00_mp(
            lambda idx:
            continued_fraction_terms_00_mp(idx, s, param, b_rate, d_rate),
            eps)
    elif i >= j:
        return mp.fmul(mu_product(i, j, param, d_rate),
                       mod_lentz_ij_mp(s, i, j, param, b_rate, d_rate, eps))
    else:
        return mp.fmul(lam_product(i, j, param, b_rate),
                       mod_lentz_ij_mp(s, j, i, param, b_rate, d_rate, eps))


def probability_ilt(z0, zt, t, param, b_rate, d_rate, lentz_eps, laplace_method, k):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *inverse Laplace transform* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'ilt'::

        birdepy.probability(z0, zt, t, param, method='ilt', eps=1e-6,
                            laplace_method='cme-talbot', precision=100, k=25)

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------
    lentz_eps : scalar, optional
        Termination threshold for Lentz algorithm computation of Laplace
        domain functions.

    laplace_method : string, optional
        Numerical inverse Laplace transform algorithm to use. Should be one of:
        'cme-talbot' (default), 'cme', 'euler', 'gaver', 'talbot', 'stehfest',
        'dehoog', 'cme-mp' or 'gwr'.

    precision : int, optional
        Numerical precision (only used for methods that invoke mpmath).

    k: int, optional
        Maximum number of terms used for Laplace transform numerical inversion.
        Only applicable if argument 'laplace_method' is set to 'cme-talbot',
        'cme', 'euler', 'gaver' or 'cme-mp'.
        See https://www.inverselaplace.org/ for more information.

    Examples
    --------
    Approximate transition probability for a Verhulst model using numerical inverse Laplace transform: ::

        import birdepy as bd
        bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='ilt')[0][0]

    Outputs: ::

        0.0027403264310572615

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
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    .. [3] Crawford, F.W. and Suchard, M.A., 2012. Transition probabilities
     for general birth-death processes with applications in ecology, genetics,
     and evolution. Journal of Mathematical Biology, 65(3), pp.553-580.

    """
    # If more than one time is requested it is easiest to divert into a different code block
    if t.size == 1:
        # Initialize an array to store output
        output = np.zeros((z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            for idx2, _zt in enumerate(zt):
                # Compute and invert Laplace transform of probability, and ensure a float is
                # returned
                pr = float(ut.laplace_invert(
                    lambda s: laplace_p(s, _z0, _zt, param, b_rate, d_rate,
                                        lentz_eps),
                    t[0], laplace_method=laplace_method, k=k,
                    f_bounds=[0.0, 1.0]))
                output[idx1, idx2] = pr
    else:
        # Initialize an array to store output
        output = np.zeros((t.size, z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            for idx2, _zt in enumerate(zt):
                for idx3, _t in enumerate(t):
                    # Compute and invert Laplace transform of probability, and ensure a float is
                    # returned
                    pr = float(ut.laplace_invert(
                        lambda s: laplace_p(s, _z0, _zt, param, b_rate, d_rate,
                                            lentz_eps),
                        _t, laplace_method=laplace_method, k=k,
                        f_bounds=[0.0, 1.0]))
                    output[idx3, idx1, idx2] = pr
    return output
