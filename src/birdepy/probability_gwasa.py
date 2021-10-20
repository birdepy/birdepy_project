import numpy as np
import birdepy.utility_probability as ut
import mpmath as mp
import warnings


def w_fun(lam, mu, _z0, _zt, t):
    """
    Computes an important part of the saddlepoint approximation for a linear
    linear-and-death process. Corresponds to $\tilde s = (2A)^{-1}(-B+\sqrt{B^2-4AC})$
    in Lemma 1 of reference [1].

    References
    ----------
    .. [1] Davison, A. C., Hautphenne, S., & Kraus, A. (2021). Parameter
    estimation for discretely observed linear birth‐and‐death processes.
    Biometrics, 77(1), 186-196.
    """
    if lam == mu:
        a = lam * t - (lam * t) ** 2
        b = 2 * (lam * t) ** 2 + _z0 / _zt - 1
        c = -lam * t - (lam * t) ** 2
    else:
        m = np.exp((lam - mu) * t)
        a = lam * (m - 1) * (lam - mu * m)
        b1 = 2 * lam * mu * (1 + m ** 2 - m - (_z0 / _zt) * m)
        b2 = m * (lam ** 2 + mu ** 2) * ((_z0 / _zt) - 1)
        b = b1 + b2
        c = mu * (m - 1) * (mu - lam * m)
    if a == 0:
        w = -c / b
    elif (b ** 2 - 4 * a * c) < 0:
        raise Exception
    elif b == 0:
        raise Exception
    else:
        w = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    return w


def w_fun_high_precision(lam, mu, _z0, _zt, t):
    """
    Computes an important part of the saddlepoint approximation for a linear
    linear-and-death process in high precision. Corresponds to
    $\tilde s = (2A)^{-1}(-B+\sqrt{B^2-4AC})$ in Lemma 1 of reference [1].

    References
    ----------
    .. [1] Davison, A. C., Hautphenne, S., & Kraus, A. (2021). Parameter
    estimation for discretely observed linear birth‐and‐death processes.
    Biometrics, 77(1), 186-196.
    """
    if lam == mu:
        a = mp.fsub(mp.fmul(lam, t),
                    mp.power(mp.fmul(lam, t), 2))
        b = mp.fsum((mp.fprod((2.0,
                               mp.power(lam, 2),
                               mp.power(t, 2))),
                     mp.fdiv(_z0, _zt),
                     -1.0))
        c = -mp.fadd(mp.fmul(lam, t),
                     mp.power(mp.fmul(lam, t), 2))
    else:
        m = mp.exp(mp.fmul(mp.fsub(lam, mu), t))
        a = mp.fprod((lam,
                      mp.expm1(mp.fmul(mp.fsub(lam, mu), t)),
                      mp.fsub(lam, mp.fmul(mu, m))))
        b1 = mp.fprod((2.0,
                       lam,
                       mu,
                       mp.fsub(1.0,
                               mp.fmul(m,
                                       mp.fadd(
                                           mp.expm1(mp.fmul(mp.fsub(lam, mu),
                                                            t)),
                                           mp.fdiv(_z0, _zt)
                                       )
                                       )
                               )
                       )
                      )
        b2 = mp.fprod((m,
                       mp.fadd(mp.power(lam, 2), mp.power(mu, 2)),
                       mp.fsub(mp.fdiv(_z0, _zt), 1)
                       )
                      )
        b = mp.fadd(b1, b2)
        c = mp.fprod((mu,
                      mp.expm1(mp.fmul(mp.fsub(lam, mu), t)),
                      mp.fsub(mu, mp.fmul(lam, m))))
    if a == 0 and b > 0:
        w = -mp.fdiv(c, b)
    elif mp.fsub(mp.power(b, 2), 4 * mp.fmul(a, c)) < 0:
        w = 1e-100
        warnings.warn("A value of b**2 - 4ac less than 0 has been "
                      "encountered and w was replaced by a default value. "
                      "The results may be unreliable.",
                      category=RuntimeWarning)
    elif b == 0:
        w = 1e-100
        warnings.warn("A value of b equal to 0 has been encountered"
                      "and w was replaced by a default value. The results"
                      "may be unreliable. ",
                      category=RuntimeWarning)
    else:
        w = mp.fdiv(mp.fadd(-b,
                            mp.sqrt(mp.fsub(mp.power(b, 2),
                                            mp.fprod((4.0, a, c))))),
                    mp.fmul(2.0, a))

    return w


def probability_gwasa(z0, zt, t, param, b_rate, d_rate, anchor):
    """Transition probabilities for continuous-time birth-and-death processes
    using the *Galton-Watson saddlepoint approximation* method.

    To use this function call :func:`birdepy.probability` with `method` set to
    'gwasa'::

        birdepy.probability(z0, zt, t, param, method='gwasa', anchor='midpoint')

    The parameters associated with this method (listed below) can be
    accessed using kwargs in :func:`birdepy.probability()`. See documentation
    of :func:`birdepy.probability` for the main arguments.

    Parameters
    ----------
    anchor : string, optional
        Determines which state z is used to determine the linear approximation.
        Should be one of: 'initial' (z0 is used), 'midpoint' (default, 0.5*(z0+zt) is used),
        'terminal' (zt is used), 'max' (max(z0, zt) is used), or 'min' (min(z0, zt) is used).

    Examples
    --------
    Approximate transition probability for a Verhulst model using the saddlepoint approximation
    to a linear approximation: ::

        import birdepy as bd
        bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='gwasa')[0][0]

    Outputs: ::

        0.002271944691896704

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
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """
    # This function computes a transition probability for a specific initial state, terminal state
    # and time combination. Later we will use it in a loop.
    def prob(_z0, _zt, _t):
        # Set the rates of the linear approximation according the argument of 'anchor'
        if anchor == 'midpoint':
            midpoint = 0.5*(_z0+_zt)
            lam = b_rate(midpoint, param) / midpoint
            mu = d_rate(midpoint, param) / midpoint
        elif anchor == 'initial':
            lam = b_rate(_z0, param) / _z0
            mu = d_rate(_z0, param) / _z0
        elif anchor == 'terminal':
            lam = b_rate(_zt, param) / _zt
            mu = d_rate(_zt, param) / _zt
        elif anchor == 'max':
            max_z = max(_z0, _zt)
            lam = b_rate(max_z, param) / max_z
            mu = d_rate(max_z, param) / max_z
        elif anchor == 'min':
            min_z = min(_z0, _zt)
            lam = b_rate(min_z, param) / min_z
            mu = d_rate(min_z, param) / min_z
        else:
            raise TypeError("Argument 'anchor' has an unknown value. Should be one of 'midpoint'"
                            " 'initial', 'terminal', 'max' or 'min'.")
        # Linear birth-and-death processes never evolve away from 0
        if _z0 == 0 and _zt > 0:
            raise TypeError("Methods 'gwa' and 'gwasa' are not suitable for "
                            "datasets that include transitions away "
                            "from the origin (i.e., z_{i-1}=0 and "
                            "z_i>0). ")
        # Linear birth-and-death processes always stay at 0
        if _z0 == 0 and _zt == 0:
            pr = 1
        elif _zt == 0:  # Transitioning to 0 is a special case
            pr = mp.fmul(_z0, np.log(max(1e-100, ut.beta1(lam, mu, _t))))
        elif _zt == 1:  # Transitioning to 1 is a special case
            pr = ut.p_lin(_z0, 1, b_rate(_z0, param) / _z0,
                          d_rate(_z0, param) / _z0, _t)
        else:
            # We attempt to compute required quantities with machine precision. If an error is
            # encountered we divert to high precision computations (which take longer).
            # The computations performed here are described in detail in Section 2.3 of
            # reference [1]
            with np.errstate(divide='raise', over='raise', under='raise',
                             invalid='raise'):
                try:
                    w = w_fun(lam, mu, _z0, _zt, _t)
                except:
                    w = w_fun_high_precision(lam, mu, _z0, _zt, _t)
            if w == 0:
                w = 1e-100
                warnings.warn(
                    "A value of w equal to 0 has been encountered and "
                    " w was replaced by a default value. The results "
                    " may be unreliable.",
                    category=RuntimeWarning)            
            with np.errstate(divide='raise', over='raise', under='raise',
                             invalid='raise'):
                try:
                    if lam == mu:
                        p1 = lam * _t * (1 - w) + w
                        p2 = 1 - lam * _t * (w - 1)
                        p3 = lam * _t * w * (
                                -lam * _t * w ** 2 + lam * _t + w ** 2 + 1)
                        p4 = (lam * _t * (w - 1) - 1) ** 2 * (
                                -lam * _t * w + lam * _t + w) ** 2
                    else:
                        m = np.exp((lam - mu) * _t)
                        p1 = mu - lam * w + mu * (w - 1) * m
                        p2 = mu - lam * w + lam * (w - 1) * m
                        p3 = -(m - 1) * m * w * (lam - mu) ** 2 * (
                                -lam ** 2 * w ** 2 +
                                lam * m * mu * (w ** 2 - 1) +
                                mu ** 2)
                        p4 = (lam * (m * (w - 1) - w) + mu) ** 2 * \
                             (lam * w + mu * (-m * w + m - 1)) ** 2
                    pr = ((p1 / p2) ** _z0 * (p3 / p4) ** (-0.5)) / \
                         (np.sqrt(2 * np.pi * _z0) * w ** _zt)
                except:
                    mp.dps = 1000
                    w = mp.mpmathify(w)
                    if lam == mu:
                        p1 = mp.fadd(mp.fprod((lam,
                                               _t,
                                               mp.fsub(1.0, w))),
                                     w)
                        p2 = mp.fsub(1.0,
                                     mp.fprod((lam,
                                               _t,
                                               mp.fsub(w, 1))
                                              )
                                     )
                        p3 = mp.fprod((lam,
                                       _t,
                                       w,
                                       mp.fsum((mp.fprod((-lam,
                                                          _t,
                                                          mp.power(w, 2))),
                                                mp.fmul(lam, _t),
                                                mp.power(w, 2),
                                                1.0))
                                       ))
                        p4 = mp.fmul(
                            mp.power(mp.fsub(mp.fprod((lam,
                                                       _t,
                                                       mp.fsub(w, 1))),
                                             1.0), 2),
                            mp.power(mp.fsum((mp.fprod((-lam,
                                                        _t,
                                                        w)),
                                              mp.fmul(lam, _t),
                                              w)), 2)
                        )
                    else:
                        m = mp.exp(mp.fmul(mp.fsub(lam, mu), _t))
                        p1 = mp.fsum((mu,
                                      mp.fmul(-lam, w),
                                      mp.fprod((mu,
                                                mp.fsub(w, 1),
                                                m))
                                      ))
                        p2 = mp.fsum((mu,
                                      mp.fmul(-lam, w),
                                      mp.fprod((lam,
                                                mp.fsub(w, 1),
                                                m))))
                        p3 = -mp.fprod(
                            (mp.expm1(mp.fmul(mp.fsub(lam, mu), _t)),
                             m,
                             w,
                             mp.power(mp.fsub(lam, mu), 2),
                             mp.fsum((-mp.power(mp.fmul(lam, w), 2),
                                      mp.fprod((lam,
                                                m,
                                                mu,
                                                mp.fsub(
                                                    mp.power(w, 2),
                                                    1.0)
                                                )),
                                      mp.power(mu, 2)
                                      ))))
                        p4 = mp.fmul(
                            mp.power(
                                mp.fadd(mp.fmul(lam,
                                                mp.fsub(
                                                    mp.fmul(m,
                                                            mp.fsub(w, 1.0)),
                                                    w)),
                                        mu),
                                2),
                            mp.power(mp.fadd(mp.fmul(lam, w),
                                             mp.fmul(mu,
                                                     mp.fsum((-mp.fmul(m, w),
                                                              m,
                                                              -1.0)))),
                                     2))
                    pr = mp.fprod((mp.fdiv(1.0,
                                           mp.fmul(mp.sqrt(mp.fprod(
                                               (2.0, mp.pi, _z0))),
                                               mp.power(w, _zt))),
                                   mp.power(mp.fdiv(p1, p2), _z0),
                                   mp.power(mp.fdiv(p3, p4), -0.5)))
        # Return a float (not a high precision mpmath object)
        if type(pr) != float:
            pr = float(mp.re(pr))

        if not 0 <= pr <= 1.0:
            warnings.warn("Probability not in [0, 1] computed, "
                          "some output has been replaced by a "
                          "default value. "
                          " Results may be unreliable.",
                          category=RuntimeWarning)
            if pr < 0:
                pr = 0.0
            else:
                pr = 1.0

        return pr

    # If more than one time is requested it is easiest to divert into a different code block
    if t.size == 1:
        # Initialize an output array to be filled in as we loop over initial and terminal states
        output = np.zeros((z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            for idx2, _zt in enumerate(zt):
                output[idx1, idx2] = prob(_z0, _zt, t[0])
    else:
        # Initialize an output array to be filled in as we loop over times, initial and terminal
        # states
        output = np.zeros((t.size, z0.size, zt.size))
        for idx1, _z0 in enumerate(z0):
            for idx2, _zt in enumerate(zt):
                for idx3, _t in enumerate(t):
                    output[idx3, idx1, idx2] = prob(_z0, _zt, _t)
    return output
