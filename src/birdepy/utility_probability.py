import numpy as np
import mpmath as mp
from scipy.special import binom


def beta1(lam, mu, t):
    """
    Computes an important part of the transition probability function for a linear
    birth-and-death process. Corresponds to $\alpha$ defined below Equation (1)
    in reference [1]. Switches into a high precision mode if an exception is
    detected using machine precision.

    References
    ----------
    .. [1] Davison, A. C., Hautphenne, S., & Kraus, A. (2021). Parameter
    estimation for discretely observed linear birth‐and‐death processes.
    Biometrics, 77(1), 186-196.
    """
    if lam == mu:
        return lam * t / (1 + lam * t)
    else:
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            try:
                output = mu * (np.exp((lam - mu) * t) - 1) \
                         / (lam * np.exp((lam - mu) * t) - mu)
                if output == 1:
                    raise Exception
            except:
                output = mp.fdiv(mp.fmul(mu,
                                         mp.fsub(mp.exp((mp.fmul(mp.fsub(lam,
                                                                         mu),
                                                                 t))),
                                                 1.0)),
                                 mp.fmul(lam,
                                         mp.fsub(mp.exp((mp.fmul(mp.fsub(lam,
                                                                         mu),
                                                                 t))),
                                                 mu)))
        return output


def beta2(lam, mu, t):
    """
    Computes an important part of the transition probability function for a linear
    birth-and-death process. Corresponds to $\beta$ defined below Equation (1)
    in reference [1]. Switches into a high precision mode if an exception is
    detected using machine precision.

    References
    ----------
    .. [1] Davison, A. C., Hautphenne, S., & Kraus, A. (2021). Parameter
    estimation for discretely observed linear birth‐and‐death processes.
    Biometrics, 77(1), 186-196.
    """
    if lam == mu:
        return beta1(lam, mu, t)
    else:
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            try:
                output = lam * beta1(lam, mu, t) / mu
                if output == 1:
                    raise Exception
            except:
                output = mp.fmul(lam, mp.fdiv(beta1(lam, mu, t), mu))
        return output


def p_lin(z0, zt, lam, mu, t):
    """
    Computes the transition probability function for a linear
    birth-and-death process, as given by Equation (2) in reference [1].
    Switches into a high precision mode if an exception is
    detected using machine precision.

    References
    ----------
    .. [1] Davison, A. C., Hautphenne, S., & Kraus, A. (2021). Parameter
    estimation for discretely observed linear birth‐and‐death processes.
    Biometrics, 77(1), 186-196.
    """
    if zt == 0:
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            try:
                pr = beta1(lam, mu, t) ** z0
            except:
                pr = float(mp.power(beta1(lam, mu, t), z0))
    else:
        pr = 0
        for j in np.arange(np.maximum(0, z0 - zt), z0, 1):
            with np.errstate(divide='raise', over='raise', under='raise',
                             invalid='raise'):
                try:
                    pr_j = binom(z0, j) * \
                           binom(zt - 1, z0 - j - 1) * \
                           (beta1(lam, mu, t) ** j) * \
                           (((1 - beta1(lam, mu, t)) * (1 - beta2(lam, mu, t)))
                            ** (z0 - j)) * \
                           (beta2(lam, mu, t) ** (zt - z0 + j))
                except:
                    pr_j = mp.fprod((mp.binomial(z0, j),
                                     mp.binomial(zt - 1.0, z0 - j - 1.0),
                                     mp.power(beta1(lam, mu, t), j),
                                     mp.power(mp.fmul(mp.fsub(1.0,
                                                              beta1(lam, mu, t)
                                                              ),
                                                      mp.fsub(1.0,
                                                              beta2(lam, mu,
                                                                    t))),
                                              z0 - j),
                                     mp.power(beta2(lam, mu, t),
                                              zt - z0 + j)))
            pr += float(pr_j)
    return pr
