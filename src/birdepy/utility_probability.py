import numpy as np
import mpmath as mp
from scipy.special import binom
from birdepy._branching import beta1, beta2, branching_probabilities


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
    z0, zt = int(z0), int(zt)

    def evaluate(alpha, beta, survival, geometric, choose):
        exp = mp.exp if isinstance(alpha, mp.mpf) else np.exp
        log1p = mp.log1p if isinstance(alpha, mp.mpf) else np.log1p

        def power(probability, complement, exponent):
            if probability > .5:
                return exp(exponent * log1p(-complement))
            return probability ** exponent

        if zt == 0:
            return power(alpha, survival, z0)
        return sum(choose(z0, j) * choose(zt - 1, z0 - j - 1)
                   * power(alpha, survival, j) * (survival * geometric) ** (z0 - j)
                   * power(beta, geometric, zt - z0 + j)
                   for j in range(max(0, z0 - zt), z0))

    with np.errstate(divide='raise', over='raise', under='raise', invalid='raise'):
        try:
            terms = branching_probabilities(lam, mu, t)
            return float(evaluate(*terms, binom))
        except (FloatingPointError, OverflowError):
            # Recompute the probabilities too: converting rounded float results
            # to mpmath cannot recover a lost complement or an underflowed tail.
            with mp.workdps(max(50, mp.mp.dps)):
                terms = branching_probabilities(*(mp.mpf(float(x)) for x in (lam, mu, t)))
                return float(evaluate(*terms, mp.binomial))
