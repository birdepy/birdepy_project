"""Shared linear birth-and-death probabilities and stable complements."""

import mpmath as mp
import numpy as np


def branching_probabilities(lam, mu, t):
    """Return alpha, beta, 1-alpha, and 1-beta without cancellation.

    Alpha is the extinction probability of one ancestor; conditional on
    survival, its descendants have a geometric distribution with success
    probability 1-beta. Accept mpmath scalars for probability calculations
    that require more precision than the simulation routines.
    """
    high_precision = any(isinstance(x, mp.mpf) for x in (lam, mu, t))
    exp = mp.exp if high_precision else np.exp
    expm1 = mp.expm1 if high_precision else np.expm1
    if lam == mu:
        complement = 1 / (1 + lam * t)
        probability = lam * t * complement
        return probability, probability, complement, complement
    if lam > mu:
        difference = lam - mu
        change = -expm1(-difference * t)
        denominator = difference + mu * change
        return (mu * change / denominator, lam * change / denominator,
                difference / denominator,
                difference * exp(-difference * t) / denominator)
    difference = mu - lam
    change = -expm1(-difference * t)
    denominator = difference + lam * change
    return (mu * change / denominator, lam * change / denominator,
            difference * exp(-difference * t) / denominator,
            difference / denominator)


def beta1(lam, mu, t):
    """Extinction probability for one ancestor in a linear birth-death process."""
    return branching_probabilities(lam, mu, t)[0]


def beta2(lam, mu, t):
    """Geometric tail parameter conditional on survival of one ancestor."""
    return branching_probabilities(lam, mu, t)[1]
