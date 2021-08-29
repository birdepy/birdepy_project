import mpmath as mp
import numpy as np
import birdepy.utility as ut
from birdepy.probability_ilt import laplace_p
from birdepy.interface_probability import probability
import birdepy.utility_em as ut_em


def aug_bld_ilt(sorted_data, param, b_rate, d_rate, likelihood, model, z_trunc,
                j_tol, h_tol, eps, laplace_method, num_terms, options):
    """
    Creates a function based on numerical Laplace transform inversion
    that returns the expected values of the number of up jumps from a
    state z, the number of down jumps from z, and the time spent in z
    when the process transitions from z0 to zt in elapsed time t.

    This function is called by :func:`bd.interface_aug.f_fun_bld()`.
    """
    def udh(z0, zt, t, z):
        """
        Computes the expected values defined by Equation (14) in
        reference [1] using numerical Laplace inversion.

        [1] Crawford, F. W., Minin, V. N., & Suchard, M. A. (2014).
        Estimation for general birth-death processes. Journal of
        the American Statistical Association, 109(506), 730-747.

        """
        pr = probability(z0, zt, t, param, model, likelihood, z_trunc=z_trunc,
                         options=options)[0][0]

        if b_rate(z, param) == 0 or pr == 0.0:
            u = 0.0
        else:
            pre_u = ut.laplace_invert(
                lambda s: laplace_p(s, z0, z, param, b_rate, d_rate, eps) *
                          laplace_p(s, z + 1, zt, param, b_rate, d_rate, eps),
                t,
                laplace_method=laplace_method,
                k=num_terms,
                f_bounds=[z_trunc[0], z_trunc[1]])
            u = float(mp.fdiv(mp.fmul(pre_u, b_rate(z, param)), pr))

        if d_rate(z, param) == 0 or pr == 0.0:
            d = 0.0
        else:
            pre_d = ut.laplace_invert(
                lambda s: laplace_p(s, z0, z, param, b_rate, d_rate, eps) *
                          laplace_p(s, z - 1, zt, param, b_rate, d_rate, eps),
                t,
                laplace_method=laplace_method,
                k=num_terms,
                f_bounds=[z_trunc[0], z_trunc[1]])
            d = float(mp.fdiv(mp.fmul(pre_d, d_rate(z, param)), pr))

        if pr == 0.0:
            h = 0.0
        else:
            pre_h = ut.laplace_invert(
                lambda s: laplace_p(s, z0, z, param, b_rate, d_rate, eps) *
                          laplace_p(s, z, zt, param, b_rate, d_rate, eps),
                t,
                laplace_method=laplace_method,
                k=num_terms,
                f_bounds=[0, t + 0.05])
            h = float(mp.fdiv(pre_h, pr))

        return np.array([u, d, h])

    aug_data = ut_em.help_bld_aug(udh, sorted_data, j_tol, h_tol, z_trunc)

    return aug_data
