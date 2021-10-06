import numpy as np
import birdepy.aug_num as _aug_num
import birdepy.aug_ilt as _aug_ilt
import birdepy.aug_expm as _aug_expm
import birdepy.utility as ut
import mpmath as mp


def f_fun_bld(sorted_data, param, b_rate, d_rate, likelihood, technique,
              idx_known_p, known_p, model, z_trunc, j_tol, h_tol, options):
    """
    Returns the surrogate likelihood function for use in the
    expectation-maximization algorithm. This function is defined inline in Section 2.1
    of reference [1].

    This function is used by the five em_*.py modules where * is cg, Lange, none,
    qn1 or qn2.

    [1] Jamshidian, M., & Jennrich, R. I. (1997). Acceleration of the EM algorithm
     by using quasi‐Newton methods. Journal of the Royal Statistical Society:
     Series B (Statistical Methodology), 59(3), 569-587.
    """
    param = ut.p_bld(param, idx_known_p, known_p)
    if technique == 'num':
        aug_data = _aug_num.aug_bld_num(sorted_data, param, b_rate, d_rate,
                                        likelihood, model, z_trunc, j_tol,
                                        h_tol, options)

    elif technique == 'ilt':
        if 'laplace_method' in options.keys():
            laplace_method = options['laplace_method']
        else:
            laplace_method = 'cme-talbot'
        if 'lentz_eps' in options.keys():
            lentz_eps = options['lentz_eps']
        else:
            lentz_eps = 1e-6
            options['lentz_eps'] = lentz_eps
        if 'k' in options.keys():
            k = options['k']
        else:
            k = 25
            options['k'] = k
        if 'precision' in options.keys():
            mp.dps = options['precision']
        else:
            mp.dps = 100
        aug_data = _aug_ilt.aug_bld_ilt(sorted_data, param, b_rate, d_rate,
                                        likelihood, model, z_trunc, j_tol,
                                        h_tol, lentz_eps, laplace_method, k,
                                        options)

    elif technique == 'expm':
        aug_data = _aug_expm.aug_bld_expm(sorted_data, param, b_rate, d_rate,
                                          likelihood, model, z_trunc, j_tol,
                                          h_tol, options)
    else:
        raise TypeError("Argument technique has an unknown value.")

    def f_fun(p_prime_prop):
        p_prime_prop = ut.p_bld(p_prime_prop, idx_known_p, known_p)
        output = 0
        for i in aug_data:
            if b_rate(i, p_prime_prop) > 0:
                output += aug_data[i][0] * np.log(b_rate(i, p_prime_prop))
            if d_rate(i, p_prime_prop) > 0:
                output += aug_data[i][1] * np.log(d_rate(i, p_prime_prop))
            output -= aug_data[i][2] * (b_rate(i, p_prime_prop) +
                                        d_rate(i, p_prime_prop))

        return output

    return f_fun, aug_data
