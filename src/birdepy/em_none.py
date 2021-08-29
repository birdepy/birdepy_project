import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld


def discrete_est_em_none(sorted_data, p0, likelihood, technique, known_p,
                         idx_known_p, model, b_rate, d_rate, z_trunc,
                         p_bounds, con, max_it, i_tol, j_tol, h_tol, display,
                         opt_method, options):
    """
    Executes an expectation-maximization algorithm (not accelerated), to estimate
    parameters for a population-size-dependent birth-and-death process.

    This function is called by :func:`bd.interface_aug.f_fun_bld()`.

    [1] Jamshidian, M., & Jennrich, R. I. (1997). Acceleration of the EM algorithm
     by using quasi‐Newton methods. Journal of the Royal Statistical Society:
     Series B (Statistical Methodology), 59(3), 569-587.
    """
    #Initialize a list to return parameters in each iteration and store the initial parameters in it
    iterations = [p0.tolist()]
    it = 1
    p_est = p0
    difference = i_tol + 1
    while difference > i_tol and it < max_it:
        # Build surrogate log-likelihood function (defined in Section 2.1 of ref [1])
        f_fun = f_fun_bld(sorted_data, p_est, b_rate, d_rate, likelihood,
                          technique, idx_known_p, known_p, model, z_trunc,
                          j_tol, h_tol, options)[0]

        # Negate the surrogate log-likelihood function so that scipy.optimize.minimize()
        # Can be used to find a maximum
        def error_fun(p_prop):
            return -f_fun(p_prop)

        # Find the estimate for this iteration
        opt = ut.minimize_(error_fun, p_est, p_bounds, con, opt_method,
                           options)
        p_est_next = opt.x

        # Compute the difference between this iteration estimate and
        # previous iteration estimate
        difference = np.sum(np.abs(p_est_next - p_est))
        # Prepare for next iteration
        p_est = p_est_next
        # Store the estimate
        iterations.append(p_est.tolist())
        # Print a progress update if requested
        if display:
            print('Iteration ', it, ' estimate is: ', p_est)
        it += 1

    return list(p_est), iterations
