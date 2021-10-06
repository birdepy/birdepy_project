import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy.interface_dnm import bld_ll_fun
import birdepy.utility_em as utem


def discrete_est_em_cg(data, sorted_data, p0, likelihood, technique, known_p,
                       idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
                       con, max_it, i_tol, j_tol, h_tol, display, opt_method,
                       options):
    """
    Executes an expectation-maximization algorithm, which is accelerated using
    conjugate gradient ideas, to estimate parameters for a
    population-size-dependent birth-and-death process.

    This function is called by :func:`bd.interface_aug.f_fun_bld()`.

    [1] Jamshidian, M., & Jennrich, R. I. (1997). Acceleration of the EM algorithm
     by using quasi‐Newton methods. Journal of the Royal Statistical Society:
     Series B (Statistical Methodology), 59(3), 569-587.

    [2] Jamshidian, M., & Jennrich, R. I. (1993). Conjugate gradient acceleration of
     the EM algorithm. Journal of the American Statistical Association, 88(421), 221-228.
    """
    #Initialize a list to return parameters in each iteration and store the initial parameters in it
    iterations = [p0]

    # Build the log-likelihood function assuming all parameters are known
    pre_ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    # Adjust the log-likelihood function to account for unknown parameters and
    # negate it so that the minimize function performs maximisation
    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    it = 0
    p_est_it_m1 = p0
    difference = i_tol + 1
    # Iterate until the difference between parameter estimates is smaller
    # than i_tol or max_it iterations have been performed
    while difference > i_tol and it < max_it:

        # Build surrogate log-likelihood (defined in Section 2.1 of ref [1])
        f_fun = f_fun_bld(sorted_data, p_est_it_m1, b_rate, d_rate, likelihood,
                          technique, idx_known_p, known_p, model, z_trunc,
                          j_tol, h_tol, options)[0]

        # Negate the surrogate log-likelihood so that scipy.optimize.minimize()
        # Can be used to find a maximum
        def error_fun(p_prop):
            return -f_fun(p_prop)

        # Find a zero of an EM step (defined as $\tilde g(\theta) at start of Section 2.2
        # of reference [1])
        g_tilde = ut.minimize_(error_fun, p_est_it_m1, p_bounds, con,
                               opt_method, options).x - p_est_it_m1

        # Reset the algorithm whenever the iteration index is an integer multiple of p.
        # This is discussed at the end of Section 2 in reference [2].
        if (it % p0.size) == 0:
            direction = g_tilde
        else:
            # Jacobian of log-likelihood at previous parameter estimate
            J_ell_it_m1 = ut.Jacobian(ll, p_est_it_m1, p_bounds)
            # Jacobian of log-likelihood at parameter estimate before previous parameter
            # estimate
            J_ell_it_m2 = ut.Jacobian(ll, p_est_it_m2, p_bounds)

            # Store current direction since new direction is in terms of current direction
            previous_direction = direction

            # Helps us to choose direction to change parameters (Equation (6) in reference [2])
            beta = np.divide(np.inner(g_tilde,
                                      np.subtract(J_ell_it_m1, J_ell_it_m2)),
                             np.inner(previous_direction,
                                      np.subtract(J_ell_it_m1, J_ell_it_m2)))
            # Tells us direction to change parameters in (Equation (7) in reference [2])
            direction = g_tilde - np.multiply(beta, previous_direction)

        # Need to store this since each iteration depends on previous 2 iterations (except
        # when a reset happens)
        p_est_it_m2 = p_est_it_m1

        # Tells us how much to change parameters by (Equation (4) in reference [2])
        a_opt = utem.line_search_ll(direction, data, p_est_it_m1, likelihood,
                                    known_p, idx_known_p, model, z_trunc,
                                    p_bounds, con, opt_method, options)[0]

        # Update parameter estimate
        p_est_it_m1 = p_est_it_m1 + np.multiply(a_opt, direction)

        # Compute difference to determine if should terminate
        difference = np.sum(np.abs(p_est_it_m1 - p_est_it_m2))
        # Store estimate
        iterations.append(p_est_it_m1)

        # Update iteration number
        it += 1
        # Print a progress update if requested
        if display:
            print('Iteration ', it, ' estimate is: ', p_est_it_m1)

    return list(p_est_it_m1), np.array(iterations)
