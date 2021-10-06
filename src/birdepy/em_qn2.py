import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy.interface_dnm import bld_ll_fun
import birdepy.utility_em as utem


def discrete_est_em_qn2(data, sorted_data, p0, likelihood, technique, known_p,
                        idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
                        con, max_it, i_tol, j_tol, h_tol, display, opt_method,
                        options):
    """
    Executes an expectation-maximization algorithm, which is accelerated using
    conjugate gradient ideas, to estimate parameters for a
    population-size-dependent birth-and-death process.

    This function is called by :func:`bd.interface_aug.f_fun_bld()`.

    """
    #Initialize a list to return parameters in each iteration and store the initial parameters in it
    iterations = [p0]

    it = 1
    difference = i_tol + 1
    p_est = p0

    # Build the log-likelihood function assuming all parameters are known
    pre_ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    # Negate the surrogate log-likelihood function so that scipy.optimize.minimize()
    # Can be used to find a maximum
    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    g = ut.Jacobian(ll, p_est, p_bounds)

    # Build surrogate log-likelihood function (defined in Section 2.1 of ref [1])
    f_fun = f_fun_bld(sorted_data, p_est, b_rate, d_rate, likelihood,
                      technique, idx_known_p, known_p, model, z_trunc,
                      j_tol, h_tol, options)[0]

    # Negate the surrogate log-likelihood function so that scipy.optimize.minimize()
    # Can be used to find a maximum
    def error_fun(p_prop):
        return -f_fun(p_prop)

    # Find a zero of an EM step (defined as $\tilde g(\theta) at start of Section 2.2
    # of reference [1])
    g_tilde = ut.minimize_(error_fun, p_est, p_bounds, con, opt_method,
                           options).x - p_est

    S = np.zeros((p0.size, p0.size))

    while difference > i_tol and it < max_it:

        # Step (a) of QN2 algorithm in reference [1]
        direction = g_tilde - np.matmul(S, g)

        # Step (b) of QN2 algorithm in reference [1]

        a_opt = utem.line_search_ll(direction, data, p_est, likelihood,
                                    known_p, idx_known_p, model, z_trunc,
                                    p_bounds, con, opt_method, options)[0]
        p_est_change = np.multiply(a_opt, direction)

        # First part of Step (c) of QN2 algorithm in reference [1]
        g_change = ut.Jacobian(ll, p_est + p_est_change, p_bounds) - g

        # Build surrogate log-likelihood function (defined in Section 2.1 of ref [1])
        f_fun = f_fun_bld(sorted_data, p_est + p_est_change, b_rate, d_rate,
                          likelihood, technique, idx_known_p, known_p, model,
                          z_trunc, j_tol, h_tol, options)[0]

        # Negate the surrogate log-likelihood function so that scipy.optimize.minimize()
        # Can be used to find a maximum
        def error_fun(p_prop):
            return -f_fun(p_prop)

        # Second part of Step (c) of QN2 algorithm in reference [1]
        g_tilde_change = ut.minimize_(error_fun, p_est, p_bounds, con,
                                      opt_method, options).x\
                         - p_est - p_est_change - g_tilde

        # First part of Step (d) of QN2 algorithm in reference [1]
        p_est_star_change = -g_tilde_change + np.matmul(S, g_change)

        # These are all part of Equation (4.1) in reference [1]
        s1 = (np.dot(g_change, p_est_star_change) /
              np.dot(g_change, p_est_change))
        s2 = np.dot(g_change, p_est_change)
        S1 = np.matmul(p_est_change, np.transpose(p_est_change))
        S2 = np.matmul(p_est_star_change, np.transpose(p_est_change))

        # Equation (4.1) (second part of Step (d) of QN2 algorithm) in reference [1]
        S_change = S1 * (1 + s1) / s2 - (S2 + np.transpose(S2)) / s2

        # Step (e) of QN2 algorithm in reference [1]
        g += g_change
        g_tilde += g_tilde_change
        S += S_change

        p_est += p_est_change

        # Compute the difference between this iteration estimate and
        # previous iteration estimate
        difference = np.sum(np.abs(p_est_change))
        # Store the estimate
        iterations.append(p_est)

        # Print a progress update if requested
        if display:
            print('Iteration ', it, ' estimate is: ', p_est)
        it += 1

    return list(p_est), np.array(iterations)
