import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld


def discrete_est_em_qn1(sorted_data, p0, likelihood, technique, known_p,
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

    """
    #Initialize a list to return parameters in each iteration and store the initial parameters in it
    iterations = [p0]

    it = 1
    # Set the difference to ensure at least one iteration occurs
    difference = i_tol + 1
    # Initilize estimate
    p_est = p0

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

    A = -np.identity(p0.size)

    while difference > i_tol and it < max_it:

        p_est_change_0 = np.matmul(-A, g_tilde)

        # First part of step (a) of QN1 algorithm in reference [1]
        # Ensure the next parameter estimate satisfies bounds and constraints
        m = 0
        feasible = False
        while not feasible:
            m += 1
            p_est_change = np.multiply(0.5 ** m, p_est_change_0)
            p_est_next = p_est + p_est_change
            feasible = True
            if type(con) == dict:
                if con['fun'](p_est_next) < 0:
                    feasible = False
            else:
                for c in con:
                    if c['fun'](p_est_next) < 0:
                        feasible = False
                        break
            if feasible:
                for idx, b in enumerate(p_bounds):
                    if p_est_next[idx] < b[0]:
                        feasible = False
                        break
                    if p_est_next[idx] > b[1]:
                        feasible = False
                        break

        # Negate the surrogate log-likelihood function so that scipy.optimize.minimize()
        # Can be used to find a maximum
        f_fun = f_fun_bld(sorted_data, p_est_next, b_rate, d_rate,
                          likelihood, technique, idx_known_p, known_p, model,
                          z_trunc, j_tol, h_tol, options)[0]

        # Negate the surrogate log-likelihood function so that scipy.optimize.minimize()
        # Can be used to find a maximum
        def error_fun(p_prop):
            return -f_fun(p_prop)

        # Second part of step (a) of QN1 algorithm in reference [1]
        g_tilde_change = ut.minimize_(error_fun, p_est_next, p_bounds,
                                      con, opt_method, options).x \
                         - p_est_next - g_tilde

        # Step (b) of QN1 algorithm in reference [1]
        A = A + np.outer((p_est_change - np.matmul(A, g_tilde_change)),
                          np.matmul(p_est_change, A)) / \
            np.inner(np.matmul(p_est_change, A), g_tilde_change)

        g_tilde += g_tilde_change

        # Compute the difference between this iteration estimate and
        # previous iteration estimate
        difference = np.sum(np.abs(p_est_next - p_est))
        # Prepare for next iteration
        p_est = p_est_next
        # Store the estimate
        iterations.append(p_est)

        # Print a progress update if requested
        if display:
            print('Iteration ', it, ' estimate is: ', p_est)
        it += 1

    return list(p_est), np.array(iterations)
