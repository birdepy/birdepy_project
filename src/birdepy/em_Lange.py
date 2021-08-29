import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy.interface_dnm import bld_ll_fun
import birdepy.utility_em as utem


def discrete_est_em_Lange(data, sorted_data, p0, likelihood, technique,
                          known_p, idx_known_p, model, b_rate, d_rate, z_trunc,
                          p_bounds, con, max_it, i_tol, j_tol, h_tol, display,
                          opt_method, options):
    """
    Executes an expectation-maximization algorithm, which is accelerated using
    conjugate gradient ideas, to estimate parameters for a
    population-size-dependent birth-and-death process.

    This function is called by :func:`bd.interface_aug.f_fun_bld()`.

    [1] Lange, K. (1995). A quasi-Newton acceleration of the EM algorithm.
     Statistica sinica, 1-18.
    """
    #Initialize a list to return parameters in each iteration and store the initial parameters in it
    iterations = [p0]

    it = 1
    p_est_it_m2 = p0

    # Build the log-likelihood function assuming all parameters are known
    pre_ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    # Adjust the log-likelihood function to account for unknown parameters and
    # negate it so that the minimize function performs maximisation.
    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    # Build surrogate log-likelihood at initial (defined in Section 2 of ref [1]).
    f_fun_it_m2 = f_fun_bld(sorted_data, p_est_it_m2, b_rate, d_rate,
                            likelihood, technique, idx_known_p, known_p, model,
                            z_trunc, j_tol, h_tol, options)[0]
    # Optimize surrogate log-likelihood to find an estimate.
    p_est_it_m1 = ut.minimize_(lambda p: -f_fun_it_m2(p), p0, p_bounds, con,
                               opt_method, options).x

    # Print a progress update if requested
    if display:
        print('Iteration ', it, ' estimate is: ', p_est_it_m1)
    it += 1

    # Initial approximation to the H quantity in Equation (3) of reference [1].
    # This choice is discussed after Equation (8) in reference [1].
    B = np.zeros((p0.size, p0.size))

    # Compute difference to determine if should terminate.
    diff_one = np.sum(np.abs(p_est_it_m1 - p_est_it_m2))
    # This is s as defined by Equation (9) in reference [1].
    s = p_est_it_m1 - p_est_it_m2

    while diff_one > i_tol and it < max_it:
        # Surrogate log-likelihood
        f_fun_it_m1 = f_fun_bld(sorted_data, p_est_it_m1, b_rate, d_rate,
                                likelihood, technique, idx_known_p, known_p,
                                model, z_trunc, j_tol, h_tol, options)[0]

        # Hessian of surrogate log-likelihood (Equation (5) in reference [1]).
        H_f = ut.Hessian(f_fun_it_m1, p_est_it_m1, p_bounds)

        # This is g as defined by Equation (9) in reference [1].
        g = ut.Jacobian(f_fun_it_m1, p_est_it_m2, p_bounds) - \
              ut.Jacobian(f_fun_it_m2, p_est_it_m2, p_bounds)

        # This is c and v as defined by Equation (11) in reference [1].
        c = 1 / np.inner(g - np.matmul(B, s), s)
        v = g - np.matmul(B, s)

        # Update approximation to the H quantity in Equation (3) of reference [1].
        # This is Equation (10) in reference [1].
        B = B + c * np.outer(v, v)

        # Modify approximation to ensure it is negative definite (as discussed
        # at the end of Section 3 of reference [1])
        m = 1
        H_f_mod = H_f - np.multiply(0.5 ** m, B)
        while not np.all(np.linalg.eigvals(H_f_mod) < 0):
            m += 1
            H_f_mod = H_f - np.multiply(0.5 ** m, B)

        # Jacobian of log-likelihood helps determine direction
        J_ell = ut.Jacobian(ll, p_est_it_m1, p_bounds)

        # Determine direction to change parameters (Equation (12) of reference [1]).
        direction = -np.matmul(np.linalg.inv(H_f_mod), J_ell)

        # Need to store this since each iteration depends on previous 2 iterations
        p_est_it_m2 = p_est_it_m1

        # Tells us how much to change parameters by (Equation (4) in reference [2])
        a_opt = utem.line_search_ll(direction, data, p_est_it_m1, likelihood,
                                    known_p, idx_known_p, model, z_trunc,
                                    p_bounds, con, opt_method, options)[0]

        p_est_it_m1 = p_est_it_m1 + np.multiply(a_opt, direction)

        # Ensure estimate satisfied bounds
        for idx in range(p_est_it_m1.size):
            p_est_it_m1[idx] = min(p_bounds[idx][1],
                                   max(p_bounds[idx][0],
                                       p_est_it_m1[idx]))

        # Store estimate
        iterations.append(p_est_it_m1)

        # Compute difference to determine if should terminate
        diff_one = np.sum(np.abs(p_est_it_m1 - p_est_it_m2))
        # This is s as defined by Equation (9) in reference [1]
        s = p_est_it_m1 - p_est_it_m2
        if display:
            print('Iteration ', it, ' estimate is: ', p_est_it_m1)
        it += 1

    return list(p_est_it_m1), np.array(iterations)
