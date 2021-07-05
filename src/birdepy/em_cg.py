import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy.interface_dnm import bld_ll_fun
import birdepy.utility_em as utem


def discrete_est_em_cg(data, sorted_data, p0, likelihood, technique, known_p,
                       idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
                       con, max_it, i_tol, j_tol, h_tol, display, opt_method,
                       options):

    iterations = [p0]

    pre_ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    it = 0
    p_est_it_m1 = p0
    difference = i_tol + 1
    while difference > i_tol and it < max_it:

        f_fun = f_fun_bld(sorted_data, p_est_it_m1, b_rate, d_rate, likelihood,
                          technique, idx_known_p, known_p, model, z_trunc,
                          j_tol, h_tol, options)[0]

        def error_fun(p_prop):
            return -f_fun(p_prop)

        g_tilde = ut.minimize_(error_fun, p_est_it_m1, p_bounds, con,
                               opt_method, options).x - p_est_it_m1

        if (it % p0.size) == 0:
            direction = g_tilde
        else:
            J_ell_it_m1 = ut.Jacobian(ll, p_est_it_m1, p_bounds)
            J_ell_it_m2 = ut.Jacobian(ll, p_est_it_m2, p_bounds)

            previous_direction = direction

            beta = np.divide(np.inner(g_tilde,
                                      np.subtract(J_ell_it_m1, J_ell_it_m2)),
                             np.inner(previous_direction,
                                      np.subtract(J_ell_it_m1, J_ell_it_m2)))
            direction = g_tilde - np.multiply(beta, previous_direction)

        p_est_it_m2 = p_est_it_m1

        a_opt = utem.line_search_ll(direction, data, p_est_it_m1, likelihood,
                                    known_p, idx_known_p, model, z_trunc,
                                    p_bounds, con, opt_method, options)[0]

        p_est_it_m1 = p_est_it_m1 + np.multiply(a_opt, direction)

        difference = np.sum(np.abs(p_est_it_m1 - p_est_it_m2))
        iterations.append(p_est_it_m1)

        it += 1
        if display:
            print('Iteration ', it, ' estimate is: ', p_est_it_m1)


    return list(p_est_it_m1), np.array(iterations)
