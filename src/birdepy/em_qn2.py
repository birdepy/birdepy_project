import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy.interface_dnm import bld_ll_fun
import birdepy.utility_em as utem


def discrete_est_em_qn2(data, sorted_data, p0, likelihood, technique, known_p,
                        idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
                        con, max_it, i_tol, j_tol, h_tol, display, opt_method,
                        options):
    iterations = [p0]

    it = 1
    difference = i_tol + 1
    p_est = p0

    pre_ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    J_ell = ut.Jacobian(ll, p_est, p_bounds)

    f_fun = f_fun_bld(sorted_data, p_est, b_rate, d_rate, likelihood,
                      technique, idx_known_p, known_p, model, z_trunc,
                      j_tol, h_tol, options)[0]

    def error_fun(p_prop):
        return -f_fun(p_prop)

    g_tilde = ut.minimize_(error_fun, p_est, p_bounds, con, opt_method,
                           options).x - p_est

    S = np.zeros((p0.size, p0.size))

    while difference > i_tol and it < max_it:

        direction = g_tilde - np.matmul(S, J_ell)

        a_opt = utem.line_search_ll(direction, data, p_est, likelihood,
                                    known_p, idx_known_p, model, z_trunc,
                                    p_bounds, con, opt_method, options)[0]

        p_est_change = np.multiply(a_opt, direction)

        J_change = ut.Jacobian(ll, p_est + p_est_change, p_bounds) - J_ell

        f_fun = f_fun_bld(sorted_data, p_est + p_est_change, b_rate, d_rate,
                          likelihood, technique, idx_known_p, known_p, model,
                          z_trunc, j_tol, h_tol, options)[0]

        def error_fun(p_prop):
            return -f_fun(p_prop)

        g_tilchange = ut.minimize_(error_fun, p_est, p_bounds, con,
                                      opt_method, options).x\
                         - p_est - p_est_change - g_tilde

        p_est_star_change = -g_tilchange + np.matmul(S, J_change)

        s1 = (np.dot(J_change, p_est_star_change) /
              np.dot(J_change, p_est_change))

        s2 = np.dot(J_change, p_est_change)

        S1 = np.matmul(p_est_change, np.transpose(p_est_change))

        S2 = np.matmul(p_est_star_change, np.transpose(p_est_change))

        S_change = S1 * (1 + s1) / s2 - (S2 + np.transpose(S2)) / s2

        J_ell += J_change

        g_tilde += g_tilchange

        S += S_change

        p_est += p_est_change

        difference = np.sum(np.abs(p_est_change))
        iterations.append(p_est)

        if display:
            print('Iteration ', it, ' estimate is: ', p_est)
        it += 1

    return list(p_est), np.array(iterations)
