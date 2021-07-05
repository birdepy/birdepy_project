import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld


def discrete_est_em_qn1(sorted_data, p0, likelihood, technique, known_p,
                        idx_known_p, model, b_rate, d_rate, z_trunc, p_bounds,
                        con, max_it, i_tol, j_tol, h_tol, display, opt_method,
                        options):
    iterations = [p0]

    it = 1
    difference = i_tol + 1
    p_est = p0

    f_fun = f_fun_bld(sorted_data, p_est, b_rate, d_rate, likelihood,
                      technique, idx_known_p, known_p, model, z_trunc,
                      j_tol, h_tol, options)[0]

    def error_fun(p_prop):
        return -f_fun(p_prop)

    g_tilde = ut.minimize_(error_fun, p_est, p_bounds, con, opt_method,
                           options).x - p_est

    A = -np.identity(p0.size)

    while difference > i_tol and it < max_it:

        p_est_change_0 = np.matmul(-A, g_tilde)

        zeta = 0
        feasible = False
        while not feasible:
            zeta += 1
            p_est_change = np.multiply(0.5 ** zeta, p_est_change_0)
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

        f_fun = f_fun_bld(sorted_data, p_est_next, b_rate, d_rate,
                          likelihood, technique, idx_known_p, known_p, model,
                          z_trunc, j_tol, h_tol, options)[0]

        def error_fun(p_prop):
            return -f_fun(p_prop)

        g_tilchange = ut.minimize_(error_fun, p_est_next, p_bounds,
                                      con, opt_method, options).x \
                         - p_est_next - g_tilde

        A = A + np.outer((p_est_change - np.matmul(A, g_tilchange)),
                          np.matmul(p_est_change, A)) / \
            np.inner(np.matmul(p_est_change, A), g_tilchange)

        g_tilde += g_tilchange

        difference = np.sum(np.abs(p_est_next - p_est))
        p_est = p_est_next
        iterations.append(p_est)

        if display:
            print('Iteration ', it, ' estimate is: ', p_est)
        it += 1

    return list(p_est), np.array(iterations)
