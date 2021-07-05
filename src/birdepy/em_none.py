import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld


def discrete_est_em_none(sorted_data, p0, likelihood, technique, known_p,
                         idx_known_p, model, b_rate, d_rate, z_trunc,
                         p_bounds, con, max_it, i_tol, j_tol, h_tol, display,
                         opt_method, options):
    iterations = [p0.tolist()]
    it = 1
    p_est = p0
    difference = i_tol + 1
    while difference > i_tol and it < max_it:
        f_fun = f_fun_bld(sorted_data, p_est, b_rate, d_rate, likelihood,
                          technique, idx_known_p, known_p, model, z_trunc,
                          j_tol, h_tol, options)[0]

        def error_fun(p_prop):
            return -f_fun(p_prop)

        opt = ut.minimize_(error_fun, p_est, p_bounds, con, opt_method,
                           options)

        p_est_next = opt.x

        difference = np.sum(np.abs(p_est_next - p_est))
        p_est = p_est_next
        iterations.append(p_est.tolist())
        if display:
            print('Iteration ', it, ' estimate is: ', p_est)
        it += 1

    return list(p_est), iterations
