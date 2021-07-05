import numpy as np
import birdepy.utility as ut
from birdepy.interface_aug import f_fun_bld as f_fun_bld
from birdepy.interface_dnm import bld_ll_fun
import birdepy.utility_em as utem


def discrete_est_em_Lange(data, sorted_data, p0, likelihood, technique,
                          known_p,
                          idx_known_p, model, b_rate, d_rate, z_trunc,
                          p_bounds,
                          con, max_it, i_tol, j_tol, h_tol, display,
                          opt_method,
                          options):
    iterations = [p0]

    it = 1
    p_est_it_m2 = p0

    pre_ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    def ll(p_prop):
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return pre_ll(param)

    f_fun_it_m2 = f_fun_bld(sorted_data, p_est_it_m2, b_rate, d_rate,
                            likelihood, technique, idx_known_p, known_p, model,
                            z_trunc, j_tol, h_tol, options)[0]

    p_est_it_m1 = ut.minimize_(lambda p: -f_fun_it_m2(p), p0, p_bounds, con,
                               opt_method, options).x

    if display:
        print('Iteration ', it, ' estimate is: ', p_est_it_m1)
    it += 1

    B = np.zeros((p0.size, p0.size))

    diff_one = np.sum(np.abs(p_est_it_m1 - p_est_it_m2))
    chi = p_est_it_m1 - p_est_it_m2

    while diff_one > i_tol and it < max_it:
        f_fun_it_m1 = f_fun_bld(sorted_data, p_est_it_m1, b_rate, d_rate,
                                likelihood, technique, idx_known_p, known_p,
                                model, z_trunc, j_tol, h_tol, options)[0]

        H_f = ut.Hessian(f_fun_it_m1, p_est_it_m1, p_bounds)
        phi = ut.Jacobian(f_fun_it_m1, p_est_it_m2, p_bounds) - \
              ut.Jacobian(f_fun_it_m2, p_est_it_m2, p_bounds)

        eta = phi - np.matmul(B, chi)

        psi = 1 / np.inner(phi - np.matmul(B, chi), chi)

        B = B + psi * np.outer(eta, eta)

        zeta = 1
        inside = H_f - np.multiply(0.5 ** zeta, B)
        while not np.all(np.linalg.eigvals(inside) < 0):
            zeta += 1
            inside = H_f - np.multiply(0.5 ** zeta, B)

        J_ell = ut.Jacobian(ll, p_est_it_m1, p_bounds)

        direction = -np.matmul(np.linalg.inv(inside), J_ell)

        p_est_it_m2 = p_est_it_m1

        a_opt = utem.line_search_ll(direction, data, p_est_it_m1, likelihood,
                                    known_p, idx_known_p, model, z_trunc,
                                    p_bounds, con, opt_method, options)[0]

        p_est_it_m1 = p_est_it_m1 + np.multiply(a_opt, direction)

        for idx in range(p_est_it_m1.size):
            p_est_it_m1[idx] = min(p_bounds[idx][1],
                                   max(p_bounds[idx][0],
                                       p_est_it_m1[idx]))

        iterations.append(p_est_it_m1)

        diff_one = np.sum(np.abs(p_est_it_m1 - p_est_it_m2))
        chi = p_est_it_m1 - p_est_it_m2
        if display:
            print('Iteration ', it, ' estimate is: ', p_est_it_m1)
        it += 1

    return list(p_est_it_m1), np.array(iterations)
