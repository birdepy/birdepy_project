from birdepy.interface_dnm import bld_ll_fun as bld_ll_fun
import functools
import numpy as np
import birdepy.utility as ut


def line_search_ll(direction, data, p0, likelihood, known_p,
                   idx_known_p, model, z_trunc, p_bounds, con, opt_method,
                   options):

    def upper_bound(a, index):
        return p_bounds[index][1] - p0[index] - a * direction[index]

    def lower_bound(a, index):
        return p0[index] + a * direction[index] - p_bounds[index][0]

    new_con = []
    if type(con) == dict:
        # If there is one constraint specified
        new_con.append({'type': con['type'],
                        'fun': lambda a:
                        con['fun'](p0 + np.multiply(a, direction))})
    else:
        # If there is strictly more than one or 0 constraints specified
        for c in con:
            new_con.append({'type': c['type'],
                            'fun': lambda a:
                            c['fun'](p0 + np.multiply(a, direction))})

    for idx in range(len(p_bounds)):
        new_con.append(
            {'type': 'ineq', 'fun': functools.partial(upper_bound, index=idx)})
        new_con.append(
            {'type': 'ineq', 'fun': functools.partial(lower_bound, index=idx)})

    new_con = tuple(new_con)

    ll = bld_ll_fun(data, likelihood, model, z_trunc, options)

    def error_fun(a):
        p_prop = p0 + np.multiply(a, direction)
        param = ut.p_bld(p_prop, idx_known_p, known_p)
        return -ll(param)

    a_opt = ut.minimize_(error_fun, 1e-6, [], new_con, opt_method, options).x

    return a_opt  # p0 + np.multiply(a_opt, direction)


def help_bld_aug(udh, sorted_data, j_tol, h_tol, z_trunc):
    z_min, z_max = z_trunc
    aug_data = {}
    for t in sorted_data:
        for zz in sorted_data[t]:
            z0 = zz[0]
            zt = zz[1]
            lower_z = min(z0, zt)
            higher_z = max(z0, zt)
            for z in range(max(lower_z, z_min + 1), higher_z):
                udh_ = udh(z0, zt, t, z)
                if z in aug_data:
                    aug_data[z][:] += sorted_data[t][zz]*udh_
                else:
                    aug_data[z] = sorted_data[t][zz]*udh_
            z = higher_z
            while z < z_max:
                udh_ = udh(z0, zt, t, z)
                if z in aug_data:
                    aug_data[z][:] += sorted_data[t][zz]*udh_
                else:
                    aug_data[z] = sorted_data[t][zz]*udh_
                if udh_[0] < j_tol and udh_[1] < j_tol and udh_[2] < h_tol:
                    break
                else:
                    z += 1
            z = lower_z - 1
            while z > z_min:
                udh_ = udh(z0, zt, t, z)
                if z in aug_data:
                    aug_data[z][:] += sorted_data[t][zz]*udh_
                else:
                    aug_data[z] = sorted_data[t][zz]*udh_
                if udh_[0] < j_tol and udh_[1] < j_tol and udh_[2] < h_tol:
                    break
                else:
                    z -= 1
    return aug_data


def help_bld_aug_2(udh, z0, zt, aug_data, j_tol, h_tol, z_trunc):
    z_min, z_max = z_trunc
    lower_z = min(z0, zt)
    higher_z = max(z0, zt)
    for z in range(max(lower_z, z_min + 1), higher_z):
        udh_ = udh(z)
        if z in aug_data:
            aug_data[z][:] += udh_
        else:
            aug_data[z] = udh_
    z = higher_z
    while z < z_max:
        udh_ = udh(z)
        if z in aug_data:
            aug_data[z][:] += udh_
        else:
            aug_data[z] = udh_
        if udh_[0] < j_tol and udh_[1] < j_tol and udh_[2] < h_tol:
            break
        else:
            z += 1
    z = lower_z - 1
    while z > z_min:
        udh_ = udh(z)
        if z in aug_data:
            aug_data[z][:] += udh_
        else:
            aug_data[z] = udh_
        if udh_[0] < j_tol and udh_[1] < j_tol and udh_[2] < h_tol:
            break
        else:
            z -= 1
    return aug_data

