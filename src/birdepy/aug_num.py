import numpy as np
from birdepy.interface_probability import probability
import birdepy.utility_em as utem
import birdepy.utility as ut
import scipy.signal


def aug_bld_num(sorted_data, param, b_rate, d_rate, likelihood, model, z_trunc,
                j_tol, h_tol, options):
    """
    Creates a function based on numerical integration that returns the
    expected values of the number of up jumps from a state z, the number
    of down jumps from z, and the time spent in z when the process
    transitions from z0 to zt in elapsed time t (for any t that is a key
    in the dictionary `sorted_data`).
    """
    if 'r' in options.keys():
        r = options['r']
    else:
        r = 10

    z_min, z_max = z_trunc

    states = np.arange(z_min, z_max+1, 1)

    p_mats = {}
    for _t in sorted_data:
        t_linspace = np.linspace(0, _t, 2 ** r + 1)
        output = np.zeros((t_linspace.size, states.size, states.size))
        p_mat_1 = probability(states, states, t_linspace[1], param, model,
                              likelihood, z_trunc=z_trunc, options=options)

        np.nan_to_num(p_mat_1, copy=False, nan=0)
        for row_idx in range(states.size):
            p_mat_1[row_idx, :] = scipy.signal.medfilt(p_mat_1[row_idx, :])
            row_sum = sum(p_mat_1[row_idx, :])
            if row_sum > 0:
                p_mat_1[row_idx, :] = p_mat_1[row_idx, :]/row_sum
            else:
                p_mat_1[row_idx, :] = 1/states.size
        output[0, :, :] = p_mat_1[np.ix_(np.array(states - z_min, dtype=np.int32),
                                         np.array(states - z_min, dtype=np.int32))]
        p_mat_power = p_mat_1
        for idx in range(1, t_linspace.size):
            p_mat_power = np.matmul(p_mat_power, p_mat_1)
            output[idx, :, :] = p_mat_power[np.ix_(np.array(states - z_min, dtype=np.int32),
                                            np.array(states - z_min, dtype=np.int32))]
        p_mats[_t] = output


    def udh(z0, zt, t, z):
        t_linspace_ = np.linspace(0, t, 2 ** r + 1)
        pr = p_mats[t][-1, z0 - z_min, zt - z_min]
        probs1 = p_mats[t][:, z0 - z_min, z - z_min]
        probs2 = np.flip(p_mats[t][:, z - z_min + 1, zt - z_min])
        u = ut.trap_int(b_rate(z, param) * np.multiply(probs1, probs2) / pr,
                        t_linspace_)[-1]
        probs3 = np.flip(p_mats[t][:, z - z_min - 1, zt - z_min])
        d = ut.trap_int(d_rate(z, param) * np.multiply(probs1, probs3) / pr,
                        t_linspace_)[-1]
        probs4 = np.flip(p_mats[t][:, z - z_min, zt - z_min])
        h = ut.trap_int(np.multiply(probs1, probs4) / pr,
                        t_linspace_)[-1]
        return np.array([u, d, h])

    aug_data = utem.help_bld_aug(udh, sorted_data, j_tol, h_tol, z_trunc)

    return aug_data
