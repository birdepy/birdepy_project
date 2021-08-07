import numpy as np
import birdepy.utility as ut
from birdepy.interface_probability import probability
from scipy.linalg import expm
import birdepy.utility_em as ut_em


def aug_bld_expm(sorted_data, param, b_rate, d_rate, likelihood, model, z_trunc,
                 j_tol, h_tol, options):
    """
    Creates a function based on matrix exponentials that returns the
    expected values of the number of up jumps from a state z, the number
    of down jumps from z, and the time spent in z when the process
    transitions from z0 to zt in elapsed time t (for any t that is a key
    in the dictionary `sorted_data`).
    """

    z_min, z_max = z_trunc

    q_mat = ut.q_mat_bld(z_min, z_max, param, b_rate, d_rate)

    num_states = int(z_max - z_min + 1)

    zero_mat = np.zeros((num_states, num_states))
    c_mat = np.vstack((np.hstack((q_mat, zero_mat)),
                       np.hstack((zero_mat, q_mat))))

    aug_data = {}
    for t in sorted_data:
        if likelihood in ['Erlang', 'expm', 'uniform']:
            p_mat = probability(np.arange(z_min, z_max + 1, 1),
                                np.arange(z_min, z_max + 1, 1), t, param,
                                method=likelihood, model=model,
                                z_trunc=z_trunc, options=options)
        for zz in sorted_data[t]:
            z0 = zz[0]
            zt = zz[1]
            if likelihood in ['Erlang', 'expm', 'uniform']:
                pr = p_mat[z0 - z_min, zt - z_min]
            elif likelihood in ['da', 'gwa', 'gwasa', 'ilt', 'oua']:
                pr = probability(z0, zt, t, param, model, likelihood,
                                 z_trunc=z_trunc, options=options)[0][0]
            c_mat[zt - z_min, num_states + z0 - z_min] = 1
            conv_mat = expm(c_mat*t)[0:num_states, num_states:]
            c_mat[zt - z_min, num_states + z0 - z_min] = 0  # Clean up for reuse

            def udh(z_):
                u = b_rate(z_, param) * \
                    conv_mat[z_ + 1 - z_min, z_ - z_min] / pr
                d = d_rate(z_, param) * \
                    conv_mat[z_ - 1 - z_min, z_ - z_min] / pr
                h = conv_mat[z_ - z_min, z_ - z_min] / pr
                return sorted_data[t][zz]*np.array([u, d, h])

            aug_data = ut_em.help_bld_aug_2(udh, z0, zt, aug_data, j_tol, h_tol,
                                            z_trunc)
    return aug_data
