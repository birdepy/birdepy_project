import numpy as np
import mpmath as mp
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import multivariate_normal
import math
from birdepy.iltcme import IltCmeParams
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from gwr_inversion import gwr


def Jacobian(fun, x, bounds):
    bounds = [[b[0] for b in bounds], [b[1] for b in bounds]]
    return approx_derivative(fun, x, bounds=bounds)


def Hessian(fun, x, bounds):
    bounds = [[b[0] for b in bounds], [b[1] for b in bounds]]
    return approx_derivative(lambda y: approx_derivative(fun,
                                                         y,
                                                         bounds=bounds),
                             x, bounds=bounds)


def confidence_region(mean, cov, obs, se_type, xlabel, ylabel, export):
    [Eval, Evec] = np.linalg.eig(cov)
    Evec = -np.hstack((Evec[:, 1].reshape((2, 1)), Evec[:, 0].reshape((2, 1))))
    xCenter = mean[0]
    yCenter = mean[1]
    theta = np.linspace(0, 2 * np.pi, 100)
    x_vec = np.array([1, 0])
    cosrotation = np.dot(x_vec, Evec[:, 1]) / \
                  (np.linalg.norm(x_vec) * np.linalg.norm(Evec[:, 1]))
    rotation = np.pi / 2 - np.arccos(cosrotation)

    R = np.array([[np.sin(rotation), np.cos(rotation)],
                  [-np.cos(rotation), np.sin(rotation)]])

    chisq = [1.368, 3.2188, 5.991]

    x = np.empty((len(theta), len(chisq)))
    y = np.empty((len(theta), len(chisq)))
    xRadius = np.empty(len(chisq))
    yRadius = np.empty(len(chisq))

    x_plot = np.empty((len(theta), len(chisq)))
    y_plot = np.empty((len(theta), len(chisq)))

    rotated_Coords = np.empty((2, len(theta), len(chisq)))

    for i in range(len(chisq)):
        xRadius[i] = np.sqrt(chisq[i] * Eval[0])
        yRadius[i] = np.sqrt(chisq[i] * Eval[1])

        x[:, i] = xRadius[i] * np.cos(theta)
        y[:, i] = yRadius[i] * np.sin(theta)

        rotated_Coords[:, :, i] = np.matmul(R, np.row_stack((x[:, i], y[:, i])))

        x_plot[:, i] = rotated_Coords[0, :, i].T + xCenter
        y_plot[:, i] = rotated_Coords[1, :, i].T + yCenter

    fig, (ax) = plt.subplots(figsize=(7, 5))

    labels = ['$50\%$', '$80\%$', '$95\%$']
    linestyles = [':', '--', '-']

    for i in range(len(chisq)):
        ax.plot(x_plot[:, i], y_plot[:, i], label=labels[i],
                linestyle=linestyles[i], color='k')
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(mean[0], mean[1], '+k', label="Estimate")
    if obs is not None:
        ax.scatter(obs[:, 0], obs[:, 1], c="b", marker="^", s=5,
                   label="Simulated\n samples")
    if se_type == "asymptotic":
        ax.set_title("Asymptotic confidence region")
    elif se_type == "simulated":
        ax.set_title("Simulated confidence region")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.set_tight_layout({
        'pad': 0
    })
    if isinstance(export, str):
        import tikzplotlib
        tikzplotlib.save(export + ".tex")
    return


def data_sort(t_data, p_data):
    """
    Data sorter to improve the efficiency of the :func:`birdepy` package.
    Counts how many times each probability_ z0 -> zt occurs, where transitions
    occurring over different elapsed times are treated as non-equal.

    Parameters
    ----------
    t_data : array_like
        Observation times of birth-and-death process.
        If one trajectory is observed, then this is a list.
        If multiple trajectories are observed, then this is a list of lists
        where each list corresponds to a trajectory.

    p_data : array_like
        Observed populations of birth-and-death process at times in argument
        't_data'.
        If one trajectory is observed, then this is a list.
        If multiple trajectories are observed, then this is a list of lists
        where each list corresponds to a traj  ectory.

    Return
    -------
    data : collections.Counter object
        A dictionary with keys corresponding to observed transitions
        (z0, zt, t) and values corresponding to the number of times
        an observed transition occurs.

    """
    if type(t_data) is np.ndarray:
        t_data = t_data.tolist()
    if type(t_data[0]) is not list:
        t_data = [t_data]
    if type(p_data) == np.ndarray:
        p_data = p_data.tolist()
    if type(p_data[0]) is not list:
        p_data = [p_data]

    all_transitions = []
    for idx1 in range(len(p_data)):
        for idx2 in range(len(p_data[idx1]) - 1):
            all_transitions.append((p_data[idx1][idx2],
                                    p_data[idx1][idx2 + 1],
                                    t_data[idx1][idx2 + 1] -
                                    t_data[idx1][idx2]))
    return Counter(all_transitions)


def data_sort_2(data):
    """
    Further sorts output of :func:`birdepy._utility.data_sort`. Separates data
    into sub-collections where all transitions in any given sub-collection
    occur in the same elapsed time (rounded to 8 decimal places).

    Parameters
    ----------
    data : collections.Counter object
        A dictionary with keys corresponding to observed transitions
        (z0, zt, t) and values corresponding to the number of times
        a probability_ is observed.

    Return
    -------
    sorted_data : dict
        A dictionary with keys corresponding to inter-observation times and
        values containing dict objects with keys corresponding to observed
        (z0, zt) pairs and values corresponding to the number of times a
        pair is observed for the relevant inter-observation time. E.g.,
        {t0 : {(a, b): 3, (c, d) : 4}, t1 : {(e, f) : 1, (g, h) : 10}}.

    """
    sorted_data = {}
    for i in data:
        if np.around(i[2], 8) not in sorted_data:
            sorted_data[np.around(i[2], 8)] = {(i[0], i[1]): data[i]}
        else:
            sorted_data[np.around(i[2], 8)][(i[0], i[1])] = data[i]
    return sorted_data


def higher_birth(model):
    if model == 'Verhulst':
        return lambda z, p: ((p[2]*z) <= 1) * p[0] * (1 - p[2]*z) * z
    elif model == 'Ricker':
        return lambda z, p: p[0] * np.exp(-(p[2] * z)**p[3]) * z
    elif model == 'Hassell':
        return lambda z, p: \
            p[0] * z / (1 + p[2] * z) ** p[3]
    elif model == 'MS-S':
        return lambda z, p: \
            p[0] * z / (1 + (p[2] * z) ** p[3])
    elif model == 'Moran':
        return lambda z, p: (z <= p[4]) * (
            (p[4] - z) * (p[0] * z * (1 - p[2]) / p[4] + p[1] * (p[4] - z) * p[3] / p[4]) / p[4])
    elif model == 'pure-birth':
        return lambda z, p: p[0] * z
    elif model == 'pure-death':
        return lambda z, p: 0
    elif model == 'Poisson':
        return lambda z, p: p[0]
    elif model == 'linear':
        return lambda z, p: p[0] * z
    elif model == 'linear-migration':
        return lambda z, p: p[0] * z + p[2]
    elif model == 'M/M/1':
        return lambda z, p: p[0]
    elif model == 'M/M/inf':
        return lambda z, p: p[0]
    elif model == 'loss-system':
        return lambda z, p: (z < p[2]) * p[0]
    else:
        raise TypeError("Argument 'model' has an unknown value.")


def higher_death(model):
    if model == 'Verhulst':
        return lambda z, p: p[1] * (1 + p[3]*z) * z
    elif model in ['Ricker', 'Hassell',
                   'MS-S', 'loss-system', 'linear',
                   'linear-migration', 'M/M/inf']:
        return lambda z, p: p[1] * z
    elif model == 'Moran':
        return lambda z, p: (z <= p[4]) * (
            z * (p[2] * (p[4] - z) * (1 - p[3]) / p[4] +
                 p[0] * z * p[2] / p[4]) / p[4])
    elif model == 'M/M/1':
        return lambda z, p: (z > 0) * p[1]
    elif model in ['pure-birth', 'Poisson']:
        return lambda z, p: 0
    elif model == 'pure-death':
        return lambda z, p: p[0] * z
    else:
        raise TypeError("Argument 'model' has an unknown value.")


def higher_h_fun(model):
    if model == 'Verhulst':
        return lambda z, p: p[0]-p[1]-2*z*(p[0]*p[2]+p[1]*p[3])
    elif model == 'Ricker':
        return lambda z, p: p[0] * np.exp(-(p[2] * z)**p[3]) * (1 - p[3]*(p[2] * z)**p[3]) - p[1]
    elif model == 'Hassell':
        return lambda z, p: (p[0] * (1 + p[2]*z - p[3] * z)) / \
                            (1 + (p[2] * z)) ** p[3] - p[1]
    elif model == 'MS-S':
        return lambda z, p: (p[0] * (1 + (p[2] * z) ** p[3] * (1 - p[3]))) / \
                            (1 + (p[2] * z) ** p[3]) ** 2
    elif model == 'Moran':
        return lambda z, p: (z*p[0]*(p[4]*(1-p[2])-z) + p[1]*(z-[4])*(z-p[4]*p[3]))\
                            / p[4]**2
    elif model == 'linear':
        return lambda z, p: (p[0] - p[1]) * (z - z + 1)
    elif model == 'linear-migration':
        return lambda z, p: (p[0] - p[1]) * (z - z + 1)
    elif model == 'M/M/1':
        return lambda z, p: 0 * (z - z + 1)
    elif model == 'M/M/inf':
        return lambda z, p: -p[1] * (z - z + 1)
    elif model == 'pure-birth':
        return lambda z, p: p[0] * (z - z + 1)
    elif model == 'pure-death':
        return lambda z, p: -p[0] * (z - z + 1)
    elif model == 'Poisson':
        return lambda z, p: 0 * (z - z + 1)
    else:
        raise TypeError("Unknown model.")


def higher_zf_bld(model):
    if model == 'Verhulst':
        return lambda p: [0, (p[0]-p[1])/(p[2]*p[0] + p[1]*p[3])]
    elif model == 'Ricker':
        return lambda p: [0, ((np.log(p[0] / p[1]))**(1/p[3])) / p[2]]
    elif model == 'Hassell':
        return lambda p: [0, ((p[0] / p[1]) ** (1 / p[3]) - 1)/p[2]]
    elif model == 'MS-S':
        return lambda p: [0, ((p[0] / p[1] - 1) ** (1 / p[3]))/p[2]]
    elif model == 'Moran':
        return lambda p: [p[4]*(p[0]*(1-p[2])-p[1]*(1+p[3]) + np.sqrt((p[2]-1)**2 * p[0]**2 + 2 * p[0] * p[1]*(p[2]+p[3]+p[2]*p[3]-1) + p[1] ** 2 *(p[3]-1)**2))/(2*(p[0]-p[1])),
                          p[4]*(p[0]*(1-p[2])-p[1]*(1+p[3]) - np.sqrt((p[2]-1)**2 * p[0]**2 + 2 * p[0] * p[1]*(p[2]+p[3]+p[2]*p[3]-1) + p[1] ** 2 *(p[3]-1)**2))/(2*(p[0]-p[1]))]
    elif model == 'linear':
        return lambda p: [0]
    elif model == 'linear-migration':
        return lambda p: [-p[2] / (p[0] - p[1])]
    elif model == 'M/M/1':
        raise TypeError("It is not advised to use this function for the"
                        "M/M/1 model.")
    elif model == 'M/M/inf':
        return lambda p: [p[0] / p[1]]
    elif model == 'pure-birth':
        return lambda z, p: [p[0]]
    elif model == 'pure-death':
        return lambda z, p: [-p[0]]
    elif model == 'Poisson':
        return lambda z, p: [0]
    else:
        raise TypeError("Unknown model.")


def q_mat_bld(z_min, z_max, p, b_rate, d_rate):
    num_states = int(z_max - z_min + 1)
    states = np.arange(z_min, z_max + 1, 1)
    q_mat = np.zeros((num_states, num_states))
    q_mat[-1, -1] = -d_rate(z_max, p)
    q_mat[-1, -2] = d_rate(z_max, p)
    q_mat[0, 0] = -b_rate(z_min, p)
    q_mat[0, 1] = b_rate(z_min, p)
    for z in np.arange(1, num_states - 1, 1):
        _state = states[z]
        q_mat[z, z] = -b_rate(_state, p) - d_rate(_state, p)
        q_mat[z, z + 1] = b_rate(_state, p)
        q_mat[z, z - 1] = d_rate(_state, p)
    return q_mat


def p_bld(p_prop, idx_known_p, known_p):
    known_p = np.array(known_p)
    p_size = known_p.size + p_prop.size
    idx_known_p = list(idx_known_p)
    if len(known_p) != len(idx_known_p):
        raise TypeError("Argument 'idx_known_p' and argument 'known_p' must"
                        " be the same size.")
    idx_unknown_p = list(set(range(p_size)) - set(idx_known_p))
    p = np.empty(p_size)
    p[idx_known_p] = known_p
    p[idx_unknown_p] = p_prop
    return p


def laplace_invert_mexp(fun, t, max_fun_evals, method="cme"):
    if method == "cme":
        # find the most steep CME satisfying max_fun_evals
        params = IltCmeParams.params[0]
        for p in IltCmeParams.params:
            if p["cv2"] < params["cv2"] and (p["n"] + 1) <= max_fun_evals:
                params = p
        eta = np.concatenate(([params["c"]],
                              np.array(params["a"]) + 1j * np.array(
                                  params["b"]))) * params["mu1"]
        # print('eta:', eta)
        beta = np.concatenate(
            ([1], 1 + 1j * np.arange(1, params["n"] + 1)
             * params["omega"])) * params["mu1"]
        # print('beta:', beta)
    elif method == "euler":
        n_euler = math.floor((max_fun_evals - 1) / 2)
        eta = np.concatenate(
            ([0.5], np.ones(n_euler), np.zeros(n_euler - 1), [2 ** -n_euler]))
        logsum = np.cumsum(np.log(np.arange(1, n_euler + 1)))
        for k in range(1, n_euler):
            eta[2 * n_euler - k] = eta[2 * n_euler - k + 1] + math.exp(
                logsum[n_euler - 1] - n_euler * math.log(2.0) - logsum[k - 1] -
                logsum[n_euler - k - 1])
        k = np.arange(2 * n_euler + 1)
        beta = n_euler * math.log(10.0) / 3.0 + 1j * math.pi * k
        eta = (10 ** ((n_euler) / 3.0)) * (1 - (k % 2) * 2) * eta
    elif method == "gaver":
        if max_fun_evals % 2 == 1:
            max_fun_evals -= 1
        ndiv2 = int(max_fun_evals / 2)
        eta = np.zeros(max_fun_evals)
        beta = np.zeros(max_fun_evals)
        logsum = np.concatenate(
            ([0], np.cumsum(np.log(np.arange(1, max_fun_evals + 1)))))
        for k in range(1, max_fun_evals + 1):
            insisum = 0.0
            for j in range(math.floor((k + 1) / 2), min(k, ndiv2) + 1):
                insisum += math.exp(
                    (ndiv2 + 1) * math.log(j) - logsum[ndiv2 - j] + logsum[
                        2 * j] - 2 * logsum[j] - logsum[k - j] - logsum[
                        2 * j - k])
            eta[k - 1] = math.log(2.0) * (-1) ** (k + ndiv2) * insisum
            beta[k - 1] = k * math.log(2.0)
    else:
        raise TypeError('Unknown Laplace inversion method.')
    res = []
    for x in t:
        res.append(eta.dot([fun(b / x) for b in beta]).real / x)
    return res


def cme(fun, t, k):
    if 'cme_params' not in globals() or globals()['cme_k'] != k:
        global cme_params
        global cme_k
        global cme_eta
        global cme_beta
        cme_k = k
        cme_params = IltCmeParams.params[0]
        for p in IltCmeParams.params:
            if p['cv2'] < cme_params['cv2'] and (p['n'] + 1) <= cme_k:
                cme_params = p
        cme_eta = [cme_params['a'][idx] + 1j * cme_params['b'][idx] for idx in
                   range(cme_params['n'])]
        cme_eta.insert(0, cme_params['c'] + 1j * 0)
        cme_eta = mp.matrix(cme_eta).T * cme_params['mu1']
        cme_beta = [1 + 1j * idx * cme_params['omega'] for idx in
                    range(1, cme_params['n'] + 1)]
        cme_beta.insert(0, 1)
        cme_beta = mp.matrix(cme_beta) * cme_params['mu1']
    foo = mp.matrix([fun(mp.fdiv(b, t)) for b in cme_beta])
    foo = cme_eta * foo
    return float(mp.fdiv(mp.re(foo[0]), t))


def laplace_invert(fun, t, **options):
    if 'laplace_method' in options.keys():
        laplace_method = options['laplace_method']
    else:
        laplace_method = 'cme-mp'
    if 'k' in options.keys():
        k = options['k']
    else:
        k = 25
    if laplace_method == 'gwr':
        return float(gwr(fun, time=t, M=k))
    elif laplace_method in ['cme', 'euler', 'gaver']:
        return laplace_invert_mexp(fun, [t], k, method=laplace_method)[0]
    elif laplace_method in ['talbot', 'stehfest', 'dehoog']:
        return float(mp.invertlaplace(fun, t, method=laplace_method))
    elif laplace_method == 'cme-mp':
        return cme(fun, t, k)
    elif laplace_method == 'cme-talbot':
        if 'f_bounds' in options.keys():
            f_min = options['f_bounds'][0]
            f_max = options['f_bounds'][1]
        else:
            raise TypeError("When 'laplace_method' is 'cme-talbot', kwarg"
                            "'f_bounds' must be given as a list [f_min, f_max]."
                            "When method 'cme-mp' returns a value outside "
                            "these bounds, method 'talbot' is used instead. ")
        output = cme(fun, t, k)
        if f_min <= output <= f_max:
            return output
        else:
            return float(mp.invertlaplace(fun, t, method='talbot'))
    else:
        raise TypeError('Unknown Laplace inversion method.')


def add_options(options):
    if 'seed' not in options.keys():
        options['seed'] = np.random.default_rng()
    if 'strategy' not in options.keys():
        options['strategy'] = 'best1bin'
    if 'maxiter' not in options.keys():
        options['maxiter'] = 1000
    if 'popsize' not in options.keys():
        options['popsize'] = 15
    if 'tol' not in options.keys():
        options['tol'] = 0.01
    if 'mutation' not in options.keys():
        options['mutation'] = 0.5
    if 'recombination' not in options.keys():
        options['recombination'] = 0.7
    if 'callback' not in options.keys():
        options['callback'] = None
    if 'disp' not in options.keys():
        options['disp'] = False
    if 'polish' not in options.keys():
        options['polish'] = True
    if 'init' not in options.keys():
        options['init'] = 'latinhypercube'
    if 'atol' not in options.keys():
        options['atol'] = 0
    if 'updating' not in options.keys():
        options['updating'] = 'immediate'
    if 'jac' not in options.keys():
        options['jac'] = '2-point'
    if 'maxcor' not in options.keys():
        options['maxcor'] = 10
    if 'ftol' not in options.keys():
        options['ftol'] = 2.220446049250313e-09
    if 'gtol' not in options.keys():
        options['gtol'] = 1e-05
    if 'eps' not in options.keys():
        options['eps'] = None
    if 'maxfun' not in options.keys():
        options['maxfun'] = 15000
    if 'maxiter' not in options.keys():
        options['maxiter'] = 15000
    if 'iprint' not in options.keys():
        options['iprint'] = -1
    if 'callback' not in options.keys():
        options['callback'] = None
    if 'maxls' not in options.keys():
        options['maxls'] = 20
    if 'finite_diff_rel_step' not in options.keys():
        options['finite_diff_rel_step'] = None
    if 'ftol' not in options.keys():
        options['ftol'] = 1e-06
    if 'eps' not in options.keys():
        options['eps'] = 1.4901161193847656e-08
    if 'disp' not in options.keys():
        options['disp'] = False
    if 'maxiter' not in options.keys():
        options['maxiter'] = 100
    if 'finite_diff_rel_step' not in options.keys():
        options['finite_diff_rel_step'] = None
    if 'jac' not in options.keys():
        options['jac'] = None
    return options


def minimize_(error_fun, p0, p_bounds, con, opt_method, options):
    if opt_method == 'differential-evolution':
        if con != ():
            old_con = con
            con= (NonlinearConstraint(con['fun'], 0, np.inf))
        sol = differential_evolution(error_fun, p_bounds, args=(),
                                     strategy=options['strategy'],
                                     maxiter=options['maxiter'],
                                     popsize=options['popsize'],
                                     tol=options['tol'],
                                     mutation=options['mutation'],
                                     recombination=options[
                                         'recombination'],
                                     seed=options['seed'],
                                     callback=options['callback'],
                                     disp=options['disp'],
                                     polish=False,
                                     init=options['init'],
                                     atol=options['atol'],
                                     updating=options['updating'],
                                     constraints=con)
        if options['polish']:
            if con != ():
                sol = minimize(error_fun, sol.x, method='SLSQP', bounds=p_bounds,
                               constraints=old_con)
            else:
                sol = minimize(error_fun, sol.x, method='L-BFGS-B', bounds=p_bounds)
    elif opt_method == 'L-BFGS-B':
        sol = minimize(error_fun, p0, method=opt_method, bounds=p_bounds,
                       callback=options['callback'],
                       jac=options['jac'],
                       options={'maxcor': options['maxcor'],
                                'ftol': options['ftol'],
                                'gtol': options['gtol'],
                                'eps': options['eps'],
                                'maxfun': options['maxfun'],
                                'maxiter': options['maxiter'],
                                'iprint': options['iprint'],
                                'maxls': options['maxls'],
                                'finite_diff_rel_step':
                                    options['finite_diff_rel_step']})
    elif opt_method == 'SLSQP':
        sol = minimize(error_fun, p0, method=opt_method, bounds=p_bounds,
                       constraints=con,
                       jac=options['jac'],
                       options={'ftol': options['ftol'],
                                'eps': options['eps'],
                                'disp': options['disp'],
                                'maxiter': options['maxiter'],
                                'finite_diff_rel_step':
                                    options['finite_diff_rel_step']})
    else:
        sol = minimize(error_fun, p0, method=opt_method, bounds=p_bounds,
                       constraints=con)
    return sol


def trap_int(y, x):
    """
    Numerical integration of sampled data using the trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Function values.

    x : array_like
        Domain values.

    Return
    -------
    res : ndarray
        Cumulative integral of `y` at points in `x`.
    """
    x = np.array(x)
    y = np.array(y)
    res = np.hstack(([0], np.cumsum(np.multiply(0.5,
                                                np.multiply(np.diff(x),
                                                            np.add(y[1:],
                                                                   y[0:-1])))))
                    )
    return res
