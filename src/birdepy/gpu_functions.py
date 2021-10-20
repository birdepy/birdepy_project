from __future__ import print_function, absolute_import
import numpy as np
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float64 as rand
from collections import Counter
import sys


def discrete(param, model, z0, t, k=1, survival=False, seed=None):
    """Simulation of continuous-time birth-and-death processes at time 't'
    using CUDA.

    Parameters
    ----------
    param : array_like
        The parameters governing the evolution of the birth-and-death
        process to be simulated.
        Array of real elements of size (m,), where ‘m’ is the number of
        parameters.
        These must be in the order given :ref:`here
        <Birth-and-death Processes>`).

    model : string
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes>`). Should be one of:

            - 'Verhulst' (default)
            - 'Ricker'
            - 'Hassell'
            - 'MS-S'
            - 'Moran'
            - 'pure-birth'
            - 'pure-death'
            - 'Poisson'
            - 'linear'
            - 'linear-migration'
            - 'M/M/1'
            - 'M/M/inf'
            - 'loss-system'
         Custom models are not available for this function. See :func:`birdepy.simulate.discrete()`
         for custom models.

    z0: int or callable
        The initial population size for each sample path.

    t : scalar
        The time at which the simulated birth-and-death is observed.

    k : int, optional
        The number of sample paths to be simulated.

    survival : bool, optional
        If set to True, then the simulated sample paths are conditioned to
        have a positive population size at the final observation time.  Since This
        can greatly increase computation time.

    seed : int, optional
        Seed for simulation.

    Return
    -------
    out : array_like
        A list containing sampled population size observations at
        time `t`, generated according to `model`.

    Examples
    --------
    Simulating 10 ** 8 sample paths of an M/M/inf queue with
    service rates 0.4 and arrival rate 0.2, with 10 items initially in the queue,
    observed at time 1.0: ::

        from birdepy import gpu_functions as bdg
        bdg.discrete([0.2, 0.4], 'M/M/inf', 10, 1.0, k=10**8)
                     array([8, 6, 3, ..., 8, 7, 9], dtype=int64)

    Notes
    -----
    This function requires a compatible Nvidia graphics processing unit and
    drivers to be installed.

    The packages `Numba` and `cudatoolkit` also need to be installed.

    If you use this function for published work, then please cite [1].

    Sample paths are generated using a discrete-event simulation algorithm.
    See, for example, Algorithm 5.8 in [2].

    For a text book treatment on the theory of birth-and-death processes
    see [3].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`

    References
    ----------
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067. 

    .. [2] Kroese, D.P., Taimre, T. and Botev, Z.I. (2013) Handbook of Monte
     Carlo methods. John Wiley & Sons.

    .. [3] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.
    """
    # Convert parameters to tuple as this is accepted by the numba/cuda functions
    param = tuple(param)
    threads_per_block = 1024
    # Determine the number of blocks such that the number of samples is at least k
    blocks = int(1 + k / threads_per_block)
    # Determine a seed for the random number generator
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(sys.maxsize)
    # Initialise the random number generator
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)
    out = np.zeros(threads_per_block * blocks, dtype=np.int64)
    # A different function is called depending on which model is of interest. The numba/cuda
    # functions do not accept functions as input (at time of writing) so we cannot pass birth and
    # death rate functions as we do for the CPU version of this code bd.simulate.discrete().
    if model == 'Verhulst':
        discrete_verhulst[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'Ricker':
        discrete_ricker[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'Hassell':
        discrete_hassell[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'MS-S':
        discrete_mss[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'Moran':
        discrete_moran[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'pure-birth':
        discrete_pb[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'pure-death':
        discrete_pd[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'Poisson':
        discrete_poisson[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'linear':
        discrete_linear[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'linear-migration':
        discrete_lm[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'M/M/1':
        discrete_mm1[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'M/M/inf':
        discrete_mminf[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    elif model == 'loss-system':
        discrete_loss[blocks, threads_per_block](out, rng_states, param, z0, t, survival)
        return out
    else:
        raise TypeError("Argument 'model' has an unknown value.")


def probability(z0, zt, t, param, model, k=10**6, seed=None):
    """Transition probabilities for continuous-time birth-and-death processes
    generated using Monte Carlo on a GPU.

    Parameters
    ----------
    z0 : array_like
        States of birth-and-death process at time 0.

    zt : array_like
        States of birth-and-death process at time(s) `t`.

    t : array_like
        Elapsed time(s) (if has size greater than 1, then must be increasing).

    param : array_like
        The parameters governing the evolution of the birth-and-death
        process.
        Array of real elements of size (n,), where ‘n’ is the number of
        parameters.
        These must be in the order given :ref:`here
        <Birth-and-death Processes>`).

    model : string, optional
        Model specifying birth and death rates of process (see :ref:`here
        <Birth-and-death Processes>`). Should be one of:

            - 'Verhulst' (default)
            - 'Ricker'
            - 'Hassell'
            - 'MS-S'
            - 'Moran'
            - 'pure-birth'
            - 'pure-death'
            - 'Poisson'
            - 'linear'
            - 'linear-migration'
            - 'MS-S'
            - 'M/M/inf'
            - 'loss-system'
         Custom models are not available for this function. See :func:`birdepy.probability()`
         for custom models.

    k: int, optional
        Minimum number of samples used to generate each probability estimate.
        (Actual number of samples will usually be higher due to the way memory
        is allocated on GPU.)
        The total number of samples used will be at least z0.size * k.

    seed : int, optional
        Seed for simulations.

    Return
    -------
    transition_probability : ndarray
        An array of transition probabilities. If t has size bigger than 1,
        then the first coordinate corresponds to `t`, the second coordinate
        corresponds to `z0` and the third coordinate corresponds to `zt`;
        for example if `z0=[1,3,5,10]`, `zt=[5,8]` and `t=[1,2,3]`, then
        `transition_probability[2,0,1]` corresponds to
        P(Z(3)=8 | Z(0)=1).
        If `t` has size 1 the first coordinate corresponds to `z0` and the second
        coordinate corresponds to `zt`.

    Examples
    --------
    Estimate transition probabilities for a Moran model: ::

        from birdepy import gpu_functions as bdg
        param = (210, 20, 0.002, 0, 100)
        t = 0.2
        z0 = [50, 60]
        zt = [55, 56, 57, 58, 59,60]
        bdg.probability(z0, zt, t, param, 'Moran', 10**6)


    Outputs: ::

        array([[3.09160e-02, 5.43120e-02, 8.09760e-02, 1.05968e-01, 1.23203e-01,1.27453e-01],
        [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.30000e-05]])

    Notes
    -----
    This function requires a compatible Nvidia graphics processing unit and
    drivers to be installed.

    The packages `Numba` and `cudatoolkit` also need to be installed.

    If you use this function for published work, then please cite [1].

    For a text book treatment on the theory of birth-and-death processes
    see [2].

    See also
    --------
    :func:`birdepy.estimate()` :func:`birdepy.probability()` :func:`birdepy.forecast()`

    :func:`birdepy.simulate.discrete()` :func:`birdepy.simulate.continuous()`


    References
    ----------
    .. [1] Hautphenne, S., & Patch, B. (2021). Birth-and-death Processes in Python:
     The BirDePy Package. arXiv preprint arXiv:2110.05067.

    .. [2] Feller, W. (1968) An introduction to probability theory and its
     applications (Volume 1) 3rd ed. John Wiley & Sons.

    """

    # Convert z0, zt, and t to common format used below
    if np.isscalar(z0):
        z0 = np.array([z0])
    else:
        z0 = np.array(z0)
    if np.isscalar(zt):
        zt = np.array([zt])
    else:
        zt = np.array(zt)
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = np.array(t)

    # Since bdg.discrete() can only accept a single time point we need to add an extra loop if more
    # than one time point is requested. This is easiest to achieve by diverting into different code
    # blocks depending on the number of time points.
    if t.size == 1:
        # Initialize an array to store output
        output = np.zeros((z0.size, zt.size))
        # Perform simulations using requested initial conditions
        for idx1, _z0 in enumerate(z0):
            sim = discrete(param, model, _z0, t[0], k, False, seed)
            # For each simulation from a specific initial condition count the number of
            # trajectories which fall into each terminal state
            counts = Counter(sim)
            # Fill in the output array according to requested terminal states
            for idx2, _zt in enumerate(zt):
                output[idx1, idx2] = counts[_zt]/k
    else:
        # Initialize an array to store output
        output = np.zeros((t.size, z0.size, zt.size))
        # Loop over requested times (note all simulations start at time 0)
        for idx3, _t in enumerate(t):
            # Perform simulations using requested initial conditions
            for idx1, _z0 in enumerate(z0):
                sim = discrete(param, model, _z0, _t, k, False, seed)
                # For each simulation from a specific intitial condition count the number of
                # trajectories which fall into each terminal state
                counts = Counter(sim)
                # Fill in the output array according to requested terminal states
                for idx2, _zt in enumerate(zt):
                    output[idx3, idx1, idx2] = counts[_zt] / k
    return output


@cuda.jit
def discrete_verhulst(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process Verhulst model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * (1 - p[2]*z) * z

    def d_rate(z):
        return p[1] * (1 + p[3]*z) * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_ricker(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process Ricker model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * math.exp(-(p[2] * z) ** p[3]) * z

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_bh(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process Beverton-Holt model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * p[2] / (z + p[2])

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_hassell(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process Hassell model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * z / (1 + z * p[2]) ** p[3]

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_mss(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process MaynardSmith-Slatkin model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * z / (1 + (z * p[2]) ** p[3])

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_moran(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process Moran model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return (p[4] - z) * (p[0] * z * (1 - p[2]) / p[4] +
                                     p[1] * (p[4] - z) * p[3] / p[4]) \
                       / p[4]

    def d_rate(z):
        return z * (p[2] * (p[4] - z) * (1 - p[3]) / p[4] +
                            p[0] * z * p[2] / p[4]) \
                       / p[4]

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_pb(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time pure birth process model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * z

    def d_rate(z):
        return 0

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_pd(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time pure death process model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return 0

    def d_rate(z):
        return p[0] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_poisson(out, rng_states, p, z0, time, survival):
    """
    Simulation of Poisson process model at discrete observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0]

    def d_rate(z):
        return 0

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop



@cuda.jit
def discrete_linear(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process linear model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * z

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_lm(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process linear-migration model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * z + p[2]

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_mm1(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process single-server queue model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0]

    def d_rate(z):
        return p[1] * (z > 0)

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_mminf(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process infinite-server queue model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0]

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop


@cuda.jit
def discrete_loss(out, rng_states, p, z0, time, survival):
    """
    Simulation of continuous-time birth-and-death process loss-system model at discrete
    observation times.

    This function is used by :func:`bd.gpu_functions.discrete()`.
    """
    thread_id = cuda.grid(1)

    def b_rate(z):
        return p[0] * (z < p[2])

    def d_rate(z):
        return p[1] * z

    while True:
        pop = z0
        rate = b_rate(pop) + d_rate(pop)
        if rate == 0:
            next_event_time = math.inf
        else:
            next_event_time = -math.log(rand(rng_states, thread_id)) / rate
        while next_event_time <= time:
            if (rand(rng_states, thread_id) * rate) <= b_rate(pop):
                pop += 1
            else:
                pop -= 1
            rate = b_rate(pop) + d_rate(pop)
            if rate > 0:
                next_event_time += -math.log(rand(rng_states, thread_id)) / rate
            else:
                next_event_time = math.inf
        if (survival and pop > 0) or not survival:
            # This conditions on survival of the population for the
            # monitoring period
            break
    out[thread_id] = pop
