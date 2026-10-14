"""Numerical forecast calculations without plotting side effects."""

import warnings

import numpy as np
from scipy.integrate import solve_ivp

from birdepy import simulate
import birdepy.utility as ut
from birdepy.results import ForecastResult


def _constraints_are_feasible(constraints, proposal):
    """Check SciPy and dictionary constraints against an unknown proposal."""
    if constraints is None or constraints == ():
        return True
    if isinstance(constraints, dict) or (hasattr(constraints, 'lb')
                                         and hasattr(constraints, 'ub')):
        constraints = (constraints,)

    for constraint in constraints:
        if isinstance(constraint, dict):
            fun = constraint.get('fun')
            if not callable(fun):
                raise ValueError("Constraint dictionaries must provide a callable 'fun'.")
            value = np.asarray(fun(proposal, *constraint.get('args', ())))
            constraint_type = constraint.get('type', 'ineq')
            if constraint_type == 'eq':
                feasible = np.all(value == 0)
            elif constraint_type == 'ineq':
                feasible = np.all(value >= 0)
            else:
                raise ValueError("Constraint dictionaries must have type 'eq' or 'ineq'.")
        elif hasattr(constraint, 'lb') and hasattr(constraint, 'ub'):
            if hasattr(constraint, 'fun'):
                value = np.asarray(constraint.fun(proposal))
            elif hasattr(constraint, 'A'):
                value = np.asarray(constraint.A @ proposal)
            else:
                raise TypeError("Unsupported SciPy constraint.")
            feasible = np.all(value >= constraint.lb) and np.all(value <= constraint.ub)
        else:
            raise TypeError("Constraints must be dictionaries or SciPy constraints.")
        if not np.all(np.isfinite(value)) or not feasible:
            return False
    return True


def parameter_sampler(param, cov, p_bounds, known_p, idx_known_p, con, rng,
                      max_rejections=10_000):
    """Draw one bounded, constrained parameter vector.

    ``max_rejections`` bounds rejection sampling so contradictory bounds or
    constraints fail explicitly rather than hanging indefinitely.
    """
    if not isinstance(max_rejections, (int, np.integer)) or max_rejections <= 0:
        raise ValueError("max_rejections must be a positive integer.")
    if p_bounds is not None:
        if len(p_bounds) != len(param):
            raise ValueError("p_bounds must contain one pair for each unknown parameter.")
        for lower, upper in p_bounds:
            if np.isnan(lower) or np.isnan(upper) or lower > upper:
                raise ValueError("Each parameter bound must not contain NaN and have lower <= upper.")

    for _ in range(max_rejections):
        param_prop = rng.multivariate_normal(param, cov)
        if p_bounds is not None and any(
                value < bound[0] or value > bound[1]
                for value, bound in zip(param_prop, p_bounds)):
            continue
        # Constraints intentionally apply to unknown parameters, before their
        # known values are expanded into the model's full parameter vector.
        if not _constraints_are_feasible(con, param_prop):
            continue
        return ut.p_bld(np.array(param_prop), idx_known_p, known_p)
    raise RuntimeError(
        "Unable to sample parameters satisfying p_bounds and con within "
        f"{max_rejections} attempts.")


def mean_curve(param, b_rate, d_rate, method, z0, solver_methods,
               shift_times, n, rng):
    """Generate a fluid or simulation-based mean forecast."""
    if method == 'fm':
        z0_ = z0() if callable(z0) else z0
        if shift_times[-1] == 0:
            return np.full(len(shift_times), z0_, dtype=float)
        for solver_method in solver_methods:
            fluid_path = solve_ivp(
                lambda t, z: b_rate(z, param) - d_rate(z, param),
                [0, shift_times[-1]], [z0_], t_eval=shift_times,
                method=solver_method)
            if fluid_path.success:
                return fluid_path.y[0]
        raise RuntimeError("Unable to solve the forecast mean trajectory.")
    if method in ['exact', 'ea', 'ma', 'gwa']:
        forecasted_mean = np.zeros(len(shift_times))
        for _ in range(n):
            forecasted_mean += simulate.discrete(
                param, 'custom', z0, shift_times, b_rate=b_rate, d_rate=d_rate,
                k=1, method=method, seed=rng)
        return np.divide(forecasted_mean, n)
    raise ValueError("Unknown method for computing a forecast mean.")


def compute_forecast(model, z0, times, param, cov=None, interval='confidence',
                     method=None, percentiles=(0, 2.5, 10, 25, 50, 75, 90, 97.5, 100),
                     p_bounds=None, con=(), known_p=(), idx_known_p=(), k=10 ** 3,
                     n=10 ** 3, seed=None, display=False, **options):
    """Compute forecast curves and intervals without creating a plot.

    Confidence samples are future *mean* trajectories. Prediction samples are
    future population trajectories. The calculation therefore preserves the
    uncertainty semantics of :func:`birdepy.forecast` while allowing callers
    to inspect or store its numerical result directly. Numerical parameters
    have the same meaning as in :func:`birdepy.forecast`. The optional
    ``max_rejections`` keyword limits proposals per uncertain parameter sample
    (default 10000), raising ``RuntimeError`` when exhausted.

    Returns a :class:`birdepy.ForecastResult` with times, central and mean
    curves, quantiles, samples (when an interval is sampled), and metadata.
    """
    times = np.asarray(times)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Forecast times must be a nonempty, increasing finite sequence.")
    percentiles = np.asarray(percentiles)
    if percentiles.ndim != 1 or percentiles.size == 0 or not np.all(np.isfinite(percentiles)) \
            or np.any(percentiles < 0) \
            or np.any(percentiles > 100) or np.any(np.diff(percentiles) < 0):
        raise ValueError("percentiles must be an increasing sequence in [0, 100].")
    if (not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer))
            or k <= 0 or n <= 0):
        raise ValueError("k and n must be positive integers.")

    if interval == 'confidence' and method is None:
        method = 'fm'
    elif interval == 'prediction' and method is None:
        method = 'gwa'
    elif interval == 'prediction' and method == 'fm':
        raise TypeError("Argument of `method` equal 'fm' not possible when argument "
                        "of `interval` equals 'prediction'.")
    elif interval not in ('confidence', 'prediction'):
        raise TypeError("Argument 'interval' has an unknown value.")

    shift_times = times - times[0]
    if cov is not None:
        cov = np.atleast_2d(cov)
    full_param = ut.p_bld(np.atleast_1d(param), idx_known_p, known_p)
    if model == 'custom':
        b_rate = options['b_rate']
        d_rate = options['d_rate']
    else:
        b_rate = ut.higher_birth(model)
        d_rate = ut.higher_death(model)
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    solver_methods = ['RK45', 'Radau', 'RK23', 'BDF', 'DOP853']
    samples = None

    if interval == 'confidence':
        if cov is None:
            warnings.warn("Confidence intervals show the likely range of the mean "
                          "future population level given the uncertainty of the "
                          "parameter values, however since argument `cov` has value "
                          "None no uncertainty in parameter values has been specified.",
                          category=RuntimeWarning)
            central = mean_curve(full_param, b_rate, d_rate, method, z0,
                                solver_methods, shift_times, n, rng)
            mean = central.copy()
            quantiles = None
        else:
            samples = np.zeros((k, times.shape[0]))
            for idx in range(k):
                param_prop = parameter_sampler(param, cov, p_bounds, known_p,
                                               idx_known_p, con, rng,
                                               options.get('max_rejections', 10_000))
                samples[idx, :] = mean_curve(param_prop, b_rate, d_rate, method, z0,
                                             solver_methods, shift_times, n, rng)
                if display:
                    print(f"Forecast is ", 100 * (idx + 1) / k, f"% complete.")
    else:
        samples = np.zeros((k, times.shape[0]))
        for idx in range(k):
            param_prop = full_param if cov is None else parameter_sampler(
                param, cov, p_bounds, known_p, idx_known_p, con, rng,
                options.get('max_rejections', 10_000))
            samples[idx, :] = simulate.discrete(
                param_prop, 'custom', z0, shift_times, b_rate=b_rate, d_rate=d_rate,
                k=1, method=method, seed=rng)
            if display:
                print(f"Forecast is ", 100 * (idx + 1) / k, f"% complete.")

    if samples is not None:
        quantiles = np.percentile(samples, percentiles, axis=0).T
        central = quantiles[:, int(np.floor((len(percentiles) - 1) / 2))]
        mean = np.mean(samples, axis=0)

    return ForecastResult(
        times=np.array(times, copy=True), central=np.array(central, copy=True),
        mean=np.array(mean, copy=True), quantiles=quantiles,
        percentiles=np.array(percentiles, copy=True), samples=samples,
        interval=interval, method=method,
        metadata={'model': model, 'k': k, 'n': n, 'has_covariance': cov is not None})
