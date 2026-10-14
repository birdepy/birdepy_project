"""Numerical forecast calculations without plotting side effects."""

import warnings
from collections.abc import Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import LinearConstraint, NonlinearConstraint

from birdepy import simulate
import birdepy.utility as ut
from birdepy.results import ForecastResult

DEFAULT_PERCENTILES = (0, 2.5, 10, 25, 50, 75, 90, 97.5, 100)
DEFAULT_MAX_REJECTIONS = 10_000


class _SamplingError(RuntimeError):
    """A parameter distribution could not produce a feasible sample."""


def _positive_integer(value, name):
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_)) or value <= 0):
        raise ValueError(f"{name} must be a positive integer.")


def _prepare_constraints(constraints):
    """Normalize public constraint formats once, before any random draws."""
    if constraints is None:
        return ()
    if isinstance(constraints, (Mapping, LinearConstraint, NonlinearConstraint)):
        constraints = (constraints,)
    try:
        constraints = tuple(constraints)
    except TypeError as error:
        raise TypeError("Constraints must be dictionaries or SciPy constraints.") from error
    prepared = []
    for constraint in constraints:
        if isinstance(constraint, Mapping):
            function = constraint.get('fun')
            kind = constraint.get('type')
            if not callable(function):
                raise ValueError("Constraint dictionaries must provide a callable 'fun'.")
            if not isinstance(kind, str) or kind.lower() not in ('ineq', 'eq'):
                raise ValueError("Constraint dictionaries must have type 'ineq' or 'eq'.")
            if kind.lower() == 'eq':
                raise ValueError("Equality constraints are not supported for forecast parameter sampling; "
                                 "use known_p or reparameterize the model.")
            try:
                args = tuple(constraint.get('args', ()))
            except TypeError as error:
                raise TypeError("Constraint dictionary 'args' must be iterable.") from error
            def evaluate(proposal, function=function, args=args):
                return function(proposal, *args)
            lower, upper = 0, np.inf
        elif isinstance(constraint, (LinearConstraint, NonlinearConstraint)):
            lower, upper = np.broadcast_arrays(constraint.lb, constraint.ub)
            if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
                raise ValueError("Constraint bounds must not contain NaN and must have lower <= upper.")
            if np.any(lower == upper):
                raise ValueError("Equality constraints are not supported for forecast parameter sampling; "
                                 "use known_p or reparameterize the model.")
            if isinstance(constraint, LinearConstraint):
                def evaluate(proposal, matrix=constraint.A):
                    return matrix @ proposal
            else:
                evaluate = constraint.fun
        else:
            raise TypeError("Constraints must be dictionaries or SciPy constraints.")
        prepared.append((evaluate, lower, upper))
    return tuple(prepared)


def _constraints_are_feasible(constraints, proposal):
    for function, lower, upper in constraints:
        value = np.asarray(function(proposal))
        # Infinities are valid for one-sided constraints; NaN is never feasible.
        if not np.all((value >= lower) & (value <= upper)):
            return False
    return True


def _prepare_parameter_sampler(param, cov, p_bounds, known_p, idx_known_p, con,
                               rng, max_rejections):
    _positive_integer(max_rejections, 'max_rejections')
    constraints = _prepare_constraints(con)
    param = np.atleast_1d(np.asarray(param, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if param.ndim != 1 or not np.all(np.isfinite(param)):
        raise ValueError("param must be a finite one-dimensional parameter vector.")
    if cov.shape != (param.size, param.size) or not np.all(np.isfinite(cov)):
        raise ValueError("cov must be a finite square matrix matching param.")
    lower = np.full(param.size, -np.inf)
    upper = np.full(param.size, np.inf)
    if p_bounds is not None:
        if len(p_bounds) != param.size:
            raise ValueError("p_bounds must contain one pair for each unknown parameter.")
        for index, (lo, hi) in enumerate(p_bounds):
            lower[index] = -np.inf if lo is None else lo
            upper[index] = np.inf if hi is None else hi
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("Each parameter bound must not contain NaN and have lower <= upper.")

    def feasible(proposal):
        return (np.all(proposal >= lower) and np.all(proposal <= upper)
                and _constraints_are_feasible(constraints, proposal))

    # Only a deterministic distribution can be rejected based on its center.
    # A nondegenerate truncated normal may have its center outside its support.
    if np.all(cov == 0) and not feasible(param):
        raise _SamplingError("Zero covariance fixes parameters at an infeasible value.")

    def draw():
        for _ in range(max_rejections):
            # Retain NumPy's covariance validation and seeded sampling algorithm.
            proposal = rng.multivariate_normal(param, cov)
            if feasible(proposal):
                return ut.p_bld(proposal, idx_known_p, known_p)
        raise _SamplingError(
            "Unable to sample parameters satisfying p_bounds and con within "
            f"{max_rejections} attempts.")
    return draw


def parameter_sampler(param, cov, p_bounds, known_p, idx_known_p, con, rng,
                      max_rejections=DEFAULT_MAX_REJECTIONS):
    """Draw one bounded, inequality-constrained parameter vector.

    Constraints apply to unknown parameters, before expanding ``known_p``.
    Exact equalities are unsupported by rejection sampling. ``max_rejections``
    bounds the proposals per sample; deterministic infeasibility fails early.
    """
    return _prepare_parameter_sampler(param, cov, p_bounds, known_p, idx_known_p,
                                      con, rng, max_rejections)()


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
                     method=None, percentiles=DEFAULT_PERCENTILES,
                     p_bounds=None, con=(), known_p=(), idx_known_p=(), k=10 ** 3,
                     n=10 ** 3, seed=None, display=False, *,
                     max_rejections=DEFAULT_MAX_REJECTIONS, **options):
    """Compute forecast curves and intervals without creating a plot.

    Confidence samples are future *mean* trajectories. Prediction samples are
    future population trajectories. The calculation therefore preserves the
    uncertainty semantics of :func:`birdepy.forecast` while allowing callers
    to inspect or store its numerical result directly. Numerical parameters
    have the same meaning as in :func:`birdepy.forecast`. The optional
    ``max_rejections`` keyword limits proposals per uncertain parameter sample
    (default 10000), raising ``RuntimeError`` with the number of completed
    samples when exhausted. Constraints apply to unknown parameters and support
    inequality dictionaries (including ``args``) and SciPy constraint objects.
    Exact equality constraints cannot be handled by rejection sampling; use
    ``known_p`` or reparameterize the model. Bounds and constraints are sampling
    controls and are only applied when ``cov`` is supplied.

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
    for name, value in (('k', k), ('n', n), ('max_rejections', max_rejections)):
        _positive_integer(value, name)

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
    quantiles = None
    draw = None
    if cov is not None:
        try:
            draw = _prepare_parameter_sampler(param, cov, p_bounds, known_p,
                                               idx_known_p, con, rng, max_rejections)
        except _SamplingError as error:
            raise RuntimeError(f"Forecast stopped after 0 of {k} samples: {error}") from error

    if interval == 'confidence' and cov is None:
        warnings.warn("Confidence intervals show the likely range of the mean "
                      "future population level given the uncertainty of the "
                      "parameter values, however since argument `cov` has value "
                      "None no uncertainty in parameter values has been specified.",
                      category=RuntimeWarning,
                      stacklevel=options.pop('_warning_stacklevel', 2))
        central = mean_curve(full_param, b_rate, d_rate, method, z0,
                             solver_methods, shift_times, n, rng)
        mean = central.copy()
    else:
        samples = np.zeros((k, times.shape[0]))
        for idx in range(k):
            try:
                proposal = full_param if draw is None else draw()
            except _SamplingError as error:
                raise RuntimeError(f"Forecast stopped after {idx} of {k} samples: {error}") from error
            if interval == 'confidence':
                samples[idx, :] = mean_curve(proposal, b_rate, d_rate, method, z0,
                                             solver_methods, shift_times, n, rng)
            else:
                samples[idx, :] = simulate.discrete(
                    proposal, 'custom', z0, shift_times, b_rate=b_rate, d_rate=d_rate,
                    k=1, method=method, seed=rng)
            if display:
                print("Forecast is ", 100 * (idx + 1) / k, "% complete.")
        quantiles = np.percentile(samples, percentiles, axis=0).T
        central = np.percentile(samples, 50, axis=0)
        mean = np.mean(samples, axis=0)

    n_samples = 0 if samples is None else samples.shape[0]
    n_trajectories = (k if interval == 'prediction' else
                      0 if method == 'fm' else n * (n_samples or 1))
    return ForecastResult(
        times=np.array(times, copy=True), central=np.array(central, copy=True),
        mean=np.array(mean, copy=True), quantiles=quantiles,
        percentiles=np.array(percentiles, copy=True), samples=samples,
        interval=interval, method=method,
        metadata={'model': model, 'n_samples': n_samples, 'n_trajectories': n_trajectories,
                  'has_covariance': cov is not None})
