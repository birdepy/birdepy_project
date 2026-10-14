"""Validation, defaults, and adapters for optimization backends."""

from collections.abc import Mapping

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

_METHOD_ALIASES = {
    "differential-evolution": "differential-evolution",
    "l-bfgs-b": "L-BFGS-B",
    "slsqp": "SLSQP",
}

_DE_DEFAULTS = {
    "seed": None,
    "strategy": "best1bin",
    "maxiter": 1000,
    "popsize": 15,
    "tol": 0.01,
    "mutation": 0.5,
    "recombination": 0.7,
    "callback": None,
    "disp": False,
    "polish": True,
    "init": "latinhypercube",
    "atol": 0,
    "updating": "immediate",
}

_LBFGSB_DEFAULTS = {
    "callback": None,
    "jac": "2-point",
    "maxcor": 10,
    "ftol": 2.220446049250313e-09,
    "gtol": 1e-05,
    "eps": None,
    "maxfun": 15000,
    "maxiter": 15000,
    "maxls": 20,
    "finite_diff_rel_step": None,
}

_SLSQP_DEFAULTS = {
    "jac": "2-point",
    "ftol": 2.220446049250313e-09,
    "eps": None,
    "disp": False,
    "maxiter": 1000,
    "finite_diff_rel_step": None,
}


def normalize_optimizer(method):
    """Validate the optimizer type and normalize BirDePy solver aliases."""
    if callable(method):
        return method
    if method is None:
        return None
    if not isinstance(method, str) or not method:
        raise ValueError(
            "Argument `opt_method` must be a SciPy minimize method name, "
            "'differential-evolution', or a callable."
        )
    return _METHOD_ALIASES.get(method.lower(), method)


def optimizer_options(options, method):
    """Return a copied option mapping with defaults for ``method`` only."""
    method = normalize_optimizer(method)
    configured = dict(options)
    if method == "differential-evolution":
        defaults = _DE_DEFAULTS
    elif method == "L-BFGS-B":
        defaults = _LBFGSB_DEFAULTS
    elif method == "SLSQP":
        defaults = _SLSQP_DEFAULTS
    else:
        defaults = {}
    for name, value in defaults.items():
        configured.setdefault(name, value)
    return configured


def differential_evolution_constraints(constraints):
    """Adapt legacy SciPy constraint dictionaries for differential evolution.

    Returns the adapted constraints and the untouched input used by the SLSQP
    polishing pass.  Native SciPy linear and nonlinear constraints are already
    accepted by differential evolution and are passed through unchanged.
    """
    if constraints == ():
        return (), constraints
    if isinstance(constraints, (Mapping, LinearConstraint, NonlinearConstraint)):
        items = (constraints,)
    elif isinstance(constraints, (list, tuple)):
        items = tuple(constraints)
        if not items:
            return (), constraints
    else:
        raise TypeError("Argument `con` must be a constraint dictionary, SciPy constraint, or sequence of them.")

    adapted = tuple(_adapt_constraint(constraint) for constraint in items)
    return adapted[0] if len(adapted) == 1 else adapted, constraints


def _adapt_constraint(constraint):
    if isinstance(constraint, (LinearConstraint, NonlinearConstraint)):
        if np.any(np.asarray(constraint.lb) == np.asarray(constraint.ub)):
            raise ValueError("Equality constraints are not supported by differential-evolution; use SLSQP instead.")
        return constraint
    if not isinstance(constraint, Mapping):
        raise TypeError("Each element of `con` must be a constraint dictionary or SciPy constraint.")
    kind = constraint.get("type")
    kind = kind.lower() if isinstance(kind, str) else kind
    function = constraint.get("fun")
    if kind == "eq":
        raise ValueError("Equality constraints are not supported by differential-evolution; use SLSQP instead.")
    if kind != "ineq" or not callable(function):
        raise ValueError("Constraint dictionaries require a callable `fun` and `type` equal to 'eq' or 'ineq'.")
    args = constraint.get("args", ())
    if not isinstance(args, tuple):
        try:
            args = tuple(args)
        except TypeError as error:
            raise TypeError("Constraint dictionary `args` must be an iterable.") from error

    def constrained_function(x, function=function, args=args):
        return function(x, *args)

    lower, upper = (0, 0) if kind == "eq" else (0, float("inf"))
    return NonlinearConstraint(constrained_function, lower, upper,
                               jac=constraint.get("jac", "2-point"))
