"""Validation, defaults, and adapters for optimization backends."""

from collections.abc import Mapping

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

_METHOD_ALIASES = {
    "differential-evolution": "differential-evolution",
    "l-bfgs-b": "L-BFGS-B",
    "slsqp": "SLSQP",
}

_DE_OPTION_NAMES = {
    "strategy", "maxiter", "popsize", "tol", "mutation", "recombination",
    "seed", "rng", "callback", "disp", "init", "atol", "updating", "workers",
    "x0", "integrality", "vectorized",
}
_LBFGSB_OPTION_NAMES = {
    "disp", "maxcor", "ftol", "gtol", "eps", "maxfun", "maxiter", "iprint",
    "maxls", "finite_diff_rel_step", "workers",
}
_SLSQP_OPTION_NAMES = {
    "ftol", "eps", "disp", "maxiter", "finite_diff_rel_step", "workers",
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


def solver_options(options, names):
    """Return only explicit controls accepted by a solver."""
    return {name: value for name, value in options.items() if name in names}


def differential_evolution_constraints(constraints):
    """Adapt legacy SciPy constraint dictionaries for differential evolution.

    Returns the adapted constraints and the untouched input used by the SLSQP
    polishing pass.  Native SciPy linear and nonlinear constraints are already
    accepted by differential evolution and are passed through unchanged.
    """
    if isinstance(constraints, (list, tuple)) and not constraints:
        return (), ()
    if isinstance(constraints, (Mapping, LinearConstraint, NonlinearConstraint)):
        items = (constraints,)
    elif isinstance(constraints, (list, tuple)):
        items = tuple(constraints)
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

    jacobian = constraint.get("jac", "2-point")
    if callable(jacobian):
        def constrained_jacobian(x, jacobian=jacobian, args=args):
            return jacobian(x, *args)
    else:
        constrained_jacobian = jacobian
    return NonlinearConstraint(constrained_function, 0, np.inf,
                               jac=constrained_jacobian)
