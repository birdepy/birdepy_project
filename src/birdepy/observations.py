"""Shared validation and normalization for observed process paths."""

from collections.abc import Sequence

import numpy as np


def normalize_observations(t_data, p_data, scheme):
    """Return observations as lists of one-dimensional numeric arrays.

    A one-dimensional input denotes one trajectory; a sequence of one-dimensional
    inputs denotes multiple, potentially ragged trajectories.  Absolute clocks
    may be translated, but each path must have finite, nondecreasing times.
    Zero-length intervals are retained because identical simultaneous
    observations carry no contradictory transition information.
    """
    if scheme not in {"discrete", "continuous"}:
        raise ValueError("Argument `scheme` has an unknown value. Should be one of: 'discrete' or 'continuous'.")
    times = _as_paths(t_data, "t_data")
    populations = _as_paths(p_data, "p_data")
    if len(times) != len(populations):
        raise ValueError("Arguments `t_data` and `p_data` must contain the same number of paths.")

    normalized_times, normalized_populations = [], []
    max_population = np.nextafter(float(np.iinfo(np.int64).max), -np.inf)
    for index, (time_path, population_path) in enumerate(zip(times, populations)):
        time_path = _as_numeric_vector(time_path, "t_data", index)
        population_path = _as_numeric_vector(population_path, "p_data", index)
        if len(time_path) != len(population_path):
            raise ValueError("Each `t_data` path must have the same length as its `p_data` path.")
        if len(time_path) == 0:
            raise ValueError("Each observation path must contain an observation.")
        if not np.all(np.isfinite(time_path)):
            raise ValueError("Observation times must be finite.")
        time_differences = np.diff(time_path)
        population_differences = np.diff(population_path)
        if np.any(time_differences < 0):
            raise ValueError("Observation times must be nondecreasing within each path.")
        if not np.all(np.isfinite(population_path)):
            raise ValueError("Observed populations must be finite.")
        if np.any(population_path < 0) or not np.all(population_path == np.floor(population_path)):
            raise ValueError("Observed populations must be nonnegative integers.")
        if np.any(population_path >= max_population):
            raise ValueError("Observed populations must fit in the platform integer range.")
        simultaneous_changes = (time_differences == 0) & (population_differences != 0)
        if scheme == "discrete" and np.any(simultaneous_changes):
            raise ValueError("Population changes require a positive elapsed observation time.")
        if scheme == "continuous" and np.any(np.abs(population_differences) > 1):
            raise ValueError("Continuously observed populations must change by one event at a time.")
        normalized_times.append(time_path)
        normalized_populations.append(population_path.astype(np.int64))
    return normalized_times, normalized_populations


def _as_paths(data, name):
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            if data.size and np.ndim(data[0]) > 0:
                return list(data)
            return [data]
        if data.ndim == 2:
            paths = list(data)
            if not paths:
                raise ValueError(f"Argument `{name}` must contain at least one observation path.")
            return paths
        raise ValueError(f"Argument `{name}` must be a one- or two-dimensional collection of paths.")
    if not isinstance(data, Sequence):
        try:
            if len(data) == 0:
                raise ValueError(f"Argument `{name}` must contain at least one observation path.")
            first = data[0]
        except (TypeError, KeyError, IndexError) as error:
            raise ValueError(f"Argument `{name}` must contain at least one observation path.") from error
    else:
        if not data:
            raise ValueError(f"Argument `{name}` must contain at least one observation path.")
        first = data[0]
    if np.ndim(first) == 0:
        return [data]
    return list(data)


def _as_numeric_vector(path, name, index):
    try:
        raw = np.asarray(path)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Path {index} in `{name}` must be numeric.") from error
    if raw.dtype.kind not in "iufb":
        raise ValueError(f"Path {index} in `{name}` must be numeric.")
    array = raw.astype(float, copy=False)
    if array.ndim != 1:
        raise ValueError(f"Path {index} in `{name}` must be one-dimensional.")
    return array
