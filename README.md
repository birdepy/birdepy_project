# BirDePy: Birth-and-death processes in Python

BirDePy is a Python package for working with continuous time birth-and-death processes. It includes functions which can be used to approximate transition probabilities (``birdepy.probability()``), estimate parameter values from data (``birdepy.estimate()``), simulate sample paths (``birdepy.simulate.discrete()`` and ``birdepy.simulate.continuous()``) and generate forecasts (``birdepy.forecast()``). The main focus of the package is the estimation of parameter values from discretely-observed sample paths, however the much more straightforward case of estimation of parameter values from continuously observed sample paths is also included.

Please visit our website: https://birdepy.github.io/ and cite our paper:

Hautphenne S, Patch B (2024). "Birth-and-Death Processes in Python: The BirDePy Package." _Journal of Statistical Software_, (111)5 1-54. [doi: 10.18637/jss.v111.i05](https://doi.org/10.18637/jss.v111.i05).

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

## Numerical reliability

The [unreleased changes](CHANGELOG.md#unreleased) correct simulation, estimation,
and forecasting defects. Results from the affected algorithms can differ from
earlier versions, including when using the same random seed.

Uniformization now chooses its Poisson-series cutoff automatically:

```python
import birdepy as bd

probability = bd.probability(
    10, 0, 10, [2], model="pure-death", method="uniform",
    z_trunc=[0, 20], uniform_tol=1e-12,
)
```

`uniform_tol` bounds the omitted Poisson tail; it does not bound the separate
error from truncating the state space with `z_trunc`. An explicit `k` is still
accepted, with a warning when the omitted probability mass exceeds the tolerance.

ABC returns weighted particle populations. Use `sample_weights` alongside
`samples` for additional posterior summaries:

```python
import numpy as np

result = bd.estimate(
    [0, 1], [1, 0], [1.0], [[0, 2]], model="pure-death",
    framework="abc", method="exact", eps_abc=[0, 0],
    eps_change=0, k=1000, seed=42,
)
particles = result.samples[-1]
weights = result.sample_weights[-1]
posterior_mean = np.average(particles, axis=0, weights=weights)
```

For ABC, `p` and `cov` use these weights, and `se` reports posterior standard
deviations. Numeric covariance outputs have shape `(n_parameters, n_parameters)`,
including for a single parameter. A covariance that cannot be estimated is
reported as a descriptive string.

## Running the tests

From the repository root, using Python 3.10 or 3.11 in a virtual environment:

```sh
python -m pip install -r requirements.txt -r requirements.dev.txt
MPLBACKEND=Agg python -m pytest
```

Regression tests include analytic likelihood and posterior checks, simulation
moments, agreement between transition-probability methods, and reproducible
bootstrap estimates. The CPU suite does not validate CUDA execution.
