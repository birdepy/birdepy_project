# BirDePy: Birth-and-death processes in Python

BirDePy is a Python package for working with continuous time birth-and-death processes. It includes functions which can be used to approximate transition probabilities (``birdepy.probability()``), estimate parameter values from data (``birdepy.estimate()``), simulate sample paths (``birdepy.simulate.discrete()`` and ``birdepy.simulate.continuous()``) and generate forecasts (``birdepy.forecast()``). The main focus of the package is the estimation of parameter values from discretely-observed sample paths, however the much more straightforward case of estimation of parameter values from continuously observed sample paths is also included.

Please visit our website: https://birdepy.github.io/ and cite our paper:

Hautphenne S, Patch B (2024). "Birth-and-Death Processes in Python: The BirDePy Package." _Journal of Statistical Software_, (111)5 1-54. [doi: 10.18637/jss.v111.i05](https://doi.org/10.18637/jss.v111.i05).

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

## Numerical reliability

See [GitHub Releases](https://github.com/birdepy/birdepy_project/releases) for release notes.
The [changelog](CHANGELOG.md) retains earlier release history.
Corrections to simulation, estimation, and forecasting
can change results from earlier versions, including when using the same random seed.

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
For large rate-time products, the automatic method uses shorter intervals and
repeated squaring, dividing the tail budget across the intervals.

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
For other estimation frameworks, `sample_weights` contains the same
"Not applicable" sentinel as `samples`.

## Running the tests

From the repository root, using Python 3.10 or 3.11 in a virtual environment:

```sh
python -m pip install -r requirements.txt -r requirements.dev.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest
```

Regression tests include analytic likelihood and posterior checks, simulation
moments, agreement between transition-probability methods, and reproducible
bootstrap estimates. The CPU suite does not validate CUDA execution.

CI checks this pinned baseline and builds a source distribution and wheel for
fresh-install tests on Python 3.10–3.15 (allowing a 3.15 prerelease until final
builds are available). The installed-package tests run outside the checkout so
that imports use the wheel, including its bundled Laplace-inversion data.
`pyproject.toml` declares the tested dependency minimums; newer compatible
versions are selected by pip for each Python version. `scikit-learn` is not used
by BirDePy and is no longer installed for tests.

## Dependency maintenance and releases

Renovate groups compatible Python updates and filters out releases that require
Python newer than 3.10 for the baseline. Major upgrades remain separate PRs.
Changing the Python support floor or CI interpreter matrix is a maintainer
decision, so Renovate does not automatically replace those interpreter versions.

The required `test` check succeeds only after both Python matrices and the
release-tooling check pass. Releases on configured release branches run the same
checks before publishing. Release tooling uses Node 24, `npm ci`, and the
repository's built-in GitHub token. The `PYPI_TOKEN` secret is still required.
Release notes are published on GitHub; releases do not push generated changelog
commits to the protected branch.

To validate release tooling locally, with Node 24.10 or newer and a Python
virtual environment active:

```sh
python -m pip install build setuptools wheel twine tomlkit
npm ci
npm run check:release
```

This check uses a temporary local Git remote, previews a breaking-change release,
and builds and validates versioned Python distributions. It does not publish,
use release credentials, or modify the working tree. Authentication with GitHub
and PyPI is verified by the actual release job. The source version `0.0.0` is a
placeholder; semantic-release sets the release version before building.
