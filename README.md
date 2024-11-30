# BirDePy: Birth-and-death processes in Python

BirDePy is a Python package for working with continuous time birth-and-death processes. It includes functions which can be used to approximate transition probabilities (``birdepy.probability()``), estimate parameter values from data (``birdepy.estimate()``), simulate sample paths (``birdepy.simulate.discrete()`` and ``birdepy.simulate.continuous()``) and generate forecasts (``birdepy.forecast()``). The main focus of the package is the estimation of parameter values from discretely-observed sample paths, however the much more straightforward case of estimation of parameter values from continuously observed sample paths is also included.

Please visit our website: https://birdepy.github.io/ and cite our paper:

Hautphenne S, Patch B (2024). "Birth-and-Death Processes in Python: The BirDePy Package." _Journal of Statistical Software_, (111)5 1-54. [doi: 10.18637/jss.v111.i05](https://doi.org/10.18637/jss.v111.i05).

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
