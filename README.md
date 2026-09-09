# funcsim
Lightweight, functional stochastic simulation

Documentation: https://funcsim.readthedocs.io/

Features:
- static (cross-sectional) and recursive-dynamic simulations from a single user-written function
- pseudo-random numbers with an explicit seed (so that results are reproducible and comparable across runs or scenarios)
- stratified/Latin hypercube sampling
- painless multi-core simulations
- fitting and comparison of univariate distributions, kernel density estimates, and goodness-of-fit tests
- multivariate normal and KDE distributions; Gaussian, Student's t, Clayton, Gumbel, and Frank copulas (implemented natively, no additional packages required); Iman-Conover
- expected utility and cumulative prospect theory calculations
- plotly-based fan charts and diagnostic plots
- highly flexible, and as simple as possible (but no more)
- functional paradigm

Requirements:
- python >= 3.11
- numpy >= 2.2.0
- scipy >= 1.16.0
- pandas >= 2.2.0 (pandas 2.2+ and 3.x are both supported)
- xarray >= 2025.4.0

For optional functionality, additional packages are needed:
- plotting: plotly (`pip install "funcsim[plotting]"`); kaleido for static image export

Additionally, to run the tests and build the docs:
- pytest
- sphinx, sphinx-autodoc-typehints (`pip install "funcsim[docs]"`)

### installation
```bash
pip install funcsim
```

### building html docs
From the top-level funcsim directory...
```bash
sphinx-build -W -b html docs/source/ docs/build/
```

### running tests
From the top-level funcsim directory...
```bash
python3 -m pytest
```
