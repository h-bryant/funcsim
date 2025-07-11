# funcsim
Lightweight, functional stochastic simulation

Features:
- cross-sectional and recursive dynamic simulations
- pseudo-random numbers (so that results are comparable across multiple runs or scenarios)
- stratified/Latin hypercube sampling
- painless multi-core simulations
- highly flexible, and as simple as possible (but no more)
- functional paradigm

Developed using:
- python >= 3.9
- numpy 2.2.0
- scipy 1.16.0
- pandas 2.2.0
- xarray 2025.4.0

For optional functionality, additinoal packages are needed:
- plotting: plotly
- some copula typed: copulae

Additionally, to run the demos, run tests, and build docs:
- pytest
- sphinx, sphinx-autodoc-typehints

### building html docs
From the top-level funcsim directory...
```bash
sphinx-build -b html docs/source/ docs/build/
```

### running tests
From the top-level funcsim directory...
```bash
python3 -m pytest
```
