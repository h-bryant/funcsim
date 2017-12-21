# funcsim
Lightweight, functional stochastic simulation

Features:
- cross-sectional and recursive dynamic simulations
- pseudo-random numbers (so that results are comparable across multiple runs or scenarios)
- stratified/Latin hypercube sampling
- painless multi-core simulations
- highly flexible, and as simple as possible (but no more)
- functional paradigm
- python 2 and 3 compatible

Developed using:
- python 3.5.3
- numpy 1.13.1
- pandas 0.20.3
- xarray 0.9.6

Additionally, to run the demos, run tests, and build docs:
- scipy 1.0.0
- pytest 3.3.0
- sphinx 1.6.5

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
