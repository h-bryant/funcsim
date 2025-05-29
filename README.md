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
- python 3.9.6
- numpy 2.0.2
- scipy 1.13.1
- pandas 2.2.3
- xarray 2024.7.0

Additionally, to run the demos, run tests, and build docs:
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
