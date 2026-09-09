funcsim
=======

**funcsim** is a lightweight Python package for stochastic simulation in a
functional style.  You write one ordinary Python function that performs a
single trial (for a static simulation) or takes one step through time (for
a recursive-dynamic simulation).  :func:`funcsim.simulate` does the rest:
it supplies pseudo-random draws, stratifies them across trials, optionally
spreads the work over all available cores, and assembles the results into a
labeled :class:`xarray.DataArray`.

This documentation describes funcsim |release|.

.. code-block:: python

   from scipy import stats
   import funcsim as fs

   def trial(ugen):
       # one trial: two independent draws, transformed by inverse CDFs
       eps = stats.norm.ppf(next(ugen))
       b = stats.bernoulli.ppf(next(ugen), 0.35)
       return {"eps": eps, "b": b}

   da = fs.simulate(f=trial, ntrials=500)
   print(da.mean(dim="trials"))


Features
--------

- Static (cross-sectional) and recursive-dynamic simulations from a single
  user-written function.
- Pseudo-random draws with an explicit seed, so results are reproducible
  and directly comparable across scenarios.
- Stratified (Latin hypercube) sampling by default, with plain Monte Carlo
  sampling as an option.
- Painless multi-core execution (``multi=True``).
- Fitting and comparison of univariate distributions across more than one
  hundred candidate families from :mod:`scipy.stats`, with optional fixed
  support bounds.
- Kernel density estimates, empirical distribution functions, and
  goodness-of-fit tests.
- Dependence modeling: multivariate normal, multivariate KDE, and Gaussian,
  Student's t, Clayton, Gumbel, and Frank copulas, all implemented natively;
  Iman-Conover rank correlation induction; covariance shrinkage.
- Expected utility and cumulative prospect theory calculations.
- Fan charts and diagnostic plots built on plotly.


Installation
------------

.. code-block:: bash

   pip install funcsim                 # core package
   pip install "funcsim[plotting]"     # adds plotly and jupyter for charts

funcsim requires Python 3.11 or later, together with NumPy 2.2+, pandas 2.2+
(pandas 3 is supported), SciPy 1.16+, and xarray 2025.4+.  Copulas are
implemented natively; no additional packages are needed for them.  Static
image export from the plotting functions (PNG, SVG, PDF) additionally
requires the ``kaleido`` package.

The source is hosted on `GitHub <https://github.com/h-bryant/funcsim>`_, and
releases are published on `PyPI <https://pypi.org/project/funcsim/>`_.


Contents
--------

.. toctree::
   :maxdepth: 2

   overview
   guide_distributions
   guide_dependence
   guide_utility
   guide_plotting
   api_reference
