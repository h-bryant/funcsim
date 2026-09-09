Simulation Guide
================

funcsim asks you to write one function.  For a *static* (cross-sectional)
simulation, that function performs a single trial.  For a
*recursive-dynamic* simulation, it takes a single step through time.
Everything else, including random number generation, stratified sampling,
multi-core execution, and the shape of the output, is handled by
:func:`funcsim.simulate`.

The trial or step function
--------------------------

The function you write, passed to ``simulate`` as its ``f`` argument, must
follow three rules.

1. **Its first argument is a generator of random draws.**  By convention it
   is called ``ugen``.  Each call to ``next(ugen)`` returns one independent
   draw from the standard uniform distribution on the unit interval (or
   from the standard normal distribution if ``stdnorm=True`` is passed to
   ``simulate``).  Transform these draws into whatever you need, typically
   with the inverse CDF (``ppf``) of a :mod:`scipy.stats` distribution, a
   distribution fitted with :func:`funcsim.fit`, or a
   :class:`funcsim.Kde`.

2. **It returns a dict.**  The keys are variable names (strings) and the
   values are floats.  The keys become the ``variables`` coordinate of the
   output.

3. **It consumes the same number of draws every time.**  Draws are
   pre-allocated and stratified across trials before any trial runs, so the
   number of times ``next(ugen)`` is called must not depend on the data or
   on earlier draws.  ``simulate`` probes ``f`` once to count its draws and
   raises :class:`RuntimeError` if any later trial or step consumes a
   different number.  If a draw is needed only conditionally, take it
   unconditionally and ignore it when it is not needed.

Draws taken indirectly count too: if ``f`` calls a helper that calls
``next(ugen)``, or the ``draw`` method of a copula or multivariate
distribution object (see :doc:`guide_dependence`), those draws are part of
the fixed count.


Static simulation
-----------------

A static simulation repeats one trial many times.  The trial function takes
only the ``ugen`` argument.

.. code-block:: python

   from scipy import stats
   import funcsim as fs

   def trial(ugen):
       # two independent standard uniform draws
       u1 = next(ugen)
       u2 = next(ugen)

       # inverse-CDF transformations
       eps = stats.norm.ppf(u1)            # standard normal
       b = stats.bernoulli.ppf(u2, 0.35)   # Bernoulli(0.35)

       # do some kind of work
       x = 1.5 * eps
       y = 34.2 + 0.95 * x + b

       return {"x": x, "y": y}

   da = fs.simulate(f=trial, ntrials=500)

The result is a three-dimensional :class:`xarray.DataArray` with
dimensions ``trials``, ``variables``, and ``steps``.  For a static
simulation the ``steps`` dimension has length one.

.. code-block:: text

   <xarray.DataArray (trials: 500, variables: 2, steps: 1)> Size: 8kB
   ...
   Coordinates:
     * trials     (trials) int64 4kB 0 1 2 3 4 5 6 ... 494 495 496 497 498 499
     * variables  (variables) <U1 8B 'x' 'y'
     * steps      (steps) int64 8B 0

xarray's labeled indexing makes the output easy to work with:

.. code-block:: python

   da.mean(dim="trials")                       # mean of each variable
   da.sel(variables="y", steps=0).values       # one variable as a 1-D array
   df = da.sel(steps=0).to_pandas()            # trials x variables DataFrame

The same simulation with plain (unstratified) Monte Carlo sampling and a
different seed:

.. code-block:: python

   da = fs.simulate(f=trial, ntrials=5000, sampling="mc", seed=123)


Recursive-dynamic simulation
----------------------------

A recursive-dynamic simulation moves through time.  In each trial,
``simulate`` calls the step function once per time step, and the step
function can look back at values from earlier steps.  To do so, it takes a
second argument, conventionally called ``hist``, and reads lagged values
with ``hist.recall(varname, lag)``.  A lag of one is the previous step.

.. code-block:: python

   import numpy as np
   import xarray as xr
   from scipy import stats
   import funcsim as fs

   def step(ugen, hist):
       # value of "p" in the previous period
       plag = hist.recall("p", lag=1)

       # an AR(1) process with standard normal innovations
       pnew = 0.25 + 0.9 * plag + stats.norm.ppf(next(ugen))

       return {"p": pnew}

Lagged values for the first simulated step have to come from somewhere.
They are supplied via the ``hist0`` argument, a two-dimensional
:class:`xarray.DataArray` with dimensions named ``steps`` and ``variables``
(in either order).  The ``steps`` coordinate must be either integers or a
:class:`pandas.PeriodIndex`.

.. code-block:: python

   hist0 = xr.DataArray(
       data=np.array([[2.4], [2.6], [2.5]]),
       dims=("steps", "variables"),
       coords={"steps": [0, 1, 2], "variables": ["p"]},
   )

   out = fs.simulate(f=step, hist0=hist0, nsteps=10, ntrials=200)

The output has the same three dimensions as before, but now the ``steps``
dimension holds the three historical observations followed by the ten
simulated steps, with the index continued from ``hist0``:

.. code-block:: text

   <xarray.DataArray (trials: 200, variables: 1, steps: 13)> Size: 21kB
   ...
   Coordinates:
     * trials     (trials) int64 2kB 0 1 2 3 4 5 6 ... 194 195 196 197 198 199
     * variables  (variables) <U1 4B 'p'
     * steps      (steps) int64 104B 0 1 2 3 4 5 6 7 8 9 10 11 12

The historical values are repeated identically in every trial, which makes
the output convenient for fan charts (see :doc:`guide_plotting`).

Dated steps
~~~~~~~~~~~

With a :class:`pandas.PeriodIndex` on the ``steps`` dimension of ``hist0``,
the simulated steps continue the calendar:

.. code-block:: python

   import pandas as pd

   hist0 = xr.DataArray(
       data=np.array([[2.4], [2.6], [2.5]]),
       dims=("steps", "variables"),
       coords={"steps": pd.period_range("2024Q1", periods=3, freq="Q"),
               "variables": ["p"]},
   )
   out = fs.simulate(f=step, hist0=hist0, nsteps=4, ntrials=200)
   print(out.sel(variables="p").mean(dim="trials").to_pandas())

.. code-block:: text

   steps
   2024Q1    2.400000
   2024Q2    2.600000
   2024Q3    2.500000
   2024Q4    2.501887
   2025Q1    2.506560
   2025Q2    2.504756
   2025Q3    2.504697
   dtype: float64

A monthly index built from a :class:`pandas.DatetimeIndex`
(``pd.date_range(...).to_period()``) works the same way.

Variables and history
~~~~~~~~~~~~~~~~~~~~~

- The ``variables`` in the output are the union of the variables in
  ``hist0`` and the keys returned by the step function, in that order.
- A variable that the step function returns but that is absent from
  ``hist0`` is ``NaN`` for the historical steps.
- A variable in ``hist0`` that the step function does not return is
  ``NaN`` for the simulated steps.  ``hist0`` may itself contain ``NaN``
  for values that are not known.
- Any lag the step function requests must be available.  Recalling a value
  that does not exist (for example, a lag of two when ``hist0`` has one
  row, or a lag of a variable that an earlier step returned as ``NaN``)
  raises :class:`funcsim.rdarrays.MissingValue` with a message naming the
  variable and lag.
- A step function that needs no history may take only the ``ugen``
  argument; ``hist0`` is then optional.  Steps are then independent across
  time, which is equivalent to a static simulation with ``nsteps``
  variables per trial.

A worked example with two variables, where a call option payoff depends on
a simulated price:

.. code-block:: python

   import math

   def step(ugen, hist):
       p_lag = hist.recall("p", lag=1)
       eps = stats.norm.ppf(next(ugen))
       # geometric Brownian motion, monthly step
       p_new = p_lag * math.exp((0.05 - 0.5 * 0.10**2) / 12.0
                                + eps * 0.10 / 12.0**0.5)
       c_new = max(0.0, p_new - 1.0)
       return {"p": p_new, "c": c_new}

   hist0 = xr.DataArray(
       data=np.array([[1.00, np.nan], [1.01, np.nan], [0.99, np.nan]]),
       dims=("steps", "variables"),
       coords={"steps": [0, 1, 2], "variables": ["p", "c"]},
   )
   out = fs.simulate(f=step, hist0=hist0, nsteps=12, ntrials=1000)
   expected_payoff = float(out.sel(variables="c", steps=14).mean())


Random draws, seeds, and sampling
---------------------------------

``simulate`` creates its own :class:`numpy.random.Generator` from the
``seed`` argument (default 6) and never touches NumPy's global random
state.  Two calls with the same seed, the same ``f``, and the same
``ntrials`` and ``nsteps`` produce identical output.

By default (``sampling="lh"``), draws are stratified across trials: for
each random variable and step, the unit interval is divided into
``ntrials`` equal strata, one draw is taken from each, and the draws are
shuffled across trials.  This Latin hypercube sampling reduces the Monte
Carlo error of estimated means and quantiles considerably for a given
number of trials.  Pass ``sampling="mc"`` for independent draws.

Pass ``stdnorm=True`` to have ``next(ugen)`` return standard normal draws
instead of standard uniform draws, which saves a call to
``stats.norm.ppf`` when every draw is normal:

.. code-block:: python

   def trial(zgen):
       z = next(zgen)              # already standard normal
       return {"z": z}

   da = fs.simulate(f=trial, ntrials=1000, stdnorm=True)

Only uniform draws can be fed to ``ppf`` methods and to the ``draw``
methods of copulas and multivariate distributions, so leave ``stdnorm`` at
its default when using those.


Scenarios and common random numbers
-----------------------------------

Because the draws depend only on the seed and the structure of the
simulation, two scenarios simulated with the same seed see exactly the
same random inputs.  Differences between their outputs are then
attributable to the scenario alone, without Monte Carlo noise.  Use
:func:`functools.partial` to hold scenario parameters fixed:

.. code-block:: python

   import functools

   def trial(ugen, prob):
       eps = stats.norm.ppf(next(ugen))
       b = stats.bernoulli.ppf(next(ugen), prob)
       return {"eps": eps, "b": b}

   low = fs.simulate(f=functools.partial(trial, prob=0.25), ntrials=500)
   high = fs.simulate(f=functools.partial(trial, prob=0.50), ntrials=500)

A ``lambda`` or a closure works as well when ``multi=False``.  Use
``functools.partial`` with a module-level function if you intend to run
the simulation on multiple cores.


Multi-core simulation
---------------------

Pass ``multi=True`` to distribute trials across all available CPU cores
with :mod:`multiprocessing`.  Two requirements follow from the way Python
starts worker processes on macOS and Windows (and, since Python 3.14, on
Linux as well):

1. ``f`` must be picklable.  Define it (and any helper it calls) at the
   top level of a module rather than inside another function, and prefer
   :func:`functools.partial` over ``lambda`` for binding parameters.
2. The call to ``simulate`` must be guarded, or the workers will re-import
   your script and start simulations of their own:

   .. code-block:: python

      if __name__ == "__main__":
          out = fs.simulate(f=trial, ntrials=100000, multi=True)

An exception raised by ``f`` inside a worker is re-raised in the parent
process as a :class:`RuntimeError` that carries the original message.


Errors you may see
------------------

``ValueError``
   ``hist0`` is not a DataArray with dimensions ``steps`` and
   ``variables``, or its ``steps`` index is neither integer nor a
   PeriodIndex; ``f`` is not callable, takes more than two arguments, or
   returns something other than a dict keyed by strings; or ``sampling`` is
   not ``"lh"`` or ``"mc"``.

``RuntimeError``
   ``f`` consumed a different number of draws in some trial or step than
   it did when first probed (see the fixed-draw rule above), or a
   multi-core worker failed.

``funcsim.rdarrays.MissingValue``
   ``hist.recall`` asked for a value that is not available.
