Multivariate Distributions and Dependence
=========================================

When the inputs to a simulation are not independent, a trial function
needs joint draws.  funcsim provides three approaches, all of which fit to
data and then produce one joint draw per call from the uniform generator
passed to the trial function:

- :class:`funcsim.MvNorm` and :class:`funcsim.MvKde` model the joint
  distribution directly (multivariate normal, or multivariate kernel
  density estimate).
- The copula classes (:class:`funcsim.CopulaGauss`,
  :class:`funcsim.CopulaStudent`, :class:`funcsim.CopulaClayton`,
  :class:`funcsim.CopulaGumbel`, :class:`funcsim.CopulaFrank`) model the
  dependence separately from the marginals.  Their draws are dependent
  standard uniforms, which you feed through the marginal ``ppf`` of your
  choice.
- :func:`funcsim.imanconover` works after the fact, rearranging
  independently generated vectors so that they exhibit a target Spearman
  correlation matrix.

All of the distribution and copula classes share the same interface: pass
the data (variables in columns, observations in rows, as a NumPy array,
:class:`pandas.DataFrame`, or :class:`xarray.DataArray`) to the
constructor, then call ``draw(ugen)`` inside the trial function.  ``draw``
returns a :class:`pandas.Series` indexed by variable name (the column
names of the input, or ``v0``, ``v1``, ... for a bare array).

Draws consumed per call
-----------------------

Because :func:`funcsim.simulate` requires a fixed number of draws per
trial (see :doc:`overview`), it is useful to know how many values each
``draw`` call consumes from ``ugen``.  With K variables:

=========================  ==================
Class                      Draws per ``draw``
=========================  ==================
``MvNorm``                 K
``MvKde``                  K + 1
``CopulaGauss``            K
``CopulaStudent``          K + 1
``CopulaClayton``          K + 1
``CopulaGumbel``           K + 2
``CopulaFrank``            K + 2
=========================  ==================

The count is constant for a given object, so the fixed-draw rule is
satisfied automatically as long as ``draw`` is called the same number of
times in every trial.


Multivariate normal
-------------------

:class:`funcsim.MvNorm` estimates a mean vector and covariance matrix from
the data.  If the sample covariance matrix is not positive definite, the
nearest positive definite matrix (Higham, 1988) is substituted with a
warning.  A warning is also issued for any variable that fails a
Shapiro-Wilk test of normality at the 5% level.

.. code-block:: python

   import pandas as pd
   import funcsim as fs

   data = pd.DataFrame(..., columns=["a", "b"])   # observations in rows
   mvn = fs.MvNorm(data)

   def trial(ugen):
       d = mvn.draw(ugen)          # pd.Series with index ["a", "b"]
       return {"a": d["a"], "b": d["b"]}

   da = fs.simulate(f=trial, ntrials=2000)
   da.sel(steps=0).to_pandas().cov()   # close to data.cov()


Multivariate kernel density estimate
------------------------------------

:class:`funcsim.MvKde` places a Gaussian kernel on each observation.  Data
are standardized internally, and the bandwidth matrix is chosen by Scott's
rule (default), Silverman's rule (``bw="silverman"``), or supplied as a
K-by-K covariance matrix in the units of the data.  A draw picks one
observation at random and perturbs it with the kernel.

.. code-block:: python

   mvk = fs.MvKde(data)                 # or fs.MvKde(data, bw="silverman")

   def trial(ugen):
       d = mvk.draw(ugen)
       return dict(d)


Copulas
-------

A copula separates the dependence structure from the marginal
distributions.  The workflow has three parts.

**1. Build pseudo-observations.**  Transform each column of the data to
the unit interval.  Either apply the CDF of a fitted marginal (see
:doc:`guide_distributions`), or use scaled ranks, which make no
distributional assumption:

.. code-block:: python

   # rank-based pseudo-observations, strictly inside (0, 1)
   udata = data.rank() / (len(data) + 1)

   # or, using fitted marginals
   fa = fs.fit(data["a"], stats.norm).dist
   fb = fs.fit(data["b"], stats.gamma, lowerBound=0.0).dist
   udata = pd.DataFrame({"a": fa.cdf(data["a"]), "b": fb.cdf(data["b"])})

Every value must lie strictly between 0 and 1; the constructors raise
:class:`ValueError` otherwise.  :func:`funcsim.utests` can be used to check
that each column is plausibly uniform.

**2. Fit the copula.**

.. code-block:: python

   cop = fs.CopulaGauss(udata)

**3. Draw dependent uniforms and apply marginal inverse CDFs.**

.. code-block:: python

   def trial(ugen):
       u = cop.draw(ugen)           # dependent standard uniforms
       return {"a": fa.ppf(u["a"]), "b": fb.ppf(u["b"])}

   da = fs.simulate(f=trial, ntrials=2000)

With this recipe, the simulated correlation matches the data closely
(0.422 simulated versus 0.418 in the data for the two-variable example
used throughout this page).

Copula families
~~~~~~~~~~~~~~~

``CopulaGauss``
   Elliptical, symmetric, no tail dependence.  The correlation matrix is
   estimated by the method of moments on the normal scores of the
   pseudo-observations.

``CopulaStudent``
   Elliptical, symmetric, with tail dependence in both tails.  The
   correlation matrix is estimated by pairwise Kendall's-tau inversion and
   the degrees of freedom by profile maximum pseudo-likelihood.

``CopulaClayton``
   Archimedean, with lower-tail dependence (joint crashes).

``CopulaGumbel``
   Archimedean, with upper-tail dependence (joint booms).

``CopulaFrank``
   Archimedean, symmetric, no tail dependence.

The three Archimedean families each have a single dependence parameter
theta, fitted by maximum pseudo-likelihood using the exact K-dimensional
copula density, starting from the Kendall's-tau inversion estimate.  They
accommodate **positive dependence only**.  If the data exhibit negative
dependence on average (mean pairwise Kendall's tau at or below zero), the
parameter is set to its minimum, which implies near independence, and a
warning suggests choosing a different representation.  All families
support any number of variables.


Iman-Conover rank correlation
-----------------------------

:func:`funcsim.imanconover` takes vectors that were generated
independently (for instance, one per variable from any distributions you
like) and reorders them so that their Spearman rank correlation matrix
matches a target, leaving each marginal distribution exactly as it was.

.. code-block:: python

   import numpy as np
   from scipy import stats

   x = stats.norm.rvs(size=500, random_state=1)
   y = stats.expon.rvs(size=500, random_state=2)
   target = np.array([[1.0, 0.7], [0.7, 1.0]])

   df = fs.imanconover(target, [x, y], names=["x", "y"], seed=0)
   df.corr(method="spearman")
   #        x      y
   # x  1.000  0.695
   # y  0.695  1.000

The method involves a random shuffle of scores; pass ``seed`` for
reproducible output.  Iman-Conover is a pre-processing tool rather than a
``draw`` object, so it is used outside of :func:`funcsim.simulate`, for
example to build a correlated input dataset or to post-process
independently simulated variables.


Correlation and covariance helpers
----------------------------------

:func:`funcsim.spearman`
   Spearman's rho for two variables, with a 95% confidence interval from
   Fisher's z-transformation.

:func:`funcsim.covtocorr`
   Convert a covariance matrix to a correlation matrix (as a DataFrame).

:func:`funcsim.shrink`
   Shrinkage estimate of a covariance matrix from data, using one of the
   six targets of Schafer and Strimmer (2005), labeled ``'A'`` through
   ``'F'``.  Useful when the number of observations is small relative to
   the number of variables.  The result has the same type as the input.

:func:`funcsim.nearestpd`
   The nearest symmetric positive definite matrix to a given matrix
   (Higham, 1988).  ``MvNorm`` and the copulas call this internally when
   needed.
