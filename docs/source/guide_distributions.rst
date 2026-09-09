Univariate Distributions
========================

A trial function needs a way to turn a standard uniform draw into a draw
from the distribution of each input variable.  This page covers the tools
for choosing that distribution from data (:func:`funcsim.compare`),
fitting it (:func:`funcsim.fit`), representing it nonparametrically
(:class:`funcsim.Kde`, :func:`funcsim.edf`), and checking the fit
(goodness-of-fit tests).  The end product in every case is a ``ppf``
(inverse CDF) callable to use inside a trial function.


Comparing candidate distributions
---------------------------------

:func:`funcsim.compare` fits every candidate family with the requested kind
of support to a data sample and returns a formatted table sorted by the
Bayesian information criterion (BIC).  The candidate list covers more than
one hundred continuous families from :mod:`scipy.stats`.

The kind of support is specified with the keyword-only arguments
``lowerBound`` and ``upperBound``:

- Omit both to compare distributions supported on the whole real line.
- Pass ``lowerBound=0.0`` (for example) to compare distributions bounded
  below, each fitted with its lower bound fixed at zero.
- Pass ``upperBound`` likewise for distributions bounded above, or both
  bounds for distributions with bounded support on both sides.

.. code-block:: python

   import numpy as np
   from scipy import stats
   import funcsim as fs

   # 200 observations of a positive quantity
   sample = stats.gamma(a=2.0, scale=3.0).rvs(size=200, random_state=0)

   print(fs.compare(sample, lowerBound=0.0))

.. code-block:: text

                     distribution,      BIC,      AIC, AD_score, CvM_score
                            Gamma, 1051.632, 1045.036,    0.954,     0.900
                            Chi^2, 1051.632, 1045.036,    0.954,     0.900
                Generalized Gamma, 1056.592, 1046.697,    0.981,     0.952
     Generalized Inverse Gaussian, 1056.597, 1046.703,    0.978,     0.953
            Exponentiated Weibull, 1056.658, 1046.763,    0.978,     0.946
                 Power log-normal, 1056.689, 1046.794,    0.977,     0.947
                                F, 1056.715, 1046.820,    0.973,     0.935
                       Beta Prime, 1056.715, 1046.820,    0.973,     0.935
        Weibull Min Extreme Value, 1056.736, 1050.139,    0.604,     0.593
                 Noncentral chi^2, 1056.931, 1047.036,    0.954,     0.900
   ...

Points to keep in mind:

- **Fixing a bound is a constraint, not truncation.**  A scipy distribution
  with standard support :math:`[a, b]` has support
  :math:`[\mathrm{loc} + a\,\mathrm{scale},\ \mathrm{loc} + b\,\mathrm{scale}]`.
  Fixing the lower bound pins one combination of location and scale.  Each
  fixed bound removes one free parameter, and the parameter counts used in
  the AIC and BIC are reduced accordingly.  Any fixed bound must be finite
  and must not exclude any observation.
- **The AD_score and CvM_score columns are not p-values.**  They are the
  nominal p-values of the Anderson-Darling and Cramer-von Mises tests,
  which assume a fully specified null distribution, whereas the parameters
  here are estimated from the same sample (the Lilliefors problem).  They
  systematically overstate goodness of fit.  Use them only as heuristic
  scores for ranking; rank by BIC or AIC when comparing families with
  different numbers of parameters.
- **Sample size.**  At least 50 observations are recommended; a warning is
  issued for smaller samples.
- **Families with shape-dependent support** (generalized extreme value,
  generalized Pareto, four-parameter kappa, and Tukey lambda) can be
  bounded on different sides depending on their shape parameters.  They
  are tried in every group they could belong to, and a fitted
  distribution whose support does not match the requested group is dropped
  with a warning.
- **A family that fails to fit** is dropped with a warning rather than
  aborting the comparison.

Legacy arguments
~~~~~~~~~~~~~~~~

Earlier versions of funcsim selected the support with the boolean
positional arguments ``lowerLimit`` and ``upperLimit``, in which case the
bound was estimated along with the other parameters rather than fixed.
These arguments still work in 0.2.3 but emit a :class:`FutureWarning` and
will be removed in a future version.  A legacy flag for one side of the
support may be combined with a fixed bound for the other side
(``compare(data, upperLimit=True, lowerBound=0.0)``), but not with a bound
on the same side.


Fitting one distribution
------------------------

:func:`funcsim.fit` fits a single :mod:`scipy.stats` family by maximum
likelihood, with the same optional fixed bounds as ``compare``.  It returns
a named tuple with the information criteria, the heuristic goodness-of-fit
scores, and the frozen fitted distribution.

.. code-block:: python

   r = fs.fit(sample, stats.gamma, lowerBound=0.0)

   r.bic, r.aic            # 1051.63, 1045.04
   r.dist.args             # (2.186, 0.0, 2.620): shape, loc, scale
   r.dist.mean()           # 5.73
   r.dist.ppf(0.5)         # 4.88

The frozen distribution in ``r.dist`` is what a trial function needs:

.. code-block:: python

   income_dist = fs.fit(sample, stats.gamma, lowerBound=0.0).dist

   def trial(ugen):
       income = income_dist.ppf(next(ugen))
       return {"income": income}

Fit the distribution once, outside the trial function, rather than
refitting it in every trial.


Nonparametric representations
-----------------------------

When no parametric family is satisfactory, :class:`funcsim.Kde` represents
a sample by a Gaussian kernel density estimate with ``pdf``, ``cdf``, and
``ppf`` methods.  The bandwidth is chosen by Scott's rule by default;
pass ``bw="silverman"``, or a positive float to set the kernel standard
deviation directly in the units of the data.

.. code-block:: python

   kde = fs.Kde(sample)
   kde.pdf(5.0), kde.cdf(5.0), kde.ppf(0.5)     # 0.103, 0.509, 4.92

   def trial(ugen):
       return {"income": kde.ppf(next(ugen))}

The ``ppf`` is computed numerically by root finding on the ``cdf``, so a
``Kde`` is slower to sample from than a parametric distribution.

:func:`funcsim.edf` returns the empirical distribution function of a sample
as a plain callable:

.. code-block:: python

   F = fs.edf(sample)
   F(5.0)                                       # 0.52


Goodness-of-fit tests
---------------------

:func:`funcsim.kstest`, :func:`funcsim.adtest`, and :func:`funcsim.cvmtest`
implement the Kolmogorov-Smirnov, Anderson-Darling, and Cramer-von Mises
tests of a sample against a fully specified distribution.  Each returns a
named tuple with ``statistic`` and ``pvalue`` fields.  The hypothesized
distribution may be given as a frozen scipy distribution, as an unfrozen
distribution with parameters in ``args``, or by name.

.. code-block:: python

   fs.kstest(sample, stats.gamma(2.0, scale=3.0))
   # GofResult(statistic=0.054, pvalue=0.580)

   fs.adtest(sample, "norm", args=(6.0, 4.0))
   # GofResult(statistic=6.696, pvalue=0.00046)

These p-values are valid when the parameters are known a priori.  When the
parameters were estimated from the same sample, the p-values are
optimistic; see the discussion of ``compare`` above.

:func:`funcsim.swtest` is the Shapiro-Wilk test of normality, and
:func:`funcsim.utests` bundles four tests of the hypothesis that a sample
is standard uniform (useful for checking pseudo-observations before
fitting a copula):

.. code-block:: python

   fs.swtest(sample)
   # ShapiroResult(statistic=0.896, pvalue=1.4e-10)

   fs.utests(stats.gamma(2.0, scale=3.0).cdf(sample))
   # {'cook_gelman_rubin_pval': 0.872, 'kolmogorov_smirnov_pval': 0.580,
   #  'anderson_darling_pval': 0.613, 'cramer_von_mises_pval': 0.547}


Screening a time series
-----------------------

Fitting a distribution to a time series presumes that the observations are
independent and identically distributed.  :func:`funcsim.screen` runs a
battery of tests for stationarity, heteroskedasticity, autocorrelation,
and randomness and returns a formatted report, flagging any rejection at
the chosen significance level (default 5%):

.. code-block:: python

   print(fs.screen(series))

.. code-block:: text

   Screening series at the 5% level:

     A) H0: series is not stationary (ADF)
              p-value: 0.000

     B) H0: series is not trend stationary (ADF)
              p-value: 0.007

     C) H0: homoskedasticity (White's)
              p-value: 0.065

     D) H0: no first-order autocorrelation (Ljung-Box)
              p-value: 0.913
        H0: no second-order autocorrelation (Ljung-Box)
              p-value: 0.935
        H0: no third-order autocorrelation (Ljung-Box)
              p-value: 0.003 (WARNING: possible non-i.i.d. properties)

     E) H0: series is i.i.d. (Wald-Wolfowitz)
              p-value: 0.714

     F) H0: series is i.i.d. (Bartels)
              p-value: 0.850

Note that the two augmented Dickey-Fuller tests (A and B) have
non-stationarity as their null hypothesis, so for those lines a small
p-value is reassuring; the report flags the opposite outcome.
