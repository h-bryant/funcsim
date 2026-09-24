Multivariate Distributions and Dependence
=========================================

Objects representing joint distributions and copulas, plus supporting
tools for correlation matrices.  Every distribution and copula class
exposes a ``draw(ugen)`` method that consumes a fixed number of values
from the uniform generator supplied to a trial or step function and
returns a :class:`pandas.Series` indexed by variable name.  ``MvNorm``
and the copula classes also provide a ``from_params`` class method that
builds the object from specified parameters instead of data, and the
copula classes expose their fitted or specified parameters as read-only
properties (``rho``, ``nu``, ``theta``; ``CopulaStudent`` also reports
``loglik_gain``, the log-likelihood gain of its fitted ``nu`` over the
Gaussian limit).  :func:`funcsim.copcompare` ranks
the copula families by information criteria.  See :doc:`guide_dependence`
for a narrative introduction with examples.

Multivariate distributions
--------------------------

.. automodule:: funcsim
   :no-index:
   :members: MvNorm, MvKde
   :undoc-members:
   :show-inheritance:

Copulas
-------

.. automodule:: funcsim
   :no-index:
   :members: CopulaGauss, CopulaStudent, CopulaClayton, CopulaGumbel, CopulaFrank, copcompare
   :undoc-members:
   :show-inheritance:

Rank correlation and covariance tools
-------------------------------------

.. automodule:: funcsim
   :no-index:
   :members: imanconover, spearman, covtocorr, shrink, nearestpd
   :undoc-members:
   :show-inheritance:
