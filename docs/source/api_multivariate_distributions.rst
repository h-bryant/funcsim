Multivariate Distributions and Dependence
=========================================

Objects representing joint distributions and copulas, plus supporting
tools for correlation matrices.  Every distribution and copula class
exposes a ``draw(ugen)`` method that consumes a fixed number of values
from the uniform generator supplied to a trial or step function and
returns a :class:`pandas.Series` indexed by variable name.  See
:doc:`guide_dependence` for a narrative introduction with examples.

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
   :members: CopulaGauss, CopulaStudent, CopulaClayton, CopulaGumbel, CopulaFrank
   :undoc-members:
   :show-inheritance:

Rank correlation and covariance tools
-------------------------------------

.. automodule:: funcsim
   :no-index:
   :members: imanconover, spearman, covtocorr, shrink, nearestpd
   :undoc-members:
   :show-inheritance:
