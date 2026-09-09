Expected Utility and Prospect Theory
====================================

Simulation output is often a distribution of monetary outcomes, and the
question is how a decision maker with given risk preferences values it.
funcsim provides the two standard frameworks: expected utility theory
(EUT) and cumulative prospect theory (CPT).  Each takes a vector of
outcomes, treated as equally likely unless probabilities are given, and
returns a named tuple with the expected (weighted) utility and the
certainty equivalent.  A typical input is one variable and step from a
simulation:

.. code-block:: python

   outcomes = da.sel(variables="income", steps=0).values


Expected utility
----------------

:func:`funcsim.eutIsoelastic` computes expected utility and the certainty
equivalent under isoelastic (constant relative risk aversion, CRRA)
utility, given the coefficient of relative risk aversion.  All outcomes
must be positive.

.. code-block:: python

   import numpy as np
   import funcsim as fs

   outcomes = np.array([80.0, 100.0, 120.0, 150.0])

   fs.eutIsoelastic(2.0, outcomes)
   # EutResult(ExpectedValue=0.9906, CertaintyEquiv=106.67)

The certainty equivalent is the sure amount that yields the same utility
as the gamble; here it is below the mean outcome of 112.5, reflecting risk
aversion.

:func:`funcsim.utilIsoelastic` is the underlying utility function, and
:func:`funcsim.eut` accepts any utility function of one argument, solving
numerically for the certainty equivalent:

.. code-block:: python

   fs.eut(lambda y: fs.utilIsoelastic(y, 2.0), outcomes)
   # EutResult(ExpectedValue=0.9906, CertaintyEquiv=106.67)

Prefer ``eutIsoelastic`` when CRRA utility is what you want; the
closed-form solution avoids the numerical root finding in ``eut``.


Cumulative prospect theory
--------------------------

Under CPT (Tversky and Kahneman, 1992), outcomes are evaluated as gains
and losses relative to a reference point, with a utility (value) function
that is concave for gains, convex for losses, and steeper for losses
(loss aversion), and with cumulative probabilities distorted by a
weighting function.  :func:`funcsim.cpt` implements the general
calculation; you supply the utility function and the weighting functions
for gains and for losses.

.. code-block:: python

   outcomes = np.array([-20.0, 0.0, 20.0, 50.0])

   r = fs.cpt(utilFunc=fs.utilPower,
              weightFuncGains=lambda p: fs.weightTK(p, 0.61),
              weightFuncLosses=lambda p: fs.weightTK(p, 0.69),
              outcomes=outcomes,
              refOutcome=0.0)
   # CptResult(ExpectedValue=1.684, CertaintyEquiv=1.809)

Explicit probabilities, which must sum to one, may be supplied for
outcomes that are not equally likely:

.. code-block:: python

   fs.cpt(fs.utilPower,
          lambda p: fs.weightPrelec1(p, 0.65),
          lambda p: fs.weightPrelec1(p, 0.65),
          outcomes=[-10.0, 30.0], refOutcome=0.0,
          probabilities=[0.5, 0.5])
   # CptResult(ExpectedValue=1.309, CertaintyEquiv=1.358)

The certainty equivalent solves ``utilFunc(CE - refOutcome) = CPT value``
to full floating-point precision; the ``precision`` argument is retained
for backward compatibility and has no effect.

Building blocks
~~~~~~~~~~~~~~~

Utility (value) functions:

:func:`funcsim.utilPower`
   The Tversky and Kahneman (1992) power function with curvature
   ``alpha`` for gains and ``beta`` for losses and loss aversion ``lamb``.
   Defaults (0.88, 0.88, 2.25) are the original estimates.

:func:`funcsim.utilNormLog`
   The normalized logarithmic function of Rachlin (1992), with curvature
   ``gamma`` for gains and ``delta`` for losses and loss aversion ``lamb``.

Probability weighting functions:

:func:`funcsim.weightTK`
   Tversky and Kahneman (1992), one parameter ``gamma`` in [0.28, 1.0].

:func:`funcsim.weightPrelec1`
   Prelec (1998) one-parameter form.

:func:`funcsim.weightPrelec2`
   Prelec (1998) two-parameter form, with sensitivity ``alpha`` and
   elevation ``beta``.

Any function of a single outcome can serve as ``utilFunc`` and any
function of a single probability as a weighting function, so these are
conveniences rather than requirements.  :func:`funcsim.twofuncs` (see
:doc:`guide_plotting`) is handy for comparing two utility or weighting
functions visually.

A ready-made specification
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`funcsim.cptBV` applies the functional forms and parameter values
recommended by Bouchouicha and Vieider (2017): ``utilNormLog`` with
gamma 1.223, delta 0, and lamb 2.25, and ``weightPrelec2`` with
(alpha, beta) of (0.53, 0.969) for gains and (0.623, 0.953) for losses.

.. code-block:: python

   fs.cptBV(np.array([-20.0, 0.0, 20.0, 50.0]), refOutcome=0.0)
   # CptResult(ExpectedValue=-12.571, CertaintyEquiv=-5.587)

Compared with the Tversky-Kahneman parameters above, this specification
weights the loss much more heavily, producing a negative certainty
equivalent for the same gamble.


References
----------

Bouchouicha, R., & Vieider, F. M. (2017). Accommodating stake effects
under prospect theory. *Journal of Risk and Uncertainty*, 55(1), 1-28.

Prelec, D. (1998). The probability weighting function. *Econometrica*,
66(3), 497-527.

Rachlin, H. (1992). Diminishing marginal value as delay discounting.
*Journal of the Experimental Analysis of Behavior*, 57(3), 407-415.

Tversky, A., & Kahneman, D. (1992). Advances in prospect theory:
Cumulative representation of uncertainty. *Journal of Risk and
Uncertainty*, 5(4), 297-323.
