Expected Utility, Prospect Theory, and Regret
=============================================

Simulation output is often a distribution of monetary outcomes, and the
question is how a decision maker with given risk preferences values it.
funcsim provides three frameworks: expected utility theory (EUT),
cumulative prospect theory (CPT), and regret-theoretical expected
utility (RTEU).  Each takes a vector of
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


Regret-theoretical expected utility
-----------------------------------

Under regret-theoretical expected utility (Braun and Muermann, 2004) the
decision maker values an outcome by its utility less a penalty for the
regret of not having chosen the alternative that turned out best in the
same state of nature:

.. math::

   RTU(y) = \psi(y) - k \, g\big(\psi(y_{\max}) - \psi(y)\big)

where :math:`\psi` is a utility function of the outcome, :math:`g` a
convex regret function of the utility gap with :math:`g(0) = 0`,
:math:`k \geq 0` the regret coefficient, and :math:`y_{\max}` the best
outcome across every scenario in the menu in that state.  RTEU is the
mean of this index over states.  Because :math:`y_{\max}` is defined
state by state, every scenario must be simulated on the same states of
nature: run each through ``simulate`` at the same seed and number of
trials, so that trial n of every scenario is the same state.

Regret theory originates with Bell (1982) and Loomes and Sugden (1982)
for pairwise choice; Sugden (1993) gives its axiomatic foundation, and
Quiggin (1994) extends it to menus of any size, where measuring regret
against the best outcome in each state is the form implied by his
irrelevance of statewise dominated alternatives.

:func:`funcsim.rteu` evaluates one scenario.  Besides the utility and
regret functions and the regret coefficient, it takes the scenario's
outcomes and ``best``, the best attainable outcome in each state across
the menu, which you compute from the scenario outputs:

.. code-block:: python

   import functools
   import numpy as np
   from scipy import stats
   import funcsim as fs

   def trial(ugen, dist):
       return {"w": float(dist.ppf(next(ugen)))}

   scenarios = {"A": stats.norm(loc=50.0, scale=15.0),
                "B": stats.chi2(df=47.0)}
   wealth = {name: fs.simulate(f=functools.partial(trial, dist=dist))
                     .sel(variables="w", steps=0).to_numpy()
             for name, dist in scenarios.items()}
   best = np.maximum(wealth["A"], wealth["B"])

   g = functools.partial(fs.regretPower, exponent=1.2)
   fs.rteu(np.log, g, 1.5, wealth["A"], best)
   # RteuResult(ExpectedValue=3.81, CertaintyEquiv=47.02)
   fs.rteu(np.log, g, 1.5, wealth["B"], best)
   # RteuResult(ExpectedValue=3.76, CertaintyEquiv=46.39)

Scenario A is preferred by ``ExpectedValue``.  At ``regretCoeff=0.0``
the results are those of expected utility, and the certainty
equivalents (47.37 and 46.01) match :func:`funcsim.eutIsoelastic` at a
coefficient of relative risk aversion of one, where the isoelastic
function is the natural log.

The certainty equivalent is the menu-based definition of the regret
literature (Loomes and Sugden, 1983; Diecidue and Somasundaram, 2017;
Gollier, 2020): the sure amount the decision maker is indifferent to
when it and the scenario are both in the menu, each evaluated against
the best attainable outcome, which becomes the larger of ``best`` and
the sure amount in every state.  A sure amount is itself regret-prone
within the menu, so the gap between a scenario's expected outcome and
its certainty equivalent is not the risk premium of expected utility
theory, and certainty equivalents can order two scenarios differently
from the choice between them; the choice rests on ``ExpectedValue``.
The regret coefficient has no accepted range, so a sensitivity analysis
over ``k`` is the practical check:

.. code-block:: python

   {k: round(fs.rteu(np.log, g, k, wealth["A"], best).CertaintyEquiv, 2)
    for k in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)}
   # {0.0: 47.37, 0.5: 47.18, 1.0: 47.08, 1.5: 47.02, 2.0: 46.98, 3.0: 46.93}

:func:`funcsim.regretPower` is the ready-made regret function,
``x ** exponent`` with ``exponent`` at least one; any function of one
non-negative float with ``g(0) = 0`` may be passed instead.


References
----------

Bell, D. E. (1982). Regret in decision making under uncertainty.
*Operations Research*, 30(5), 961-981.

Bouchouicha, R., & Vieider, F. M. (2017). Accommodating stake effects
under prospect theory. *Journal of Risk and Uncertainty*, 55(1), 1-28.

Braun, M., & Muermann, A. (2004). The impact of regret on the demand for
insurance. *Journal of Risk and Insurance*, 71(4), 737-767.

Diecidue, E., & Somasundaram, J. (2017). Regret theory: A new foundation.
*Journal of Economic Theory*, 172, 88-119.

Gollier, C. (2020). Aversion to risk of regret and preference for
positively skewed risks. *Economic Theory*, 70(4), 913-941.

Loomes, G., & Sugden, R. (1982). Regret theory: An alternative theory of
rational choice under uncertainty. *Economic Journal*, 92(368), 805-824.

Loomes, G., & Sugden, R. (1983). A rationale for preference reversal.
*American Economic Review*, 73(3), 428-432.

Prelec, D. (1998). The probability weighting function. *Econometrica*,
66(3), 497-527.

Quiggin, J. (1994). Regret theory with general choice sets. *Journal of
Risk and Uncertainty*, 8(2), 153-165.

Rachlin, H. (1992). Diminishing marginal value as delay discounting.
*Journal of the Experimental Analysis of Behavior*, 57(3), 407-415.

Sugden, R. (1993). An axiomatic foundation for regret theory. *Journal
of Economic Theory*, 60(1), 159-180.

Tversky, A., & Kahneman, D. (1992). Advances in prospect theory:
Cumulative representation of uncertainty. *Journal of Risk and
Uncertainty*, 5(4), 297-323.
