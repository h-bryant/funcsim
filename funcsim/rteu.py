"""regret-theoretical expected utility (RTEU) and certainty equivalents"""

import math
from collections import namedtuple
from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray
import scipy
from . import conversions


# container for results
RteuResult = namedtuple("RteuResult", "ExpectedValue CertaintyEquiv")


def regretPower(x: Union[float, NDArray],
                exponent: float,
                ) -> Union[float, NDArray]:
    """
    Compute power regret for a utility gap.

    Parameters
    ----------
    x : float or numpy.ndarray
        The utility gap psi(y_max) - psi(y), non-negative; an array is
        evaluated elementwise.
    exponent : float
        Curvature of regret, at least 1.0.  Values above one make the regret
        function strictly convex, so a few large regrets weigh more than
        many small ones; at exactly one, regret is linear in the gap and the
        choice between scenarios coincides with expected utility's, although
        the certainty equivalents do not.

    Returns
    -------
    float or numpy.ndarray
        ``x ** exponent``, a float for a float and an array for an array.

    Raises
    ------
    ValueError
        If ``x`` (or any element of it) is negative or ``exponent`` is below
        1.0.

    Notes
    -----
    This is the one ready-made regret function; any function of one
    non-negative float with g(0) = 0 may be passed as ``regretFunc``.  It
    accepts arrays as well as floats so that a user's own per-state code can
    call it on whole vectors; ``rteu`` evaluates it through
    ``numpy.vectorize`` like any other ``regretFunc``.  There is no
    canonical parameterization in the literature comparable to Tversky and
    Kahneman's for cumulative prospect theory, so no preset analogous to
    ``cptBV`` is provided.
    """
    if exponent < 1.0:
        raise ValueError(f"exponent (={exponent}) must be at least 1.0")
    xa = np.asarray(x, dtype=float)
    if np.any(xa < 0.0):
        raise ValueError(f"x (={x}) must be non-negative: the argument of "
                         "a regret function is a utility gap")
    powered = xa ** exponent
    return float(powered) if powered.ndim == 0 else powered


def _rtu(psiY, psiBest, vg, k):
    # psi(y) - k g(psi(best) - psi(y)), elementwise, from utilities that
    # have already been evaluated; vg is the vectorized regret function.
    # psiY may be a scalar (a sure amount), in which case it broadcasts
    return psiY - k * vg(psiBest - psiY)


def _finite_or_raise(values, where, label):
    # raise if any evaluated utility is not finite, naming the first culprit
    bad = ~np.isfinite(values)
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(f"utilFunc is not finite at {label} {where[i]} "
                         f"(position {i})")


def rteu(utilFunc: Callable[[float], float],
         regretFunc: Callable[[float], float],
         regretCoeff: float,
         outcomes: conversions.VectorLike,
         best: conversions.VectorLike,
         ) -> RteuResult:
    """
    Compute regret-theoretical expected utility (RTEU) and the certainty
    equivalent of one scenario, given the best attainable outcome in each
    state of nature across the decision maker's menu.

    Parameters
    ----------
    utilFunc : Callable
        Utility function of the monetary outcome, psi, taking a float and
        returning a float.  It must be strictly increasing; the theory also
        assumes it concave.  See ``utilIsoelastic``.
    regretFunc : Callable
        Regret function of the utility gap, g, taking a non-negative float
        and returning a float, with g(0) = 0, g' > 0, and g'' > 0.  See
        ``regretPower``.
    regretCoeff : float
        Regret coefficient k, non-negative.  At zero the results are those
        of expected utility theory.
    outcomes : VectorLike
        The scenario's outcome in each state of nature (one value per
        simulation trial).
    best : VectorLike
        The best attainable outcome in the same state of nature across every
        scenario in the menu, the same length as ``outcomes`` and no smaller
        than it in any state: typically the row-wise maximum across the
        scenario columns, which includes this scenario's own outcome.
        Element n of ``outcomes`` and of ``best`` must belong to the same
        state of nature, which ``simulate`` delivers when every scenario is
        run at the same seed and number of trials.

    Returns
    -------
    RteuResult
        Named tuple with fields:

        ExpectedValue : float
            The regret-theoretical expected utility of the scenario, the
            mean over states of psi(y) - k g(psi(best) - psi(y)).
        CertaintyEquiv : float
            The certainty equivalent of the scenario within the menu: the
            sure amount c such that the decision maker is indifferent
            between c and the scenario when both are in the menu, each
            evaluated against the best attainable outcome max(best, c) in
            every state.

    Raises
    ------
    ValueError
        If ``outcomes`` and ``best`` differ in length, ``best`` is smaller
        than ``outcomes`` in any state, ``regretCoeff`` is negative,
        ``regretFunc(0.0)`` is not zero, or ``utilFunc`` is not finite at
        every outcome (for example a logarithmic psi applied to a
        non-positive outcome).

    Notes
    -----
    The per-state index follows Braun and Muermann (2004); measuring regret
    against the best attainable outcome in each state is the form implied
    by Quiggin's (1994) irrelevance of statewise dominated alternatives,
    and it is why the rest of the menu enters only through ``best``.  The
    certainty equivalent is the menu-based definition of the regret theory
    literature (Loomes and Sugden, 1983; Diecidue and Somasundaram, 2017;
    Gollier, 2020).  It is found by Brent's method between the scenario's
    smallest outcome and the largest value of ``best``, where the
    indifference condition has opposite signs; the sure-amount side is
    strictly increasing in c and the scenario side non-increasing, so the
    root is unique.  At ``regretCoeff`` = 0 the certainty equivalent is the
    expected utility certainty equivalent, psi^-1(E[psi(y)]), and
    ``ExpectedValue`` is E[psi(y)].  States are equally likely.  Passing the
    scenario's own outcomes as ``best`` describes a menu of the scenario
    and the sure amount alone, the two-element setting of Loomes and Sugden
    (1983).  Because a sure amount is itself regret-prone within the menu,
    the gap between a scenario's expected outcome and its certainty
    equivalent is not the risk premium of expected utility theory (Gollier,
    2020), and the certainty equivalents can order two scenarios
    differently from the choice between them; the choice rests on
    ``ExpectedValue``.  The scale of ``utilFunc`` matters: k weights regret
    measured in utility units, so k is not comparable across differently
    scaled psi.  The utility and regret functions are evaluated through
    ``numpy.vectorize``, so any function of one float works.
    """
    if regretCoeff < 0.0:
        raise ValueError(f"regretCoeff (={regretCoeff}) must be non-negative")
    g0 = float(regretFunc(0.0))
    if not math.isclose(g0, 0.0, abs_tol=1e-12):
        raise ValueError(f"regretFunc(0.0) must be zero (got {g0})")

    y = conversions.vlToArray(outcomes).astype(float)
    b = conversions.vlToArray(best).astype(float)
    if len(y) != len(b):
        raise ValueError(f"outcomes (length {len(y)}) and best (length "
                         f"{len(b)}) must have the same length")
    below = b < y
    if np.any(below):
        i = int(np.argmax(below))
        raise ValueError("best must be the best attainable outcome in each "
                         "state, so no smaller than outcomes; at position "
                         f"{i} best={b[i]} is below outcomes={y[i]}")

    vpsi = np.vectorize(utilFunc, otypes=[float])
    vg = np.vectorize(regretFunc, otypes=[float])
    with np.errstate(invalid="ignore", divide="ignore"):
        psiY = vpsi(y)
        psiB = vpsi(b)
    _finite_or_raise(psiY, y, "outcome")
    _finite_or_raise(psiB, b, "best")
    k = float(regretCoeff)

    expected = float(np.mean(_rtu(psiY, psiB, vg, k)))

    # certainty equivalent: the sure amount c at which the decision maker is
    # indifferent between c and the scenario, both in the menu, so that the
    # best attainable outcome is max(best, c) in every state.  Because psi
    # is increasing, psi(max(best, c)) = max(psi(best), psi(c)), so psi is
    # evaluated on one scalar per iteration and g on the vector
    def gap(c):
        psiC = float(utilFunc(c))
        psiBC = np.maximum(psiB, psiC)
        sure = np.mean(_rtu(psiC, psiBC, vg, k))
        scenario = np.mean(_rtu(psiY, psiBC, vg, k))
        return float(sure - scenario)

    # the gap is <= 0 at the scenario's worst outcome and >= 0 at the menu's
    # best, so the root lies between them; a sure scenario (constant
    # outcomes) has gap(lo) == 0 exactly, which brentq would reject
    lo = float(np.min(y))
    hi = float(np.max(b))
    if gap(lo) >= 0.0:
        certequiv = lo
    elif gap(hi) <= 0.0:
        certequiv = hi
    else:
        certequiv = scipy.optimize.brentq(f=gap, a=lo, b=hi, maxiter=200)

    return RteuResult(expected, float(certequiv))
