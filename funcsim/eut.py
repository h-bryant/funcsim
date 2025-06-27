"""calculate isoelastic utility expected value and certainty equiv"""

import math
import numpy as np
from collections import namedtuple
import conversions


# container for results
EutResult = namedtuple("EutResult", "ExpectedValue CertaintyEquiv")


def utility(y, crra):
    # isoelastic utility function: return the utility level associated
    # with the monetary outcome "y" given the constant relative
    # risk aversion coefficient "crra"
    #print(type(y))
    # assert type(y) in [float, np.float64], "y must be a float"
    assert type(y) in [float, np.float64], "y must be a float"
    assert y >= 1.0, "y must be >= one"
    assert crra >= 0.0, "crra must be non-negative"
    if crra == 1.0:
        return math.log(y)
    else:
        num = y**(1.0-crra) - 1.0
        denom = 1.0 - crra
        return num / denom


def _util_inv(util, crra):
    # inverse isoelastic utility function: return the monetary outcome
    # associated with a given "utility" level, given the constant
    # relative risk aversion coefficient "crra"
    # assert type(utility) in [float, np.float64], \
    #    "util must be a float"
    assert type(util) in [float, np.float64], "util must be a float"
    assert util >= 0.0, "util must be non-negative"
    assert crra >= 0.0, "crra must be non-negative"
    if crra == 1.0:
        return math.exp(util)
    else:
        return (1.0 + (1.0 - crra) * util)**(1.0/(1.0-crra))


def eut(crra: float,
        outcomes: conversions.VectorLike
        ) -> EutResult:
    """
    Compute expected isoelastic utility and certainty equivalent income.

    Parameters
    ----------
    crra : float
        Constant relative risk aversion coefficient (must be non-negative).
    outcomes : VectorLike
        Sequence of monetary outcomes (all must be positive).

    Returns
    -------
    EutResult
        Named tuple with fields:

        ExpectedValue : float
            The mean isoelastic utility of the outcomes.
        CertaintyEquiv : float
            The certainty equivalent income.
    """

    # check inputs
    if not isinstance(crra, float):
        raise ValueError("crra must be a float")
    if crra < 0.0:
        raise ValueError("crra must be non-negative")

    outcomesA = conversions.vlToArray(outcomes)

    # find scaling factor such that min adjusted outcome is one
    scale = min(outcomesA)

    # scale outcomes
    y_scaled = [i / scale for i in outcomesA]

    # calculations
    N = float(len(outcomesA))
    eUtil = sum(map(lambda v: utility(v, crra), y_scaled)) / N
    ce = scale * _util_inv(eUtil, crra)
    return EutResult(ExpectedValue=eUtil, CertaintyEquiv=ce)