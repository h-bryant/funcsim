import math
import numpy as np
import scipy
from collections import namedtuple
from typing import Callable
import conversions


# container for results
EutResult = namedtuple("EutResult", "ExpectedValue CertaintyEquiv")


def utilIsoelastic(y, crra):
    """
    Compute isoelastic (CRRA) utility for a monetary outcome.

    Parameters
    ----------
    y : float
        Monetary outcome (must be >= 1.0).
    crra : float
        Constant relative risk aversion coefficient (must be non-negative).

    Returns
    -------
    float
        Isoelastic utility value for the given outcome and CRRA.
    """
    assert type(y) in [float, np.float64], f"y (={y}) must be a float"
    assert y >= 1.0, "y must be >= one"
    assert crra >= 0.0, "crra must be non-negative"
    if crra == 1.0:
        ret = math.log(y)
    else:
        num = y**(1.0-crra) - 1.0
        denom = 1.0 - crra
        ret = num / denom
    return ret


def eut(util: Callable[[float], float],
        outcomes: conversions.VectorLike,
        precision: float = 0.001
        ) -> EutResult:
    """
    Compute expected utility and certainty equivalent income.

    Parameters
    ----------
    util : Callable
        Utility function that takes a float as its sole argument
        and returns a float
    outcomes : VectorLike
        Sequence of monetary outcomes (all must be positive).
    precition : float, optional
        Precision parameter used in solving for the certainty equivalent.
        The default value is 0.001

    Returns
    -------
    EutResult
        Named tuple with fields:

        ExpectedValue : float
            The mean isoelastic utility of the outcomes.
        CertaintyEquiv : float
            The certainty equivalent income.
    """
    
    # outcomes to numpy vec 
    outcomesA = conversions.vlToArray(outcomes)

    # expected util
    n = len(outcomes)
    eutil = float(np.sum(np.vectorize(util)(outcomesA)) / n)

    # certainty equiv: find C.E. such than U(C.E.) = E(U)
    meanOutcome = np.mean(outcomesA)
    
    def obj(ce):
        return util(ce) - eutil
    
    solveout = scipy.optimize.root_scalar(f=obj,
                                          bracket = (np.min(outcomesA),
                                                    np.max(outcomesA)),
                                          x0=meanOutcome,
                                          x1=(meanOutcome * 0.99),
                                          maxiter=1000,
                                          rtol=precision)
    certequiv = solveout.root if solveout.converged is True else None
 
    return EutResult(eutil, certequiv)
