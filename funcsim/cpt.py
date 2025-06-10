"""calculate cumulative prospect theory expected value and certainty equiv"""

import math
import itertools
from collections import namedtuple
from typing import Optional, Callable, Iterable
import scipy


class InferenceError(Exception):
    """Error indicating that solving for the certainty equivalent failed."""


def utilPower(x: float,
              alpha: float = 0.88,
              beta: float = 0.88,
              lamb: float = 2.25,
              ) -> float:
    """
    Compute the power utility for gains and losses with loss aversion.

    This function implements the power utility function used in cumulative
    prospect theory in Tversky and Kahneman (1992), with separate risk aversion
    parameters for gains and losses, and a loss aversion parameter.

    Parameters
    ----------
    x : float
        The outcome value.
    alpha : float, optional
        Risk aversion parameter for gains (default is 0.88).
        Must be in [0.0, 1.0].
    beta : float, optional
        Risk aversion parameter for losses (default is 0.88).
        Must be in [0.0, 1.0].
    lamb : float, optional
        Loss aversion parameter (default is 2.25).
        Must be >= 1.0.

    Returns
    -------
    float
        The utility value for the given outcome.

    Notes
    -----
    For x >= 0, utility is x**alpha (or log(x) if alpha == 0).
    For x < 0, utility is -lamb * (-x)**beta (or -lamb * log(-x) if beta == 0).
    Default parameter values are from Tversky and Kahneman (1992).
    """
    if not isinstance(alpha, float):
        raise ValueError('alpha must be a float between 0.0 and 1.0')
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be between 0.0 and 1.0')
    if not isinstance(beta, float):
        raise ValueError('beta must be a float between 0.0 and 1.0')
    if beta < 0.0 or beta > 1.0:
        raise ValueError('beta must be between 0.0 and 1.0')
    if not isinstance(lamb, float):
        raise ValueError('lamb must be a float >= 1.0')
    if lamb < 1.0:
        raise ValueError('lamb must be greater or equal to 1.0')

    if x >= 0.0:
        return math.log(x) if alpha == 0.0 else x**alpha
    return -lamb * math.log(-x) if beta == 0.0 else -lamb * (-x)**beta


def utilNormLog(x: float,
                gamma: float = 1.223,
                delta: float = 0.000,
                lamb: float = 2.25,
                ) -> float:
    """
    Compute normalized logarithmic utility from Rachlin (1992) for gains and
    losses with loss aversion.

    This function implements the normalized log utility used in cumulative
    prospect theory, with separate curvature parameters for gains and losses,
    and a loss aversion parameter.

    Parameters
    ----------
    x : float
        The outcome value.
    gamma : float, optional
        Curvature parameter for gains (default is 1.223).
    delta : float, optional
        Curvature parameter for losses (default is 0.0).
    lamb : float, optional
        Loss aversion parameter (default is 2.25).

    Returns
    -------
    float
        The utility value for the given outcome.

    Notes
    -----
    For x >= 0, utility is x if gamma == 0, else log(1 + gamma * x) / gamma.
    For x < 0, utility is lamb * x if delta == 0, else
    -lamb * log(1 - delta * x) / delta.
    Default parameter values are from Bouchoiuicha & Vieider (2017) and
    Tversky and Kahneman (1992).
    """
    if x >= 0.0:
        return x if gamma == 0.0 else math.log(1.0 + gamma * x) / gamma
    return lamb * x if delta == 0.0 else \
        -lamb * math.log(1.0 - delta * x) / delta


def weightTK(p: float,
             gamma: float
             ) -> float:
    """
    Compute the Tversky and Kahneman (1992) probability weighting function.

    This function applies the probability distortion function used in
    cumulative prospect theory, as proposed by Tversky and Kahneman (1992).

    Parameters
    ----------
    p : float
        Probability value in the range [0.0, 1.0].
    gamma : float
        Distortion parameter, must be between 0.28 and 1.0.

    Returns
    -------
    float
        The weighted (distorted) probability.

    Raises
    ------
    ValueError
        If p is not in [0.0, 1.0] or gamma is not in [0.28, 1.0].

    Notes
    -----
    For gamma < 1, small probabilities are overweighted and large
    probabilities are underweighted.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError('probabilities must be within [0.0, 1.0]')
    if gamma < 0.28 or gamma > 1.0:
        raise ValueError('gamma must be between 0.28 and 1')

    num = p ** gamma
    denom = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
    return num / denom


def weightPrelec1(p: float,
                  gamma: float
                  ) -> float:
    """
    Compute the Prelec (1998) one-parameter probability weighting function.

    This function applies the one-parameter Prelec probability distortion
    function, commonly used in cumulative prospect theory, to a probability
    value.

    Parameters
    ----------
    p : float
        Probability value in the range [0.0, 1.0].
    gamma : float
        Distortion parameter. Typical values are in (0, 1].

    Returns
    -------
    float
        The weighted (distorted) probability.

    Raises
    ------
    ValueError
        If p is not in [0.0, 1.0].

    Notes
    -----
    For gamma < 1, small probabilities are overweighted and large
    probabilities are underweighted.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError('probabilities must be within [0.0, 1.0]')

    if p == 0.0:
        return 0.0
    inner = -1.0 * (-1.0 * math.log(p)) ** gamma
    return math.exp(inner)


def weightPrelec2(p: float,
                  alpha: float,
                  beta: float
                  ) -> float:
    """
    Compute the Prelec (1998) two-parameter probability weighting function.

    This function applies the two-parameter Prelec probability distortion
    function, commonly used in cumulative prospect theory, to a probability
    value.

    Parameters
    ----------
    p : float
        Probability value in the range [0.0, 1.0].
    alpha : float
        Slope or sensitivity parameter. Typical values are in (0, 1].
    beta : float
        Elevation parameter. Typical values are positive.

    Returns
    -------
    float
        The weighted (distorted) probability.

    Raises
    ------
    ValueError
        If p is not in [0.0, 1.0].

    Notes
    -----
    For alpha < 1, small probabilities are overweighted and large
    probabilities are underweighted.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError('probabilities must be within [0.0, 1.0]')

    return 0.0 if p == 0.0 else math.exp(-beta * (-math.log(p))**alpha)


class _accDict:
    # accumulator dictionary

    def __init__(self):
        self._d = {}

    def add(self, key, val):
        """add a value to dict if the key is not already present"""
        if key not in self._d.keys():
            self._d[key] = 0.0
        self._d[key] += val

    @property
    def final(self):
        """return the dictionary"""
        return self._d


def _makeDistinct(outcomes, probs):
    # given an iterable of possibly non-unique outcomes and an iterable of
    # their corresponding probabilities, return a list of *distinct* outcomes
    # and a list of their corresponding *total* probabilities
    assert len(outcomes) == len(probs)

    if len(set(outcomes)) == len(outcomes):
        return (outcomes, probs)

    # warning: mutating d
    d = _accDict()
    _ = [d.add(k, v) for k, v in zip(outcomes, probs)]
    final = d.final

    # return lists, being sure to preserve correspondence
    keys = list(final.keys())
    return (keys, [final[k] for k in keys])


def _validate_psum(psum):
    # due to numerical imprecision, we may occasional get a cumulative
    # probability that is 1.00000000007 or something.  Filter these out
    # so they do not cause trouble in the probability weighting functions
    return [min(1.0, p) for p in psum]


# container for results
CptResult = namedtuple("CptResult", "ExpectedValue CertaintyEquiv")


def cpt(utilFunc: Callable,
        weightFuncGains: Callable,
        weightFuncLosses: Callable,
        outcomes: Iterable[float],
        refOutcome: float,
        probabilities: Optional[Iterable[float]] = None,
        precision: float = 1.0
        ) -> CptResult:
    """
    Compute cumulative prospect theory (CPT) value and certainty equivalent.

    This function calculates the CPT expected value and certainty equivalent
    for a set of outcomes, using user-supplied utility and probability
    weighting functions. It supports arbitrary outcome distributions and
    reference points.

    Parameters
    ----------
    utilFunc : Callable
        Utility function taking a single outcome as input.  See ``utilPower``
        and ``utilNormLog``.
    weightFuncGains : Callable
        Probability weighting function for gains.  See ``weightTK``,
        ``weightPrelec1``, and ``weightPrelec2``.
    weightFuncLosses : Callable
        Probability weighting function for losses.
    outcomes : Iterable
        Sequence of possible stochastic outcomes.
    refOutcome : float
        Reference point for gains and losses.
    probabilities : float or Iterable, optional
        Probabilities for each outcome. If None, outcomes are assumed
        equally likely.
    precision : float, optional
        Precision for numerically finding the certainty equivalent.
        Default is 1.0, which is suitable for outcomes with magnitudes up to
        perhaps 100,000.  For outcomes with smaller maximum magnitudes, smaller
        values for precision should probably be used.

    Returns
    -------
    CptResult
        A named tuple with fields:

        ExpectedValue : float
            The expected CPT value.
        CertaintyEquiv : float
            The certainty equivalent of the gamble.

    Raises
    ------
    ValueError
        If probabilities are negative or do not sum to 1.

    Notes
    -----
    The function supports both unique and repeated outcomes. Probabilities
    are assumed equal for all outcomes if the user does not provide
    probabilities. Certainty equivalent is found by
    solving utilFunc(C.E. - refOutcome) = CPT value.
    """
    if probabilities is not None:
        if min(probabilities) < 0.0:
            raise ValueError('probabilities must be non-negative')

    # assume all outcomes are equally likely if no probabilities were passed
    outcomesList = list(outcomes)  # in case a generator was passed
    probs_orig = probabilities if probabilities is not None \
        else len(outcomesList) * [1.0 / float(len(outcomesList))]

    # be sure that outcomes are distinct/unique
    (out_distinct, probs_tot) = _makeDistinct(outcomesList, probs_orig)

    # apply reference point if it is not zero
    out = [o - refOutcome for o in out_distinct]

    # pair conditioned outcomes and their probabilities, sort by outcome value
    pairs = sorted(zip(out, probs_tot))

    # main calcs
    pos_pairs = list(reversed(list(filter(lambda p: p[0] >= 0.0, pairs))))
    neg_pairs = list(filter(lambda p: p[0] < 0.0, pairs))

    def pairsToVf(pares, wfunc):
        psum_prelim = list(itertools.accumulate(p[1] for p in pares))
        psum = _validate_psum(psum_prelim)
        psum_lag = [0.0] + psum[:-1]
        return sum((wfunc(psum[i]) - wfunc(psum_lag[i]))
                   * utilFunc(pares[i][0]) for i in range(len(psum)))

    Vfpos = pairsToVf(pos_pairs, weightFuncGains)
    Vfneg = pairsToVf(neg_pairs, weightFuncLosses)
    cptEval = Vfpos + Vfneg

    # calculate certainty equiv: find ce such that
    # utilFunc(ce - refOutcome) = cptEval
    def trySolve(x0, x1):

        def obj(x):
            return utilFunc(x - refOutcome) - cptEval

        return scipy.optimize.root_scalar(f=obj,
                                          x0=x0,
                                          x1=x1,
                                          maxiter=1000,
                                          rtol=precision)

    # using ugly mutation to try to find certainty equivalent using
    # different sets of starting points
    decrement = 0.05 * (max(outcomesList) - min(outcomesList))
    xtop = max(outcomesList)
    result = trySolve(x0=min(outcomesList), x1=xtop)
    while result.converged is False:
        xtop = xtop - decrement
        if xtop <= min(outcomesList):
            msg = (f"could not find certainty equivalent"
                   f" min(outcome)={min(outcomesList)}"
                   f" max(outcome)={max(outcomesList)}"
                   f" obj@minOutcome="
                   f"{utilFunc(min(outcomesList) - refOutcome) - cptEval}"
                   f" obj@maxOutcome="
                   f"{utilFunc(max(outcomesList) - refOutcome) - cptEval}")
            raise InferenceError(msg)
        result = trySolve(x0=min(outcomesList), x1=xtop)

    # return if we found the certainty equivalent
    certEquiv = result.root
    return CptResult(ExpectedValue=cptEval, CertaintyEquiv=certEquiv)


def cptBV(outcomes: Iterable,
          refOutcome: float,
          probabilities: Optional[Iterable[float]] = None
          ) -> CptResult:
    """
    Compute cumulative prospect theory value and certainty equivalent using
    Bouchoiuicha & Vieider (2017) recommended functional forms and parameters.

    This function calculates the cumulative prospect theory (CPT) expected value
    and certainty equivalent for a set of outcomes, using preset utility and
    probability weighting functions and parameters from Bouchoiuicha & Vieider
    (2017).

    Parameters
    ----------
    outcomes : Iterable
        Sequence of possible stochastic outcomes.
    refOutcome : float
        Reference point for gains and losses.
    probabilities : float or Iterable, optional
        Probabilities for each outcome. If None, outcomes are assumed equally
        likely.

    Returns
    -------
    CptResult
        A named tuple with fields:

        ExpectedValue : float
            The expected CPT value.
        CertaintyEquiv : float
            The certainty equivalent of the gamble.

    Notes
    -----
    Uses utilNormLog for utility, weightPrelec2 for gains and losses, and
    parameter values recommended by Bouchoiuicha & Vieider (2017).
    """
    return cpt(utilFunc=lambda x: utilNormLog(x, gamma=1.223, delta=0.0,
                                              lamb=2.25),
               weightFuncGains=lambda p: weightPrelec2(p, alpha=0.53,
                                                       beta=0.969),
               weightFuncLosses=lambda p: weightPrelec2(p, alpha=0.623,
                                                        beta=0.953),
               outcomes=outcomes,
               refOutcome=refOutcome,
               probabilities=probabilities)
