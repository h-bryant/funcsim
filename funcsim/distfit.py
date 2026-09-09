
import numbers
from collections import namedtuple
import scipy.stats as stats
from scipy import optimize
from scipy.stats._distn_infrastructure import rv_continuous
import numpy as np
import warnings
from .ecdfgof import adtest, cvmtest
from . import conversions
from typing import Optional, Tuple


# "name", "scipy_name", "lower_bound", "upper_bound"
#
# The two flags describe the support of the *fitted* distribution.  A few
# families realize different kinds of support in different regions of their
# shape-parameter space (e.g., the generalized extreme value distribution is
# bounded below for negative shape and bounded above for positive shape).
# Such families are listed once per kind of support they can realize on a
# region of positive measure; `compare` checks the support of every fitted
# distribution and drops any whose support does not match its group.
candidates = [
    ("alpha", stats.alpha, True, False),
    ("anglit", stats.anglit, True, True),
    ("arcsine", stats.arcsine, True, True),
    ("argus", stats.argus, True, True),
    ("Beta", stats.beta, True, True),
    ("Beta Prime", stats.betaprime, True, False),
    ("Bradford", stats.bradford, True, True),
    ("Burr", stats.burr, True, False),
    ("Burr Type XII", stats.burr12, True, False),
    ("Cauchy", stats.cauchy, False, False),
    ("Chi", stats.chi, True, False),
    ("Chi^2", stats.chi2, True, False),
    ("Cosine", stats.cosine, True, True),
    ("Crystal Ball", stats.crystalball, False, False),
    ("Double Gamma", stats.dgamma, False, False),
    # ("Double Pareto Lognormal",stats.dpareto_lognorm,True,False),
    ("Double Weibull", stats.dweibull, False, False),
    ("Erlang", stats.erlang, True, False),
    ("Exponential", stats.expon, True, False),
    ("Expone. Modified Normal", stats.exponnorm, False, False),
    ("Exponentiated Weibull", stats.exponweib, True, False),
    ("Exponential Power", stats.exponpow, True, False),
    ("F", stats.f, True, False),
    ("Fatigue Life", stats.fatiguelife, True, False),
    ("Fisk", stats.fisk, True, False),
    ("Folded Cauchy", stats.foldcauchy, True, False),
    ("Folded Normal", stats.foldnorm, True, False),
    ("Generalized Logistic", stats.genlogistic, False, False),
    ("Generalized Normal", stats.gennorm, False, False),
    # support [0, inf) for shape c >= 0, [0, -1/c] for c < 0
    ("Generalized Pareto", stats.genpareto, True, False),
    ("Generalized Pareto", stats.genpareto, True, True),
    ("Generalized Exponential", stats.genexpon, True, False),
    # support [1/c, inf) for shape c < 0, (-inf, 1/c] for c > 0; never both
    ("Generalized Extreme Value", stats.genextreme, True, False),
    ("Generalized Extreme Value", stats.genextreme, False, True),
    ("Gauss Hypergeometric", stats.gausshyper, True, True),
    ("Gamma", stats.gamma, True, False),
    ("Generalized Gamma", stats.gengamma, True, False),
    ("Generalized Half-Logistic", stats.genhalflogistic, True, True),
    ("Generalized Hyperbolic", stats.genhyperbolic, False, False),
    ("Generalized Inverse Gaussian", stats.geninvgauss, True, False),
    ("Gibrat", stats.gibrat, True, False),
    ("Gompertz", stats.gompertz, True, False),
    ("Gumbel Right-skewed", stats.gumbel_r, False, False),
    ("Gumbel Left-skewed", stats.gumbel_l, False, False),
    ("Half-Cauchy", stats.halfcauchy, True, False),
    ("Half-Logistic", stats.halflogistic, True, False),
    ("Half-Normal", stats.halfnorm, True, False),
    ("Half Generalized Normal", stats.halfgennorm, True, False),
    ("Hyperbolic Secant", stats.hypsecant, False, False),
    ("Inverse Gamma", stats.invgamma, True, False),
    ("Inverse Gaussian", stats.invgauss, True, False),
    ("Inverse Weibull", stats.invweibull, True, False),
    # ("Irwin-Hall",stats.irwinhall,True,True),
    ("Jones and Faddy Skew-T", stats.jf_skew_t, False, False),
    ("Johnson's S_B", stats.johnsonsb, True, True),
    ("Johnson's S_U", stats.johnsonsu, False, False),
    # support depends on the signs of both shapes h and k: bounded on both
    # sides (h > 0, k > 0), below only (k <= 0 with h > 0, or k < 0), or
    # above only (h <= 0, k > 0); unbounded only on the line h <= 0, k = 0
    ("Four-parameter Kappa", stats.kappa4, True, True),
    ("Four-parameter Kappa", stats.kappa4, True, False),
    ("Four-parameter Kappa", stats.kappa4, False, True),
    ("3-Param Kappa Distribution", stats.kappa3, True, False),
    # ("Landau",stats.landau,False,False),
    ("Laplace", stats.laplace, False, False),
    ("Asymmetric Laplace", stats.laplace_asymmetric, False, False),
    ("Lévy", stats.levy, True, False),
    ("Left-skewed Lévy", stats.levy_l, False, True),
    ("Lévy stable", stats.levy_stable, False, False),
    ("Logistic", stats.logistic, False, False),
    ("Log-gamma", stats.loggamma, False, False),
    ("Log-Laplace", stats.loglaplace, True, False),
    ("Log-normal", stats.lognorm, True, False),
    ("Log-uniform", stats.loguniform, True, True),
    ("Lomax", stats.lomax, True, False),
    ("Maxwell", stats.maxwell, True, False),
    ("Mielke's Beta-Kappa", stats.mielke, True, False),
    ("Moyal", stats.moyal, False, False),
    ("Nakagami", stats.nakagami, True, False),
    ("Noncentral chi^2", stats.ncx2, True, False),
    ("Noncentral F", stats.ncf, True, False),
    ("Noncentral Student’s t", stats.nct, False, False),
    ("Normal", stats.norm, False, False),
    ("Normal inverse Gaussian", stats.norminvgauss, False, False),
    ("Pareto", stats.pareto, True, False),
    ("Pearson Type III", stats.pearson3, False, False),
    ("Power-function", stats.powerlaw, True, True),
    ("Power log-normal", stats.powerlognorm, True, False),
    ("Power normal", stats.powernorm, False, False),
    ("symmetric beta", stats.rdist, True, True),
    ("Rayleigh", stats.rayleigh, True, False),
    ("Relativistic Breit–Wigner", stats.rel_breitwigner, True, False),
    ("Rice", stats.rice, True, False),
    ("Reciprocal inverse Gaussian", stats.recipinvgauss, True, False),
    ("Semicircular", stats.semicircular, True, True),
    ("Skewed Cauchy", stats.skewcauchy, False, False),
    ("Skew-normal", stats.skewnorm, False, False),
    ("Studentized range", stats.studentized_range, True, False),
    ("Student's t", stats.t, False, False),
    ("Trapezoidal", stats.trapezoid, True, True),
    ("Triangular", stats.triang, True, True),
    ("Truncated exponential", stats.truncexpon, True, True),
    ("Truncated normal", stats.truncnorm, True, True),
    ("Truncated Pareto", stats.truncpareto, True, True),
    ("Truncated Weibull minimum", stats.truncweibull_min, True, True),
    # support [-1/lam, 1/lam] for shape lam > 0, the whole line otherwise
    ("Tukey lambda", stats.tukeylambda, True, True),
    ("Tukey lambda", stats.tukeylambda, False, False),
    ("Uniform", stats.uniform, True, True),
    # scipy's `vonmises` is defined on the whole real line (it is periodic);
    # `vonmises_line` is the same density restricted to [-pi, pi]
    ("Von Mises", stats.vonmises_line, True, True),
    ("Wald", stats.wald, True, False),
    ("Weibull Min Extreme Value", stats.weibull_min, True, False),
    ("Weibull Max Extreme Value", stats.weibull_max, False, True),
    ("Wrapped Cauchy", stats.wrapcauchy, True, True),
]


# container for "fit" results
FitResult = namedtuple("FitResult",
                       "bic aic ad_pval cvm_pval dist distName warnings")


# per-observation penalty for data outside the support, matching scipy
_PENALTY = 100.0 * np.log(np.finfo(float).max)


# ---------------------------------------------------------------------------
# argument validation


def _validate_bound(value, name: str) -> Optional[float]:
    # None (bound not fixed) or a finite real number; bools are rejected
    # because they are the legacy "which group" flags of `compare`
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a float or None, not a bool")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a float or None, "
                        f"not {type(value).__name__}")
    bound = float(value)
    if not np.isfinite(bound):
        raise ValueError(f"{name} must be finite, not {value}")
    return bound


def _check_bounds_against_data(dataA: np.ndarray,
                               lower: Optional[float],
                               upper: Optional[float]) -> None:
    if lower is not None and upper is not None and not lower < upper:
        raise ValueError(f"lowerBound ({lower}) must be strictly less than "
                         f"upperBound ({upper})")
    if lower is not None and dataA.min() < lower:
        raise ValueError(f"lowerBound ({lower}) exceeds the smallest "
                         f"observation ({dataA.min()})")
    if upper is not None and dataA.max() > upper:
        raise ValueError(f"upperBound ({upper}) is below the largest "
                         f"observation ({dataA.max()})")


# ---------------------------------------------------------------------------
# support metadata for scipy distributions


def _shape_dependent_support(scipydist) -> bool:
    # scipy signals a support that depends on the shape parameters by
    # overriding `_get_support`.  This is private but has been stable since
    # scipy 1.2; if it ever disappears, every distribution is treated as
    # shape-dependent, which is slower but still correct.
    base = getattr(rv_continuous, "_get_support", None)
    own = getattr(type(scipydist), "_get_support", None)
    if base is None or own is None:
        return True
    return own is not base


def _natural_bounds(scipydist) -> Tuple[bool, bool]:
    # (has lower bound, has upper bound) for the standard form of the
    # distribution.  For a shape-dependent support the answer is not known
    # until the distribution is fitted, so both are reported as possible.
    if _shape_dependent_support(scipydist):
        return True, True
    return bool(np.isfinite(scipydist.a)), bool(np.isfinite(scipydist.b))


def _check_natural_bounds(scipydist, distName: str,
                          lower: Optional[float],
                          upper: Optional[float]) -> None:
    has_lower, has_upper = _natural_bounds(scipydist)
    if lower is not None and not has_lower:
        raise ValueError(f"the {distName} distribution has no natural lower "
                         f"bound, so lowerBound cannot be fixed")
    if upper is not None and not has_upper:
        raise ValueError(f"the {distName} distribution has no natural upper "
                         f"bound, so upperBound cannot be fixed")


def _fitted_support_flags(dist) -> Optional[Tuple[bool, bool]]:
    # (lower bound is finite, upper bound is finite) for a frozen
    # distribution, or None if the object cannot report its support
    support = getattr(dist, "support", None)
    if support is None:
        return None
    try:
        lo, hi = support()
    except Exception:
        return None
    return bool(np.isfinite(lo)), bool(np.isfinite(hi))


# ---------------------------------------------------------------------------
# maximum likelihood with one or both bounds of the support fixed
#
# A scipy distribution with standard support [a, b] has support
# [loc + a*scale, loc + b*scale].  Fixing the lower bound at L therefore
# imposes loc + a*scale = L, and fixing the upper bound at U imposes
# loc + b*scale = U.  When a constraint pins a single scipy parameter
# (a == 0 or b == 0, or both bounds fixed with constant a and b), scipy's
# own `fit` is used with `floc`/`fscale`.  Otherwise the constraint couples
# loc and scale (or depends on the shape parameters through a and b), and the
# penalized negative log-likelihood is minimized directly over the remaining
# free parameters, mirroring scipy's optimizer and penalty.


def _start_values(scipydist, dataA: np.ndarray) -> Tuple[float, ...]:
    # (shapes..., loc, scale) starting values; `_fitstart` is what scipy's
    # own `fit` uses and is the only source of its per-distribution
    # heuristics, so it is used here with a plain moment-based fallback
    fitstart = getattr(scipydist, "_fitstart", None)
    if fitstart is not None:
        try:
            start = tuple(float(v) for v in fitstart(dataA))
            if len(start) == scipydist.numargs + 2:
                return start
        except Exception:
            pass
    shapes = (1.0,) * scipydist.numargs
    return shapes + (float(np.mean(dataA)), float(np.std(dataA)) or 1.0)


def _verified(scipydist, params: Tuple[float, ...],
              lower: Optional[float], upper: Optional[float]
              ) -> Tuple[float, ...]:
    # confirm that the fitted distribution really has the requested bounds
    with np.errstate(all="ignore"):
        lo, hi = scipydist.support(*params)
    scale = params[-1]
    if not (np.isfinite(scale) and scale > 0.0):
        raise stats.FitError("Optimization converged to parameters that "
                             "are outside the range allowed by the "
                             "distribution.")
    tol = 1e-7 * (1.0 + abs(scale))
    if lower is not None and not np.isclose(lo, lower, rtol=1e-7, atol=tol):
        raise stats.FitError(f"fitted lower bound {lo} does not match the "
                             f"requested lowerBound {lower}")
    if upper is not None and not np.isclose(hi, upper, rtol=1e-7, atol=tol):
        raise stats.FitError(f"fitted upper bound {hi} does not match the "
                             f"requested upperBound {upper}")
    return tuple(params)


def _fit_coupled(scipydist, dataA: np.ndarray,
                 lower: Optional[float], upper: Optional[float]
                 ) -> Tuple[float, ...]:
    # free parameters `theta` are the shapes, followed by the scale when only
    # one bound is fixed; loc (and scale, when both bounds are fixed) follow
    # from the constraint(s) and the standard support for those shapes
    numargs = scipydist.numargs
    both = lower is not None and upper is not None

    def expand(theta) -> Tuple[float, ...]:
        shapes = tuple(float(t) for t in theta[:numargs])
        with np.errstate(all="ignore"):
            a, b = scipydist.support(*shapes)
            a, b = np.float64(a), np.float64(b)
            if both:
                scale = np.float64(upper - lower) / (b - a)
                loc = lower - a * scale
            elif lower is not None:
                scale = np.float64(theta[numargs])
                loc = lower - a * scale
            else:
                scale = np.float64(theta[numargs])
                loc = upper - b * scale
        return shapes + (float(loc), float(scale))

    def nnlf(theta) -> float:
        params = expand(theta)
        loc, scale = params[-2:]
        if not (np.isfinite(loc) and np.isfinite(scale)) or scale <= 0.0:
            return np.inf
        with np.errstate(all="ignore"):
            logpdf = scipydist.logpdf(dataA, *params)
        finite = np.isfinite(logpdf)
        nbad = int(np.count_nonzero(~finite))
        return -float(logpdf[finite].sum()) + nbad * _PENALTY

    def initial_theta(shapes, scale0: float):
        if both:
            return list(shapes)
        if not (np.isfinite(scale0) and scale0 > 0.0):
            scale0 = float(np.std(dataA)) or 1.0
        # widen the starting scale, if needed, so that the constrained
        # support covers the data and the optimizer starts unpenalized
        with np.errstate(all="ignore"):
            a, b = scipydist.support(*shapes)
        if np.isfinite(a) and np.isfinite(b) and b > a:
            if lower is not None:
                needed = (dataA.max() - lower) / (b - a)
            else:
                needed = (upper - dataA.min()) / (b - a)
            if np.isfinite(needed) and needed > 0.0:
                scale0 = max(scale0, 1.1 * needed)
        return list(shapes) + [scale0]

    start = _start_values(scipydist, dataA)
    shapes0, scale0 = start[:numargs], start[-1]
    # a sign flip of the shapes rescues families whose support is bounded on
    # one side or the other depending on the sign of a shape parameter
    trials = [shapes0]
    if any(s != 0.0 for s in shapes0):
        trials.append(tuple(-s for s in shapes0))
    for shapes in trials:
        theta0 = initial_theta(shapes, scale0)
        if np.isfinite(nnlf(theta0)):
            break
    else:
        raise stats.FitError("could not find starting values consistent "
                             "with the requested bounds")

    if len(theta0) == 0:
        # nothing free to optimize: both bounds fixed, no shape parameters
        theta = theta0
    else:
        theta = optimize.fmin(nnlf, theta0, disp=0)
    if not np.isfinite(nnlf(theta)):
        raise stats.FitError("Optimization converged to parameters that "
                             "are outside the range allowed by the "
                             "distribution.")
    return _verified(scipydist, expand(theta), lower, upper)


def _fit_with_bounds(scipydist, dataA: np.ndarray,
                     lower: Optional[float], upper: Optional[float]
                     ) -> Tuple[float, ...]:
    if not _shape_dependent_support(scipydist):
        a, b = float(scipydist.a), float(scipydist.b)
        if lower is not None and upper is not None:
            scale = (upper - lower) / (b - a)
            loc = lower - a * scale
            if scipydist.numargs == 0:
                params = (loc, scale)
            else:
                params = scipydist.fit(dataA, floc=loc, fscale=scale)
            return _verified(scipydist, params, lower, upper)
        if lower is not None and a == 0.0:
            params = scipydist.fit(dataA, floc=lower)
            return _verified(scipydist, params, lower, upper)
        if upper is not None and b == 0.0:
            params = scipydist.fit(dataA, floc=upper)
            return _verified(scipydist, params, lower, upper)
    return _fit_coupled(scipydist, dataA, lower, upper)


# ---------------------------------------------------------------------------
# public API


def fit(data: conversions.VectorLike,
        scipydist: rv_continuous,
        distName: Optional[str] = None,
        lowerBound: Optional[float] = None,
        upperBound: Optional[float] = None
        ) -> FitResult:
    """
    Fit a univariate distribution for a continuous random variable.

    This function fits a given scipy.stats univariate distribution to the
    provided data using maximum likelihood estimation. It returns a named
    tuple containing information criteria, goodness-of-fit p-values, the
    frozen distribution, and the distribution name.

    Parameters
    ----------
    data : VectorLike
        The data to fit, as a one-dimensional array or similar.
    scipydist : scipy.stats.rv_continuous
        The scipy.stats distribution object to fit (e.g., stats.norm).
    distName : str, optional
        Name of the distribution. If None, the distribution's name is used.
    lowerBound : float, optional
        If given, the lower bound of the support of the fitted distribution
        is fixed at this value rather than estimated.  The distribution must
        have a natural lower bound (e.g., ``stats.expon`` or ``stats.beta``,
        but not ``stats.norm``).  Must be finite and no greater than the
        smallest observation.  If omitted, all parameters are estimated.
    upperBound : float, optional
        If given, the upper bound of the support of the fitted distribution
        is fixed at this value rather than estimated.  The distribution must
        have a natural upper bound (e.g., ``stats.beta`` or
        ``stats.weibull_max``).  Must be finite, no less than the largest
        observation, and greater than `lowerBound` if both are given.  If
        omitted, all parameters are estimated.

    Returns
    -------
    FitResult
        A Named tuple with fields:

        bic : float
            Bayesian Information Criterion for the fit.
        aic : float
            Akaike Information Criterion for the fit.
        ad_pval : float
            Anderson-Darling heuristic goodness-of-fit score (see Notes).
        cvm_pval : float
            Cramer-von Mises heuristic goodness-of-fit score (see Notes).
        dist : scipy.stats.rv_continuous
            The frozen fitted distribution object.
        distName : str
            Name of the fitted distribution.

    Raises
    ------
    TypeError
        If `lowerBound` or `upperBound` is not a real number or None
        (bools are rejected).
    ValueError
        If a bound is not finite, if the data fall outside a fixed bound,
        if ``lowerBound >= upperBound``, or if a bound is requested for a
        distribution without the corresponding natural bound.

    Notes
    -----
    The function uses maximum likelihood estimation for parameter fitting.
    Goodness of fit is assessed using the Anderson-Darling and
    Cramer-von Mises statistics.

    Fixing a bound is a constraint on the distribution's location and scale
    parameters (a scipy distribution with standard support ``[a, b]`` has
    support ``[loc + a*scale, loc + b*scale]``); it is not truncation.  Each
    fixed bound removes one free parameter, and the parameter counts used
    for AIC and BIC are reduced accordingly.

    The `ad_pval` and `cvm_pval` values are computed from tests that assume
    a fully specified (a priori) null distribution, but the parameters here
    are estimated from the same sample being tested.  Consequently these
    values systematically overstate goodness of fit (the Lilliefors
    problem) and are *not* valid p-values.  Treat them only as heuristic
    scores for comparing candidate distributions with equal numbers of
    parameters; do not use them for formal hypothesis tests.
    """

    dataA = conversions.vlToArray(data)

    lower = _validate_bound(lowerBound, "lowerBound")
    upper = _validate_bound(upperBound, "upperBound")
    nfixed = int(lower is not None) + int(upper is not None)
    if nfixed > 0:
        _check_bounds_against_data(dataA, lower, upper)
        name = distName if distName is not None else \
            getattr(scipydist, "name", type(scipydist).__name__)
        _check_natural_bounds(scipydist, name, lower, upper)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # deprecation chatter from third-party libraries is not evidence
        # of a bad fit; do not let it disqualify a candidate
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)

        # fit distribution using maximum likelihood
        if nfixed == 0:
            params = scipydist.fit(data)
        else:
            params = _fit_with_bounds(scipydist, dataA, lower, upper)

        # create a "frozen" distribution object
        dist = scipydist(*params)

        # calculate log likelihood function and info criteria; each fixed
        # bound removes one free parameter
        loglike = dist.logpdf(dataA).sum()
        k = len(params) - nfixed
        bic = np.log(len(dataA)) * k - 2.0 * loglike  # Schwarz
        aic = 2.0 * k - 2.0 * loglike                # Akaike

        # p-values for GOF tests
        ad_pval = adtest(dataA, dist)[1]  # Anderson-Darling
        cvm_pval = cvmtest(dataA, dist)[1]  # Cramer-von Mises

    return FitResult(bic=bic, aic=aic, ad_pval=ad_pval, cvm_pval=cvm_pval,
                     dist=dist, distName=distName, warnings=w)


def _fit_all(data, dist_list, lower=None, upper=None):
    # fit every candidate, isolating failures so that one distribution
    # raising (e.g., scipy.stats.FitError) cannot abort the comparison
    results = []
    failures = []
    for name, dist, *_ in dist_list:
        try:
            results.append(fit(data, dist, name,
                               lowerBound=lower, upperBound=upper))
        except Exception as e:
            failures.append((name, e))
    results = sorted(results, key=lambda r: r.bic)  # lowest BIC to highest
    return results, failures


def _fstr(value, nchars=8):
    return ("%.3f" % value).rjust(nchars)


def _result_line(r, header=False):
    if header is True:
        return ("                  distribution,"
                "      BIC,      AIC, AD_score, CvM_score\n")
    else:
        return ("%s, %s, %s,   %s,   %s\n" %
                (r.distName.rjust(30), _fstr(r.bic), _fstr(r.aic),
                 _fstr(r.ad_pval, 6), _fstr(r.cvm_pval, 7)))


def _resolve_side(limit_name: str, limit, bound_name: str, bound
                  ) -> Tuple[bool, Optional[float]]:
    # reconcile the legacy bool flag and the new float bound for one side of
    # the support -> (distributions must be bounded on this side, value the
    # bound is fixed at or None)
    if limit is not None and not isinstance(limit, (bool, np.bool_)):
        raise TypeError(f"{limit_name} must be a bool or None, not "
                        f"{type(limit).__name__}; to fix the bound at a "
                        f"value, use {bound_name}")
    fixed = _validate_bound(bound, bound_name)
    if limit is not None and fixed is not None:
        raise ValueError(f"{limit_name} and {bound_name} cannot both be "
                         f"given")
    if fixed is not None:
        return True, fixed
    return (bool(limit) if limit is not None else False), None


def compare(data: conversions.VectorLike,
            lowerLimit: Optional[bool] = None,
            upperLimit: Optional[bool] = None,
            *,
            lowerBound: Optional[float] = None,
            upperBound: Optional[float] = None
            ) -> str:
    """
    Compare fits of univariate distributions for a continuous random variable.

    This function fits all candidate distributions with the specified support
    (bounded below and/or above, or not) to the provided data. It returns a
    formatted string summarizing the Bayesian Information Criterion (BIC),
    Akaike Information Criterion (AIC), and goodness-of-fit p-values for
    each distribution.

    Parameters
    ----------
    data : array-like
        The data to fit, as a one-dimensional array or sequence.
    lowerLimit : bool, optional
        Deprecated; use `lowerBound` (see Notes).  If True, only
        distributions with a lower bound are considered, and the bound is
        estimated along with the other parameters.  If False, only
        distributions without a lower bound are considered.  Passing this
        argument emits a ``FutureWarning``.
    upperLimit : bool, optional
        Deprecated; use `upperBound` (see Notes).  If True, only
        distributions with an upper bound are considered, and the bound is
        estimated along with the other parameters.  If False, only
        distributions without an upper bound are considered.  Passing this
        argument emits a ``FutureWarning``.
    lowerBound : float, optional
        Keyword only.  If given, only distributions with a lower bound are
        considered, and each is fitted with its lower bound fixed at this
        value (which must be finite and no greater than the smallest
        observation).  If omitted, and `lowerLimit` is not given, only
        distributions without a lower bound are considered.
    upperBound : float, optional
        Keyword only.  If given, only distributions with an upper bound are
        considered, and each is fitted with its upper bound fixed at this
        value (which must be finite, no less than the largest observation,
        and greater than `lowerBound` if that is also given).  If omitted,
        and `upperLimit` is not given, only distributions without an upper
        bound are considered.

    Returns
    -------
    str
        A formatted string summarizing the fit statistics for each candidate
        distribution, sorted by BIC (best to worst).

    Raises
    ------
    TypeError
        If `lowerLimit` or `upperLimit` is not a bool, or if `lowerBound` or
        `upperBound` is not a real number.
    ValueError
        If a bound is not finite, if the data fall outside a fixed bound, if
        ``lowerBound >= upperBound``, or if the legacy flag and the bound
        are both given for the same side of the support.

    Notes
    -----
    For reliable results, at least 50 observations are recommended. The summary
    includes BIC, AIC, and Anderson-Darling and Cramer-von Mises heuristic
    scores for each distribution.

    Fixing a bound is a constraint on a distribution's location and scale
    parameters, not truncation.  Each fixed bound removes one free parameter
    from the count used for AIC and BIC.

    A few candidate families (the generalized extreme value, generalized
    Pareto, four-parameter kappa, and Tukey lambda distributions) are
    bounded on different sides in different regions of their shape-parameter
    space.  These are considered in every group they can belong to, and a
    fitted distribution whose support does not match the requested group is
    dropped from the results with a warning.

    `lowerLimit` and `upperLimit` form the legacy interface, in which the
    bounds are estimated rather than fixed.  They still work but emit a
    ``FutureWarning`` and will be removed in a future version; use
    `lowerBound` and `upperBound` instead.  The legacy flag for one side of
    the support may be combined with the bound for the other side (e.g.,
    ``compare(data, upperLimit=True, lowerBound=0.0)``), but not with the
    bound for the same side.

    The AD_score and CvM_score columns are nominal p-values from tests that
    assume a fully specified (a priori) null distribution, but the
    parameters are estimated from the same sample being tested.  These
    values therefore systematically overstate goodness of fit (the
    Lilliefors problem) and are *not* valid p-values.  Use them only as
    heuristic scores for ranking candidate distributions; rankings by BIC
    or AIC additionally account for differing numbers of parameters.
    """
    has_lower, lower = _resolve_side("lowerLimit", lowerLimit,
                                     "lowerBound", lowerBound)
    has_upper, upper = _resolve_side("upperLimit", upperLimit,
                                     "upperBound", upperBound)
    legacy = [name for name, value in (("lowerLimit", lowerLimit),
                                       ("upperLimit", upperLimit))
              if value is not None]
    if legacy:
        msg = (f"the {' and '.join(legacy)} argument"
               f"{'s' if len(legacy) > 1 else ''} of compare "
               f"{'are' if len(legacy) > 1 else 'is'} deprecated and will "
               f"be removed in a future version of funcsim; use lowerBound "
               f"and upperBound instead: pass a float to restrict the "
               f"comparison to distributions with that bound, fixed at the "
               f"given value, or omit the argument for distributions "
               f"without that bound")
        warnings.warn(msg, FutureWarning, stacklevel=2)

    dataA = conversions.vlToArray(data)
    _check_bounds_against_data(dataA, lower, upper)

    if len(dataA) < 50:
        msg = (f"using 'compare' with only {len(dataA)} observations "
               f"can produce unreliable results. Interpret "
               f"with caution.")
        warnings.warn(msg, UserWarning)
    dist_list = [d for d in candidates if d[2] == has_lower and
                 d[3] == has_upper]
    results, failures = _fit_all(dataA, dist_list, lower, upper)
    results_edit = []
    mismatched = []
    for r in results:
        if len(r.warnings) > 0:
            msg = (f"encountered a problem "
                   f"while fitting the {r.distName} distribution. "
                   f"It will not be included in the results.")
            warnings.warn(msg, RuntimeWarning)
            continue
        flags = _fitted_support_flags(r.dist)
        if flags is not None and flags != (has_lower, has_upper):
            mismatched.append(r.distName)
            continue
        results_edit.append(r)
    if mismatched:
        msg = (f"the fitted {', '.join(mismatched)} distribution(s) do not "
               f"have the requested support (lower bound: {has_lower}, "
               f"upper bound: {has_upper}) and will not be included in the "
               f"results.")
        warnings.warn(msg, RuntimeWarning)
    for name, exc in failures:
        msg = (f"fitting the {name} distribution raised "
               f"{type(exc).__name__}: {exc}. It will not be included "
               f"in the results.")
        warnings.warn(msg, RuntimeWarning)
    lines = [_result_line(None, header=True)] + \
        list(map(_result_line, results_edit))
    return "".join(lines)
