
import sys
from collections import namedtuple
import scipy.stats as stats
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
import numpy as np
import warnings
from ecdfgof import adtest, cvmtest
import conversions
from typing import Optional


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

warnings.showwarning = custom_warning_format


# "name","scipy_name","lower_limit","upper_limit
candidates = [
	("alpha",stats.alpha,True,False),
	("anglit",stats.anglit,True,True),
	("arcsine",stats.arcsine,True,True),
	("argus",stats.argus,True,True),
	("Beta",stats.beta,True,True),
	("Beta Prime",stats.betaprime,True,False),
	("Bradford",stats.bradford,True,True),
	("Burr",stats.burr,True,False),
	("Burr Type XII",stats.burr12,True,False),
	("Cauchy",stats.cauchy,False,False),
	("Chi",stats.chi,True,False),
	("Chi^2",stats.chi2,True,False),
	("Cosine",stats.cosine,True,True),
	("Crystal Ball",stats.crystalball,False,False),
	("Double Gamma",stats.dgamma,False,False),
	# ("Double Pareto Lognormal",stats.dpareto_lognorm,True,False),
	("Double Weibull",stats.dweibull,False,False),
	("Erlang",stats.erlang,True,False),
	("Exponential",stats.expon,True,False),
	("Expone. Modified Normal",stats.exponnorm,False,False),
	("Exponentiated Weibull",stats.exponweib,True,False),
	("Exponential Power",stats.exponpow,True,False),
	("F",stats.f,True,False),
	("Fatigue Life",stats.fatiguelife,True,False),
	("Fisk",stats.fisk,True,False),
	("Folded Cauchy",stats.foldcauchy,True,False),
	("Folded Normal",stats.foldnorm,True,False),
	("Generalized Logistic",stats.genlogistic,False,False),
	("Generalized Normal",stats.gennorm,False,False),
	("Generalized Pareto",stats.genpareto,True,False),
	("Generalized Exponential",stats.genexpon,True,False),
	("Generalized Extreme Value",stats.genextreme,True,True),
	("Gauss Hypergeometric",stats.gausshyper,True,True),
	("Gamma",stats.gamma,True,False),
	("Generalized Gamma",stats.gengamma,True,False),
	("Generalized Half-Logistic",stats.genhalflogistic,True,True),
	("Generalized Hyperbolic",stats.genhyperbolic,False,False),
	("Generalized Inverse Gaussian",stats.geninvgauss,True,False),
	("Gibrat",stats.gibrat,True,False),
	("Gompertz",stats.gompertz,True,False),
	("Gumbel Right-skewed",stats.gumbel_r,False,False),
	("Gumbel Left-skewed",stats.gumbel_l,False,False),
	("Half-Cauchy",stats.halfcauchy,True,False),
	("Half-Logistic",stats.halflogistic,True,False),
	("Half-Normal",stats.halfnorm,True,False),
	("Half Generalized Normal",stats.halfgennorm,True,False),
	("Hyperbolic Secant",stats.hypsecant,False,False),
	("Inverse Gamma",stats.invgamma,True,False),
	("Inverse Gaussian",stats.invgauss,True,False),
	("Inverse Weibull",stats.invweibull,True,False),
	# ("Irwin-Hall",stats.irwinhall,True,True),
	("Jones and Faddy Skew-T",stats.jf_skew_t,False,False),
	("Johnson's S_B",stats.johnsonsb,True,True),
	("Johnson's S_U",stats.johnsonsu,False,False),
	("Four-parameter Kappa",stats.kappa4,False,False),
	("3-Param Kappa Distribution",stats.kappa3,True,False),
	# ("Landau",stats.landau,False,False),
	("Laplace",stats.laplace,False,False),
	("Asymmetric Laplace",stats.laplace_asymmetric,False,False),
	("Lévy",stats.levy,True,False),
	("Left-skewed Lévy",stats.levy_l,False,True),
	("Lévy stable",stats.levy_stable,False,False),
	("Logistic",stats.logistic,False,False),
	("Log-gamma",stats.loggamma,False,False),
	("Log-Laplace",stats.loglaplace,True,False),
	("Log-normal",stats.lognorm,True,False),
	("Log-uniform",stats.loguniform,True,True),
	("Lomax",stats.lomax,True,False),
	("Maxwell",stats.maxwell,True,False),
	("Mielke's Beta-Kappa",stats.mielke,True,False),
	("Moyal",stats.moyal,False,False),
	("Nakagami",stats.nakagami,True,False),
	("Noncentral chi^2",stats.ncx2,True,False),
	("Noncentral F",stats.ncf,True,False),
	("Noncentral Student’s t",stats.nct,False,False),
	("Normal",stats.norm,False,False),
	("Normal inverse Gaussian",stats.norminvgauss,False,False),
	("Pareto",stats.pareto,True,False),
	("Pearson Type III",stats.pearson3,False,False),
	("Power-function",stats.powerlaw,True,True),
	("Power log-normal",stats.powerlognorm,True,False),
	("Power normal",stats.powernorm,False,False),
	("symmetric beta",stats.rdist,True,True),
	("Rayleigh",stats.rayleigh,True,False),
	("Relativistic Breit–Wigner",stats.rel_breitwigner,True,False),
	("Rice",stats.rice,True,False),
	("Reciprocal inverse Gaussian",stats.recipinvgauss,True,False),
	("Semicircular",stats.semicircular,True,True),
	("Skewed Cauchy",stats.skewcauchy,False,False),
	("Skew-normal",stats.skewnorm,False,False),
	("Studentized range",stats.studentized_range,True,False),
	("Student's t",stats.t,False,False),
	("Trapezoidal",stats.trapezoid,True,True),
	("Triangular",stats.triang,True,True),
	("Truncated exponential",stats.truncexpon,True,True),
	("Truncated normal",stats.truncnorm,True,True),
	("Truncated Pareto",stats.truncpareto,True,True),
	("Truncated Weibull minimum",stats.truncweibull_min,True,True),
	("Tukey lambda",stats.tukeylambda,True,True),
	("Uniform",stats.uniform,True,True),
	("Von Mises",stats.vonmises,True,True),
	("Wald",stats.wald,True,False),
	("Weibull Min Extreme Value",stats.weibull_min,True,False),
	("Weibull Max Extreme Value",stats.weibull_max,False,True),
	("Wrapped Cauchy",stats.wrapcauchy,True,True),
]


# container for "fit" results
FitResult = namedtuple("FitResult",
                       "bic aic ad_pval cvm_pval dist distName warnings")


def fit(data: conversions.VectorLike,
        scipydist: rv_continuous,
        distName: Optional[str] = None
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

    Returns
    -------
    FitResult
    	A Named tuple with fields:

        bic : float
            Bayesian Information Criterion for the fit.
        aic : float
            Akaike Information Criterion for the fit.
        ad_pval : float
            Anderson-Darling test p-value.
        cvm_pval : float
            Cramer-von Mises test p-value.
        dist : scipy.stats.rv_continuous
            The frozen fitted distribution object.
        distName : str
            Name of the fitted distribution.

    Notes
    -----
    The function uses maximum likelihood estimation for parameter fitting.
    Goodness-of-fit is assessed using Anderson-Darling and Kolmogorov-Smirnov
    tests.
    """

    dataA = conversions.vlToArray(data)

    with warnings.catch_warnings(record=True) as w:
        # warnings.simplefilter("always", category=RuntimeWarning)
        warnings.simplefilter("always")

        # fit distribution using maximum likelihood
        params = scipydist.fit(data)

	    # create a "frozen" distribution object
        dist = scipydist(*params)

        # calculate log likelihood function and info criteria
        loglike = dist.logpdf(dataA).sum()
        bic = np.log(len(dataA)) * len(params) - 2.0 * loglike  # Schwarz
        aic = 2.0 * len(params) - 2.0 * loglike                # Akaike
        
		# p-values for GOF tests
        ad_pval = adtest(dataA, dist)[1]  # Anderson-Darling
        cvm_pval = cvmtest(dataA, dist)[1]  # Cramer-von Mises

    return FitResult(bic=bic, aic=aic, ad_pval=ad_pval, cvm_pval=cvm_pval,
                     dist=dist, distName=distName, warnings=w)


def _fit_all(data, dist_list):
    results = list(map(lambda x: fit(data, x[1], x[0]), dist_list))
    return sorted(results, key=lambda r: r.bic)  # lowest BIC to highest


def _fstr(value, nchars=8):
    return ("%.3f" % value).rjust(nchars)


def _result_line(r, header=False):
    if header is True:
        return ("                  distribution,"
                "      BIC,      AIC, AD_p-val, CvM_p-val\n")
    else:
        return ("%s, %s, %s,   %s,   %s\n" %
                (r.distName.rjust(30), _fstr(r.bic), _fstr(r.aic),
                 _fstr(r.ad_pval, 6), _fstr(r.cvm_pval, 7)))

    
def compare(data: conversions.VectorLike,
            lowerLimit: bool,
            upperLimit: bool
            ) -> str:
    """
    Compare fits of univariate distributions for a continuous random variable.

    This function fits all candidate distributions with the specified support
    (lower and/or upper limit) to the provided data. It returns a formatted
    string summarizing the Bayesian Information Criterion (BIC), Akaike
    Information Criterion (AIC), and goodness-of-fit p-values for each
    distribution.

    Parameters
    ----------
    data : array-like
        The data to fit, as a one-dimensional array or sequence.
    lowerLimit : bool
        If True, only distributions with a lower bound are considered.  If
        False, only distributiuons without a lower bound are considered.
    upperLimit : bool
        If True, only distributions with an upper bound are considered. If
        False, only distributions without an upper bound are considered.

    Returns
    -------
    str
        A formatted string summarizing the fit statistics for each candidate
        distribution, sorted by BIC (best to worst).

    Notes
    -----
    For reliable results, at least 50 observations are recommended. The summary
    includes BIC, AIC, Cramer-von Mises p-value, and Anderson-Darling
    p-value for each distribution.
    """
    dataA = conversions.vlToArray(data)
    
    if len(dataA) < 50:
        msg = (f"using 'compare' with only {len(dataA)} observations "
               f"can produce unreliable results. Interpret "
               f"with caution.")
        warnings.warn(msg, UserWarning)
    dist_list = [d for d in candidates if d[2] == lowerLimit and
                 d[3] == upperLimit]
    results = _fit_all(dataA, dist_list)
    results_edit = [r for r in results if len(r.warnings) == 0]
    for r in results:
        if len(r.warnings) > 0:
            msg = (f"encountered a problem "
				   f"while fitting the {r.distName} distribution. "
                   f"It will not be included in the results.")
            warnings.warn(msg, RuntimeWarning)
    lines = [_result_line(None, header=True)] + \
        list(map(_result_line, results_edit))
    return "".join(lines)