

import scipy.stats as stats
import numpy as np
import warnings
from ecdfgof import adtest, kstest

warnings.filterwarnings("ignore")


_long = [
         ("alpha",              stats.alpha),
         ("anglit",             stats.anglit),
         ("arcsine",            stats.arcsine),
         ("argus",              stats.argus),
         ("beta",               stats.beta),
         ("betaprime",          stats.betaprime),
         ("bradford",           stats.bradford),
         ("burr",               stats.burr),
         ("burr12",             stats.burr12),
         ("cauchy",             stats.cauchy),
         ("chi",                stats.chi),
         ("chi2",               stats.chi2),
         ("cosine",             stats.cosine),
         ("crystalball",        stats.crystalball),
         ("dgamma",             stats.dgamma),
         ("dweibull",           stats.dweibull),
         # ("erlang",             stats.erlang),
         ("expon",              stats.expon),
         ("exponnorm",          stats.exponnorm),
         ("exponweib",          stats.exponweib),
         ("exponpow",           stats.exponpow),
         ("f",                  stats.f),
         ("fatiguelife",        stats.fatiguelife),
         ("fisk",               stats.fisk),
         ("foldcauchy",         stats.foldcauchy),
         ("foldnorm",           stats.foldnorm),
         # ("frechet_r",          stats.frechet_r),
         # ("frechet_l",          stats.frechet_l),
         ("genlogistic",        stats.genlogistic),
         ("gennorm",            stats.gennorm),
         ("genpareto",          stats.genpareto),
         ("genexpon",           stats.genexpon),
         ("genextreme",         stats.genextreme),
         ("gausshyper",         stats.gausshyper),
         ("gamma",              stats.gamma),
         ("gengamma",           stats.gengamma),
         ("genhalflogistic",    stats.genhalflogistic),
         ("gilbrat",            stats.gilbrat),
         ("gompertz",           stats.gompertz),
         ("gumbel_r",           stats.gumbel_r),
         ("gumbel_l",           stats.gumbel_l),
         ("halfcauchy",         stats.halfcauchy),
         ("halflogistic",       stats.halflogistic),
         ("halfnorm",           stats.halfnorm),
         ("halfgennorm",        stats.halfgennorm),
         ("hypsecant",          stats.hypsecant),
         ("invgamma",           stats.invgamma),
         ("invgauss",           stats.invgauss),
         ("invweibull",         stats.invweibull),
         ("johnsonsb",          stats.johnsonsb),
         ("johnsonsu",          stats.johnsonsu),
         ("kappa4",             stats.kappa4),
         ("kappa3",             stats.kappa3),
         ("ksone",              stats.ksone),
         ("kstwobign",          stats.kstwobign),
         ("laplace",            stats.laplace),
         ("levy",               stats.levy),
         ("levy_l",             stats.levy_l),
         ("levy_stable",        stats.levy_stable),
         ("logistic",           stats.logistic),
         ("loggamma",           stats.loggamma),
         ("loglaplace",         stats.loglaplace),
         ("lognorm",            stats.lognorm),
         ("lomax",              stats.lomax),
         ("maxwell",            stats.maxwell),
         ("mielke",             stats.mielke),
         ("moyal",              stats.moyal),
         ("nakagami",           stats.nakagami),
         ("ncx2",               stats.ncx2),
         ("ncf",                stats.ncf),
         ("nct",                stats.nct),
         ("norm",               stats.norm),
         ("norminvgauss",       stats.norminvgauss),
         ("pareto",             stats.pareto),
         ("pearson3",           stats.pearson3),
         ("powerlaw",           stats.powerlaw),
         ("powerlognorm",       stats.powerlognorm),
         ("powernorm",          stats.powernorm),
         # ("rdist",              stats.rdist),
         # ("reciprocal",         stats.reciprocal),
         ("rayleigh",           stats.rayleigh),
         ("rice",               stats.rice),
         ("recipinvgauss",      stats.recipinvgauss),
         ("semicircular",       stats.semicircular),
         ("skewnorm",           stats.skewnorm),
         ("t",                  stats.t),
         ("trapz",              stats.trapz),
         ("triang",             stats.triang),
         ("truncexpon",         stats.truncexpon),
         # ("truncnorm",          stats.truncnorm),
         ("tukeylambda",        stats.tukeylambda),
         ("uniform",            stats.uniform),
         # ("vonmises",           stats.vonmises),
         ("vonmises_line",      stats.vonmises_line),
         ("wald",               stats.wald),
         ("weibull_min",        stats.weibull_min),
         ("weibull_max",        stats.weibull_max),
         # ("wrapcauchy",         stats.wrapcauchy),
        ]

_short = [
          ("alpha",              stats.alpha),
          ("beta",               stats.beta),
          ("cauchy",             stats.cauchy),
          ("chi2",               stats.chi2),
          ("cosine",             stats.cosine),
          ("expon",              stats.expon),
          ("exponnorm",          stats.exponnorm),
          ("f",                  stats.f),
          ("gamma",              stats.gamma),
          ("laplace",            stats.laplace),
          ("levy",               stats.levy),
          ("levy_stable",        stats.levy_stable),
          ("logistic",           stats.logistic),
          ("loggamma",           stats.loggamma),
          ("loglaplace",         stats.loglaplace),
          ("lognorm",            stats.lognorm),
          ("norm",               stats.norm),
          ("pareto",             stats.pareto),
          ("powerlaw",           stats.powerlaw),
          ("t",                  stats.t),
          ("triang",             stats.triang),
          ("uniform",            stats.uniform),
          ("weibull_min",        stats.weibull_min),
          ("weibull_max",        stats.weibull_max),
         ]


def fit(data, scipydist, name=None):

    # fit distribution using maximum likelihood
    params = scipydist.fit(data)

    # create a "frozen" distribution object
    dist = scipydist(*params)

    # calculate log likelihood function and info criteria
    loglike = dist.logpdf(data).sum()
    bic = np.log(len(data)) * len(params) - 2.0 * loglike  # Schwarz
    aic = 2.0 * len(params) - 2.0 * loglike                # Akaike

    # p-values for GOF tests
    ad_pval = adtest(data, dist)[1]  # Anderson-Darling
    ks_pval = kstest(data, dist)[1]  # Kolmogorov-Smirnov

    return {"bic": bic, "aic": aic, "ad_pval": ad_pval,
            "ks_pval": ks_pval, "dist": dist, "name": name}


def _fit_all(data, dist_list):
    results = list(map(lambda x: fit(data, x[1], x[0]), dist_list))
    return sorted(results, key=lambda r: r["bic"])  # lowest BIC to highest


def _fstr(value):
    return ("%.3f" % value).rjust(8)


def _print_result(r, header=False):
    if header is True:
        print("   distribution,      BIC,      AIC,   KS p-val,   AD p-val")
    else:
        print("%s, %s, %s,   %s,   %s" %
              (r["name"].rjust(15), _fstr(r["bic"]), _fstr(r["aic"]),
               _fstr(r["ks_pval"]), _fstr(r["ad_pval"])))


def compare(data, long=False, disp=False):
    dist_list = _long if long is True else _short

    if disp is True:
        print("Fitting %s probability distributions...\n" % len(dist_list))

    results = _fit_all(data, dist_list)

    if disp is True:
        _print_result(None, header=True)
        list(map(_print_result, results))

    return results


if __name__ == "__main__":

    # example: compare fit of possible distributions
    data = stats.norm.rvs(size=30)  # true loc=0.0, true scale=1.0
    results = compare(data, disp=True)
    print("\nBest fit according to BIC: %s" % results[0]["name"])

    # example: decide that normal looks best, get "frozen" (parameterized)
    # instance of normal
    mydist = fit(data, stats.norm)["dist"]

    # invoke the PPF
    v = mydist.ppf(0.5)
    print("\nv=%s" % v)

    # invoke the CDF
    u = mydist.cdf(v)
    print("u=%s" % u)

