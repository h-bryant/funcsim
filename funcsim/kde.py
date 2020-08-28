import math
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats


class Kde():
    # create PDF, CDF, and PPF functions via Gaussian KDE.
    # "sample" should be one of the following
    # types: list, tuple, numpy.array, or pandas.Series.
    # "bw" is the bandwidth method or parameter.  Alternatives
    # to bw="scott" are 'silverman' or a float

    # resulting object has methods analogous to a scipy "frozen" (parameterized)
    # parametric distribution object: pdf(), cdf(), and ppf()

    def __init__(self, sample, bw='scott'):

        # raw kde object
        self.gkde = stats.gaussian_kde(sample, bw)

        # lower limit of integreation for CDF
        self.cdf_low = float(min(sample)) - 1.0 * (max(sample) - min(sample))

        # initial guess for PPF optimization: the sample mean
        self.ppf_x0 = float(sum(sample)) / float(len(sample))

        # "bracket" for PPF root finding
        self.ppf_low = float(min(sample)) - 3.0 * (max(sample) - min(sample))
        self.ppf_high = float(max(sample)) + 3.0 * (max(sample) - min(sample))

    def pdf(self, v):
        # probability density function
        return float(self.gkde(v))

    def cdf(self, v):
        # cumulative distribution fucntion
        return float(self.gkde.integrate_box_1d(self.cdf_low, v))

    def ppf(self, u):
        # numerical percent point function (inverse CDF)
        assert u >= 0.0 and u <= 1.0, \
            "u must be within the range [0.0, 1.0]"

        if int(scipy.__version__[0]) > 0:
            # version for newer versions of scipy.optimize
            try:
                result0 = scipy.optimize.root_scalar(f=lambda x: self.cdf(x) - u,
                                                     x0=self.ppf_x0,
                                                     fprime=self.pdf,
                                                     bracket=(self.ppf_low,
                                                              self.ppf_high),
                                                     # method="bisect",
                                                     # method="brentq",
                                                     # method="brenth",
                                                     # method="ridder",
                                                     # method="toms748",
                                                     # method="newton",
                                                     # method="secant",
                                                     # method="halley",
                                                     xtol=0.0001)

            except ValueError:
                if u > 0.98:
                    return self.ppf_high
                elif u < 0.02:
                    return self.ppf_low
                else:
                    raise ValueError("kde ppf could not find solution")

            if result0.converged is True:
                return result0.root
            else:
                msg = "numerical optimization for PPF failed"
                raise ValueError(msg)

        else:  # scipy version is < 1.0
            # version that works with scipy 0.13.3
            try:
                (x0, r) = scipy.optimize.brentq(f=lambda x: self.cdf(x) - u,
                                                a=self.ppf_low,
                                                b=self.ppf_high,
                                                xtol=0.0001,
                                                full_output=True,
                                                disp=True)
            except ValueError:
                if u > 0.98:
                    return self.ppf_high
                elif u < 0.02:
                    return self.ppf_low
                else:
                    raise ValueError("kde ppf could not find solution")

            if r.converged is True:
                return x0
            else:
                msg = "numerical optimization for PPF failed"
                raise ValueError(msg)


def fitkde(sample, bw="scott"):
    return Kde(sample, bw)