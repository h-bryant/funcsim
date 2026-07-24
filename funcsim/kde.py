import functools
from typing import Union
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
from . import conversions


def vectorized_method(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # vectorize the bound method (bound to `self`)
        return np.vectorize(lambda *a, **k: func(self, *a, **k))(*args, **kwargs)
    return wrapper


class Kde():
    """
    Univariate kernel density estimator (KDE).

    Parameters
    ----------
    data : conversions.VectorLike
        1-D data vector (list, tuple, np.ndarray, xr.DataArray, or pd.Series).
    bw : str or float, optional
        Bandwidth selection method ('scott', 'silverman') or a positive
        float to use as the bandwidth (the standard deviation of the
        Gaussian kernel), in the same units as the data.
        Default is 'scott'.
    """
    def __init__(self,
                 data : conversions.VectorLike,
                 bw : Union[str, float] = 'scott'):

        sampleA = conversions.vlToArray(data)

        # raw kde object.  a numeric bw is the kernel standard deviation
        # in data units; scipy's float bw_method is instead a factor
        # multiplied by the sample std. dev., so convert
        if isinstance(bw, (int, float)) and not isinstance(bw, bool):
            if bw <= 0.0:
                raise ValueError("a numeric 'bw' must be positive")
            bw = float(bw) / float(np.std(sampleA, ddof=1))
        self.gkde = stats.gaussian_kde(sampleA, bw)

        # initial guess for PPF optimization: the sample mean
        self.ppf_x0 = float(sum(sampleA)) / float(len(sampleA))

        # "bracket" for PPF root finding
        self.ppf_low = float(min(sampleA)) - \
            3.0 * (max(sampleA) - min(sampleA))
        self.ppf_high = float(max(sampleA)) + \
            3.0 * (max(sampleA) - min(sampleA))

        # x tolerance for PPF root finding, proportional to the data
        # scale (an absolute tolerance would make ppf meaningless for
        # data on scales much smaller than the tolerance)
        self.ppf_xtol = max((self.ppf_high - self.ppf_low) * 1e-9, 1e-300)

    @vectorized_method
    def pdf(self,
            v: float
           ) -> float:
        """
        Probability density function of the KDE at value v.

        Parameters
        ----------
        v : float
            Value at which to evaluate the PDF.

        Returns
        -------
        float
            The estimated probability density at v.
        """
        return float(self.gkde(v))

    @vectorized_method
    def cdf(self,
            v: float
           ) -> float:
        """
        Cumulative distribution function of the KDE at value v.

        Parameters
        ----------
        v : float
            Value at which to evaluate the CDF.

        Returns
        -------
        float
            The estimated cumulative probability at v.
        """
        # integrate from -inf (not from an arbitrary finite cutoff, which
        # loses tail mass); clip to [0, 1] against numerical error
        raw = self.gkde.integrate_box_1d(-np.inf, v)
        return float(min(1.0, max(0.0, raw)))

    @vectorized_method
    def ppf(self,
            u: float
           ) -> float:
        """
        Numerical percent point function (inverse CDF) for the KDE.

        Parameters
        ----------
        u : float
            Probability value in the range [0.0, 1.0].

        Returns
        -------
        float
            The value x such that CDF(x) = u.

        Raises
        ------
        ValueError
            If u is not in [0.0, 1.0] or optimization fails.
        """
        if u < 0.0 or u > 1.0:
            raise ValueError("u must be within the range [0.0, 1.0]")

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
                                                     xtol=self.ppf_xtol)

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
                                                xtol=self.ppf_xtol,
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