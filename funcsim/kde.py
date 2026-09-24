import math
from typing import Union
import numpy as np
import scipy.optimize
import scipy.stats as stats
from scipy import interpolate, special
from . import conversions


# inverse-CDF table: number of grid points, and the margin beyond the data
# range in kernel standard deviations (the CDF is within 6e-16 of 0 or 1 at
# eight standard deviations from every observation)
_PPF_GRID = 4096
_PPF_MARGIN = 8.0


class _PpfTable:
    # monotone inverse of a Gaussian KDE's CDF.  The CDF F and survival
    # function S are evaluated on a uniform x grid spanning the data plus a
    # margin, and x is interpolated as a function of the normal score
    # q = Phi^-1(F) (computed from S in the upper half, where 1 - F loses
    # precision) by a shape-preserving cubic (PCHIP).  On that scale the
    # inverse CDF is smooth and nearly linear in the tails, so the
    # interpolant is accurate to about 1e-9 of the data range with 4096
    # points (see tests/test_kde_ppf_table.py), and monotone in u
    def __init__(self, gkde: stats.gaussian_kde) -> None:
        data = np.asarray(gkde.dataset[0], dtype=float)
        weights = np.asarray(gkde.weights, dtype=float)
        h = math.sqrt(float(gkde.covariance[0, 0]))
        lo = float(data.min()) - _PPF_MARGIN * h
        hi = float(data.max()) + _PPF_MARGIN * h
        xg = np.linspace(lo, hi, _PPF_GRID)
        F = np.empty(_PPF_GRID)
        S = np.empty(_PPF_GRID)
        # chunk the grid so the (grid x data) intermediate stays modest
        step = max(1, 2_000_000 // max(1, data.size))
        for i in range(0, _PPF_GRID, step):
            z = (xg[i:i + step, None] - data[None, :]) / h
            F[i:i + step] = special.ndtr(z) @ weights
            S[i:i + step] = special.ndtr(-z) @ weights
        with np.errstate(divide="ignore"):
            q = np.where(F <= 0.5, special.ndtri(F), -special.ndtri(S))
        # PCHIP needs strictly increasing abscissae; drop grid points where
        # the CDF is flat to machine precision (a gap between data
        # clusters) or the score is not finite
        keep = np.isfinite(q)
        keep[1:] &= np.diff(q) > 0.0
        self._q = q[keep]
        self._x = xg[keep]
        self._interp = interpolate.PchipInterpolator(self._q, self._x,
                                                     extrapolate=False)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        # inverse CDF for probabilities inside the table's range; NaN outside
        with np.errstate(divide="ignore", invalid="ignore"):
            q = special.ndtri(u)
        return self._interp(q)


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
    exact_ppf : bool, optional
        If True, :meth:`ppf` finds every value by root finding on
        :meth:`cdf`, as funcsim did before version 0.2.7 (about 0.3 ms per
        value).  If False (the default), :meth:`ppf` is served from an
        inverse-CDF table built once, on the first call; see Notes.

    Notes
    -----
    ``pdf``, ``cdf``, and ``ppf`` return a Python float for a scalar
    argument and a NumPy array of the same shape for an array-like argument
    (a pandas Series or a list gives an array).

    The inverse-CDF table evaluates the CDF on 4096 points spanning the data
    range plus eight kernel standard deviations on each side, and
    interpolates x against the normal score of the CDF with a
    shape-preserving cubic (PCHIP).  The interpolant is monotone, so ``ppf``
    is nondecreasing in ``u``, and in tests over samples of 20 to 1000
    observations (unimodal, bimodal, and on scales from 1e-6 upward) it
    agrees with the root finder to better than 1e-8 of the data range; the
    documented accuracy target is 1e-6 of the data range.  Probabilities
    outside the table's range (below about 1e-15 or above 1 - 1e-15) fall
    back to root finding.  Because the table and the root finder differ in
    the last digits, simulations that pass draws through ``ppf`` change
    slightly at that level when upgrading; pass ``exact_ppf=True`` to
    reproduce earlier results exactly.
    """
    def __init__(self,
                 data : conversions.VectorLike,
                 bw : Union[str, float] = 'scott',
                 exact_ppf : bool = False):

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

        self._exact_ppf = bool(exact_ppf)
        self._table = None  # inverse-CDF table, built on first use

    @staticmethod
    def _apply(func, x):
        # evaluate a 1-D-array function on scalar or array-like input:
        # scalar in, Python float out; array-like in, array of that shape out
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return float(func(arr.reshape(1))[0])
        return func(arr.ravel()).reshape(arr.shape)

    def pdf(self,
            v: Union[float, conversions.VectorLike]
           ) -> Union[float, np.ndarray]:
        """
        Probability density function of the KDE at value v.

        Parameters
        ----------
        v : float or array-like
            Value(s) at which to evaluate the PDF.

        Returns
        -------
        float or numpy.ndarray
            The estimated probability density at v: a float for a scalar
            `v`, an array of the same shape otherwise.
        """
        return self._apply(lambda a: np.asarray(self.gkde(a), dtype=float), v)

    def _cdf_array(self, v: np.ndarray) -> np.ndarray:
        # integrate from -inf (not from an arbitrary finite cutoff, which
        # loses tail mass); clip to [0, 1] against numerical error
        return np.array([min(1.0, max(0.0, self.gkde.integrate_box_1d(-np.inf, x)))
                         for x in v])

    def cdf(self,
            v: Union[float, conversions.VectorLike]
           ) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function of the KDE at value v.

        Parameters
        ----------
        v : float or array-like
            Value(s) at which to evaluate the CDF.

        Returns
        -------
        float or numpy.ndarray
            The estimated cumulative probability at v: a float for a scalar
            `v`, an array of the same shape otherwise.
        """
        return self._apply(self._cdf_array, v)

    def _ppf_root(self, u: float) -> float:
        # inverse CDF of one probability by bracketed root finding on the
        # CDF (Brent's method), the method used for every value before 0.2.7
        try:
            return float(scipy.optimize.brentq(
                lambda x: self._cdf_array(np.array([x]))[0] - u,
                self.ppf_low, self.ppf_high, xtol=self.ppf_xtol))
        except ValueError:
            # the bracket does not contain a root: u is at or beyond the
            # CDF's value at an end of the bracket
            if u > 0.98:
                return self.ppf_high
            elif u < 0.02:
                return self.ppf_low
            raise ValueError("kde ppf could not find solution")

    def _ppf_array(self, u: np.ndarray) -> np.ndarray:
        if np.any(np.isnan(u)) or np.any((u < 0.0) | (u > 1.0)):
            raise ValueError("u must be within the range [0.0, 1.0]")
        if self._exact_ppf:
            return np.array([self._ppf_root(x) for x in u])
        if self._table is None:
            self._table = _PpfTable(self.gkde)
        out = self._table(u)
        for i in np.flatnonzero(~np.isfinite(out)):
            out[i] = self._ppf_root(u[i])  # outside the table's range
        return out

    def ppf(self,
            u: Union[float, conversions.VectorLike]
           ) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF) of the KDE.

        Parameters
        ----------
        u : float or array-like
            Probability value(s) in the range [0.0, 1.0].

        Returns
        -------
        float or numpy.ndarray
            The value x such that CDF(x) = u: a float for a scalar `u`, an
            array of the same shape otherwise.

        Raises
        ------
        ValueError
            If a value of u is not in [0.0, 1.0], or root finding (used
            when `exact_ppf` is True, or for a probability outside the
            table's range) fails.

        Notes
        -----
        By default values are read from a monotone inverse-CDF table built
        on the first call (see the class notes for its accuracy); with
        ``exact_ppf=True`` each value is found by root finding on
        :meth:`cdf`.  A value of exactly 0 or 1 lies outside the table and
        returns the lower or upper end of the root-finding bracket (the
        data range extended by three times its width on each side), as in
        earlier versions.
        """
        return self._apply(self._ppf_array, u)
