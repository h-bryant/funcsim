import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Generator, Optional, Tuple
import warnings
from . import conversions
from . import copfit
from . import nearby
from . import shapiro


def _goodUvec(uvec: np.ndarray) -> bool:
    # check that values in uvec are in (0, 1)
    return bool(np.all((uvec > 0.0) & (uvec <1.0)))


def _checkcov(cov, name):
    # sanity check a covariace matrix.  Use "name" in any error/exception msg
    if not np.allclose(cov, cov.T):
        raise ValueError(f"{name} must be symmetrical")
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError(f"{name} must be positive definite")
    return cov


def _rand_int(u, M):
    # given a standard uniform draw "u", select a
    # random integer from a length "M" sequece: 0, 1, ..., M-1
    # (clamp at M - 1 so that u == 1.0 cannot index out of range)
    return int(min(math.floor(M * u), M - 1))


def _logser_draw(theta: float, u1: float, u2: float) -> float:
    # draw from the logarithmic-series distribution with parameter
    # p = 1 - exp(-theta), given two independent standard uniform draws,
    # via Kemp's "LK" algorithm.  Working directly with log(1 - p) = -theta
    # keeps the sampler exact and finite even when p rounds to 1.0 in
    # floats (theta > ~37), where stats.logser breaks down; the O(1) cost
    # also avoids logser.ppf's O(k) scan, which is impractically slow for
    # theta > ~20.
    #
    # Kemp, A. W. (1981). Efficient generation of logarithmically
    # distributed pseudo-random variables. Applied Statistics, 30(3),
    # 249-253.

    # clip draws away from {0, 1} so the logs below stay finite
    u1 = min(max(u1, 1e-300), 1.0 - 1e-16)
    u2 = min(max(u2, 1e-300), 1.0 - 1e-16)

    p = -math.expm1(-theta)  # p rounding to exactly 1.0 is harmless here
    if u2 >= p:
        return 1.0
    x = theta * u1
    q = -math.expm1(-x)  # q = 1 - (1 - p)**u1
    if u2 <= q * q:
        # log(q), using log1p(-exp(-x)) where q itself would round to 1.0
        logq = (math.log(q) if x <= 0.6931471805599453
                else math.log1p(-math.exp(-x)))
        return max(1.0, math.floor(1.0 + math.log(u2) / logq))
    elif u2 >= q:
        return 1.0
    return 2.0


def _skew_stable_draw(draw, alpha, beta, gamma, delta):
    # random draw based on Nolan (1997) appraoch to
    # general stable distributions
    #
    # adapted from the MIT-licensed code at:
    # https://github.com/DanielBok/copulae/blob/master/copulae/stats/stable/stable.py

    if not alpha.is_integer():
        omega = np.tan(alpha * np.pi / 2)
    elif alpha == 1:
        omega = 2 / np.pi * np.log(gamma)
    else:
        omega = 0

    if np.isclose(alpha, 1) and np.isclose(beta, 0):
        z = stats.cauchy.ppf(next(draw))
    else:
        thetu = np.pi * (next(draw) - 0.5)

        w = stats.expon.ppf(next(draw))

        bt = beta * np.tan(alpha * np.pi / 2)
        t0 = min(max(-np.pi / 2.0, np.arctan(bt) / alpha), np.pi / 2.0)
        at = alpha * (thetu + t0)

        c = (1 + bt ** 2) ** (1 / (2 * alpha))

        z = (c * np.sin(at)
             * (np.cos(thetu) ** (-1 / alpha))
             * ((np.cos(thetu - at) / w) ** ((1 - alpha) / alpha))
             - bt)

    return z * gamma + delta + beta * gamma * omega


def covtocorr(cov: conversions.ArrayLike) -> pd.DataFrame:
    """
    Convert a covariance matrix to a correlation matrix.

    This function takes a symmetric, positive definite covariance matrix
    and returns the corresponding correlation matrix.

    Parameters
    ----------
    cov : ArrayLike
        Covariance matrix (square, symmetric).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame representing the correlation matrix.

    Raises
    ------
    ValueError
        If the input is not a valid covariance matrix.

    Notes
    -----
    The correlation matrix is computed by normalizing the covariance
    matrix by the standard deviations of each variable.
    """
    cov_np = conversions.alToArray(cov)
    names = conversions.alColNames(cov)
    _checkcov(cov_np, "covariance matrix")    

    N = cov.shape[0]
    sinv = np.identity(N) * np.sqrt(1.0 / np.diag(cov))
    corr = sinv.dot(cov).dot(sinv)
    return pd.DataFrame(corr, index=names, columns=names)


def spearman(array: conversions.ArrayLike) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Spearman's rank correlation coefficient and its 95% confidence
    interval.

    This function computes Spearman's rho for two variables and returns the
    correlation coefficient along with the lower and upper bounds of the 95%
    confidence interval.

    Parameters
    ----------
    array : ArrayLike
        Input data as an (N, 2) array-like object, where N is the number of
        observations and each column represents a variable.

    Returns
    -------
    tuple of (float, tuple of float)
        A tuple (rho, ci), where rho is Spearman's rank correlation coefficient,
        and ci is a tuple of (lower, upper) bounds for the 95% confidence
        interval.

    Raises
    ------
    AssertionError
        If the input array does not have shape (N, 2).

    Notes
    -----
    The confidence interval is computed using Fisher's z-transformation.
    """
    a = conversions.alToArray(array)
    assert len(a.shape) == 2, "a must have exactly two dimensions"
    assert a.shape[1] == 2, "a must have exactly two columns"

    rho_s = stats.spearmanr(a)[0]

    N = a.shape[0]
    if N <= 3:
        raise ValueError("At least 4 observations are required for confidence interval.")

    stderr = 1.0 / math.sqrt(N - 3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(rho_s) - delta)
    upper = math.tanh(math.atanh(rho_s) + delta)

    return (rho_s, (lower, upper))


class MvKde():
    """
    A multivariate KDE distribution object.

    Parameters
    ----------
    data : ArrayLike
        Input data array with variables in columns and observations
        in rows.
    bw : str or ArrayLike, optional
        Bandwidth selection method ('scott' or 'silverman'), or a K-by-K
        bandwidth covariance matrix in the units of the data.
        Default is 'scott'.

    Notes
    -----
    Each call to :meth:`draw` consumes K + 1 values from ``ugen``, where K
    is the number of variables.
    """
    def __init__(self,
                 data: conversions.ArrayLike,
                 bw: str = 'scott'
                ) -> None:
        self._data = conversions.alToArray(data)
        self._names = conversions.alColNames(data)
        (self._M, self._K) = self._data.shape

        # standardize data (store mean/std to undo later)
        self._means = self._data.mean(axis=0)
        self._stds  = self._data.std(axis=0)
        self._stds[self._stds == 0.0] = 1.0   # for columns w/no variability
        self._data = (self._data - self._means) / self._stds
        self._data_std = self._data  # sample standard deviations
        stdevs = self._data_std.std(axis=0)

        # rule-of-thumb scott bandwidth
        mult = self._M**(-1.0/(self._K+4.0))
        self._scott =  np.square(mult * np.diagflat(stdevs))

        # rule-of-thumb silverman bandwidth
        smult = ((4.0 * self._M)/ (self._K+2.0))**(-1.0 / (self._K+4.0))
        self._silverman = np.square(smult * np.diagflat(stdevs))

        if bw is None or bw == 'scott':
            self._bw = self._scott
        elif bw == 'silverman':
            self._bw = self._silverman
        elif isinstance(bw, str):
            raise ValueError(f"unknown bandwidth method '{bw}'; expected "
                             f"'scott', 'silverman', or a "
                             f"{self._K}-by-{self._K} bandwidth matrix")
        else:
            H = np.asarray(bw, dtype=float)
            if H.shape != (self._K, self._K):
                raise ValueError(f"a bandwidth matrix must have shape "
                                 f"({self._K}, {self._K}); got {H.shape}")
            # convert user's H (orig. units) into std units: D^{-1} H D^{-1}
            Dinv = np.diag(1.0 / self._stds)
            self._bw = Dinv @ H @ Dinv

        # cholesky decomp of the bandwidth (covariance) matrix, so that
        # draws apply covariance H (not H @ H.T)
        self._chol = np.linalg.cholesky(nearby.nearestpd(self._bw))

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the multivariate distribution.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator that yields independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the KDE.  The index
            values are the variable names, and the values are the random
            values.
        """
        # hist obs about which we will sample
        m = _rand_int(next(ugen), self._M)

        # means for this sample
        mu = self._data[m]

        # generate joint standard normal draw from the obs m kernel
        uvec = [next(ugen) for i in range(self._K)]
        retA_std = mu + np.dot(self._chol, stats.norm.ppf(uvec))

        # de-standardize back to original units
        retA = self._means + self._stds * retA_std

        return pd.Series(retA, index=self._names)


class MvNorm():
    """
    A multivariate normal distribution object. A vector of means
    and a covariance matrix are computed from the input data.  If the sample
    covariance matrix is not positive definite, the Higham
    method is used to calculate the nearest positive definite matrix.

    Parameters
    ----------
    data : ArrayLike
        Input data array with variables in columns and observations
        in rows.

    Notes
    -----
    Each call to :meth:`draw` consumes K values from ``ugen``, where K is
    the number of variables.  A warning is issued for any variable whose
    Shapiro-Wilk test rejects normality at the 5% level.
    """
    def __init__(self,
                 data: conversions.ArrayLike,
                ) -> None:

        self._data = conversions.alToArray(data)
        self._names = conversions.alColNames(data)
        (self._M, self._K) = self._data.shape

        # fit mean and covariance
        self._mu = self._data.mean(axis=0)

        # compute covariance matrix, ensure it is positive definite
        self._sigma = nearby.nearestpd(np.cov(self._data, rowvar=False))

        # get cholesky decomposition of covariance matrix
        self._A = np.linalg.cholesky(self._sigma)

        # warn if data seem non-normally distributed
        for k in range(self._K):
            swp = shapiro.swtest(self._data[:, k])[1]
            if not swp > 0.05:
                msg = (f"variable '{self._names[k]}' may not be "
                       f" normally distributed. (Shapiro-Wilk p-value"
                       f"={swp:.3f})")
                warnings.warn(msg, UserWarning)

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the multivariate distribution.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the multivariate
            normal distribution.  The index values are the variable names,
            and the values are the random values.  If no variable names were
            provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            order of the columns in the input data.
        """
        uvec = [next(ugen) for i in range(self._K)]
        retA = self._mu + np.dot(self._A, stats.norm.ppf(uvec))
        return pd.Series(retA, index=self._names)


class CopulaGauss():
    """
    A Gaussian copula object. 

    Parameters
    ----------
    udata : ArrayLike
        Input array of probability values from individual marginal
        distributions ("pseudo-observations") with variables in
        columns and observations in rows.  That is, each column
        represents the result of applying the fitted CDF of some marginal
        distribution to the raw data for that variable.  The
        values in each column should be in the range (0, 1). The correlation
        matrix is fit using the method of moments on the normal scores of
        the pseudo-observations.  If it is not positive definite, the Higham
        method is used to calculate the nearest positive definite matrix.

    Notes
    -----
    Each call to :meth:`draw` consumes K values from ``ugen``, where K is
    the number of variables.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:
        self._data = conversions.alToArray(udata)
        self._names = conversions.alColNames(udata)
        (self._M, self._K) = self._data.shape

        # check that data are in (0, 1)
        for k in range(self._K):
            if not _goodUvec(self._data[:, k]):
                raise ValueError(f"Column {k} of the input data, with name "
                                 f"{self._names[k]}, has values that are not "
                                 f"in the range (0, 1)")

        # normal scores of the pseudo-observations
        self._z = stats.norm.ppf(self._data)

        # a Gaussian copula's only parameter is a correlation matrix: the
        # mean is zero and the diagonal is one by construction, so fit
        # second moments about zero and normalize to unit diagonal
        # (fitting a full MvNorm here would let imperfect pseudo-
        # observations distort the uniform marginals of the draws)
        m2 = (self._z.T @ self._z) / float(self._M)
        d = np.sqrt(np.diag(m2))
        corr = m2 / np.outer(d, d)
        corr = nearby.nearestpd(0.5 * (corr + corr.T))
        s = np.sqrt(np.diag(corr))
        corr = corr / np.outer(s, s)  # renormalize to unit diagonal
        np.fill_diagonal(corr, 1.0)
        self._rho = corr

        # cholesky decomposition of the correlation matrix, for draws
        self._A = np.linalg.cholesky(self._rho)

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the Gaussian copula.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the Gaussian copula.
            The index reflects the variable names, and non-independent
            standard uniform draws are the values in the Series.
            If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            order of the columns in the input data.

        """
        uvec = [next(ugen) for i in range(self._K)]
        z = np.dot(self._A, stats.norm.ppf(uvec))
        return pd.Series(stats.norm.cdf(z), index=self._names)


class CopulaStudent():
    """
    A Student's t copula object. 

    Parameters
    ----------
    udata : ArrayLike
        Input array of probability values from individual marginal
        distributions ("pseudo-observations") with variables in
        columns and observations in rows.  That is, each column
        represents the result of applying the fitted CDF of some marginal
        distribution to the raw data for that variable.  The
        values in each column should be in the range (0, 1). The correlation
        matrix is fit by pairwise Kendall's-tau inversion, and the degrees
        of freedom are then fit by profile maximum pseudo-likelihood.

    Notes
    -----
    Each call to :meth:`draw` consumes K + 1 values from ``ugen``, where K
    is the number of variables.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:

        self._data = conversions.alToArray(udata)
        self._names = conversions.alColNames(udata)
        (self._M, self._K) = self._data.shape

        # check that data are in (0, 1)
        for k in range(self._K):
            if not _goodUvec(self._data[:, k]):
                raise ValueError(f"Column {k} of the input data, with name "
                                 f"{self._names[k]}, has values that are not "
                                 f"in the range (0, 1)")

        # fit parameters
        (self._rho, self._nu) = copfit.fit_student(self._data)

        # cholesky decomposition of the correlation matrix, for draws
        self._A = np.linalg.cholesky(self._rho)

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the Student's t copula.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the Student's t
            copula. The index reflects the variable names, and non-independent
            standard uniform draws are the values in the Series.
            If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            order of the columns in the input data.
        
        """
        uvec = [next(ugen) for i in range(self._K)]
        z = np.dot(self._A, stats.norm.ppf(uvec))
        chi2 = stats.chi2.ppf(next(ugen), df=self._nu)
        mult = (self._nu / chi2)**0.5
        retA = stats.t.cdf(mult * z, df=self._nu)
        return pd.Series(retA, index=self._names)


class CopulaClayton():
    """
    A Clayton copula object. 

    Parameters
    ----------
    udata : ArrayLike
        Input array of probability values from individual marginal
        distributions ("pseudo-observations") with variables in
        columns and observations in rows.  That is, each column
        represents the result of applying the fitted CDF of some marginal
        distribution to the raw data for that variable.  The
        values in each column should be in the range (0, 1). The dependence
        parameter theta is fit by maximum pseudo-likelihood, starting from
        the Kendall's-tau inversion estimate.  This implementation accommodates
        only positive dependence, so the fitted value for theta is
        constrained to be > 0.

    Notes
    -----
    Each call to :meth:`draw` consumes K + 1 values from ``ugen``, where K
    is the number of variables.  If the data exhibit negative dependence
    (mean pairwise Kendall's tau <= 0), theta is set to its minimum
    (near independence) and a warning is issued.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:

        self._data = conversions.alToArray(udata)
        self._names = conversions.alColNames(udata)
        (self._M, self._K) = self._data.shape

        # check that data are in (0, 1)
        for k in range(self._K):
            if not _goodUvec(self._data[:, k]):
                raise ValueError(f"Column {k} of the input data, with name "
                                 f"{self._names[k]}, has values that are not "
                                 f"in the range (0, 1)")

        # fit parameters
        (self._theta, taub, clamped) = copfit.fit_clayton(self._data)
        if clamped:
            warnings.warn(f"The Clayton copula implementation in funcsim "
                          f"accommodates only positive dependence, but the "
                          f"data exhibit negative dependence (mean pairwise "
                          f"Kendall's tau = {taub:.3f}).  A theta value of "
                          f"{copfit.THETA_MIN_CLAYTON} is being used rather "
                          f"than a fitted value, which implies (near) "
                          f"independence among the variables. You "
                          f"should probably choose a different dependence "
                          f"representation for your data.",
                          UserWarning)

    def _Ftilde(self, t):
        return (1.0 + t)**(-1.0 / self._theta)

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the Clayton copula.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the Clayton
            copula. The index reflects the variable names, and non-independent
            standard uniform draws are the values in the Series.
            If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            order of the columns in the input data.
        
        """
        # floor the gamma frailty: for large theta the shape parameter is
        # tiny and the ppf underflows to 0.0 for small u, which would
        # raise ZeroDivisionError below.  the floored value yields the
        # correct limiting draw (all components near 0)
        v = max(stats.gamma.ppf(next(ugen), (1.0/self._theta)), 1e-300)
        retA = np.array([self._Ftilde(-math.log(next(ugen))/v)
                         for i in range(self._K)])
        return pd.Series(retA, index=self._names)


class CopulaGumbel():
    """
    A Gumbel copula object. 

    Parameters
    ----------
    udata : ArrayLike
        Input array of probability values from individual marginal
        distributions ("pseudo-observations") with variables in
        columns and observations in rows.  That is, each column
        represents the result of applying the fitted CDF of some marginal
        distribution to the raw data for that variable.  The
        values in each column should be in the range (0, 1). The dependence
        parameter theta is fit by maximum pseudo-likelihood, starting from
        the Kendall's-tau inversion estimate.  This implementation accommodates
        only positive dependence, so the fitted value for theta is
        constrained to be > 1.0.

    Notes
    -----
    Each call to :meth:`draw` consumes K + 2 values from ``ugen``, where K
    is the number of variables.  If the data exhibit negative dependence
    (mean pairwise Kendall's tau <= 0), theta is set to its minimum
    (near independence) and a warning is issued.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:

        self._data = conversions.alToArray(udata)
        self._names = conversions.alColNames(udata)
        (self._M, self._K) = self._data.shape

        # check that data are in (0, 1)
        for k in range(self._K):
            if not _goodUvec(self._data[:, k]):
                raise ValueError(f"Column {k} of the input data, with name "
                                 f"{self._names[k]}, has values that are not "
                                 f"in the range (0, 1)")

        # fit parameters
        (self._theta, taub, clamped) = copfit.fit_gumbel(self._data)
        if clamped:
            warnings.warn(f"The Gumbel copula implementation in funcsim "
                          f"accommodates only positive dependence, but the "
                          f"data exhibit negative dependence (mean pairwise "
                          f"Kendall's tau = {taub:.3f}).  A theta value of "
                          f"(approximately) 1.0 is being used rather than a "
                          f"fitted value, which implies (near) independence "
                          f"among the variables. You "
                          f"should probably choose a different dependence "
                          f"representation for your data.",
                          UserWarning)

    def _Ftilde(self, t):
        return math.exp(-(t**(1.0/self._theta)))

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the Gumbel copula.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the Gumbel
            copula. The index reflects the variable names, and non-independent
            standard uniform draws are the values in the Series.
            If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            order of the columns in the input data.

        """
        gamma = math.cos(0.5 * math.pi / self._theta)**self._theta
        alpha = 1.0 / self._theta
        v = _skew_stable_draw(ugen, alpha, 1.0, gamma, 0.0)
        retA = np.array([self._Ftilde(-math.log(next(ugen))/v)
                         for i in range(self._K)])
        return pd.Series(retA, index=self._names)


class CopulaFrank():
    """
    A Frank copula object. 

    Parameters
    ----------
    udata : ArrayLike
        Input array of probability values from individual marginal
        distributions ("pseudo-observations") with variables in
        columns and observations in rows.  That is, each column
        represents the result of applying the fitted CDF of some marginal
        distribution to the raw data for that variable.  The
        values in each column should be in the range (0, 1). The dependence
        parameter theta is fit by maximum pseudo-likelihood, starting from
        the Kendall's-tau inversion estimate.  This implementation accommodates
        only positive dependence, so the fitted value for theta is
        constrained to be > 0.

    Notes
    -----
    Each call to :meth:`draw` consumes K + 2 values from ``ugen``, where K
    is the number of variables.  If the data exhibit negative dependence
    (mean pairwise Kendall's tau <= 0), theta is set to its minimum
    (near independence) and a warning is issued.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:

        self._data = conversions.alToArray(udata)
        self._names = conversions.alColNames(udata)
        (self._M, self._K) = self._data.shape

        # check that data are in (0, 1)
        for k in range(self._K):
            if not _goodUvec(self._data[:, k]):
                raise ValueError(f"Column {k} of the input data, with name "
                                 f"{self._names[k]}, has values that are not "
                                 f"in the range (0, 1)")

        # fit parameters
        (self._theta, taub, clamped) = copfit.fit_frank(self._data)
        if clamped:
            warnings.warn(f"The Frank copula implementation in funcsim "
                          f"accommodates only positive dependence, but the "
                          f"data exhibit negative dependence (mean pairwise "
                          f"Kendall's tau = {taub:.3f}).  A theta value of "
                          f"{copfit.THETA_MIN_FRANK} is being used rather "
                          f"than a fitted value, which implies (near) "
                          f"independence among the variables. You "
                          f"should probably choose a different dependence "
                          f"representation for your data.",
                          UserWarning)

    def draw(self,
             ugen: Generator[float, None, None]
             ) -> pd.Series:
        """
        Generate a joint random draw from the Frank copula.

        Parameters
        ----------
        ugen : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        pd.Series
            A pandas Series representing a joint draw from the Frank
            copula. The index reflects the variable names, and non-independent
            standard uniform draws are the values in the Series.
            If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            order of the columns in the input data.
        
        """
        # Generate d uniform random variables
        uA = np.array([next(ugen) for _ in range(self._K)])

        # log-series "frailty" draw via Kemp's algorithm (exactly two
        # uniform draws, so the total draw count stays fixed)
        v = _logser_draw(self._theta, next(ugen), next(ugen))

        # final draws: -log(1 - (1 - exp(-theta)) * uA**(1/v)) / theta,
        # evaluated in log space so that huge frailty values (which are
        # routine for large theta) cannot round uA**(1/v) to 1.0 and
        # produce infs or NaNs
        with np.errstate(divide="ignore"):
            t = -np.log(uA) / v  # >= 0
            # log(1 - exp(-t)), split at log(2) for accuracy (log1mexp)
            log1mexp_t = np.where(t <= 0.6931471805599453,
                                  np.log(-np.expm1(-t)),
                                  np.log1p(-np.exp(-t)))
        retA = -np.logaddexp(log1mexp_t, -t - self._theta) / self._theta

        return pd.Series(retA, index=self._names)
