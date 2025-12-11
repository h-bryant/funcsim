import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Generator, Optional, Tuple
import warnings
import conversions
import nearby
import shapiro


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

warnings.showwarning = custom_warning_format


def _goodUvec(uvec: np.ndarray) -> bool:
    # check that values in uvec are in (0, 1)
    return bool(np.all((uvec > 0.0) & (uvec <1.0)))


def _memoize(f):
    # custom memoization decorator for _makeA, which works around the
    # fact that a numpy array is not inherently hashable
    class memodict(dict):
        def __init__(self, f):
            self.f = f

        def __call__(self, sig):
            key = hash(sig.tobytes())
            if key not in self.keys():
                self[key] = self.f(sig)
            return self[key]

    return memodict(f)


@_memoize
def _makeA(sigma):
    # get cholesky decomp of a covar matrix, after some sanity checks
    return np.linalg.cholesky(sigma)


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
    return int(min(math.floor((M)*u), M))


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
        Input data array of with variables in columns and observations
        in rows.
    bw : str, optional
        Bandwidth selection method, 'scott' or 'silverman'.
        Default is 'scott'.    
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

        if type(bw) == str:
            if bw == 'scott' or bw is None:
                self._bw = self._scott
            elif bw == 'silverman':
                self._bw = self._silverman
        else:
            # convert user's H (orig. units) into std units: D^{-1} H D^{-1}
            Dinv = np.diag(1.0 / self._stds)
            self._bw = Dinv @ bw @ Dinv

        # cholestky decomp of bandwidth matrix
        self._chol = nearby.nearestpd(self._bw)

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
        Input data array of with variables in columns and observations
        in rows.
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
            A pandas Series representing a joint draw from the KDE.  The index
            values are the variable names, and the values are the random
            values. If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            oreder of the columns in the input data.
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
        values in each column should be in the range [0, 1]. The parameters
        are fit using the method of moments.  If the sample covariance
        matrix is not positive definite, the Higham method is used to
        calculate the nearest positive definite matrix.
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

        # standardize the data
        self._z = pd.DataFrame(stats.norm.ppf(self._data), columns=self._names)

        # fit MV normal dist to the standardized data
        self._mvnorm = MvNorm(self._z)

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
            oreder of the columns in the input data.
        
        """
        return self._mvnorm.draw(ugen).apply(stats.norm.cdf)


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
        values in each column should be in the range [0, 1]. Parameters are
        fit using maximum likelihood.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:
        
        # check that copulae package is installed
        try:
            import copulae
        except ImportError as e:
            raise ImportError("Optional dependency 'copulae' is required for "
                              "CopulaStudent. Install with `pip install "
                              "copulae`.") from e

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
        self._cop = copulae.elliptical.StudentCopula(dim=self._K)
        self._cop.fit(self._data)
        self._rho = self._cop.sigma
        self._nu = self._cop.params.df

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
            oreder of the columns in the input data.
        
        """
        uvec = [next(ugen) for i in range(self._K)]
        z = np.dot(self._rho, stats.norm.ppf(uvec))
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
        values in each column should be in the range [0, 1]. Parameters are
        fit using maximum likelihood.  This implementation accomodates
        only positive dependence, so the fitted value for theta is
        constrained to be >= 2.0.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:
        
        # check that copulae package is installed
        try:
            import copulae
        except ImportError as e:
            raise ImportError("Optional dependency 'copulae' is required for "
                              "CopulaClayton. Install with `pip install "
                              "copulae`.") from e

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
        self._cop = copulae.archimedean.ClaytonCopula()
        self._cop.fit(self._data)
        self._theta = max(self._cop.params, 2.000)
        if self._cop.params < 2.00:
            warnings.warn(f"The Clayton copula implementation in funcsim "
                          f"accomodates only positive dependence. The fitted "
                          f"value for theta is {self._theta}, implying "
                          f"negative dependence.  A theta value of 2.0 is "
                          f"being used rather that the fitted value, but this "
                          f"implies no dependence among the variables. You "
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
            oreder of the columns in the input data.
        
        """
        v = stats.gamma.ppf(next(ugen), (1.0/self._theta))
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
        values in each column should be in the range [0, 1]. Parameters are
        fit using maximum likelihood.  This implementation accomodates
        only positive dependence, so the fitted value for theta is
        constrained to be > 1.0.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:
        
        # check that copulae package is installed
        try:
            import copulae
        except ImportError as e:
            raise ImportError("Optional dependency 'copulae' is required for "
                              "CopulaGumbel. Install with `pip install "
                              "copulae`.") from e

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
        self._cop = copulae.archimedean.GumbelCopula()
        self._cop.fit(self._data)
        self._theta = max(self._cop.params, 1.0000038089)  # ensure theta > 1.0
        if self._cop.params <= 1.0:
            warnings.warn(f"The Gumbel copula implementation in funcsim "
                          f"accomodates only positive dependence. The fitted "
                          f"value for theta is {self._params}, implying "
                          f"negative dependence.  A theta value of 1.0 is "
                          f"being used rather that the fitted value, but this "
                          f"implies no dependence among the variables. You "
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
            A pandas Series representing a joint draw from the Clayton
            copula. The index reflects the variable names, and non-independent
            standard uniform draws are the values in the Series.
            If no variable names were provided in the input data,
            the variables will be named 'v0', 'v1', ..., reflecting the
            oreder of the columns in the input data.
        
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
        values in each column should be in the range [0, 1]. Parameters are
        fit using maximum likelihood.  This implementation accomodates
        only positive dependence, so the fitted value for theta is
        constrained to be >= 0.
    """

    def __init__(self,
                 udata: conversions.ArrayLike,
                 ) -> None:
        
        # check that copulae package is installed
        try:
            import copulae
        except ImportError as e:
            raise ImportError("Optional dependency 'copulae' is required for "
                              "CopulaFrank. Install with `pip install "
                              "copulae`.") from e

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
        self._cop = copulae.archimedean.FrankCopula()
        self._cop.fit(self._data)
        self._theta = self._cop.params

        # ensure that theta is non-negative
        self._theta = max(0.0, self._theta)

        if self._cop.params <= 0.0:
            warnings.warn(f"The Frank copula implementation in funcsim "
                          f"accomodates only positive dependence. The fitted "
                          f"value for theta is {self._params}, implying "
                          f"negative dependence.  A theta value of 0.0 is "
                          f"being used rather that the fitted value, but this "
                          f"implies no dependence among the variables. You "
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
            oreder of the columns in the input data.
        
        """
        # Generate d uniform random variables
        uA = np.array([next(ugen) for _ in range(self._K)])

        # generate "frailty" variable random draw
        uval = next(ugen)
        try:
            v = stats.logser.ppf(p=(1.0 - np.exp(-self._theta)), q=uval)
        except RuntimeError:
            # logser.ppf sometimes throws an error for some combinations of
            # p and q, even though q values on either side of the problematic
            # q value seem to work just fine...
            v = stats.logser.ppf(p=(1.0 - np.exp(-self._theta)), q=(uval-0.01))

        # generate final draws
        retA = -1.0 / self._theta * np.log(1.0 + np.exp(-(-np.log(uA) / v))
                                           * (np.exp(-self._theta) - 1.0))

        return pd.Series(retA, index=self._names)
