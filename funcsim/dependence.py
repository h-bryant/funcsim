import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Generator, Optional, Tuple
import warnings
import conversions
import nearby
import shapiro


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

        # sample standard deviations
        stdevs = data.std(axis=0)

        # rule-of-thumb bandwidths
        mult = self._M**(-1.0/(self._K+1.0))
        self._scott =  np.square(mult * np.diagflat(stdevs))
        smult = (4.0 / (self._K+2.0))**(2.0 / (self._K+4.0))
        self._silverman = smult * self._scott

        if bw == 'scott' or bw is None:
            self._bw = self._scott
        elif bw == 'silverman':
            self._bw = self._silverman
        else:
            self._bw = bw

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
        m = _rand_int(next(draw), self._M)

        # means for this sample
        mu = self._data[m]

        return normal(draw, self._bw, mu)


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

        # compute covariance matrix
        self._sigma = np.cov(self._data, rowvar=False)

        # ensure covariance matrix is positive definite
        if not np.all(np.linalg.eigvals(self._sigma) > 0):
            # use the Higam method to ensure positive definiteness
            self._sigma = nearby.nearestpd(self._data)

        # get cholesky decomposition of covariance matrix
        self._A = np.linalg.cholesky(self._sigma)

        # warn if data seem non-normally distributed
        for k in range(self._K):
            swp = shapiro.swtest(self._data[:, k])[1]
            if not swp > 0.05:
                msg = (f"Warning: variable '{self._names[k]}' may not be "
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
    A Gaussian Copula object. 

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
    A Student's t Copula object. 

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
                              "CopulaStudentst. Install with `pip install "
                              "copulae`.") from e

        self._data = conversions.alToArray(udata)
        self._names = conversions.alColNames(udata)
        (self._M, self._K) = self._data.shape

        # fit parameters
        # self._cop = copulae.elliptical.StudentCopula(dim=self._K)
        self._cop = copulae.elliptical.StudentCopula()
        self._cop.fit(self._data)

        # create a multivariate normal distribution object
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





def cclayton(draw: Generator[float, None, None],
             nvars: int,
             theta: float
            ) -> np.ndarray:
    """
    Generate joint uniform draws from a Clayton copula.

    Parameters
    ----------
    draw : Generator[float, None, None]
        A generator yielding independent standard uniform random numbers.
    nvars : int
        Number of variables (dimension), must be >= 2.
    theta : float
        Copula parameter, must be > 0.0.

    Returns
    -------
    np.ndarray
        A 1-D NumPy array of length nvars with joint uniform draws from
        the Clayton copula.

    Raises
    ------
    ValueError
        If theta is not a float or theta <= 0.0.

    Notes
    -----
    See SAS documentation for the Clayton copula construction.
    """
    if type(theta) != float:
        raise ValueError('"theta" must be a float')
    if theta <= 0.0:
        raise ValueError('"theta" must be greater than 0.0')

    def Ftilde(t):
        return (1.0 + t)**(-1.0 / theta)

    v = stats.gamma.ppf(next(draw), (1.0/theta))
    return np.array([Ftilde(-math.log(next(draw))/v) for i in range(nvars)])


def cgumbel(draw: Generator[float, None, None],
            nvars: int,
            theta: float
           ) -> np.ndarray:
    """
    Generate joint uniform draws from a Gumbel copula.

    Parameters
    ----------
    draw : Generator[float, None, None]
        A generator yielding independent standard uniform random numbers.
    nvars : int
        Number of variables (dimension), must be >= 2.
    theta : float
        Copula parameter, must be > 1.0.

    Returns
    -------
    np.ndarray
        A 1-D NumPy array of length nvars with joint uniform draws from
        the Gumbel copula.

    Raises
    ------
    ValueError
        If theta is not a float or theta <= 1.0.

    Notes
    -----
    See SAS documentation for the Gumbel copula construction.
    """
    if type(theta) != float:
        raise ValueError('"theta" must be a float')
    if theta <= 1.0:
        raise ValueError('"theta" must be greater than 1.0')

    def Ftilde(t):
        return math.exp(-(t**(1.0/theta)))

    gamma = math.cos(0.5 * math.pi / theta)**theta
    alpha = 1.0 / theta
    v = _skew_stable_draw(draw, alpha, 1.0, gamma, 0.0)
    return np.array([Ftilde(-math.log(next(draw))/v) for i in range(nvars)])

