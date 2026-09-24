import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Generator, Optional, Sequence, Tuple
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


def _cop_names(K: int,
               names: Optional[Sequence[str]],
               source: Optional[pd.Index] = None,
               ) -> pd.Index:
    # variable names for a copula built from parameters rather than data:
    # the explicit `names` if given, else the labels carried by the
    # parameter object (`source`, e.g. a DataFrame's columns), else v0, v1, ...
    if names is not None:
        nms = pd.Index(list(names))
    elif source is not None:
        nms = pd.Index(source)
    else:
        nms = pd.Index([f"v{k}" for k in range(K)])
    if len(nms) != K:
        raise ValueError(f"names must have length {K}; got {len(nms)}")
    return nms


def _corr_from_params(rho: conversions.ArrayLike) -> np.ndarray:
    # validate a user-supplied correlation matrix for an elliptical copula:
    # square, symmetric, unit diagonal.  If it is not positive definite the
    # nearest positive definite matrix (Higham, 1988) is substituted and
    # renormalized to a unit diagonal
    if isinstance(rho, (list, tuple)):
        R = np.asarray(rho, dtype=float)
    else:
        R = np.asarray(conversions.alToArray(rho), dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"rho must be a square matrix; got shape {R.shape}")
    if not np.allclose(R, R.T):
        raise ValueError("rho must be symmetric")
    if not np.allclose(np.diag(R), 1.0):
        raise ValueError("rho must have ones on its diagonal")
    Rpd = nearby.nearestpd(0.5 * (R + R.T))
    s = np.sqrt(np.diag(Rpd))
    Rn = Rpd / np.outer(s, s)
    np.fill_diagonal(Rn, 1.0)
    Rn = 0.5 * (Rn + Rn.T)
    # Higham's projection of an indefinite matrix has a zero eigenvalue at
    # machine precision, and rescaling can push it slightly negative, so the
    # Cholesky factorization used for draws may fail.  Shrink toward the
    # identity by the smallest power of ten that restores positive
    # definiteness; the convex combination keeps the unit diagonal exactly
    Id = np.eye(R.shape[0])
    candidates = ((1.0 - eps) * Rn + eps * Id
                  for eps in (0.0, *(10.0 ** p for p in range(-14, -3))))
    Rout = next(filter(nearby.is_positive_definite, candidates), None)
    if Rout is None:
        raise ValueError("rho could not be repaired to a positive definite "
                         "correlation matrix")
    return Rout


def _param_labels(rho: conversions.ArrayLike) -> Optional[pd.Index]:
    # column labels carried by a parameter matrix, if any
    if isinstance(rho, pd.DataFrame):
        return rho.columns
    return None


def _corr_from_tau(tau: conversions.ArrayLike) -> np.ndarray:
    # correlation matrix of an elliptical copula from Kendall's tau: a scalar
    # (two variables) or a K-by-K matrix of pairwise taus, inverted entrywise
    # through rho = sin(pi tau / 2) (Lindskog, McNeil, and Schmock, 2003).
    # The diagonal of a matrix is ignored.  The result is a symmetric matrix
    # with a unit diagonal; positive definiteness is left to
    # _corr_from_params, which repairs it with a warning
    if isinstance(tau, (list, tuple)):
        T = np.asarray(tau, dtype=float)
    elif np.ndim(tau) == 0:
        T = np.asarray(float(tau))
    else:
        T = np.asarray(conversions.alToArray(tau), dtype=float)
    if T.ndim == 0:
        t = float(T)
        T = np.array([[1.0, t], [t, 1.0]])
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"tau must be a scalar or a square matrix; got "
                         f"shape {T.shape}")
    if not np.allclose(T, T.T, equal_nan=False):
        raise ValueError("tau must be symmetric")
    off = ~np.eye(T.shape[0], dtype=bool)
    if not np.all(np.abs(T[off]) < 1.0):
        raise ValueError("every pairwise tau must lie strictly between -1 "
                         "and 1")
    R = np.sin(0.5 * np.pi * T)
    np.fill_diagonal(R, 1.0)
    return R


def _positive_scalar(x: float, what: str, minimum: float = 0.0,
                     allow_inf: bool = False) -> float:
    # validate a scalar copula parameter that must exceed `minimum`; with
    # `allow_inf`, +inf is accepted as well (e.g., the Gaussian limit of the
    # Student's t degrees of freedom)
    xf = float(x)
    ok = xf > minimum and (math.isfinite(xf) or (allow_inf and xf > 0.0))
    if not ok:
        kind = "a number" if allow_inf else "a finite number"
        raise ValueError(f"{what} must be {kind} greater than {minimum}; "
                         f"got {x}")
    return xf


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
    A multivariate normal distribution object.  Constructed from data, a
    vector of means and a covariance matrix are computed from the input.
    Constructed with :meth:`from_params`, the mean vector and covariance
    matrix are supplied directly.  Either way, if the covariance matrix is
    not positive definite, the Higham method is used to calculate the
    nearest positive definite matrix.

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
                       f"normally distributed. (Shapiro-Wilk p-value"
                       f"={swp:.3f})")
                warnings.warn(msg, UserWarning)

    @classmethod
    def from_params(cls,
                    mu: conversions.VectorLike,
                    sigma: conversions.ArrayLike,
                    names: Optional[Sequence[str]] = None,
                    ) -> "MvNorm":
        """
        Create a multivariate normal distribution object from a mean vector
        and a covariance matrix, rather than from data.

        Parameters
        ----------
        mu : VectorLike
            Mean vector of length K.
        sigma : ArrayLike
            K-by-K covariance matrix (a nested list is also accepted).  It
            must be symmetric.  If it is not positive definite, the nearest
            positive definite matrix (Higham, 1988) is substituted with a
            warning.
        names : sequence of str, optional
            Variable names, length K.  If omitted, the index of `mu` (if it
            is a pandas Series) or the column labels of `sigma` (if it is a
            pandas DataFrame) are used; otherwise the variables are named
            'v0', 'v1', ....

        Returns
        -------
        MvNorm
            A distribution object whose :meth:`draw` method returns joint
            draws with the given means and covariances.

        Notes
        -----
        No normality check is performed, since there are no data to check.

        Examples
        --------
        >>> dist = fs.MvNorm.from_params([3.0, 4.0], [[1.0, 0.5], [0.5, 1.0]])
        """
        muA = np.asarray(conversions.vlToArray(mu), dtype=float)
        if isinstance(sigma, (list, tuple)):
            sigmaA = np.asarray(sigma, dtype=float)
        else:
            sigmaA = np.asarray(conversions.alToArray(sigma), dtype=float)
        K = muA.shape[0]
        if sigmaA.shape != (K, K):
            raise ValueError(f"sigma must be {K}-by-{K} to match mu; "
                             f"got shape {sigmaA.shape}")
        if not np.allclose(sigmaA, sigmaA.T):
            raise ValueError("sigma must be symmetric")
        if names is not None:
            nms = pd.Index(list(names))
        elif isinstance(mu, pd.Series):
            nms = pd.Index(mu.index)
        elif isinstance(sigma, pd.DataFrame):
            nms = pd.Index(sigma.columns)
        else:
            nms = pd.Index([f"v{k}" for k in range(K)])
        if len(nms) != K:
            raise ValueError(f"names must have length {K}; got {len(nms)}")

        obj = cls.__new__(cls)
        obj._data = None
        obj._names = nms
        (obj._M, obj._K) = (0, K)
        obj._mu = muA
        obj._sigma = nearby.nearestpd(sigmaA)
        obj._A = np.linalg.cholesky(obj._sigma)
        return obj

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
            provided (as column labels of the input data, or through the
            `names` argument of :meth:`from_params`), the variables are
            named 'v0', 'v1', ..., in column order.
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

    @classmethod
    def from_params(cls,
                    rho: conversions.ArrayLike,
                    names: Optional[Sequence[str]] = None,
                    ) -> "CopulaGauss":
        """
        Create a Gaussian copula object from a correlation matrix, rather
        than from pseudo-observations.

        Parameters
        ----------
        rho : ArrayLike
            K-by-K correlation matrix (a nested list is also accepted).  It
            must be symmetric with ones on its diagonal.  If it is not
            positive definite, the nearest positive definite matrix
            (Higham, 1988) is substituted and rescaled to a unit diagonal.
        names : sequence of str, optional
            Variable names, length K.  If omitted, the column labels of
            `rho` (if it is a pandas DataFrame) are used; otherwise the
            variables are named 'v0', 'v1', ....

        Returns
        -------
        CopulaGauss
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Gaussian dependence.

        Examples
        --------
        >>> cop = fs.CopulaGauss.from_params([[1.0, 0.9], [0.9, 1.0]])
        """
        R = _corr_from_params(rho)
        K = R.shape[0]
        obj = cls.__new__(cls)
        obj._data = None
        obj._z = None
        obj._names = _cop_names(K, names, _param_labels(rho))
        (obj._M, obj._K) = (0, K)
        obj._rho = R
        obj._A = np.linalg.cholesky(obj._rho)
        return obj

    @classmethod
    def from_tau(cls,
                 tau: conversions.ArrayLike,
                 names: Optional[Sequence[str]] = None,
                 ) -> "CopulaGauss":
        """
        Create a Gaussian copula object from Kendall's tau, rather than from
        pseudo-observations or a correlation matrix.

        Parameters
        ----------
        tau : float or ArrayLike
            Kendall's tau: a scalar for two variables, or a K-by-K matrix of
            pairwise values (a nested list is also accepted) whose diagonal
            is ignored.  Every pairwise value must lie strictly between -1
            and 1.  Each is inverted through ``rho = sin(pi * tau / 2)``; a
            resulting matrix that is not positive definite is repaired as in
            :meth:`from_params`, with a warning.
        names : sequence of str, optional
            Variable names, length K.  If omitted, the column labels of
            `tau` (if it is a pandas DataFrame) are used; otherwise the
            variables are named 'v0', 'v1', ....

        Returns
        -------
        CopulaGauss
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Kendall's tau.

        Examples
        --------
        >>> cop = fs.CopulaGauss.from_tau(0.6, names=["yield", "price"])
        >>> cop.rho.iloc[0, 1]    # sin(0.3 pi) = 0.809
        """
        R = _corr_from_tau(tau)
        nms = _cop_names(R.shape[0], names, _param_labels(tau))
        return cls.from_params(R, names=nms)

    @property
    def rho(self) -> pd.DataFrame:
        """
        The copula's correlation matrix parameter, as a DataFrame whose
        index and columns are the variable names.
        """
        return pd.DataFrame(self._rho.copy(), index=self._names,
                            columns=self._names)

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

    The degrees of freedom are searched over [2, 200].  If the profile
    likelihood is maximized at the ceiling of that range, the data do not
    distinguish the fitted copula from its Gaussian limit, and ``nu`` is
    reported as ``inf`` rather than as a number near 200: the object then
    behaves exactly as :class:`CopulaGauss` with the same ``rho`` (while
    still consuming K + 1 draws).  :attr:`loglik_gain` reports how much
    log-likelihood the fitted ``nu`` adds over that Gaussian limit, which is
    small whenever ``nu`` is identified only as "large".
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
        (self._rho, self._nu, self._loglik_gain) = \
            copfit.fit_student(self._data)

        # cholesky decomposition of the correlation matrix, for draws
        self._A = np.linalg.cholesky(self._rho)

    @classmethod
    def from_params(cls,
                    rho: conversions.ArrayLike,
                    nu: float,
                    names: Optional[Sequence[str]] = None,
                    ) -> "CopulaStudent":
        """
        Create a Student's t copula object from a correlation matrix and a
        degrees-of-freedom parameter, rather than from pseudo-observations.

        Parameters
        ----------
        rho : ArrayLike
            K-by-K correlation matrix (a nested list is also accepted).  It
            must be symmetric with ones on its diagonal.  If it is not
            positive definite, the nearest positive definite matrix
            (Higham, 1988) is substituted and rescaled to a unit diagonal.
        nu : float
            Degrees of freedom, greater than zero.  As `nu` grows the copula
            approaches the Gaussian copula with the same `rho`; ``math.inf``
            is accepted and gives that Gaussian limit exactly.
        names : sequence of str, optional
            Variable names, length K.  If omitted, the column labels of
            `rho` (if it is a pandas DataFrame) are used; otherwise the
            variables are named 'v0', 'v1', ....

        Returns
        -------
        CopulaStudent
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Student's t dependence.

        Examples
        --------
        >>> cop = fs.CopulaStudent.from_params([[1.0, 0.9], [0.9, 1.0]], nu=4)
        """
        R = _corr_from_params(rho)
        K = R.shape[0]
        obj = cls.__new__(cls)
        obj._data = None
        obj._names = _cop_names(K, names, _param_labels(rho))
        (obj._M, obj._K) = (0, K)
        obj._rho = R
        obj._nu = _positive_scalar(nu, "nu", allow_inf=True)
        obj._loglik_gain = math.nan
        obj._A = np.linalg.cholesky(obj._rho)
        return obj

    @classmethod
    def from_tau(cls,
                 tau: conversions.ArrayLike,
                 nu: float,
                 names: Optional[Sequence[str]] = None,
                 ) -> "CopulaStudent":
        """
        Create a Student's t copula object from Kendall's tau and a
        degrees-of-freedom parameter, rather than from pseudo-observations
        or a correlation matrix.

        Parameters
        ----------
        tau : float or ArrayLike
            Kendall's tau: a scalar for two variables, or a K-by-K matrix of
            pairwise values (a nested list is also accepted) whose diagonal
            is ignored.  Every pairwise value must lie strictly between -1
            and 1.  Each is inverted through ``rho = sin(pi * tau / 2)``,
            which holds for every elliptical copula and so does not involve
            `nu`; a resulting matrix that is not positive definite is
            repaired as in :meth:`from_params`, with a warning.
        nu : float
            Degrees of freedom, greater than zero (``math.inf`` gives the
            Gaussian limit), as in :meth:`from_params`.
        names : sequence of str, optional
            Variable names, length K.  If omitted, the column labels of
            `tau` (if it is a pandas DataFrame) are used; otherwise the
            variables are named 'v0', 'v1', ....

        Returns
        -------
        CopulaStudent
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Kendall's tau and tail behavior.

        Examples
        --------
        >>> cop = fs.CopulaStudent.from_tau(0.6, nu=4.0)
        """
        R = _corr_from_tau(tau)
        nms = _cop_names(R.shape[0], names, _param_labels(tau))
        return cls.from_params(R, nu, names=nms)

    @property
    def rho(self) -> pd.DataFrame:
        """
        The copula's correlation matrix parameter, as a DataFrame whose
        index and columns are the variable names.
        """
        return pd.DataFrame(self._rho.copy(), index=self._names,
                            columns=self._names)

    @property
    def nu(self) -> float:
        """
        The copula's degrees-of-freedom parameter.  ``inf`` means the
        Gaussian limit: either the fit reached the ceiling of the search
        range (see the class notes) or ``math.inf`` was passed to
        :meth:`from_params`.
        """
        return float(self._nu)

    @property
    def loglik_gain(self) -> float:
        """
        Log-likelihood gain of the fitted degrees of freedom over the
        Gaussian limit (``nu = inf``) at the same correlation matrix, i.e.,
        the copula log-likelihood at the fitted ``nu`` minus the Gaussian
        copula log-likelihood at ``rho``.  Zero when ``nu`` is ``inf``, and
        small whenever the data pin ``nu`` down only as "large".  ``nan``
        for an object built with :meth:`from_params`, which has no data.
        """
        return float(self._loglik_gain)

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
        uchi = next(ugen)
        if math.isinf(self._nu):
            # Gaussian limit: the mixing variable is identically one, but
            # its draw is still consumed so the count per call is fixed
            return pd.Series(stats.norm.cdf(z), index=self._names)
        chi2 = stats.chi2.ppf(uchi, df=self._nu)
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

    @classmethod
    def from_params(cls,
                    theta: float,
                    K: int = 2,
                    names: Optional[Sequence[str]] = None,
                    ) -> "CopulaClayton":
        """
        Create a Clayton copula object from its dependence parameter,
        rather than from pseudo-observations.

        Parameters
        ----------
        theta : float
            Dependence parameter, greater than zero.  Kendall's tau is theta / (theta + 2).
        K : int, optional
            Number of variables.  Default is 2.  Ignored when `names` is
            given, in which case K is the number of names.
        names : sequence of str, optional
            Variable names.  If omitted, the variables are named 'v0',
            'v1', ....

        Returns
        -------
        CopulaClayton
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Clayton dependence.

        Examples
        --------
        >>> cop = fs.CopulaClayton.from_params(theta=2.0)
        """
        th = _positive_scalar(theta, "theta", 0.0)
        k = len(names) if names is not None else int(K)
        if k < 2:
            raise ValueError("a copula needs at least two variables")
        obj = cls.__new__(cls)
        obj._data = None
        obj._names = _cop_names(k, names)
        (obj._M, obj._K) = (0, k)
        obj._theta = th
        return obj

    @classmethod
    def from_tau(cls,
                 tau: float,
                 K: int = 2,
                 names: Optional[Sequence[str]] = None,
                 ) -> "CopulaClayton":
        """
        Create a Clayton copula object from Kendall's tau, rather than from
        pseudo-observations or from theta.

        Parameters
        ----------
        tau : float
            Kendall's tau, strictly between 0 and 1 (this implementation
            represents positive dependence only).  Inverted through
            ``theta = 2 * tau / (1 - tau)``.
        K : int, optional
            Number of variables.  Default is 2.  Ignored when `names` is
            given, in which case K is the number of names.
        names : sequence of str, optional
            Variable names.  If omitted, the variables are named 'v0',
            'v1', ....

        Returns
        -------
        CopulaClayton
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Kendall's tau.

        Examples
        --------
        >>> cop = fs.CopulaClayton.from_tau(0.5)    # theta = 2.0
        """
        t = float(tau)
        if not (0.0 < t < 1.0):
            raise ValueError("Kendall's tau for a Clayton copula must lie "
                             "strictly between 0 and 1 (this implementation "
                             f"represents positive dependence only); got {tau}")
        return cls.from_params(2.0 * t / (1.0 - t), K=K, names=names)

    @property
    def theta(self) -> float:
        """The copula's dependence parameter."""
        return float(self._theta)

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

    @classmethod
    def from_params(cls,
                    theta: float,
                    K: int = 2,
                    names: Optional[Sequence[str]] = None,
                    ) -> "CopulaGumbel":
        """
        Create a Gumbel copula object from its dependence parameter,
        rather than from pseudo-observations.

        Parameters
        ----------
        theta : float
            Dependence parameter, greater than one (one is independence, which this sampler cannot represent).  Kendall's tau is 1 - 1 / theta.
        K : int, optional
            Number of variables.  Default is 2.  Ignored when `names` is
            given, in which case K is the number of names.
        names : sequence of str, optional
            Variable names.  If omitted, the variables are named 'v0',
            'v1', ....

        Returns
        -------
        CopulaGumbel
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Gumbel dependence.

        Examples
        --------
        >>> cop = fs.CopulaGumbel.from_params(theta=2.0)
        """
        th = _positive_scalar(theta, "theta", 1.0)
        k = len(names) if names is not None else int(K)
        if k < 2:
            raise ValueError("a copula needs at least two variables")
        obj = cls.__new__(cls)
        obj._data = None
        obj._names = _cop_names(k, names)
        (obj._M, obj._K) = (0, k)
        obj._theta = th
        return obj

    @classmethod
    def from_tau(cls,
                 tau: float,
                 K: int = 2,
                 names: Optional[Sequence[str]] = None,
                 ) -> "CopulaGumbel":
        """
        Create a Gumbel copula object from Kendall's tau, rather than from
        pseudo-observations or from theta.

        Parameters
        ----------
        tau : float
            Kendall's tau, strictly between 0 and 1 (this implementation
            represents positive dependence only).  Inverted through
            ``theta = 1 / (1 - tau)``.
        K : int, optional
            Number of variables.  Default is 2.  Ignored when `names` is
            given, in which case K is the number of names.
        names : sequence of str, optional
            Variable names.  If omitted, the variables are named 'v0',
            'v1', ....

        Returns
        -------
        CopulaGumbel
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Kendall's tau.

        Examples
        --------
        >>> cop = fs.CopulaGumbel.from_tau(0.5)    # theta = 2.0
        """
        t = float(tau)
        if not (0.0 < t < 1.0):
            raise ValueError("Kendall's tau for a Gumbel copula must lie "
                             "strictly between 0 and 1 (this implementation "
                             f"represents positive dependence only); got {tau}")
        return cls.from_params(1.0 / (1.0 - t), K=K, names=names)

    @property
    def theta(self) -> float:
        """The copula's dependence parameter."""
        return float(self._theta)

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

    @classmethod
    def from_params(cls,
                    theta: float,
                    K: int = 2,
                    names: Optional[Sequence[str]] = None,
                    ) -> "CopulaFrank":
        """
        Create a Frank copula object from its dependence parameter,
        rather than from pseudo-observations.

        Parameters
        ----------
        theta : float
            Dependence parameter, greater than zero.  Kendall's tau is 1 - 4 (1 - D_1(theta)) / theta, with D_1 the first Debye function.
        K : int, optional
            Number of variables.  Default is 2.  Ignored when `names` is
            given, in which case K is the number of names.
        names : sequence of str, optional
            Variable names.  If omitted, the variables are named 'v0',
            'v1', ....

        Returns
        -------
        CopulaFrank
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Frank dependence.

        Examples
        --------
        >>> cop = fs.CopulaFrank.from_params(theta=5.0)
        """
        th = _positive_scalar(theta, "theta", 0.0)
        k = len(names) if names is not None else int(K)
        if k < 2:
            raise ValueError("a copula needs at least two variables")
        obj = cls.__new__(cls)
        obj._data = None
        obj._names = _cop_names(k, names)
        (obj._M, obj._K) = (0, k)
        obj._theta = th
        return obj

    @classmethod
    def from_tau(cls,
                 tau: float,
                 K: int = 2,
                 names: Optional[Sequence[str]] = None,
                 ) -> "CopulaFrank":
        """
        Create a Frank copula object from Kendall's tau, rather than from
        pseudo-observations or from theta.

        Parameters
        ----------
        tau : float
            Kendall's tau, strictly between 0 and 1 (this implementation
            represents positive dependence only).  The relation
            ``tau = 1 - 4 (1 - D_1(theta)) / theta``, with ``D_1`` the first
            Debye function, has no closed-form inverse; it is solved
            numerically to full floating-point precision.
        K : int, optional
            Number of variables.  Default is 2.  Ignored when `names` is
            given, in which case K is the number of names.
        names : sequence of str, optional
            Variable names.  If omitted, the variables are named 'v0',
            'v1', ....

        Returns
        -------
        CopulaFrank
            A copula object whose :meth:`draw` method returns joint standard
            uniform draws with the given Kendall's tau.

        Examples
        --------
        >>> cop = fs.CopulaFrank.from_tau(0.6)    # theta = 7.930
        """
        t = float(tau)
        if not (0.0 < t < 1.0):
            raise ValueError("Kendall's tau for a Frank copula must lie "
                             "strictly between 0 and 1 (this implementation "
                             f"represents positive dependence only); got {tau}")
        return cls.from_params(copfit.frank_theta_from_tau(t), K=K,
                               names=names)

    @property
    def theta(self) -> float:
        """The copula's dependence parameter."""
        return float(self._theta)

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
