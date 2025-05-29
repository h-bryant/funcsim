import math
import numpy as np
from scipy import stats
from typing import Generator, Optional
import conversions


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
    return min(math.floor((M)*u), M)


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


def normal(draw: Generator[float, None, None],
           sigma: conversions.ArrayLike,
           mu: Optional[conversions.VectorLike] = None
          ) -> np.ndarray:    
    """
    Generate joint normal random draws.

    Given a covariance matrix `sigma`, this function generates a vector of
    joint normal random variables using the provided random number generator
    `draw`. If `mu` is provided, it is used as the mean vector; otherwise,
    the mean is assumed to be zero.

    Parameters
    ----------
    draw : generator
        A generator that yields independent standard uniform random numbers.
    sigma : array_like
        Covariance matrix (K x K) for the joint normal distribution.
    mu : vector_like, optional
        Mean vector (length K) for the joint normal distribution.
        If None, the mean is zero.

    Returns
    -------
    ndarray
        A 1-D NumPy array of length K containing a sample from the specified
        joint normal distribution.

    Notes
    -----
    The function transforms standard uniform draws into standard normal draws
    using the inverse CDF, then applies the Cholesky decomposition of the
    covariance matrix to induce the desired correlation structure.
    """
    A_prelim = conversions.alToArray(sigma)
    A = _makeA(_checkcov(A_prelim, "sigma"))
    K = len(A)  # number of variables
    prod = np.dot(A, stats.norm.ppf([next(draw) for i in range(K)]))
    if mu is None:
        return prod
    else:
        return mu + prod


def cgauss(draw: Generator[float, None, None],
           rho: conversions.ArrayLike
          ) -> np.ndarray:
    """
    Generate joint uniform draws from a Gaussian copula.

    Given a correlation matrix `rho`, this function generates a vector of joint
    uniform random variables (copula samples) using the Gaussian copula
    construction. The random number generator `draw` should yield independent
    standard uniform random numbers.

    Parameters
    ----------
    draw : generator
        A generator that yields independent standard uniform random numbers.
    rho : array_like
        Correlation matrix (K x K) for the Gaussian copula.

    Returns
    -------
    ndarray
        A 1-D NumPy array of length K containing joint uniform draws from the
        Gaussian copula.

    Notes
    -----
    The function first generates a joint normal random vector with correlation
    structure `rho`, then applies the standard normal CDF to each component to
    obtain standard uniform draws reflecting the Gaussian copula dependence
    structure.
    """
    rho_prelim = conversions.alToArray(rho)
    return stats.norm.cdf(normal(draw, _checkcov(rho_prelim, "rho")))


def cstudent(draw: Generator[float, None, None],
             rho: conversions.ArrayLike,
             nu: float
            ) -> np.ndarray:
    """
    Generate joint uniform draws from a Student's t copula.

    Given a correlation matrix `rho` and degrees of freedom `nu`, this
    function generates a vector of joint uniform random variables (copula
    samples) using the Student's t copula construction. The random number
    generator `draw` should yield independent standard uniform random
    numbers.

    Parameters
    ----------
    draw : generator
        A generator that yields independent standard uniform random numbers.
    rho : array_like
        Correlation matrix (K x K) for the Student's t copula.
    nu : float
        Degrees of freedom for the Student's t distribution.

    Returns
    -------
    ndarray
        A 1-D NumPy array of length K containing joint uniform draws from
        the Student's t copula.

    Notes
    -----
    The function generates a joint normal random vector with correlation
    structure `rho`, scales it by a chi-squared random variable, and then
    applies the Student's t CDF to each component to obtain uniform
    marginals.
    """
    x = normal(draw, rho)
    chi2 = stats.chi2.ppf(next(draw), df=nu)
    mult = (nu / chi2)**0.5
    return stats.t.cdf(mult * x, df=nu)


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


class MvKde():

    def __init__(self,
                 data: conversions.ArrayLike,
                 bw: str = 'scott'
                ) -> None:
        """
        Initialize a multivariate KDE object.

        Parameters
        ----------
        data : array_like
            Input data array of shape (n_samples, n_features).
        bw : str, optional
            Bandwidth selection method, 'scott' or 'silverman'.
            Default is 'scott'.

        Returns
        -------
        None
        """
        self._data = conversions.alToArray(data)
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

    def sample(self,
               draw: Generator[float, None, None]
              ) -> np.ndarray:
        """
        Generate a random sample from the multivariate KDE.

        Parameters
        ----------
        draw : Generator[float, None, None]
            A generator yielding independent standard uniform random numbers.

        Returns
        -------
        np.ndarray
            A 1-D NumPy array representing a sample from the KDE.
        """
        # hist obs about which we will sample
        m = _rand_int(next(draw), self._M)

        # means for this sample
        mu = self._data.iloc[m]

        return normal(draw, self._bw, mu)


def kdemv(data: conversions.ArrayLike,
          bw: str = 'scott'
         ) -> MvKde:
    """
    Create a multivariate KDE object.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    bw : str, optional
        Bandwidth selection method, 'scott' or 'silverman'.
        Default is 'scott'.

    Returns
    -------
    MvKde
        An instance of the MvKde class.
    """
    return MvKde(data, bw)