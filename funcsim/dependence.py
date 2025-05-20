import math
import numpy as np
from scipy import stats


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
    if type(cov) != np.ndarray:
        raise ValueError("%s must be of type numpy.ndarray" % name)
    if len(cov.shape) != 2:
        raise ValueError("%s must be two-dimensional" % name)
    if not np.allclose(cov, cov.T):
        raise ValueError("%s must be symmetrical" % name)
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError("%s must be positive definite" % name)
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


def normal(draw, sigma, mu=None):
    # perform joint normal draws.  "sigma" should be a covariance matrix as
    # a numpy.array
    A = _makeA(_checkcov(sigma, "sigma"))
    K = len(A)  # number of variables
    prod = np.dot(A, stats.norm.ppf([next(draw) for i in range(K)]))
    if mu is None:
        return prod
    else:
        return mu + prod


def cgauss(draw, rho):
    # joint u draws from a Gaussian copula.
    # "rho" should be a correlation matrix as a numpy.array
    return stats.norm.cdf(normal(draw, _checkcov(rho, "rho")))


def cstudent(draw, rho, nu):
    # joint u draws from a Student's t copula.
    # "rho" is a correlation  matrix
    # "nu" is the degrees-of-freedom parameter
    x = normal(draw, rho)
    chi2 = stats.chi2.ppf(next(draw), df=nu)
    mult = (nu / chi2)**0.5
    return stats.t.cdf(mult * x, df=nu)


def cclayton(draw, nvars, theta):
    # joint u draws from a clayton copula
    # "nvars" is an integer >= 2
    # "theta" is a float > 0.0
    #
    # see: https://support.sas.com/documentation/cdl/en/etsug/63939/HTML/default/viewer.htm#etsug_copula_sect017.htm

    if type(theta) != float:
        raise ValueError('"theta" must be a float')
    if theta <= 0.0:
        raise ValueError('"theta" must be greater than 0.0')

    def Ftilde(t):
        return (1.0 + t)**(-1.0 / theta)

    v = stats.gamma.ppf(next(draw), (1.0/theta))
    return np.array([Ftilde(-math.log(next(draw))/v) for i in range(nvars)])


def cgumbel(draw, nvars, theta):
    # joint u draws from a gumbel copula
    # "nvars" is an integer >= 2
    # "theta" is a float > 0.0
    #
    # see: https://support.sas.com/documentation/cdl/en/etsug/63939/HTML/default/viewer.htm#etsug_copula_sect017.htm

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

    def __init__(self, data, bw='scott'):
        # sample should be a numpy array

        self._data = data
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

    def sample(self, draw):
        # random draw from the KDE

        # hist obs about which we will sample
        m = _rand_int(next(draw), self._M)

        # means for this sample
        mu = self._data.iloc[m]

        return normal(draw, self._bw, mu)


def fitkdemv(data, bw='scott'):
    return MvKde(data, bw)
