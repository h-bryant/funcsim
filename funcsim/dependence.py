import numpy as np
from scipy import stats


def _memoize(f):
    # custom memoization decorator for _makeA, which works around the
    # fact that a numpy array is not inherently hashable
    class memodict(dict):
        def __init__(self, f):
            self.f = f

        def __call__(self, sig):
            key = hash(sig.tostring())
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
    # draws from a Gaussian copula.  "rho" should be a correlation matrix
    # as a numpy.array
    return stats.norm.cdf(normal(draw, _checkcov(rho, "rho")))


def cstudent(draw, sigma, nu):
    # draws from a Student's t copula.  "rho" should be a correlation matrix
    # as a numpy.array.  "sigma" is a covariance-like matrix; "nu" is the
    # degrees-of-freedom parameter.
    x = normal(draw, sigma)
    chi2 = stats.chi2.ppf(next(draw), df=nu)
    mult = (nu / chi2)**0.5
    return stats.t.cdf(mult * x, df=nu)
