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
    if type(sigma) != np.ndarray:
        raise ValueError("sigma must be of type numpy.ndarray")
    if len(sigma.shape) != 2:
        raise ValueError("sigma must be two-dimensional")
    if not np.allclose(sigma, sigma.T):
        raise ValueError("sigma must be symmetrical")
    if not np.all(np.linalg.eigvals(sigma) > 0):
        raise ValueError("sigma must be positive definite")
    return np.linalg.cholesky(sigma)


def normal(sigma, draw):
    # return a function that performs joint normal draws
    A = _makeA(sigma)
    K = len(A)  # number of variables
    return np.dot(A, stats.norm.ppf([next(draw) for i in range(K)]))


def corru(sigma, draw):
    # return uniform draws implied by correlated standard normal draws
    return stats.norm.cdf(normal(sigma, draw))
