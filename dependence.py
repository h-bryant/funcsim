import funcsim as fs
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


def test_normal_0():
    # specify correlation matrix
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])

    # set up function to perform a single trial
    def f(draw):
        eps = normal(rho, draw)  # vector of two correlated stand. normal draws
        return {"eps0": eps[0], "eps1": eps[1]}

    # perform simulations
    out = fs.crossec(trial=f, trials=2000)
    sampcorr = np.corrcoef(out, rowvar=False)[0, 1]
    assert abs(sampcorr - 0.5) < 0.05


test_normal_0()
