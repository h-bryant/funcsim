import numpy as np
import funcsim as fs


def is_positive_definite(A):
    """Checks whether a matrix is positive definite"""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def test_nearestpd():
    K = 5
    M = 5
    vh  = np.random.normal(size=(M, K))
    cov = fs.nearestpd(vh)
    assert is_positive_definite(cov) is True