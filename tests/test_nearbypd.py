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
    # K = 5
    # M = 5
    # vh  = np.random.normal(size=(M, K))
    # vh = np.array([[1, -20], [-20, 1]])

    # symmetric matrix, but not positive definite 
    A = [[1.0, 0.99, 0.35], 
        [0.99,  1.0, 0.80],
        [0.35, 0.80,  1.0]]
    array = np.array(A)
    assert is_positive_definite(array) is False

    cov = fs.nearestpd(array)
    assert is_positive_definite(cov) is True, f"cov=\n{cov}"
