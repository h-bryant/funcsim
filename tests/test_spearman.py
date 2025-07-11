import numpy as np
import funcsim as fs


def test_nearestpd():
    K = 2
    M = 300
    vh  = np.random.normal(size=(M, K))
    corr = fs.spearman(vh)
    assert corr[0] < 0.1