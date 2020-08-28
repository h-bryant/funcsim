import numpy as np
import funcsim as fs


def test_kde():
    sample = np.random.rand(1000)
    kde = fs.fitkde(sample)
    assert abs(kde.cdf(0.5) - 0.5) < 0.01
    assert abs(kde.ppf(0.5) - 0.5) < 0.01