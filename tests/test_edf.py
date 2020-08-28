import numpy as np
import funcsim as fs


def test_edf():
    np.random.seed(45678)
    sample = np.random.rand(1000)
    edf = fs.makeedf(sample)
    assert abs(edf(0.5) - 0.489) < 0.01
