import numpy as np
import scipy.stats as stats
import funcsim as fs


def test_0():
    vh  = stats.norm().rvs(size=100)
    result = fs.compare(vh, False, False)


def test_1():
    vh  = stats.uniform().rvs(size=100)
    result = fs.compare(vh, True, True)
