import funcsim as fs
import numpy as np
from scipy import stats


def test_normal_0():
    # specify correlation matrix
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])

    # set up function to perform a single trial
    def f(draw):
        eps = fs.normal(draw, rho)  # vector of two correlated stand. normal draws
        return {"eps0": eps[0], "eps1": eps[1]}

    # perform simulations
    out = fs.static(trial=f, trials=2000)
    sampcorr = np.corrcoef(out, rowvar=False)[0, 1]
    assert abs(sampcorr - 0.5) < 0.05


def test_normal_1():
    # specify correlation matrix
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu = np.array([3.0, 4.0])

    # set up function to perform a single trial
    def f(draw):
        eps = fs.normal(draw, rho, mu)  # vector of two correlated stand. normal draws
        return {"eps0": eps[0], "eps1": eps[1]}

    # perform simulations
    out = fs.static(trial=f, trials=2000)
    sampcorr = np.corrcoef(out, rowvar=False)[0, 1]
    assert abs(sampcorr - 0.5) < 0.05
