import math
import numpy as np
from scipy import stats
import xarray as xr
import funcsim as fs


def gbm(s0, dt, mu, sig, eps):
    # update a variable via a standard geometric Brownian motion
    return s0 * math.exp((mu - 0.5 * sig**2) * dt + eps * sig * dt ** 0.5)


def step(data, draw):
    # take one step through time

    # value of p in previous period
    pLag1 = fs.recall(data, "p", lag=1)

    # uniform draw --> standard normal draw
    eps = next(draw)

    # update all intermediate variables
    pNew = gbm(s0=pLag1, dt=1.0 / 12.0, mu=0.05, sig=0.10, eps=eps)
    cNew = max(0.0, pNew - 1.0)

    # return updated price history
    dataNew = fs.chron(data, {"p": pNew, "c": cNew})
    return dataNew


def data0():
    # set up existing/historical data
    steps = [0, 1, 2]
    variables = ["p", "c"]
    a = np.array([[1.0, np.nan], [1.01, np.nan], [0.99, np.nan]])
    d0 = xr.DataArray(data=a, coords=(('steps', steps),
                                      ('variables', variables)))
    return d0


def test_0():  # basic
    out = fs.recdyn(step=step, data0=data0(), steps=10, trials=500,
                    stdnorm=True)
    assert type(out) == xr.DataArray
    print(out)
    print(out[:, 0, 10].mean())
    assert abs(float(out[:, 0, 10].mean()) - 1.0234) < 0.01


def test_1():  # use multi
    out = fs.recdyn(step=step, data0=data0(), steps=10, trials=500, multi=True,
                    stdnorm=True)
    assert type(out) == xr.DataArray
    assert abs(float(out[:, 0, 10].mean()) - 1.0234) < 0.01


def test_2():  # alternative seed
    out = fs.recdyn(step=step, data0=data0(), steps=10, trials=500, seed=123,
                    stdnorm=True)
    assert type(out) == xr.DataArray
    assert abs(float(out[:, 0, 10].mean()) - 1.0234) < 0.01


def test_3():  # many steps (check that recursion does not bust stack)
    out = fs.recdyn(step=step, data0=data0(), steps=2000, trials=10,
                    stdnorm=True)
    assert type(out) == xr.DataArray
