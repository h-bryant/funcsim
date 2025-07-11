import math
import numpy as np
from scipy import stats
import xarray as xr
import funcsim as fs


def gbm(s0, dt, mu, sig, eps):
    # update a variable via a standard geometric Brownian motion
    return s0 * math.exp((mu - 0.5 * sig**2) * dt + eps * sig * dt ** 0.5)


def step(draw, hist):
    # take one step through time

    # value of p in previous period
    pLag1 = hist.recall("p", lag=1)

    # uniform draw --> standard normal draw
    u = next(draw)
    eps = stats.norm.ppf(u)

    # update all intermediate variables
    pNew = gbm(s0=pLag1, dt=1.0 / 12.0, mu=0.05, sig=0.10, eps=eps)
    cNew = max(0.0, pNew - 1.0)

    # return new values for this step
    return {"p": pNew, "c": cNew, "unused": np.nan}


def data0():
    # set up existing/historical data
    steps = [0, 1, 2]
    variables = ["p", "c"]
    a = np.array([[1.0, np.nan], [1.01, np.nan], [0.99, np.nan]])
    d0 = xr.DataArray(data=a, coords=(('steps', steps),
                                      ('variables', variables)))
    return d0


def test_00():  # basic
    # vars in data0 and f match
    out = fs.simulate(f=step, hist0=data0(), nsteps=10, ntrials=500)
    assert type(out) == xr.DataArray
    value = float(out.sel(steps=12, variables="c").mean())
    assert abs(value - 0.05) < 0.01

