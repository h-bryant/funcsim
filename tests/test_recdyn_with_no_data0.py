import math
import numpy as np
from scipy import stats
import xarray as xr
import funcsim as fs


def gbm(s0, dt, mu, sig, eps):
    # update a variable via a standard geometric Brownian motion
    return s0 * math.exp((mu - 0.5 * sig**2) * dt + eps * sig * dt ** 0.5)


def step(draw, data):
    # take one step through time

    # value of p in previous period
    # pLag1 = data.recall("p", lag=1)a
    pLag1 = 0.99

    # uniform draw --> standard normal draw
    u = next(draw)
    eps = stats.norm.ppf(u)

    # update all intermediate variables
    pNew = gbm(s0=pLag1, dt=1.0 / 12.0, mu=0.05, sig=0.10, eps=eps)
    cNew = max(0.0, pNew - 1.0)

    # return new values for this step
    return {"p": pNew, "c": cNew, "unused": np.nan}




def test_01():  # basic
    # extra var returned by f (compared to contents of hist0)
    out = fs.simulate(f=step, nsteps=10, ntrials=500)
