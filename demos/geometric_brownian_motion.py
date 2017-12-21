#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../"))

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
    u = next(draw)
    eps = stats.norm.ppf(u)

    # update all intermediate variables
    pNew = gbm(s0=pLag1, dt=1.0 / 12.0, mu=0.01, sig=0.10, eps=eps)
    cNew = max(0.0, pNew - 1.0)

    # return updated price history
    dataNew = fs.chron(data, {"p": pNew, "c": cNew})
    return dataNew


# set up existing/historical data
steps = [0, 1, 2]
variables = ["p", "c"]
a = np.array([[1.0, np.nan], [1.01, np.nan], [0.99, np.nan]])
data0 = \
    xr.DataArray(data=a, coords=(('steps', steps), ('variables', variables)))

# simulate
out = fs.recdyn(step=step, data0=data0, steps=10, trials=20, multi=True)
print(out)
