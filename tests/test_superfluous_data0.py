import numpy as np
import xarray as xr
from scipy import stats
import funcsim as fs


def data0():
    # set up existing/historical data
    steps = [0, 1, 2]
    variables = ["b", "c"]
    a = np.array([[1.0, np.nan], [1.01, np.nan], [0.99, np.nan]])
    d0 = xr.DataArray(data=a, coords=(('steps', steps),
                                      ('variables', variables)))
    return d0


def trial(draw):
    # function to perform one trial.
    # simulate one std. norm variable, and one Bernoilli variable

    # independent uniform draws
    u1 = next(draw)
    u2 = next(draw)

    # inverse CDF transformations
    eps = stats.norm.ppf(u1)
    b = stats.bernoulli.ppf(u2, 0.35)

    # return dict with var names and values
    return {"eps": eps, "b": b}


def test_1():
    # superfluous data0
    out = fs.simulate(f=trial, ntrials=500, hist0=data0(), sampling='mc')
    meanB = float(out.sel(variables="b", steps=3).mean(dim="trials"))
    print(meanB)
    assert abs(meanB - 0.35) < 0.03

test_1()
