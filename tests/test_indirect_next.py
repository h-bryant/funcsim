from scipy import stats
import funcsim as fs
import xarray as xr


def getu(draw):
    return next(draw)


def trial(draw):
    # function to perform one trial.
    # simulate one std. norm variable, and one Bernoilli variable

    # independent uniform draws
    u1 = getu(draw)
    u2 = next(draw)

    # inverse CDF transformations
    eps = stats.norm.ppf(u1)
    b = stats.bernoulli.ppf(u2, 0.35)

    # return dict with var names and values
    return {"eps": eps, "b": b}


def test_0():
    out = fs.static(trial=trial, trials=15)
    assert type(out) == xr.DataArray
