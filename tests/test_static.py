from scipy import stats
import funcsim as fs


def trial(ugen):
    # function to perform one trial.
    # simulate one std. norm variable, and one Bernoilli variable

    # independent uniform draws
    u1 = next(ugen)
    u2 = next(ugen)

    # inverse CDF transformations
    eps = stats.norm.ppf(u1)
    b = stats.bernoulli.ppf(u2, 0.35)

    # return dict with var names and values
    return {"eps": eps, "b": b}


def test_0():
    out = fs.simulate(f=trial, ntrials=500)
    meanEps = float(out.sel(steps=0, variables="eps").mean())
    meanB = float(out.sel(steps=0, variables="b").mean())
    assert abs(meanEps) < 0.01
    assert abs(meanB - 0.35) < 0.01


def test_1():
    out = fs.simulate(f=trial, ntrials=500, sampling='mc')
    meanEps = float(out.sel(steps=0, variables="eps").mean())
    meanB = float(out.sel(steps=0, variables="b").mean())
    assert abs(meanEps) < 0.03
    assert abs(meanB - 0.35) < 0.03
