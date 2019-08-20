from scipy import stats
import funcsim as fs


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


def test_0():
    out = fs.static(trial=trial, trials=500)
    means = out.mean(dim='trials')
    meanEps = float(means.loc['eps'])
    meanB = float(means.loc['b'])
    assert abs(meanEps) < 0.01
    assert abs(meanB - 0.35) < 0.01
