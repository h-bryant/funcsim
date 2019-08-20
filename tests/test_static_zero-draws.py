import funcsim as fs


def trial(draw):
    # function to perform one trial.
    # no stochastic draws

    # inverse CDF transformations
    eps = 13.0
    b = 2.0 * eps

    # return dict with var names and values
    return {"eps": eps, "b": b}


def test_0():
    out = fs.static(trial=trial, trials=500)
    means = out.mean(dim='trials')
    meanEps = float(means.loc['eps'])
    meanB = float(means.loc['b'])
    assert abs(meanEps - 13.0) < 13.0
    assert abs(meanB - 26.0) < 0.01
