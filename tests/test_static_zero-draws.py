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
    out = fs.simulate(f=trial, ntrials=500)
    meanEps = float(out.sel(steps=0, variables="eps").mean())
    meanB = float(out.sel(steps=0, variables="b").mean())
    assert abs(meanEps - 13.0) < 13.0
    assert abs(meanB - 26.0) < 0.01
