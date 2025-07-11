import funcsim as fs
import numpy as np
import pandas as pd
from scipy import stats
from core import simulate


def test_MvNorm():
    data = np.random.normal(size=(1000, 2))
    mvn = fs.MvNorm(data)

    def f(ugen):
        samp = mvn.draw(ugen)
        assert isinstance(samp, pd.Series)
        return {"samp0": samp.iloc[0], "samp1": samp.iloc[1]}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)

    sampcorr = np.corrcoef(sampl, rowvar=False)[0, 1]
    assert abs(sampcorr) < 0.10  # correlation should be close to zero


def test_MvNorm_2():
    data = np.random.normal(size=(1000, 2))
    mvn = fs.MvNorm(data)

    def f(ugen):
        draw = mvn.draw(ugen)
        return {"samp0": draw.v0, "samp1": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)

    sampcorr = np.corrcoef(sampl, rowvar=False)[0, 1]
    assert abs(sampcorr) < 0.10  # correlation should be close to zero

def test_MvNorm_3():
    data = np.random.normal(size=(1000, 2))
    dataPd = pd.DataFrame(data, columns=["rain", "temp"])
    mvn = fs.MvNorm(dataPd)

    def f(ugen):
        draw = mvn.draw(ugen)
        return {"rain": draw.rain, "temp": draw.temp}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)

    sampcorr = np.corrcoef(sampl, rowvar=False)[0, 1]
    assert abs(sampcorr) < 0.10  # correlation should be close to zero


def test_MvNorm_4():
    data = np.random.random(size=(1000, 2))
    dataPd = pd.DataFrame(data, columns=["rain", "temp"])
    mvn = fs.MvNorm(dataPd)

    def f(ugen):
        draw = mvn.draw(ugen)
        return {"rain": draw.rain, "temp": draw.temp}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)

    sampcorr = np.corrcoef(sampl, rowvar=False)[0, 1]
    assert abs(sampcorr) < 0.10  # correlation should be close to zero
