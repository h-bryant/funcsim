import funcsim as fs
import numpy as np
import pandas as pd
from scipy import stats
from core import simulate


def test_normal_0():
    # specify correlation matrix
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])

    # set up function to perform a single trial
    def f(draw):
        eps = fs.normal(draw, rho)  # vector of two correlated stand. normal draws
        return {"eps0": eps[0], "eps1": eps[1]}

    # perform simulations
    out = fs.simulate(f=f, ntrials=2000).sel(steps=0)
    sampcorr = np.corrcoef(out, rowvar=False)[0, 1]
    assert abs(sampcorr - 0.5) < 0.05


def test_normal_1():
    # specify correlation matrix
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu = np.array([3.0, 4.0])

    # set up function to perform a single trial
    def f(draw):
        eps = fs.normal(draw, rho, mu)  # vector of two correlated stand. normal draws
        return {"eps0": eps[0], "eps1": eps[1]}

    # perform simulations
    out = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    sampcorr = np.corrcoef(out, rowvar=False)[0, 1]
    assert abs(sampcorr - 0.5) < 0.05


def test_MvNorm():
    data = np.random.normal(size=(1000, 2))
    mvn = fs.MvNorm(data)

    def f(ugen):
        samp = mvn.draw(ugen)
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
