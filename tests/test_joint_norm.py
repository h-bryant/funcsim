import funcsim as fs
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from funcsim.core import simulate


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


def test_MvNorm_from_params():
    mu = np.array([3.0, 4.0])
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    mvn = fs.MvNorm.from_params(mu, sigma)

    def f(ugen):
        draw = mvn.draw(ugen)
        assert isinstance(draw, pd.Series)
        assert list(draw.index) == ["v0", "v1"]
        return {"v0": draw.v0, "v1": draw.v1}

    sampl = fs.simulate(f=f, ntrials=4000).sel(steps=0).values
    assert sampl.shape == (4000, 2)
    assert np.allclose(sampl.mean(axis=0), mu, atol=0.1)
    assert np.allclose(sampl.std(axis=0), 1.0, atol=0.1)
    assert abs(np.corrcoef(sampl, rowvar=False)[0, 1] - 0.5) < 0.05


def test_MvNorm_from_params_names():
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu = pd.Series([3.0, 4.0], index=["rain", "temp"])
    assert list(fs.MvNorm.from_params(mu, sigma)._names) == ["rain", "temp"]
    df = pd.DataFrame(sigma, index=["a", "b"], columns=["a", "b"])
    assert list(fs.MvNorm.from_params([0.0, 0.0], df)._names) == ["a", "b"]
    explicit = fs.MvNorm.from_params([0.0, 0.0], sigma, names=["x", "y"])
    assert list(explicit._names) == ["x", "y"]
    nested = fs.MvNorm.from_params([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]])
    assert nested._sigma.shape == (2, 2)
    with pytest.raises(ValueError):
        fs.MvNorm.from_params([0.0, 0.0], sigma, names=["x"])
    with pytest.raises(ValueError):
        fs.MvNorm.from_params([0.0, 0.0, 0.0], sigma)
    with pytest.raises(ValueError):
        fs.MvNorm.from_params([0.0, 0.0], [[1.0, 0.5], [0.2, 1.0]])


def test_MvNorm_from_params_indefinite():
    # an internally inconsistent correlation matrix: one negative eigenvalue
    rho = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.5], [0.9, -0.5, 1.0]])
    assert np.linalg.eigvalsh(rho).min() < 0.0
    with pytest.warns(UserWarning, match="Higham"):
        mvn = fs.MvNorm.from_params([0.0, 0.0, 0.0], rho)
    # the nearest positive semidefinite matrix is singular by construction;
    # the repaired matrix must be numerically positive semidefinite and
    # must have a Cholesky factor (its construction would have raised)
    assert np.linalg.eigvalsh(mvn._sigma).min() > -1e-10
    assert mvn._A.shape == (3, 3)

    def f(ugen):
        draw = mvn.draw(ugen)
        return {"a": draw.v0, "b": draw.v1, "c": draw.v2}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 3)
    assert np.all(np.isfinite(sampl))
