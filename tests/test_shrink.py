import numpy as np
import pandas as pd
import xarray as xr
import funcsim as fs


targets = ["A", "B", "C", "D", "E", "F"]

def calc_all_norms(seed, mu, r, n):
    np.random.seed(seed)
    x = np.random.multivariate_normal(mu, r, size=n)
    return np.array([np.linalg.norm(fs.shrink(x, tgt) - r)
                     for tgt in targets])

def test_0():
    mu = np.array([10.0, 5.0, 0.0])

    rho = np.array([
        [1, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0]])

    variances = ([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 5.0]])

    r = np.dot(np.dot(variances, rho), variances)

    all_norms = calc_all_norms(seed=1, mu=mu, r=r, n=20)
    print(all_norms)
    print("sum: %s" % sum(all_norms))
    assert abs(sum(all_norms)) < 35.0
    print("test_0 passed")


def test_1():
    rho = np.array([
        [1, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0]])
    answer = fs.shrink(rho, "F")
    assert isinstance(answer, np.ndarray)


def test_2():
    rho = np.array([
        [1, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0]])
    df = pd.DataFrame(rho, columns=["a", "b", "c"], index=["a", "b", "c"])
    answer = fs.shrink(df, "F")
    assert isinstance(answer, pd.DataFrame)


def test_3():
    rho = np.array([
        [1, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0]])
    index=["a", "b", "c"]
    da = xr.DataArray(rho, dims={'rows': index, 'cols': index})
    answer = fs.shrink(da, "F")
    assert isinstance(answer, xr.DataArray)
