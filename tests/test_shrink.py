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


def test_return_intensity():
    np.random.seed(3)
    x = np.random.normal(size=(8, 10))
    sig = fs.shrink(x, "D")
    sig2, lam = fs.shrink(x, "D", return_intensity=True)
    assert isinstance(lam, float) and 0.0 <= lam <= 1.0
    assert np.allclose(sig, sig2)
    # for a diagonal target the off-diagonals are scaled by (1 - lam)
    s = np.cov(x, rowvar=False)
    off = ~np.eye(10, dtype=bool)
    assert np.allclose(sig2[off], (1.0 - lam) * s[off])
    # more data, less shrinkage
    x_big = np.random.normal(size=(2000, 10))
    _, lam_big = fs.shrink(x_big, "D", return_intensity=True)
    assert lam_big < lam
    # other return types carry the same intensity
    df = pd.DataFrame(x, columns=list("abcdefghij"))
    sig_df, lam_df = fs.shrink(df, "D", return_intensity=True)
    assert isinstance(sig_df, pd.DataFrame) and lam_df == lam
