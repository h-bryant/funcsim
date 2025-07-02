import numpy as np
import pandas as pd
import funcsim as fs


def test_covtocorr_0():
    data = 100.0 * np.random.random((30, 3))
    cov = np.cov(data, rowvar=False)
    corr = fs.covtocorr(cov)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)


def test_covtocorr_1():
    data = 100.0 * np.random.random((30, 3))
    covA = np.cov(data, rowvar=False)
    covPd = pd.DataFrame(covA, columns=['a', 'b', 'c'], index=['a', 'b', 'c'])
    corr = fs.covtocorr(covPd)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.index.equals(covPd.index)