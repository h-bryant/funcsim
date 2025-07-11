import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
import funcsim as fs

def test_ic_0():
    v1 = stats.norm(4.0, 0.2).rvs(30)
    v2 = stats.norm(0.0, 3.2).rvs(30)
    rho_s = np.array([[1.0, 0.89], [0.89, 1.0]])
    ic = fs.imanconover(rho_s, [v1, v2])
    assert isinstance(ic, pd.DataFrame)
    assert ic.shape == (30, 2)
    assert ic.columns.tolist() == ['v1', 'v2']
    assert ic.index.tolist() == list(range(30))

def test_ic_1():
    v1 = pd.Series(stats.norm(4.0, 0.2).rvs(30))
    v2 = stats.norm(0.0, 3.2).rvs(30)
    rho_s = np.array([[1.0, 0.89], [0.89, 1.0]])
    ic = fs.imanconover(rho_s, [v1, v2])
    assert isinstance(ic, pd.DataFrame)
    assert ic.shape == (30, 2)
    assert ic.columns.tolist() == ['v1', 'v2']
    assert ic.index.tolist() == list(range(30))
    
def test_ic_2():
    idx = pd.Index(list(range(30)))
    v1 = pd.Series(stats.norm(4.0, 0.2).rvs(30), index=idx)
    v2 = stats.norm(0.0, 3.2).rvs(30)
    rho_s = np.array([[1.0, 0.89], [0.89, 1.0]])
    ic = fs.imanconover(rho_s, [v1, v2])
    assert isinstance(ic, pd.DataFrame)
    assert ic.shape == (30, 2)
    assert ic.columns.tolist() == ['v1', 'v2']
    assert ic.index.tolist() == list(range(30))
    



