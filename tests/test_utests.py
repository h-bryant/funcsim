import numpy as np
import scipy.stats as stats
import funcsim as fs


def test_0():
    vh  = stats.chi2(df=1).rvs(size=10000)
    results = fs.utests(np.clip(vh, a_min=0.02, a_max=0.99))
    assert results["cook_gelman_rubin_pval"] < 0.30  # it's a weak test
    assert results["kolmogorov_smirnov_pval"] < 0.10
    assert results["anderson_darling_pval"] < 0.10
    assert results["cramer_von_mises_pval"] < 0.10


def test_1():
    vh = stats.uniform().rvs(size=1000)
    results = fs.utests(vh)
    assert results["cook_gelman_rubin_pval"] > 0.05
    assert results["kolmogorov_smirnov_pval"] > 0.05
    assert results["anderson_darling_pval"] > 0.05
    assert results["cramer_von_mises_pval"] > 0.05