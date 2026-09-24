"""
Item 1 of the 2026-09-24 course handoff: a Student's t copula whose profile
likelihood peaks at the ceiling of the degrees-of-freedom search is reported
with nu = inf and behaves as the Gaussian copula with the same rho.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import funcsim as fs
from funcsim import copfit


def _gauss_udata(seed, M, rho):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    return stats.norm.cdf(rng.standard_normal((M, 2)) @ L.T)


def _ugen(seed):
    rng = np.random.default_rng(seed)
    while True:
        yield float(rng.random())


def test_fit_on_gaussian_data_reports_inf():
    # the handoff's evidence: pairs from a Gaussian copula with rho 0.9 used
    # to give nu = 199.9987 (a value at the search ceiling read as an estimate)
    udata = pd.DataFrame(_gauss_udata(1, 1000, 0.9), columns=["a", "b"])
    ct = fs.CopulaStudent(udata)
    assert math.isinf(ct.nu)
    assert ct.loglik_gain == 0.0
    assert ct.rho.loc["a", "b"] == pytest.approx(0.9, abs=0.03)


def test_fit_on_heavy_tailed_data_stays_finite_with_positive_gain():
    cop = fs.CopulaStudent.from_params([[1.0, 0.7], [0.7, 1.0]], nu=3.0)
    gen = _ugen(2)
    sample = np.array([cop.draw(gen).to_numpy() for _ in range(2000)])
    refit = fs.CopulaStudent(sample)
    assert math.isfinite(refit.nu)
    assert 2.0 < refit.nu < 6.0
    assert refit.loglik_gain > 0.0
    # the gain is what it says: loglik at nu minus loglik at the Gaussian limit
    R = refit.rho.to_numpy()
    want = (copfit.student_loglik(R, refit.nu, sample)
            - copfit.gauss_loglik(R, sample))
    assert refit.loglik_gain == pytest.approx(want, rel=1e-9)


def test_draw_with_inf_nu_matches_gauss_and_consumes_one_extra():
    rho = [[1.0, 0.6], [0.6, 1.0]]
    ct = fs.CopulaStudent.from_params(rho, nu=math.inf, names=["x", "y"])
    cg = fs.CopulaGauss.from_params(rho, names=["x", "y"])
    assert math.isinf(ct.nu)
    assert math.isnan(ct.loglik_gain)
    for seed in range(5):
        u = list(np.random.default_rng(seed).random(3))
        dt = ct.draw(iter(u))
        dg = cg.draw(iter(u[:2]))
        np.testing.assert_allclose(dt.to_numpy(), dg.to_numpy(), rtol=0,
                                   atol=1e-15)
        assert list(dt.index) == ["x", "y"]
    # exactly K + 1 draws are consumed
    gen = iter(np.random.default_rng(9).random(3))
    ct.draw(gen)
    with pytest.raises(StopIteration):
        next(gen)


def test_from_params_rejects_bad_nu_still():
    with pytest.raises(ValueError):
        fs.CopulaStudent.from_params([[1.0, 0.6], [0.6, 1.0]], nu=-math.inf)
    with pytest.raises(ValueError):
        fs.CopulaStudent.from_params([[1.0, 0.6], [0.6, 1.0]], nu=math.nan)
    with pytest.raises(ValueError):
        fs.CopulaStudent.from_params([[1.0, 0.6], [0.6, 1.0]], nu=0.0)


def test_student_loglik_at_inf_is_gauss_loglik():
    u = _gauss_udata(3, 300, 0.5)
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert (copfit.student_loglik(R, math.inf, u)
            == pytest.approx(copfit.gauss_loglik(R, u)))


def test_copcompare_prints_inf():
    table = fs.copcompare(_gauss_udata(3, 1000, 0.9))
    assert table.loc["Student", "parameters"].endswith("nu=inf")
    assert np.isfinite(table.loc["Student", "loglik"])
    assert table.loc["Student", "n_params"] == 2
    # the Student row is the Gaussian row plus the parameter penalty
    assert (table.loc["Student", "BIC"] - table.loc["Gaussian", "BIC"]
            == pytest.approx(math.log(1000), rel=0.3))
