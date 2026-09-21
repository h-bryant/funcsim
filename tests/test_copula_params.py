import numpy as np
import pandas as pd
import pytest
from scipy import stats

import funcsim as fs
from funcsim import copfit


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ugen(seed):
    rng = np.random.default_rng(seed)
    while True:
        yield float(rng.random())


def _draws(cop, M, seed):
    gen = _ugen(seed)
    return np.array([cop.draw(gen).to_numpy() for _ in range(M)])


def _tau(d):
    return stats.kendalltau(d[:, 0], d[:, 1]).statistic


def _gauss_udata(seed, M, rho):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    return stats.norm.cdf(rng.standard_normal((M, 2)) @ L.T)


# ---------------------------------------------------------------------------
# from_params: elliptical copulas
# ---------------------------------------------------------------------------

def test_gauss_from_params_roundtrip():
    cop = fs.CopulaGauss.from_params([[1.0, 0.8], [0.8, 1.0]],
                                     names=["a", "b"])
    assert list(cop.rho.columns) == ["a", "b"]
    assert cop.rho.loc["a", "b"] == pytest.approx(0.8)
    d = _draws(cop, 3000, 1)
    assert abs(_tau(d) - 2.0 / np.pi * np.arcsin(0.8)) < 0.03
    assert fs.utests(d[:, 0])["anderson_darling_pval"] > 0.01
    assert fs.utests(d[:, 1])["anderson_darling_pval"] > 0.01


def test_gauss_from_params_dataframe_names():
    rho = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]],
                       index=["x", "y"], columns=["x", "y"])
    cop = fs.CopulaGauss.from_params(rho)
    assert list(cop.draw(_ugen(0)).index) == ["x", "y"]


def test_gauss_from_params_validation():
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_params([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2]])
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_params([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_params([[2.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_params([[1.0, 0.5], [0.5, 1.0]], names=["a"])


def test_gauss_from_params_repairs_indefinite():
    rho = [[1.0, 0.9, -0.5], [0.9, 1.0, 0.9], [-0.5, 0.9, 1.0]]
    with pytest.warns(UserWarning):
        cop = fs.CopulaGauss.from_params(rho)
    R = cop.rho.to_numpy()
    assert np.allclose(np.diag(R), 1.0)
    assert np.all(np.linalg.eigvalsh(R) > 0.0)


def test_student_from_params_roundtrip():
    cop = fs.CopulaStudent.from_params([[1.0, 0.6], [0.6, 1.0]], nu=4.0)
    assert cop.nu == pytest.approx(4.0)
    d = _draws(cop, 3000, 2)
    assert abs(_tau(d) - 2.0 / np.pi * np.arcsin(0.6)) < 0.03
    assert fs.utests(d[:, 0])["anderson_darling_pval"] > 0.01


def test_student_from_params_validation():
    with pytest.raises(ValueError):
        fs.CopulaStudent.from_params([[1.0, 0.6], [0.6, 1.0]], nu=0.0)
    with pytest.raises(ValueError):
        fs.CopulaStudent.from_params([[1.0, 0.6], [0.6, 1.0]], nu=float("nan"))


# ---------------------------------------------------------------------------
# from_params: Archimedean copulas
# ---------------------------------------------------------------------------

def test_clayton_from_params_roundtrip():
    cop = fs.CopulaClayton.from_params(theta=2.0)
    assert cop.theta == pytest.approx(2.0)
    d = _draws(cop, 3000, 3)
    assert abs(_tau(d) - 2.0 / (2.0 + 2.0)) < 0.03
    assert fs.utests(d[:, 1])["anderson_darling_pval"] > 0.01


def test_gumbel_from_params_roundtrip():
    cop = fs.CopulaGumbel.from_params(theta=2.0, names=["p", "q", "r"])
    assert cop.theta == pytest.approx(2.0)
    d = _draws(cop, 3000, 4)
    assert d.shape == (3000, 3)
    assert abs(_tau(d) - (1.0 - 1.0 / 2.0)) < 0.03


def test_frank_from_params_roundtrip():
    theta = copfit.frank_theta0(0.5)
    cop = fs.CopulaFrank.from_params(theta=theta, K=2)
    d = _draws(cop, 3000, 5)
    assert abs(_tau(d) - 0.5) < 0.03


def test_archimedean_from_params_validation():
    with pytest.raises(ValueError):
        fs.CopulaClayton.from_params(theta=0.0)
    with pytest.raises(ValueError):
        fs.CopulaGumbel.from_params(theta=1.0)
    with pytest.raises(ValueError):
        fs.CopulaFrank.from_params(theta=-1.0)
    with pytest.raises(ValueError):
        fs.CopulaClayton.from_params(theta=1.0, K=1)


# ---------------------------------------------------------------------------
# parameter properties on fitted copulas
# ---------------------------------------------------------------------------

def test_fitted_properties():
    udata = pd.DataFrame(_gauss_udata(6, 800, 0.7), columns=["a", "b"])
    cg = fs.CopulaGauss(udata)
    assert list(cg.rho.index) == ["a", "b"]
    assert abs(cg.rho.loc["a", "b"] - 0.7) < 0.06
    ct = fs.CopulaStudent(udata)
    assert abs(ct.rho.loc["a", "b"] - 0.7) < 0.06
    assert ct.nu > 0.0
    assert fs.CopulaClayton(udata).theta > 0.0
    assert fs.CopulaGumbel(udata).theta > 1.0
    assert fs.CopulaFrank(udata).theta > 0.0


# ---------------------------------------------------------------------------
# log-likelihoods and copcompare
# ---------------------------------------------------------------------------

def test_gauss_loglik_identity_is_zero():
    u = np.random.default_rng(7).random((200, 3))
    assert copfit.gauss_loglik(np.eye(3), u) == pytest.approx(0.0)


def test_student_loglik_approaches_gauss_for_large_nu():
    u = _gauss_udata(8, 300, 0.5)
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    llg = copfit.gauss_loglik(R, u)
    llt = copfit.student_loglik(R, 1e6, u)
    assert abs(llg - llt) < 0.5


def test_copcompare_prefers_elliptical_for_gaussian_data():
    table = fs.copcompare(_gauss_udata(9, 1500, 0.6))
    assert set(table.index) == {"Gaussian", "Student", "Clayton", "Gumbel",
                                "Frank"}
    assert list(table.columns) == ["parameters", "n_params", "loglik",
                                   "AIC", "BIC"]
    assert table["BIC"].is_monotonic_increasing
    assert table.index[0] in ("Gaussian", "Student")
    assert table.loc["Gaussian", "BIC"] < table.loc["Clayton", "BIC"]
    assert table.loc["Student", "n_params"] == 2
    assert table.loc["Gaussian", "n_params"] == 1


def test_copcompare_prefers_clayton_for_clayton_data():
    d = _draws(fs.CopulaClayton.from_params(theta=3.0), 1500, 10)
    table = fs.copcompare(d)
    assert table.index[0] == "Clayton"


def test_copcompare_rejects_single_column():
    with pytest.raises(ValueError):
        fs.copcompare(np.random.default_rng(11).random((100, 1)))
