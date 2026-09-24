"""
Item 3 of the 2026-09-24 course handoff: lambda_lower and lambda_upper
tail-dependence properties on all five copula classes.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import funcsim as fs


def _ugen(seed):
    rng = np.random.default_rng(seed)
    while True:
        yield float(rng.random())


def _student_lambda(rho, nu):
    return 2.0 * stats.t.cdf(-math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)),
                             df=nu + 1.0)


def test_chapter_12_table_at_rho_09():
    rho = [[1.0, 0.9], [0.9, 1.0]]
    for nu, want in ((2.0, 0.72), (4.0, 0.63), (10.0, 0.46)):
        ct = fs.CopulaStudent.from_params(rho, nu=nu)
        assert ct.lambda_lower == pytest.approx(want, abs=0.005)
        assert ct.lambda_upper == ct.lambda_lower
        assert ct.lambda_lower == pytest.approx(_student_lambda(0.9, nu))
    cg = fs.CopulaGauss.from_params(rho)
    assert cg.lambda_lower == 0.0
    assert cg.lambda_upper == 0.0
    assert isinstance(cg.lambda_lower, float)


def test_chapter_13_archimedean_values():
    assert fs.CopulaGumbel.from_params(theta=6.67).lambda_upper \
        == pytest.approx(0.890, abs=5e-4)
    assert fs.CopulaGumbel.from_params(theta=6.67).lambda_lower == 0.0
    assert fs.CopulaClayton.from_params(theta=11.33).lambda_lower \
        == pytest.approx(0.941, abs=5e-4)
    assert fs.CopulaClayton.from_params(theta=11.33).lambda_upper == 0.0
    cf = fs.CopulaFrank.from_params(theta=8.0)
    assert cf.lambda_lower == 0.0 and cf.lambda_upper == 0.0


def test_student_inf_nu_and_negative_rho():
    ct = fs.CopulaStudent.from_params([[1.0, 0.9], [0.9, 1.0]], nu=math.inf)
    assert ct.lambda_lower == 0.0
    ct = fs.CopulaStudent.from_params([[1.0, -0.5], [-0.5, 1.0]], nu=3.0)
    assert ct.lambda_lower == pytest.approx(_student_lambda(-0.5, 3.0))
    assert 0.0 < ct.lambda_lower < 0.1


def test_elliptical_k3_return_labeled_dataframes():
    rho = pd.DataFrame([[1.0, 0.9, 0.3], [0.9, 1.0, 0.5], [0.3, 0.5, 1.0]],
                       index=list("abc"), columns=list("abc"))
    ct = fs.CopulaStudent.from_params(rho, nu=4.0)
    lam = ct.lambda_upper
    assert isinstance(lam, pd.DataFrame)
    assert list(lam.index) == ["a", "b", "c"]
    assert list(lam.columns) == ["a", "b", "c"]
    assert np.allclose(np.diag(lam.to_numpy()), 1.0)
    assert lam.loc["a", "b"] == pytest.approx(_student_lambda(0.9, 4.0))
    assert lam.loc["b", "c"] == pytest.approx(_student_lambda(0.5, 4.0))
    assert lam.loc["c", "a"] == lam.loc["a", "c"]
    cg = fs.CopulaGauss.from_params(rho)
    lg = cg.lambda_lower
    assert isinstance(lg, pd.DataFrame)
    np.testing.assert_array_equal(lg.to_numpy(), np.eye(3))
    assert list(lg.columns) == ["a", "b", "c"]
    # the Archimedean classes stay scalar at any K
    assert isinstance(fs.CopulaClayton.from_params(theta=2.0, K=4).lambda_lower,
                      float)


def test_properties_on_fitted_copulas():
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky([[1.0, 0.7], [0.7, 1.0]])
    u = pd.DataFrame(stats.norm.cdf(rng.standard_normal((500, 2)) @ L.T),
                     columns=["x", "y"])
    ct = fs.CopulaStudent(u)
    assert 0.0 <= ct.lambda_lower <= 1.0
    assert ct.lambda_lower == pytest.approx(
        _student_lambda(ct.rho.loc["x", "y"], ct.nu) if math.isfinite(ct.nu)
        else 0.0)
    cc = fs.CopulaClayton(u)
    assert cc.lambda_lower == pytest.approx(2.0 ** (-1.0 / cc.theta))
    cgu = fs.CopulaGumbel(u)
    assert cgu.lambda_upper == pytest.approx(2.0 - 2.0 ** (1.0 / cgu.theta))


def test_clayton_lower_tail_probability_matches_coefficient():
    # P(U1 < q, U2 < q) / q for Clayton is 1 / sqrt(2 - q^2) at theta = 2,
    # within 0.002 of the limit 2^(-1/2) already at q = 0.1.  With 20000
    # draws the standard error of the ratio is about 0.018, so the tolerance
    # is roughly three standard errors
    cop = fs.CopulaClayton.from_params(theta=2.0)
    gen = _ugen(8)
    d = np.array([cop.draw(gen).to_numpy() for _ in range(20000)])
    q = 0.1
    both = np.mean((d[:, 0] < q) & (d[:, 1] < q))
    assert both / q == pytest.approx(cop.lambda_lower, abs=0.06)
