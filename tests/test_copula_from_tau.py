"""
Item 2 of the 2026-09-24 course handoff: from_tau constructors on all five
copula classes, with exact inversion of Frank's Kendall's tau.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import funcsim as fs
from funcsim import copfit


def _ugen(seed):
    rng = np.random.default_rng(seed)
    while True:
        yield float(rng.random())


def _draws(cop, M, seed):
    gen = _ugen(seed)
    return np.array([cop.draw(gen).to_numpy() for _ in range(M)])


def _tau(d):
    return stats.kendalltau(d[:, 0], d[:, 1]).statistic


# ---------------------------------------------------------------------------
# elliptical copulas
# ---------------------------------------------------------------------------

def test_gauss_from_tau_scalar():
    cop = fs.CopulaGauss.from_tau(0.6, names=["a", "b"])
    assert list(cop.rho.index) == ["a", "b"]
    assert cop.rho.loc["a", "b"] == pytest.approx(math.sin(0.3 * math.pi))
    assert cop.rho.loc["a", "a"] == 1.0


def test_gauss_from_tau_matrix_labels_and_ignored_diagonal():
    tau = pd.DataFrame([[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]],
                       index=list("xyz"), columns=list("xyz"))
    cop = fs.CopulaGauss.from_tau(tau)
    assert list(cop.rho.columns) == ["x", "y", "z"]
    np.testing.assert_allclose(cop.rho.to_numpy(),
                               np.sin(0.5 * np.pi * tau.to_numpy()))
    # the diagonal of tau is ignored: zeros there give the same copula
    tau0 = tau.to_numpy().copy()
    np.fill_diagonal(tau0, 0.0)
    cop0 = fs.CopulaGauss.from_tau(tau0, names=["x", "y", "z"])
    np.testing.assert_allclose(cop0.rho.to_numpy(), cop.rho.to_numpy())
    # a nested list works too, and np.eye-like independence is fine
    assert fs.CopulaGauss.from_tau([[1, 0], [0, 1]]).rho.iloc[0, 1] == 0.0


def test_gauss_from_tau_negative_and_roundtrip_through_draws():
    cop = fs.CopulaGauss.from_tau(-0.4)
    assert cop.rho.iloc[0, 1] == pytest.approx(math.sin(-0.2 * math.pi))
    d = _draws(cop, 3000, 1)
    assert abs(_tau(d) + 0.4) < 0.03
    assert fs.utests(d[:, 0])["anderson_darling_pval"] > 0.01


def test_gauss_from_tau_validation():
    for bad in (1.0, -1.0, 1.2, math.nan):
        with pytest.raises(ValueError):
            fs.CopulaGauss.from_tau(bad)
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_tau([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2]])
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_tau([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError):
        fs.CopulaGauss.from_tau(0.5, names=["a", "b", "c"])


def test_gauss_from_tau_repairs_indefinite():
    tau = [[1.0, 0.8, -0.8], [0.8, 1.0, 0.8], [-0.8, 0.8, 1.0]]
    with pytest.warns(UserWarning):
        cop = fs.CopulaGauss.from_tau(tau)
    R = cop.rho.to_numpy()
    assert np.allclose(np.diag(R), 1.0)
    assert np.all(np.linalg.eigvalsh(R) > 0.0)


def test_student_from_tau():
    cop = fs.CopulaStudent.from_tau(0.6, nu=4.0, names=["p", "q"])
    assert cop.nu == 4.0
    assert cop.rho.loc["p", "q"] == pytest.approx(math.sin(0.3 * math.pi))
    d = _draws(cop, 3000, 2)
    assert abs(_tau(d) - 0.6) < 0.03
    # nu is validated as in from_params, and the Gaussian limit is accepted
    with pytest.raises(ValueError):
        fs.CopulaStudent.from_tau(0.6, nu=0.0)
    assert math.isinf(fs.CopulaStudent.from_tau(0.6, nu=math.inf).nu)


# ---------------------------------------------------------------------------
# Archimedean copulas: closed forms and the chapter's Frank values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.1, 0.5, 0.85])
def test_clayton_from_tau_closed_form(tau):
    cop = fs.CopulaClayton.from_tau(tau, names=["a", "b", "c"])
    assert cop.theta == pytest.approx(2.0 * tau / (1.0 - tau))
    assert list(cop.draw(_ugen(0)).index) == ["a", "b", "c"]


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.85])
def test_gumbel_from_tau_closed_form(tau):
    cop = fs.CopulaGumbel.from_tau(tau, K=3)
    assert cop.theta == pytest.approx(1.0 / (1.0 - tau))
    assert len(cop.draw(_ugen(0))) == 3


def test_chapter_13_parameter_values():
    # tau 0.85: Gumbel theta 6.67 and Clayton theta 11.33 in the chapter text
    assert fs.CopulaGumbel.from_tau(0.85).theta == pytest.approx(6.667, abs=5e-4)
    assert fs.CopulaClayton.from_tau(0.85).theta == pytest.approx(11.333, abs=5e-4)


def test_frank_from_tau_matches_chapter_values():
    # chapter 12 quotes 7.93 (tau 0.6) and 24.9 (tau 0.85), from the Debye
    # inversion; the handoff asks for 7.930 and 24.905
    assert fs.CopulaFrank.from_tau(0.6).theta == pytest.approx(7.930, abs=5e-4)
    assert fs.CopulaFrank.from_tau(0.85).theta == pytest.approx(24.905, abs=5e-4)
    # and the exact inverse agrees with the (clamped) fitting start value
    # inside the latter's range
    assert (fs.CopulaFrank.from_tau(0.5).theta
            == pytest.approx(copfit.frank_theta0(0.5), rel=1e-9))


@pytest.mark.parametrize("theta", [1e-4, 0.01, 0.3, 0.5, 0.5000001, 2.0,
                                   8.0, 50.0, 400.0, 5000.0])
def test_frank_tau_inversion_roundtrip(theta):
    tau = copfit.frank_tau(theta)
    assert 0.0 < tau < 1.0
    assert copfit.frank_theta_from_tau(tau) == pytest.approx(theta, rel=1e-9)


def test_frank_tau_is_odd_and_matches_debye_formula():
    for theta in (0.2, 3.0, 40.0):
        assert copfit.frank_tau(-theta) == -copfit.frank_tau(theta)
    for theta in (1.0, 3.0, 40.0):
        direct = 1.0 - (4.0 / theta) * (1.0 - copfit._debye1(theta))
        assert copfit.frank_tau(theta) == pytest.approx(direct, rel=1e-12)
    # the series and the quadrature agree where they hand over
    series = copfit.frank_tau(0.5)
    direct = 1.0 - (4.0 / 0.5) * (1.0 - copfit._debye1(0.5))
    assert series == pytest.approx(direct, rel=1e-11)
    # tau ~ theta / 9 for small theta, without cancellation
    assert copfit.frank_tau(1e-6) == pytest.approx(1e-6 / 9.0, rel=1e-9)


def test_debye_asymptote_is_finite_and_continuous():
    assert copfit._debye1(1000.0) == pytest.approx(math.pi ** 2 / 6000.0,
                                                   rel=1e-12)
    assert copfit._debye1(200.0) == pytest.approx(copfit._debye1(200.0001),
                                                  rel=1e-6)


@pytest.mark.parametrize("cls", [fs.CopulaClayton, fs.CopulaGumbel,
                                 fs.CopulaFrank])
def test_archimedean_from_tau_rejects_out_of_range(cls):
    for bad in (0.0, 1.0, 1.5, math.nan):
        with pytest.raises(ValueError):
            cls.from_tau(bad)
    with pytest.raises(ValueError):
        cls.from_tau(0.5, K=1)


@pytest.mark.parametrize("cls", [fs.CopulaClayton, fs.CopulaGumbel,
                                 fs.CopulaFrank])
def test_archimedean_from_tau_draws_have_that_tau(cls):
    d = _draws(cls.from_tau(0.45), 3000, 7)
    assert abs(_tau(d) - 0.45) < 0.03
    assert fs.utests(d[:, 1])["anderson_darling_pval"] > 0.01
