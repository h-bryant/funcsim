"""
Item 4 of the 2026-09-24 course handoff: negative dependence for the Frank
copula with two variables (theta < 0) in from_params, from_tau, the fit,
the log-likelihood, and draws by conditional inversion.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

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


def _frank_biv_density(theta, u, v):
    # textbook bivariate Frank density, valid for either sign of theta
    p = 1.0 - math.exp(-theta)
    num = theta * p * math.exp(-theta * (u + v))
    den = (p - (1.0 - math.exp(-theta * u)) * (1.0 - math.exp(-theta * v))) ** 2
    return num / den


def _frank_cond_cdf(theta, u1, v):
    # P(U2 <= v | U1 = u1) = dC/du1
    D = math.exp(-theta) - 1.0
    x = math.exp(-theta * u1) - 1.0
    y = math.exp(-theta * v) - 1.0
    return math.exp(-theta * u1) * y / (D + x * y)


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------

def test_from_params_negative_theta_two_variables():
    cop = fs.CopulaFrank.from_params(theta=-3.0, names=["yield", "price"])
    assert cop.theta == -3.0
    assert list(cop.draw(_ugen(0)).index) == ["yield", "price"]
    assert cop.lambda_lower == 0.0 and cop.lambda_upper == 0.0


def test_from_params_rejections():
    with pytest.raises(ValueError, match="two variables"):
        fs.CopulaFrank.from_params(theta=-3.0, K=3)
    with pytest.raises(ValueError, match="two variables"):
        fs.CopulaFrank.from_params(theta=-3.0, names=["a", "b", "c"])
    with pytest.raises(ValueError):
        fs.CopulaFrank.from_params(theta=0.0)
    with pytest.raises(ValueError):
        fs.CopulaFrank.from_params(theta=math.nan)
    # positive theta with K > 2 is unaffected
    assert fs.CopulaFrank.from_params(theta=3.0, K=4).theta == 3.0


def test_from_tau_negative():
    neg = fs.CopulaFrank.from_tau(-0.5)
    pos = fs.CopulaFrank.from_tau(0.5)
    assert neg.theta == -pos.theta
    assert neg.theta == pytest.approx(-5.736, abs=5e-4)
    assert copfit.frank_tau(neg.theta) == pytest.approx(-0.5, abs=1e-12)
    with pytest.raises(ValueError, match="two variables"):
        fs.CopulaFrank.from_tau(-0.5, K=3)
    with pytest.raises(ValueError):
        fs.CopulaFrank.from_tau(-1.0)
    with pytest.raises(ValueError):
        fs.CopulaFrank.from_tau(0.0)


# ---------------------------------------------------------------------------
# log-likelihood
# ---------------------------------------------------------------------------

def test_negative_loglik_matches_closed_form_density():
    grid = [(0.3, 0.7), (0.05, 0.9), (0.5, 0.5), (0.95, 0.97), (0.01, 0.02)]
    for theta in (-0.5, -2.5, -25.0):
        for (u, v) in grid:
            got = copfit.frank_loglik(theta, np.array([[u, v]]))
            want = math.log(_frank_biv_density(theta, u, v))
            assert got == pytest.approx(want, rel=1e-8)
    # large |theta| stays finite where the direct formula overflows
    assert np.isfinite(copfit.frank_loglik(-900.0, np.array([[0.2, 0.9]])))


def test_loglik_limits_and_rejections():
    u = np.random.default_rng(1).random((50, 2))
    assert copfit.frank_loglik(0.0, u) == 0.0
    # continuity through zero
    assert abs(copfit.frank_loglik(1e-6, u) - copfit.frank_loglik(-1e-6, u)) < 1e-3
    with pytest.raises(ValueError, match="two variables"):
        copfit.frank_loglik(-2.0, np.random.default_rng(2).random((50, 3)))


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def test_fit_recovers_negative_theta_without_warning():
    theta_star = fs.CopulaFrank.from_tau(-0.5).theta       # -5.736
    sample = _draws(fs.CopulaFrank.from_params(theta=theta_star), 2000, 3)
    assert _tau(sample) < -0.4
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cop = fs.CopulaFrank(pd.DataFrame(sample, columns=["y", "p"]))
    assert cop.theta < 0.0
    assert abs(cop.theta - theta_star) / abs(theta_star) < 0.15
    assert list(cop.draw(_ugen(4)).index) == ["y", "p"]
    # the fitted value beats the tau-inversion start on the likelihood
    assert (copfit.frank_loglik(cop.theta, copfit._clipu(sample))
            >= copfit.frank_loglik(theta_star, copfit._clipu(sample)) - 1e-6)


def test_three_variables_still_clamp_with_a_warning():
    rng = np.random.default_rng(5)
    u1 = rng.random(400)
    u2 = np.clip(1.0 - u1 + rng.normal(0.0, 0.05, 400), 1e-6, 1.0 - 1e-6)
    u3 = rng.random(400)
    with pytest.warns(UserWarning, match="more than two variables"):
        cop = fs.CopulaFrank(np.column_stack([u1, u2, u3]))
    assert cop.theta == copfit.THETA_MIN_FRANK


def test_copcompare_handles_negative_pair():
    sample = _draws(fs.CopulaFrank.from_params(theta=-4.0), 800, 6)
    with pytest.warns(UserWarning):   # Clayton and Gumbel still warn
        table = fs.copcompare(sample)
    assert table.loc["Frank", "parameters"].startswith("theta=-")
    assert np.isfinite(table.loc["Frank", "loglik"])
    assert table.index[0] in ("Frank", "Gaussian", "Student")
    assert table.loc["Frank", "BIC"] < table.loc["Clayton", "BIC"]


# ---------------------------------------------------------------------------
# sampling by conditional inversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [-1.5, -5.0, -20.0])
def test_sampled_tau_matches_debye_formula(theta):
    cop = fs.CopulaFrank.from_params(theta=theta)
    d = _draws(cop, 3000, 7)
    assert np.all((d > 0.0) & (d < 1.0))
    assert abs(_tau(d) - copfit.frank_tau(theta)) < 0.03
    assert fs.utests(d[:, 0])["anderson_darling_pval"] > 0.01
    assert fs.utests(d[:, 1])["anderson_darling_pval"] > 0.01


def test_conditional_inversion_matches_numerical_inverse():
    for theta in (-0.7, -3.0, -12.0, -60.0):
        cop = fs.CopulaFrank.from_params(theta=theta)
        for (u1, a) in ((0.2, 0.9), (0.5, 0.5), (0.9, 0.05), (0.01, 0.99)):
            got = cop.draw(iter([u1, a, 0.3, 0.6])).to_numpy()
            assert got[0] == u1
            want = optimize.brentq(lambda v: _frank_cond_cdf(theta, u1, v) - a,
                                   1e-12, 1.0 - 1e-12, xtol=1e-14)
            assert got[1] == pytest.approx(want, abs=1e-9)


def test_negative_draw_consumes_k_plus_2():
    cop = fs.CopulaFrank.from_params(theta=-3.0)
    gen = iter(np.random.default_rng(8).random(4))
    cop.draw(gen)
    with pytest.raises(StopIteration):
        next(gen)
    # and the positive path is untouched: same uniforms, same frailty draws
    pos = fs.CopulaFrank.from_params(theta=3.0)
    u = list(np.random.default_rng(9).random(4))
    before = pos.draw(iter(u)).to_numpy()
    uA = np.array(u[:2])
    v = fs.dependence._logser_draw(3.0, u[2], u[3])
    t = -np.log(uA) / v
    log1mexp_t = np.where(t <= math.log(2.0), np.log(-np.expm1(-t)),
                          np.log1p(-np.exp(-t)))
    want = -np.logaddexp(log1mexp_t, -t - 3.0) / 3.0
    np.testing.assert_allclose(before, want, rtol=0, atol=1e-15)
