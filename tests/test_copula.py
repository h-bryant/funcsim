import math

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
    # deterministic generator of independent standard uniform draws
    rng = np.random.default_rng(seed)
    while True:
        yield float(rng.random())


def _dependent_udata(seed, M=1000, K=2, rho=0.5):
    # pseudo-observations with positive dependence (via a Gaussian copula)
    rng = np.random.default_rng(seed)
    R = np.full((K, K), rho)
    np.fill_diagonal(R, 1.0)
    z = rng.standard_normal((M, K)) @ np.linalg.cholesky(R).T
    return stats.norm.cdf(z)


def _draws(cop, M, seed):
    # M deterministic draws from a fitted copula object
    gen = _ugen(seed)
    return np.array([cop.draw(gen).to_numpy() for _ in range(M)])


# ---------------------------------------------------------------------------
# public-API smoke tests (marginal uniformity of simulated draws)
# ---------------------------------------------------------------------------

def test_cg_0():
    udata = np.random.default_rng(1).random(size=(1000, 2))
    udataPd = pd.DataFrame(udata, columns=["rain", "temp"])
    cg = fs.CopulaGauss(udataPd)

    def f(ugen):
        draw = cg.draw(ugen)
        return {"rain": draw.rain, "temp": draw.temp}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


def test_cg_1():
    udata = np.random.default_rng(2).random(size=(1000, 2))
    cg = fs.CopulaGauss(udata)

    def f(ugen):
        draw = cg.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


def test_ct_1():
    udata = _dependent_udata(3, M=1000, K=2)
    ct = fs.CopulaStudent(udata)

    def f(ugen):
        draw = ct.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


def test_ct_2():
    udata = _dependent_udata(4, M=1000, K=5)
    ct = fs.CopulaStudent(udata)

    def f(ugen):
        draw = ct.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1, "v2": draw.v2,
                "v3": draw.v3, "v4": draw.v4}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 5)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


def test_cc_0():
    udata = _dependent_udata(5, M=1000, K=2)
    cc = fs.CopulaClayton(udata)

    def f(ugen):
        draw = cc.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


def test_cgu_0():
    # note: formerly (mis)named test_cg_0, which shadowed the Gaussian test
    udata = _dependent_udata(6, M=1000, K=2)
    cg = fs.CopulaGumbel(udata)

    def f(ugen):
        draw = cg.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000, seed=666).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


def test_cf_0():
    udata = _dependent_udata(7, M=1000, K=2)
    cf = fs.CopulaFrank(udata)

    def f(ugen):
        draw = cf.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:, 0])
    assert u_test_results["anderson_darling_pval"] > 0.03


# ---------------------------------------------------------------------------
# parameter recovery: sample from a known copula via the class draw
# machinery, refit, and check the estimate
# ---------------------------------------------------------------------------

def _recover_theta(cls, theta_star, K, M=2000, seed=11):
    cop = cls(_dependent_udata(seed, M=500, K=K))
    cop._theta = theta_star
    sample = _draws(cop, M, seed=seed + 1)
    refit = cls(sample)
    assert abs(refit._theta - theta_star) / theta_star < 0.20


def test_clayton_recover_k2():
    _recover_theta(fs.CopulaClayton, 2.0, K=2)


def test_clayton_recover_k5():
    _recover_theta(fs.CopulaClayton, 2.0, K=5)


def test_gumbel_recover_k2():
    _recover_theta(fs.CopulaGumbel, 2.0, K=2)


def test_gumbel_recover_k5():
    _recover_theta(fs.CopulaGumbel, 2.0, K=5)


def test_frank_recover_k2():
    _recover_theta(fs.CopulaFrank, 5.0, K=2)


def test_frank_recover_k5():
    _recover_theta(fs.CopulaFrank, 5.0, K=5)


def test_student_fit_recover():
    R = np.array([[1.0, 0.8, 0.3],
                  [0.8, 1.0, 0.5],
                  [0.3, 0.5, 1.0]])
    cop = fs.CopulaStudent(_dependent_udata(13, M=500, K=3))
    cop._rho = R
    cop._nu = 5.0
    cop._A = np.linalg.cholesky(R)
    sample = _draws(cop, 4000, seed=14)
    refit = fs.CopulaStudent(sample)
    assert np.max(np.abs(refit._rho - R)) < 0.08
    assert 2.5 < refit._nu < 12.0


# ---------------------------------------------------------------------------
# rank-correlation preservation for the Student's t copula (these fail if
# draw() multiplies by the correlation matrix rather than its Cholesky
# factor, as it once did)
# ---------------------------------------------------------------------------

def test_student_rankcorr_k2():
    R = np.array([[1.0, 0.8], [0.8, 1.0]])
    cop = fs.CopulaStudent(_dependent_udata(15, M=500, K=2))
    cop._rho = R
    cop._nu = 5.0
    cop._A = np.linalg.cholesky(R)
    s = _draws(cop, 4000, seed=16)
    tau = stats.kendalltau(s[:, 0], s[:, 1]).statistic
    assert abs(tau - (2.0 / math.pi) * math.asin(0.8)) < 0.05


def test_student_rankcorr_k3():
    R = np.array([[1.0, 0.8, 0.3],
                  [0.8, 1.0, 0.5],
                  [0.3, 0.5, 1.0]])
    cop = fs.CopulaStudent(_dependent_udata(17, M=500, K=3))
    cop._rho = R
    cop._nu = 5.0
    cop._A = np.linalg.cholesky(R)
    s = _draws(cop, 4000, seed=18)
    implied = np.sin(0.5 * np.pi * copfit.taumatrix(s))
    np.fill_diagonal(implied, 1.0)
    assert np.max(np.abs(implied - R)) < 0.07


# ---------------------------------------------------------------------------
# log-density oracles: K-dimensional implementations vs textbook bivariate
# closed forms, and the Gumbel coefficient recurrence vs the
# Hofert-Machler-McNeil Stirling-number formula
# ---------------------------------------------------------------------------

def _clayton_biv(theta, u, v):
    return ((1.0 + theta) * (u * v) ** (-(1.0 + theta))
            * (u ** -theta + v ** -theta - 1.0) ** (-(2.0 + 1.0 / theta)))


def _gumbel_biv(theta, u, v):
    lu, lv = -math.log(u), -math.log(v)
    t = lu ** theta + lv ** theta
    C = math.exp(-t ** (1.0 / theta))
    return (C / (u * v) * (lu * lv) ** (theta - 1.0)
            * t ** (1.0 / theta - 2.0) * (t ** (1.0 / theta) + theta - 1.0))


def _frank_biv(theta, u, v):
    p = 1.0 - math.exp(-theta)
    num = theta * p * math.exp(-theta * (u + v))
    den = (p - (1.0 - math.exp(-theta * u))
           * (1.0 - math.exp(-theta * v))) ** 2
    return num / den


def test_density_bivariate_closed_forms():
    grid = [(0.3, 0.7), (0.05, 0.9), (0.5, 0.5), (0.95, 0.97)]
    for theta in (0.5, 2.5, 25.0):
        for (u, v) in grid:
            arr = np.array([[u, v]])
            for loglik, biv, shift in (
                    (copfit.clayton_loglik, _clayton_biv, 0.0),
                    (copfit.gumbel_loglik, _gumbel_biv, 1.0),
                    (copfit.frank_loglik, _frank_biv, 0.0)):
                th = theta + shift
                got = loglik(th, arr)
                want = math.log(biv(th, u, v))
                assert abs(got - want) / max(abs(want), 1.0) < 1e-8


def _stirling1_unsigned(n):
    s = [[0] * (n + 1) for _ in range(n + 1)]
    s[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            s[i][j] = s[i - 1][j - 1] + (i - 1) * s[i - 1][j]
    return s


def _stirling2(n):
    S = [[0] * (n + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            S[i][j] = S[i - 1][j - 1] + j * S[i - 1][j]
    return S


def test_gumbel_coefs_vs_stirling():
    for d in (2, 5, 10, 15):
        for alpha in (0.3, 0.7, 0.95):
            s1 = _stirling1_unsigned(d)
            s2 = _stirling2(d)
            want = np.array([
                sum(alpha ** j * (-1) ** (j - k) * s1[d][j] * s2[j][k]
                    for j in range(k, d + 1))
                for k in range(1, d + 1)])
            got = np.exp(copfit._gumbel_logcoefs(d, alpha))
            assert np.allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# negative-dependence gate: warn, clamp to the family floor, and still
# produce finite draws in (0, 1)
# ---------------------------------------------------------------------------

def test_gate_negative_dependence():
    rng = np.random.default_rng(19)
    u1 = rng.random(500)
    u2 = np.clip(1.0 - u1 + rng.normal(0.0, 0.05, 500), 1e-6, 1.0 - 1e-6)
    neg = np.column_stack([u1, u2])
    for cls, floor in ((fs.CopulaClayton, copfit.THETA_MIN_CLAYTON),
                       (fs.CopulaGumbel, copfit.THETA_MIN_GUMBEL),
                       (fs.CopulaFrank, copfit.THETA_MIN_FRANK)):
        with pytest.warns(UserWarning):
            cop = cls(neg)
        assert cop._theta == floor
        d = _draws(cop, 20, seed=20)
        assert np.all(np.isfinite(d))
        assert np.all((d > 0.0) & (d < 1.0))
