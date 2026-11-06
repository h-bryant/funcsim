"""
MvKde bandwidth options added after 0.2.9: full-matrix rules of thumb
(full=True), least-squares and smoothed cross-validation (bw='lscv',
bw='scv'), and the sample-size checks by family and structure.  See
kde-plan-2026-11-06.md, step 4.
"""
import math
import warnings

import numpy as np
import pytest
from scipy import stats

import funcsim as fs
from funcsim import mvbandwidth as mb
from funcsim.samplesize import check_sample_size, SILVERMAN_TABLE


def _sample60():
    C = np.array([[1.0, 0.6, -0.3], [0.6, 2.0, 0.2], [-0.3, 0.2, 0.5]])
    return np.random.default_rng(42).multivariate_normal(
        [0, 1, -1], C, size=60, method="cholesky")


def _sample200():
    return np.random.default_rng(43).multivariate_normal(
        [0, 1], [[1.0, 0.5], [0.5, 2.0]], size=200, method="cholesky")


# ---------------------------------------------------------------------------
# full-matrix rules of thumb

@pytest.mark.parametrize("rule", ["scott", "silverman"])
def test_full_rule_is_factor_times_covariance(rule):
    x = _sample60()
    M, K = x.shape
    H = fs.MvKde(x, bw=rule, full=True).bandwidth.values
    factor = mb.rule_factor(rule, M, K)
    np.testing.assert_allclose(H, factor * np.cov(x, rowvar=False, bias=True),
                               rtol=1e-12)
    # the diagonal rule is the diagonal of the full one
    D = fs.MvKde(x, bw=rule, full=False).bandwidth.values
    np.testing.assert_allclose(D, np.diag(np.diag(H)), rtol=1e-12)


def test_full_scott_matches_scipy_up_to_ddof():
    x = _sample60()
    M = x.shape[0]
    H = fs.MvKde(x, bw="scott", full=True).bandwidth.values
    g = stats.gaussian_kde(x.T)     # covariance uses ddof=1
    np.testing.assert_allclose(H, g.covariance * (M - 1) / M, rtol=1e-12)


def test_full_ignored_for_user_matrix():
    x = _sample60()
    H = np.diag([0.2, 0.3, 0.1])
    a = fs.MvKde(x, bw=H, full=True).bandwidth.values
    b = fs.MvKde(x, bw=H).bandwidth.values
    np.testing.assert_array_equal(a, b)


def test_draws_with_full_matrix():
    x = _sample200()
    kde = fs.MvKde(x, bw="scott", full=True)
    rng = np.random.default_rng(0)
    ugen = (u for u in rng.random(10 ** 5))
    draws = np.array([kde.draw(ugen).values for _ in range(20000)])
    assert draws.shape == (20000, 2)
    # the draws' covariance is the data's plus the bandwidth (to within
    # sampling noise: the ratio was within 0.3% at 100,000 draws)
    want = np.cov(x, rowvar=False, bias=True) + kde.bandwidth.values
    assert np.all(np.abs(np.cov(draws, rowvar=False) / want - 1.0) < 0.1)


# ---------------------------------------------------------------------------
# criteria against R's ks (ks 1.15.3; values from ks:::lscv.mat(x, H),
# ks:::scv.mat(x, H, G, binned = FALSE), and ks:::Qr(...) on the sample
# written with 17 significant digits)

H_REF = np.array([[0.5, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.3]])
G_REF = np.array([[0.3, 0.05, 0.0], [0.05, 0.4, 0.02], [0.0, 0.02, 0.2]])


def test_criteria_match_ks():
    pairs = mb._Pairs(_sample60())
    assert mb.lscv(pairs, H_REF) == pytest.approx(-0.0325907171710156,
                                                  rel=1e-12)
    assert mb.scv(pairs, H_REF, G_REF) == pytest.approx(0.00498638305273066,
                                                        rel=1e-12)


def test_chunked_pairs_match_held_pairs(monkeypatch):
    x = _sample60()
    held = mb._Pairs(x)
    monkeypatch.setattr(mb, "_PAIR_ELEMENTS", 100)
    chunked = mb._Pairs(x)
    assert chunked._held is None
    assert mb.lscv(chunked, H_REF) == pytest.approx(mb.lscv(held, H_REF),
                                                    rel=1e-12)
    assert mb.scv(chunked, H_REF, G_REF) == pytest.approx(
        mb.scv(held, H_REF, G_REF), rel=1e-12)


def _mixture_pdf(points, data, H):
    # Gaussian KDE with bandwidth H at an array of points (rows)
    return np.mean([stats.multivariate_normal(mu, H).pdf(points)
                    for mu in data], axis=0)


def _grid_integral(func, data, n=801):
    # integral over a box six units beyond the data by the trapezoid rule
    # on an n-by-n grid: for a smooth integrand that has decayed to
    # nothing at the edges this converges far faster than any power of
    # the spacing
    lo = data.min(axis=0) - 6.0
    hi = data.max(axis=0) + 6.0
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    values = func(np.column_stack([X.ravel(), Y.ravel()])).reshape(n, n)
    return np.trapezoid(np.trapezoid(values, ys, axis=1), xs)


def test_lscv_matches_brute_force_in_two_dimensions():
    data = _sample200()[:15]
    H = np.array([[0.4, 0.1], [0.1, 0.6]])
    M = data.shape[0]
    sq = _grid_integral(lambda pts: _mixture_pdf(pts, data, H) ** 2, data)
    loo = sum(float(np.atleast_1d(_mixture_pdf(data[i:i + 1],
                                               np.delete(data, i, axis=0),
                                               H))[0]) for i in range(M))
    want = sq - 2.0 / M * loo
    assert mb.lscv(mb._Pairs(data), H) == pytest.approx(want, rel=1e-8)


def test_scv_matches_brute_force_in_two_dimensions():
    # the pairwise bias term equals the integral of (f_{H+G} - f_G)^2,
    # the squared difference between the pilot estimate smoothed by the
    # kernel and the pilot estimate itself
    data = _sample200()[:15]
    H = np.array([[0.4, 0.1], [0.1, 0.6]])
    G = np.array([[0.2, 0.0], [0.0, 0.2]])
    M, K = data.shape
    bias2 = _grid_integral(
        lambda pts: (_mixture_pdf(pts, data, H + G)
                     - _mixture_pdf(pts, data, G)) ** 2, data)
    want = (4 * math.pi) ** (-K / 2) / (M * math.sqrt(np.linalg.det(H))) + bias2
    assert mb.scv(mb._Pairs(data), H, G) == pytest.approx(want, rel=1e-8)


# ---------------------------------------------------------------------------
# SCV pilot against ks:::gdscalar(x, d, r = 0, n, nstage = 1, scv = TRUE)

def test_scv_pilot_matches_ks():
    x = _sample60()
    M, K = x.shape
    xs = x / x.std(axis=0, ddof=1)             # ks's pre.scale
    assert mb.scv_pilot(np.cov(xs, rowvar=False), M) == pytest.approx(
        0.46408132544354, rel=1e-12)
    assert mb.scv_pilot(np.eye(K), M) == pytest.approx(0.697759996860378,
                                                       rel=1e-12)
    for K, want in [(2, 0.644501760416292), (3, 0.666097799946016),
                    (4, 0.685258343772666), (5, 0.702335605412378)]:
        assert mb.scv_pilot(np.eye(K), 100) == pytest.approx(want, rel=1e-12)


# ---------------------------------------------------------------------------
# selectors against ks (Hlscv, Hlscv.diag, Hscv(nstage = 1,
# pilot = "dscalar", pre = "sphere"), Hscv.diag(nstage = 1,
# pilot = "dscalar", pre = "scale"), all binned = FALSE).  ks minimizes
# with optim's default tolerances and uses the ddof=1 covariance in the
# pilot, so agreement is to about a percent; the criterion value at
# funcsim's matrix is also required to be no worse than at ks's

KS_SELECTORS = {
    ("lscv", True, 60): [[0.3664185, 0.1998472, -0.1205760],
                         [0.1998472, 0.8853755, 0.2179725],
                         [-0.1205760, 0.2179725, 0.2158659]],
    ("lscv", False, 60): np.diag([0.1688861, 0.4314384, 0.1359491]),
    ("scv", True, 60): [[0.4007749, 0.1913568, -0.1443194],
                        [0.1913568, 0.8040257, 0.1836128],
                        [-0.1443194, 0.1836128, 0.2461130]],
    ("scv", False, 60): np.diag([0.2431097, 0.499319, 0.1396867]),
    ("lscv", True, 200): [[0.3170455, 0.1156631], [0.1156631, 0.2490551]],
    ("lscv", False, 200): np.diag([0.2964176, 0.2403874]),
    ("scv", True, 200): [[0.28870347, 0.09201616],
                         [0.09201616, 0.43332275]],
    ("scv", False, 200): np.diag([0.2714509, 0.4083515]),
}


@pytest.mark.parametrize("family, full, M", list(KS_SELECTORS))
def test_selectors_agree_with_ks(family, full, M):
    x = _sample60() if M == 60 else _sample200()
    want = np.asarray(KS_SELECTORS[(family, full, M)], dtype=float)
    kde = fs.MvKde(x, bw=family, full=full)
    got = kde.bandwidth.values
    scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
    assert np.max(np.abs(got - want) / scale) < 0.03
    # at least as good a minimum as ks's, by funcsim's own criterion in
    # the data's units (the pilot, for scv, as funcsim computes it)
    pairs = mb._Pairs(x)
    if family == "lscv":
        crit = lambda H: mb.lscv(pairs, H)
    else:
        s = x.std(axis=0)
        S0 = np.cov(x, rowvar=False, bias=True)
        if full:
            G = mb.scv_pilot(np.eye(x.shape[1]), M) ** 2 * S0
        else:
            R = S0 / np.outer(s, s)
            G = mb.scv_pilot(R, M) ** 2 * np.diag(s ** 2)
        crit = lambda H: mb.scv(pairs, H, G)
    assert crit(got) <= crit(want) + 1e-6 * abs(crit(want))


def test_selector_result_equals_the_same_matrix():
    x = _sample200()
    kde = fs.MvKde(x, bw="scv", full=True)
    same = fs.MvKde(x, bw=kde.bandwidth)
    np.testing.assert_allclose(same.bandwidth.values, kde.bandwidth.values,
                               rtol=1e-12)


def test_collapse_warning(monkeypatch):
    # many exact duplicates: LSCV collapses toward zero bandwidth
    base = np.random.default_rng(1).normal(size=(12, 2))
    x = np.repeat(base, 10, axis=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fs.MvKde(x, bw="lscv")
    messages = [str(w.message) for w in caught]
    assert any("tied" in m for m in messages)


# ---------------------------------------------------------------------------
# sample-size checks by family and structure

def test_hard_minimums():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="at least two"):
        fs.MvKde(rng.normal(size=(1, 2)))
    with pytest.raises(ValueError, match="singular"):
        fs.MvKde(rng.normal(size=(3, 3)), bw="scott", full=True)
    fs.MvKde(rng.normal(size=(4, 3)), bw="scott", full=True)   # M > K
    with pytest.raises(ValueError, match="bw='lscv'"):
        fs.MvKde(rng.normal(size=(4, 3)), bw="lscv")           # p = 3
    with pytest.raises(ValueError, match="bw='scv'"):
        fs.MvKde(rng.normal(size=(7, 3)), bw="scv", full=True)  # p = 6
    with pytest.raises(ValueError):
        fs.Kde([1.0])
    with pytest.raises(ValueError):
        fs.Kde([1.0, 2.0], bw="sj")
    # a user matrix only needs two observations
    fs.MvKde(rng.normal(size=(2, 3)), bw=np.eye(3))


def test_soft_minimums_warn_by_family_and_structure():
    rng = np.random.default_rng(3)

    def caught(data, **kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fs.MvKde(data, **kw)
        return [str(m.message) for m in w if "recommended" in str(m.message)]

    # diagonal rule: Silverman's table (67 for K = 3)
    assert caught(rng.normal(size=(50, 3)), bw="scott")
    assert not caught(rng.normal(size=(70, 3)), bw="scott")
    # full rule: at least ten per variable, here the table still binds
    assert caught(rng.normal(size=(50, 3)), bw="scott", full=True)
    # selector, full: ten per parameter (60 for K = 3) and the table (67)
    assert caught(rng.normal(size=(65, 3)), bw="scv", full=True)
    assert not caught(rng.normal(size=(70, 3)), bw="scv", full=True)
    # K = 4: table says 223
    assert caught(rng.normal(size=(100, 4)), bw="silverman")
    assert not caught(rng.normal(size=(230, 4)), bw="silverman")
    # univariate: 4 for a rule, 10 for a selector
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fs.Kde(rng.normal(size=3))
        fs.Kde(rng.normal(size=8), bw="sj")
        fs.Kde(rng.normal(size=12), bw="sj")
    assert sum("recommended" in str(m.message) for m in w) == 2


def test_table_and_beyond_ten_dimensions():
    assert SILVERMAN_TABLE[1] == 4 and SILVERMAN_TABLE[10] == 842_000
    with pytest.warns(UserWarning, match="beyond 10 variables"):
        check_sample_size(10 ** 6, 11, "scott", False)
