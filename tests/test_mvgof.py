"""Tests of the two-sample energy and MMD goodness-of-fit tests."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist, pdist

import funcsim as fs
from funcsim.mvgof import _chi2_pvalue, _prepare


rng = np.random.default_rng(12345)
X = rng.normal(size=(60, 2))
Y_SAME = rng.normal(size=(90, 2))
Y_SHIFT = rng.normal(size=(90, 2)) + np.array([1.5, 0.0])


def energy_direct(x, y, unbiased=True):
    m, n = len(x), len(y)
    dxy = cdist(x, y).mean()
    sxx, syy = 2.0 * pdist(x).sum(), 2.0 * pdist(y).sum()
    if unbiased:
        return 2 * dxy - sxx / (m * (m - 1)) - syy / (n * (n - 1))
    return 2 * dxy - sxx / (m * m) - syy / (n * n)


def mmd_direct(x, y, h, unbiased=True):
    m, n = len(x), len(y)
    k = lambda a, b: np.exp(-cdist(a, b) ** 2 / (2 * h * h))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    if unbiased:
        return ((kxx.sum() - m) / (m * (m - 1)) + (kyy.sum() - n) / (n * (n - 1))
                - 2 * kxy.mean())
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def test_energy_statistic_matches_formula():
    for unbiased in (True, False):
        res = fs.energy(X, Y_SHIFT, reps=10, standardize=False,
                            unbiased=unbiased)
        assert res.statistic == pytest.approx(
            energy_direct(X, Y_SHIFT, unbiased), rel=1e-12)


def test_mmd_statistic_matches_formula():
    h = 0.8
    for unbiased in (True, False):
        res = fs.mmd(X, Y_SHIFT, reps=10, standardize=False,
                         unbiased=unbiased, bandwidth=h)
        assert res.statistic == pytest.approx(
            mmd_direct(X, Y_SHIFT, h, unbiased), rel=1e-12)


def test_median_bandwidth_default():
    pooled = np.vstack([X, Y_SHIFT])
    h = np.median(pdist(pooled))
    default = fs.mmd(X, Y_SHIFT, reps=10, standardize=False)
    explicit = fs.mmd(X, Y_SHIFT, reps=10, standardize=False, bandwidth=h)
    assert default.statistic == pytest.approx(explicit.statistic, rel=1e-12)


def test_biased_statistics_nonnegative():
    for f in (fs.energy, fs.mmd):
        assert f(X, Y_SAME, reps=10, unbiased=False).statistic >= 0.0


def test_detects_shift_and_accepts_same():
    for f in (fs.energy, fs.mmd):
        assert f(X, Y_SHIFT, reps=500, seed=1).pvalue < 0.01
        assert f(X, Y_SAME, reps=500, seed=1).pvalue > 0.05


def test_chi2_method_agrees_with_permutation():
    for f in (fs.energy, fs.mmd):
        perm = f(X, Y_SHIFT, reps=2000, seed=2)
        chi = f(X, Y_SHIFT, method="chi2")
        assert chi.statistic == perm.statistic
        assert chi.pvalue < 0.01
        # under the null the approximation is conservative: it returns 1.0
        # whenever the unbiased statistic is negative
        chi = f(X, Y_SAME, method="chi2")
        assert chi.pvalue > 0.05


def test_seed_reproducible():
    a = fs.energy(X, Y_SAME, reps=200, seed=7)
    b = fs.energy(X, Y_SAME, reps=200, seed=7)
    c = fs.energy(X, Y_SAME, reps=200, seed=8)
    assert a == b
    assert a.pvalue != c.pvalue


def test_pvalue_bounds():
    res = fs.energy(X, Y_SAME, reps=9, seed=0)
    assert 1.0 / 10 <= res.pvalue <= 1.0


def test_shift_invariance_without_standardizing():
    shift = np.array([100.0, -50.0])
    a = fs.energy(X, Y_SHIFT, method="chi2", standardize=False)
    b = fs.energy(X + shift, Y_SHIFT + shift, method="chi2",
                      standardize=False)
    assert a.statistic == pytest.approx(b.statistic, rel=1e-9)
    assert a.pvalue == pytest.approx(b.pvalue, rel=1e-9)


def test_pooled_standardization_is_affine_invariant():
    # rescaling one variable in both samples changes nothing when the
    # pooled standardization is on
    scale = np.array([1000.0, 0.01])
    a = fs.mmd(X, Y_SHIFT, method="chi2")
    b = fs.mmd(X * scale, Y_SHIFT * scale, method="chi2")
    assert a.statistic == pytest.approx(b.statistic, rel=1e-9)
    assert a.pvalue == pytest.approx(b.pvalue, rel=1e-9)


def test_standardize_options():
    pooled, m = _prepare(X, Y_SHIFT, True)
    assert m == len(X)
    assert np.allclose(pooled.mean(axis=0), 0.0)
    assert np.allclose(pooled.std(axis=0, ddof=1), 1.0)
    first, _ = _prepare(X, Y_SHIFT, "first")
    assert np.allclose(first[:m].mean(axis=0), 0.0)
    assert np.allclose(first[:m].std(axis=0, ddof=1), 1.0)
    raw, _ = _prepare(X, Y_SHIFT, False)
    assert np.array_equal(raw, np.vstack([X, Y_SHIFT]))
    with pytest.raises(ValueError):
        _prepare(X, Y_SHIFT, "each")


def test_symmetric_in_arguments_with_pooled_standardization():
    a = fs.energy(X, Y_SHIFT, method="chi2")
    b = fs.energy(Y_SHIFT, X, method="chi2")
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
    assert a.pvalue == pytest.approx(b.pvalue, rel=1e-9)


def test_dataframe_and_vector_inputs():
    dx = pd.DataFrame(X, columns=["a", "b"])
    dy = pd.DataFrame(Y_SHIFT, columns=["a", "b"])
    a = fs.energy(dx, dy, method="chi2")
    b = fs.energy(X, Y_SHIFT, method="chi2")
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
    assert a.pvalue == pytest.approx(b.pvalue, rel=1e-9)
    one = fs.energy(X[:, 0], Y_SHIFT[:, 0], method="chi2")
    two = fs.energy(X[:, :1], Y_SHIFT[:, :1], method="chi2")
    assert one.statistic == pytest.approx(two.statistic, rel=1e-12)
    assert one.pvalue == pytest.approx(two.pvalue, rel=1e-9)


def test_input_errors():
    with pytest.raises(ValueError):
        fs.energy(X, Y_SHIFT[:, :1], reps=5)
    with pytest.raises(ValueError):
        fs.energy(X, np.vstack([Y_SHIFT, [[np.nan, 0.0]]]), reps=5)
    with pytest.raises(ValueError):
        fs.energy(X[:1], Y_SHIFT, reps=5)
    with pytest.raises(ValueError):
        fs.energy(X, Y_SHIFT, method="bootstrap")
    with pytest.raises(ValueError):
        fs.energy(X, Y_SHIFT, reps=0)
    with pytest.raises(ValueError):
        fs.mmd(X, Y_SHIFT, reps=5, bandwidth=0.0)
    const = np.hstack([X, np.ones((len(X), 1))])
    const_y = np.hstack([Y_SHIFT, np.ones((len(Y_SHIFT), 1))])
    with pytest.raises(ValueError):
        fs.energy(const, const_y, reps=5)
    assert fs.energy(const, const_y, reps=5, standardize=False).pvalue <= 1


def test_chi2_matches_unbiased_dcor_reference():
    # independent re-derivation of the chi-square p-value from the unbiased
    # distance correlation (Shen, Panda, and Vogelstein, 2022)
    from scipy.stats import chi2
    pooled = np.vstack([X, Y_SHIFT]); N = len(pooled); m = len(X)
    D = cdist(pooled, pooled)
    L = (np.arange(N)[:, None] >= m) != (np.arange(N)[None, :] >= m)
    L = L.astype(float)

    def uc(A):
        r = A.sum(1, keepdims=True) / (N - 2)
        c = A.sum(0, keepdims=True) / (N - 2)
        t = A.sum() / ((N - 1) * (N - 2))
        B = A - r - c + t
        np.fill_diagonal(B, 0.0)
        return B
    Dc, Lc = uc(D), uc(L)
    dcor = (Dc * Lc).sum() / np.sqrt((Dc * Dc).sum() * (Lc * Lc).sum())
    assert _chi2_pvalue(D, m) == pytest.approx(chi2.sf(dcor * N + 1, 1),
                                               rel=1e-12)


def test_against_hyppo_if_available():
    hyppo = pytest.importorskip("hyppo.ksample")
    e = hyppo.Energy().test(X, Y_SHIFT, auto=True)
    ours = fs.energy(X, Y_SHIFT, method="chi2", standardize=False)
    # hyppo's statistic is a rescaled unbiased distance covariance, which
    # differs from the distinct-pairs energy distance by O(1/N)
    assert ours.statistic == pytest.approx(e.stat, rel=0.05)
    assert ours.pvalue == pytest.approx(e.pvalue, rel=1e-6)
    mm = hyppo.MMD().test(X, Y_SHIFT, auto=True)
    ours = fs.mmd(X, Y_SHIFT, method="chi2", standardize=False)
    assert ours.pvalue == pytest.approx(mm.pvalue, rel=1e-6)
