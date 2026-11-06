"""
Sheather-Jones plug-in bandwidth, bw='sj', added after 0.2.9: see
kde-plan-2026-11-06.md, step 3.
"""
import numpy as np
import pytest
from scipy import stats

import funcsim as fs
from funcsim import kde as kde_module
from funcsim.kde import _centers, _psi, _sj_bandwidth


@pytest.mark.parametrize("r", [4, 6])
def test_functional_matches_double_loop(r):
    data = np.random.default_rng(0).normal(size=30)
    g = 0.4
    z = (data[:, None] - data[None, :]) / g
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    if r == 4:
        term = (z ** 4 - 6 * z ** 2 + 3) * phi
    else:
        term = (z ** 6 - 15 * z ** 4 + 45 * z ** 2 - 15) * phi
    want = term.sum() / (30 * 29 * g ** (r + 1))
    assert abs(_psi(r, g, data) / want - 1.0) < 1e-12


# R reference values, R 4.6.1, from samples written to CSV with 17
# significant digits and
#     bw.SJ(x, nb = 1e6, method = "ste", tol = 1e-10)
#     bw.SJ(x, nb = 1e6, method = "dpi")
# R bins pairwise distances into nb bins, which limits agreement to a few
# parts per million
R_REFERENCE = [
    ("normal100", lambda: np.random.default_rng(20).normal(size=100),
     0.4744476144, 0.4738359295),
    ("lognormal500", lambda: stats.lognorm(s=0.6).rvs(500, random_state=21),
     0.1196520013, 0.1279796508),
    ("bimodal2000", lambda: np.concatenate(
        [np.random.default_rng(22).normal(-2, 1, 1000),
         np.random.default_rng(23).normal(3, 0.7, 1000)]),
     0.2296159238, 0.2385946269),
]


@pytest.mark.parametrize("name, sample, ste, dpi", R_REFERENCE)
def test_agrees_with_r_bw_sj(name, sample, ste, dpi):
    data = sample()
    assert abs(fs.Kde(data, bw="sj").bandwidth / ste - 1.0) < 2e-5
    assert abs(_sj_bandwidth(data, None, None, "dpi") / dpi - 1.0) < 2e-5


def test_normal_sample_near_normal_reference():
    data = np.random.default_rng(2).normal(size=1000)
    h = fs.Kde(data, bw="sj").bandwidth
    assert abs(h / (1.06 * np.std(data, ddof=1) * 1000 ** -0.2) - 1.0) < 0.2


def test_folded_normal_with_bound():
    # for Z ~ N(0, 1), int (f'')^2 over [0, inf) for |Z| is twice that of
    # the normal density, so the bounded plug-in bandwidth of |Z| should be
    # about 2^(-1/5) = 0.87 times the unbounded bandwidth of Z
    Z = np.random.default_rng(1).normal(size=2000)
    h_z = fs.Kde(Z, bw="sj").bandwidth
    h_abs = fs.Kde(np.abs(Z), bw="sj", lowerBound=0.0).bandwidth
    assert abs(h_abs / h_z / 2.0 ** -0.2 - 1.0) < 0.1


def test_reflected_centers_of_folded_sample():
    # the centers of |Z| with a lower bound at 0 are exactly Z and -Z, so
    # the functional estimate is that of the symmetrized sample
    Z = np.random.default_rng(3).normal(size=200)
    centers = _centers(np.abs(Z), 0.0, None)
    assert np.allclose(np.sort(centers), np.sort(np.concatenate([Z, -Z])))
    assert _psi(4, 0.3, centers) == pytest.approx(
        _psi(4, 0.3, np.concatenate([Z, -Z])), rel=1e-12)


def test_bounded_exponential_is_not_undersmoothed():
    # the unbounded plug-in rule sees the edge at zero as curvature and
    # halves the bandwidth; the reflected heuristic is close to the
    # cross-validated choice for the reflected estimate
    data = np.random.default_rng(3).exponential(size=500)
    h_plain = fs.Kde(data, bw="sj").bandwidth
    h_bound = fs.Kde(data, bw="sj", lowerBound=0.0).bandwidth
    h_lscv = fs.Kde(data, bw="lscv", lowerBound=0.0).bandwidth
    assert h_bound > 1.5 * h_plain
    assert abs(h_bound / h_lscv - 1.0) < 0.25


def test_degenerate_samples_raise():
    with pytest.raises(ValueError, match="interquartile"):
        fs.Kde(np.repeat(1.0, 50), bw="sj")
    with pytest.raises(ValueError, match="interquartile"):
        # zero IQR from ties, though the sd is positive
        fs.Kde(np.concatenate([np.zeros(40), [1.0, 2.0]]), bw="sj")
    with pytest.raises(ValueError):
        fs.Kde([1.0, 2.0], bw="sj")


def test_no_root_in_bracket_raises(monkeypatch):
    # two tight clusters far apart: the root lies well below a tenth of
    # the normal-reference bandwidth, so without widening there is none
    data = np.concatenate([np.random.default_rng(4).normal(0, 0.01, 100),
                           np.random.default_rng(5).normal(100, 0.01, 100)])
    assert fs.Kde(data, bw="sj").bandwidth < 1.0     # found by widening
    monkeypatch.setattr(kde_module, "_SJ_MAX_WIDEN", 0)
    with pytest.raises(ValueError, match="no solution"):
        fs.Kde(data, bw="sj")


def test_result_equals_float_bandwidth():
    data = np.random.default_rng(6).exponential(size=200)
    kde = fs.Kde(data, bw="sj", lowerBound=0.0)
    same = fs.Kde(data, bw=kde.bandwidth, lowerBound=0.0)
    x = np.linspace(0.0, 5.0, 51)
    np.testing.assert_allclose(kde.pdf(x), same.pdf(x), rtol=1e-12)
    np.testing.assert_allclose(kde.cdf(x), same.cdf(x), rtol=1e-12)
    np.testing.assert_allclose(kde.ppf([0.01, 0.5, 0.99]),
                               same.ppf([0.01, 0.5, 0.99]), rtol=1e-12)


def test_unknown_method_message_lists_sj():
    with pytest.raises(ValueError, match="'sj'"):
        fs.Kde(np.random.default_rng(7).normal(size=20), bw="plugin")
