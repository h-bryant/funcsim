"""
Least-squares cross-validated bandwidth, bw='lscv', added after 0.2.9: see
kde-plan-2026-11-06.md, step 2.
"""
import warnings

import numpy as np
import pytest
from scipy import integrate, stats

import funcsim as fs
from funcsim.kde import _centers, _lscv_criterion


def _brute_force(h, data, lower, upper):
    # int f^2 by quadrature over the support, and the leave-one-out term
    # by refitting without each observation, both through the public API.
    # The criterion is defined for the unnormalized reflected estimate, so
    # the public pdf is multiplied back by its two-bound mass Z (1 exactly
    # with at most one bound)
    M = data.size
    kde = fs.Kde(data, bw=h, lowerBound=lower, upperBound=upper)
    a = lower if lower is not None else data.min() - 10.0 * h
    b = upper if upper is not None else data.max() + 10.0 * h
    sq, _ = integrate.quad(lambda x: (kde._Z * kde.pdf(x)) ** 2, a, b,
                           limit=400, epsabs=1e-13, epsrel=1e-12)
    loo = 0.0
    for i in range(M):
        other = fs.Kde(np.delete(data, i), bw=h, lowerBound=lower,
                       upperBound=upper)
        loo += other._Z * other.pdf(data[i])
    return sq - 2.0 / M * loo


@pytest.mark.parametrize("lower, upper", [(None, None), (0.0, None),
                                          (None, 3.0), (0.0, 3.0)])
@pytest.mark.parametrize("h", [0.1, 0.3, 0.8])
def test_closed_form_matches_brute_force(lower, upper, h):
    data = np.random.default_rng(0).random(40) * 3.0
    centers = _centers(data, lower, upper)
    got = _lscv_criterion(h, data, centers, lower, upper)
    want = _brute_force(h, data, lower, upper)
    assert abs(got - want) < 1e-10


def test_normal_sample_near_scott():
    data = np.random.default_rng(1).normal(size=500)
    kde = fs.Kde(data, bw="lscv")
    ratio = kde.bandwidth / fs.Kde(data).bandwidth
    assert 0.5 < ratio < 2.0


def test_selected_bandwidth_is_a_local_minimum():
    data = stats.lognorm(s=0.6).rvs(300, random_state=2)
    for lower in (None, 0.0):
        kde = fs.Kde(data, bw="lscv", lowerBound=lower)
        h = kde.bandwidth
        centers = _centers(data, lower, None)
        crit = lambda hh: _lscv_criterion(hh, data, centers, lower, None)
        assert crit(h) <= crit(h * 1.001) and crit(h) <= crit(h / 1.001)


# R reference values, R 4.6.1, from samples written to CSV with 17
# significant digits and
#     hmax <- 1.144 * sd(x) * n^(-1/5)
#     bw.ucv(x, nb = 1e6, lower = 0.05 * hmax, upper = 4 * hmax, tol = 1e-10)
# R's criterion is the Scott-Terrell form with 1 / n^2 in both terms,
# while funcsim's leave-one-out term has 1 / (n (n - 1)), so the two agree
# only to order 1 / n (observed: -1.2%, -0.28%, -0.16%)
R_REFERENCE = [
    ("normal100", lambda: np.random.default_rng(20).normal(size=100),
     0.5773801011, 0.02),
    ("lognormal500", lambda: stats.lognorm(s=0.6).rvs(500, random_state=21),
     0.1226727164, 0.005),
    ("bimodal2000", lambda: np.concatenate(
        [np.random.default_rng(22).normal(-2, 1, 1000),
         np.random.default_rng(23).normal(3, 0.7, 1000)]),
     0.2484238140, 0.003),
]


@pytest.mark.parametrize("name, sample, want, rtol", R_REFERENCE)
def test_agrees_with_r_bw_ucv(name, sample, want, rtol):
    kde = fs.Kde(sample(), bw="lscv")
    assert abs(kde.bandwidth / want - 1.0) < rtol


def test_bounded_selection_differs_from_unbounded():
    # with a sharp edge at zero the unbounded criterion undersmooths
    data = np.random.default_rng(3).exponential(size=500)
    bounded = fs.Kde(data, bw="lscv", lowerBound=0.0).bandwidth
    plain = fs.Kde(data, bw="lscv").bandwidth
    assert bounded > plain


def test_ties_warn():
    data = np.round(np.random.default_rng(4).normal(size=300), 1)
    with pytest.warns(UserWarning, match="tied"):
        fs.Kde(data, bw="lscv")


def test_heavy_ties_warn_at_lower_end():
    # with many exact duplicates the criterion goes to minus infinity as
    # h goes to zero, so the grid's lower end is returned, with warnings
    data = np.repeat(np.random.default_rng(5).normal(size=20), 10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kde = fs.Kde(data, bw="lscv")
    messages = [str(w.message) for w in caught]
    assert any("tied" in m for m in messages)
    assert any("lower end" in m for m in messages)
    assert kde.bandwidth == pytest.approx(fs.Kde(data).bandwidth / 20.0)


def test_result_equals_float_bandwidth():
    data = np.random.default_rng(6).exponential(size=200)
    kde = fs.Kde(data, bw="lscv", lowerBound=0.0)
    same = fs.Kde(data, bw=kde.bandwidth, lowerBound=0.0)
    x = np.linspace(0.0, 5.0, 51)
    np.testing.assert_allclose(kde.pdf(x), same.pdf(x), rtol=1e-12)
    np.testing.assert_allclose(kde.cdf(x), same.cdf(x), rtol=1e-12)
    u = np.array([0.01, 0.5, 0.99])
    np.testing.assert_allclose(kde.ppf(u), same.ppf(u), rtol=1e-12)


def test_too_few_observations():
    with pytest.raises(ValueError):
        fs.Kde([1.0, 2.0], bw="lscv")
