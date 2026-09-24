"""
Item 5 of the 2026-09-24 course handoff: Kde.ppf served from a monotone
inverse-CDF table (with root finding as fallback and opt-out), and Python
float returns for scalar arguments of pdf, cdf, and ppf.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import funcsim as fs


U_EXTREME = np.array([1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.5, 0.95,
                      0.99, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 1 - 1e-7])


def _reference_ppf(kde, u):
    # tightly converged Brent root finding on the exact CDF
    return np.array([optimize.brentq(lambda x: kde.cdf(x) - uu, kde.ppf_low,
                                     kde.ppf_high, xtol=1e-14, rtol=8.9e-16,
                                     maxiter=500) for uu in u])


@pytest.mark.parametrize("name, data", [
    ("lognormal", stats.lognorm(s=0.8).rvs(500, random_state=1)),
    ("bimodal", np.concatenate([np.random.default_rng(2).normal(-5, 1, 150),
                                np.random.default_rng(3).normal(5, 0.5, 150)])),
    ("tiny_scale", 1e-6 * stats.gamma(2).rvs(400, random_state=4)),
    ("n20", np.random.default_rng(5).normal(size=20)),
])
def test_table_matches_root_finder_within_1e6_of_range(name, data):
    kde = fs.Kde(data)
    u = np.concatenate([U_EXTREME, np.random.default_rng(6).uniform(size=150)])
    got = kde.ppf(u)
    ref = _reference_ppf(kde, u)
    R = data.max() - data.min()
    assert np.max(np.abs(got - ref)) / R < 1e-6
    # and in fact far better than the documented target
    assert np.max(np.abs(got - ref)) / R < 1e-7


def test_return_types_and_shapes():
    data = np.random.default_rng(0).uniform(size=300)
    kde = fs.Kde(data)
    for method, arg in ((kde.pdf, 0.5), (kde.cdf, 0.5), (kde.ppf, 0.5)):
        out = method(arg)
        assert type(out) is float
        out = method(np.float64(arg))
        assert type(out) is float
        arr = method(np.array([0.2, 0.5, 0.8]))
        assert isinstance(arr, np.ndarray) and arr.shape == (3,)
        arr2 = method(np.array([[0.2, 0.5], [0.6, 0.8]]))
        assert arr2.shape == (2, 2)
        ser = method(pd.Series([0.2, 0.5, 0.8]))
        assert isinstance(ser, np.ndarray) and ser.shape == (3,)
        lst = method([0.2, 0.5])
        assert isinstance(lst, np.ndarray) and lst.shape == (2,)


def test_monotone_and_round_trip():
    data = stats.lognorm(s=0.8).rvs(500, random_state=1)
    kde = fs.Kde(data)
    u = np.sort(np.random.default_rng(7).uniform(size=2000))
    x = kde.ppf(u)
    assert np.all(np.diff(x) >= 0.0)
    np.testing.assert_allclose(kde.cdf(x), u, atol=1e-9)
    # single values agree with the vectorized evaluation
    assert kde.ppf(u[10]) == x[10]


def test_exact_ppf_reproduces_the_root_finder():
    data = stats.lognorm(s=0.8).rvs(300, random_state=8)
    exact = fs.Kde(data, exact_ppf=True)
    table = fs.Kde(data)
    u = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    want = np.array([optimize.brentq(lambda x: exact.cdf(x) - uu,
                                     exact.ppf_low, exact.ppf_high,
                                     xtol=exact.ppf_xtol) for uu in u])
    np.testing.assert_array_equal(exact.ppf(u), want)
    assert type(exact.ppf(0.5)) is float
    # the table agrees with it to the root finder's own tolerance scale
    np.testing.assert_allclose(table.ppf(u), want,
                               atol=1e-6 * (data.max() - data.min()))


def test_bounds_and_invalid_input():
    data = np.random.default_rng(9).normal(size=200)
    kde = fs.Kde(data)
    # 0 and 1 lie outside the table and keep the pre-0.2.7 fallbacks
    assert kde.ppf(0.0) == kde.ppf_low
    assert kde.ppf(1.0) == kde.ppf_high
    for bad in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            kde.ppf(bad)
    with pytest.raises(ValueError):
        kde.ppf(np.array([0.5, 1.5]))
    # extreme but representable probabilities are served (finite, ordered)
    x = kde.ppf(np.array([1e-12, 1e-9, 0.5, 1 - 1e-9]))
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)


def test_cdf_values_unchanged_and_pdf_matches_scipy():
    data = np.random.default_rng(10).uniform(size=500)
    kde = fs.Kde(data)
    x = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(
        kde.cdf(x),
        [kde.gkde.integrate_box_1d(-np.inf, v) for v in x])
    np.testing.assert_allclose(kde.pdf(x), kde.gkde(x))
    assert kde.cdf(0.5) == kde.gkde.integrate_box_1d(-np.inf, 0.5)


def test_table_is_built_lazily_and_once():
    kde = fs.Kde(np.random.default_rng(11).normal(size=100))
    assert kde._table is None
    kde.cdf(0.3)
    assert kde._table is None
    kde.ppf(0.3)
    first = kde._table
    kde.ppf(np.array([0.2, 0.7]))
    assert kde._table is first
