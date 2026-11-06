"""
Bounded support for Kde by reflection (lowerBound / upperBound), added in
0.2.10: see kde-plan-2026-11-06.md, step 1.
"""
import numpy as np
import pytest
from scipy import integrate, optimize, stats

import funcsim as fs


RNG = np.random.default_rng(2026)


def _reference_ppf(kde, u):
    # tightly converged Brent root finding on the exact CDF
    R = max(1.0, kde.ppf_high - kde.ppf_low)
    return np.array([optimize.brentq(lambda x: kde.cdf(x) - uu, kde.ppf_low,
                                     kde.ppf_high, xtol=1e-14 * R,
                                     rtol=8.9e-16, maxiter=500) for uu in u])


CASES = {
    # name: (data, lowerBound, upperBound)
    "halfnormal": (np.abs(RNG.normal(size=2000)), 0.0, None),
    "exponential": (RNG.exponential(size=500), 0.0, None),
    "uniform": (RNG.random(300), 0.0, 1.0),
    "beta_on_bound": (np.concatenate([[0.0] * 5, RNG.beta(0.5, 2, 400)]),
                      0.0, 1.0),
    "beta22": (RNG.beta(2, 2, 400), 0.0, 1.0),
    "upper_only": (-np.abs(RNG.normal(size=300)), None, 0.0),
    "tiny_scale": (1e-6 * RNG.exponential(size=300), 0.0, None),
    "shifted": (5.0 + RNG.exponential(size=200), 5.0, None),
}


@pytest.fixture(params=list(CASES))
def case(request):
    data, lo, hi = CASES[request.param]
    return fs.Kde(data, lowerBound=lo, upperBound=hi), data, lo, hi


def test_ends_of_support(case):
    kde, data, lo, hi = case
    if lo is not None:
        assert kde.cdf(lo) == 0.0
        assert kde.ppf(0.0) == lo
        assert kde.pdf(lo - 1e-9) == 0.0 and kde.cdf(lo - 1e-9) == 0.0
        assert kde.pdf(lo) > 0.0
    if hi is not None:
        assert kde.cdf(hi) == 1.0
        assert kde.ppf(1.0) == hi
        assert kde.pdf(hi + 1e-9) == 0.0 and kde.cdf(hi + 1e-9) == 1.0
    assert kde.lowerBound == lo and kde.upperBound == hi


def test_mass_is_one(case):
    kde, data, lo, hi = case
    a = lo if lo is not None else data.min() - 10 * kde.bandwidth
    b = hi if hi is not None else data.max() + 10 * kde.bandwidth
    assert abs(kde.cdf(b) - kde.cdf(a) - 1.0) < 1e-12
    mass, err = integrate.quad(kde.pdf, a, b, limit=400)
    assert abs(mass - 1.0) < 1e-8


def test_table_matches_root_finder(case):
    kde, data, lo, hi = case
    low_u = 10.0 ** -np.arange(2, 13) if lo is not None \
        else 10.0 ** -np.arange(2, 10)
    high_u = 1 - 10.0 ** -np.arange(2, 13) if hi is not None \
        else 1 - 10.0 ** -np.arange(2, 10)
    u = np.concatenate([low_u, high_u, np.linspace(0.0005, 0.9995, 200)])
    got = kde.ppf(u)
    ref = _reference_ppf(kde, u)
    R = data.max() - data.min()
    assert np.max(np.abs(got - ref)) / R < 1e-7
    # all served from the table except, at an unbounded end, the most
    # extreme probabilities
    assert np.sum(~np.isfinite(kde._table(u))) <= 2


def test_monotone_and_round_trip(case):
    kde, data, lo, hi = case
    u = np.sort(RNG.random(5000))
    x = kde.ppf(u)
    assert np.all(np.diff(x) >= 0.0)
    # the table is accurate to about 1e-8 of the data range near a bound,
    # which a peaked density turns into a larger error in u
    np.testing.assert_allclose(kde.cdf(x), u, atol=1e-6)
    assert kde.ppf(u[10]) == x[10]


def test_exact_ppf_with_bounds():
    data, lo, hi = CASES["uniform"]
    exact = fs.Kde(data, lowerBound=lo, upperBound=hi, exact_ppf=True)
    assert exact.ppf(0.0) == 0.0 and exact.ppf(1.0) == 1.0
    u = np.array([0.001, 0.3, 0.999])
    np.testing.assert_allclose(exact.ppf(u), _reference_ppf(exact, u),
                               atol=1e-8)


def test_uniform_is_nearly_flat():
    # reflection about both ends of a uniform sample gives an almost
    # exactly flat estimate; the unbounded KDE is near 0.5 at the ends
    data = np.random.default_rng(3).random(2000)
    kde = fs.Kde(data, lowerBound=0.0, upperBound=1.0)
    x = np.linspace(0.001, 0.999, 999)
    assert np.all(np.abs(kde.pdf(x) - 1.0) < 0.12)
    plain = fs.Kde(data)
    assert plain.pdf(0.0) < 0.6 and plain.pdf(1.0) < 0.6


def test_half_normal_at_the_bound():
    data = np.abs(np.random.default_rng(4).normal(size=5000))
    kde = fs.Kde(data, lowerBound=0.0)
    assert abs(kde.pdf(0.0) - 2.0 * stats.norm.pdf(0.0)) < 0.05
    assert abs(kde.cdf(1.0) - (2.0 * stats.norm.cdf(1.0) - 1.0)) < 0.02


def test_matches_augmented_sample_construction():
    # with a lower bound the estimate equals twice the unbounded KDE of
    # the sample augmented by its mirror image, at the same bandwidth
    data = np.abs(np.random.default_rng(5).normal(size=500))
    kde = fs.Kde(data, lowerBound=0.0)
    aug = np.concatenate([data, -data])
    g = stats.gaussian_kde(aug, bw_method=kde.bandwidth / np.std(aug, ddof=1))
    x = np.linspace(0.0, 3.0, 301)
    np.testing.assert_allclose(kde.pdf(x), 2.0 * g(x), rtol=1e-10)
    np.testing.assert_allclose(
        kde.cdf(x), [2.0 * g.integrate_box_1d(-np.inf, v) - 1.0 for v in x],
        atol=1e-12)


def test_two_bounds_normalization_on_a_narrow_interval():
    # interval narrower than eight bandwidths: the omitted higher-order
    # images matter, and the normalization keeps the mass at one
    data = 0.1 * np.random.default_rng(6).random(30)
    kde = fs.Kde(data, lowerBound=0.0, upperBound=0.1, bw=0.05)
    assert kde._Z < 1.0 - 1e-6
    mass, _ = integrate.quad(kde.pdf, 0.0, 0.1)
    assert abs(mass - 1.0) < 1e-10
    assert kde.cdf(0.1) == 1.0 and kde.cdf(0.0) == 0.0


def test_bandwidth_unchanged_by_bounds():
    data = np.random.default_rng(7).exponential(size=300)
    for bw in ("scott", "silverman", 0.2):
        assert fs.Kde(data, bw=bw, lowerBound=0.0).bandwidth == \
            fs.Kde(data, bw=bw).bandwidth


def test_unbounded_is_unchanged():
    data = np.random.default_rng(8).normal(size=300)
    kde = fs.Kde(data)
    assert kde.lowerBound is None and kde.upperBound is None
    assert kde.ppf(0.0) == kde.ppf_low and kde.ppf(1.0) == kde.ppf_high
    assert kde.pdf(data.min() - 100.0) >= 0.0


def test_validation():
    data = np.random.default_rng(9).random(50)
    with pytest.raises(TypeError):
        fs.Kde(data, lowerBound=True)
    with pytest.raises(TypeError):
        fs.Kde(data, upperBound="0")
    with pytest.raises(ValueError):
        fs.Kde(data, lowerBound=np.inf)
    with pytest.raises(ValueError):
        fs.Kde(data, lowerBound=0.5, upperBound=0.5)
    with pytest.raises(ValueError):
        fs.Kde(data, lowerBound=0.5, upperBound=0.2)
    with pytest.raises(ValueError):
        fs.Kde(data, lowerBound=data.min() + 1e-9)
    with pytest.raises(ValueError):
        fs.Kde(data, upperBound=data.max() - 1e-9)
    # data exactly on a bound is accepted
    kde = fs.Kde(data, lowerBound=data.min(), upperBound=data.max())
    assert kde.cdf(data.min()) == 0.0
    # bw validation
    with pytest.raises(ValueError):
        fs.Kde(data, bw="gauss")
    with pytest.raises(ValueError):
        fs.Kde(data, bw=-1.0)
    with pytest.raises(TypeError):
        fs.Kde(data, bw=True)


def test_return_types_with_bounds():
    data, lo, hi = CASES["uniform"]
    kde = fs.Kde(data, lowerBound=lo, upperBound=hi)
    for method in (kde.pdf, kde.cdf, kde.ppf):
        assert type(method(0.5)) is float
        assert method(np.array([[0.2, 0.5], [0.6, 0.8]])).shape == (2, 2)
