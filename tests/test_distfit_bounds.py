"""
Tests for fixed support bounds in `fit` and `compare`, the deprecation of
the bool `lowerLimit`/`upperLimit` flags in `compare`, and the corrected
support categorization of the candidate table in `distfit`.
"""

import warnings

import numpy as np
import pytest
import scipy.stats as stats

import funcsim as fs
from funcsim import distfit


# ---------------------------------------------------------------------------
# data


@pytest.fixture(scope="module")
def expon_data():
    return np.random.default_rng(1).exponential(2.0, size=300)


@pytest.fixture(scope="module")
def beta_data():
    return np.random.default_rng(2).beta(2.0, 3.0, size=300)


@pytest.fixture(scope="module")
def wmax_data():
    return stats.weibull_max(1.5).rvs(size=300,
                                      random_state=np.random.default_rng(3))


@pytest.fixture(scope="module")
def pareto_data():
    return stats.pareto(3.0).rvs(size=300,
                                 random_state=np.random.default_rng(4))


@pytest.fixture(scope="module")
def norm_data():
    return np.random.default_rng(5).normal(size=200)


def _ll(dist, x):
    return float(dist.logpdf(x).sum())


def _future_warnings(records):
    return [str(r.message) for r in records
            if issubclass(r.category, FutureWarning)]


# ---------------------------------------------------------------------------
# fit: a fixed bound that pins a single scipy parameter reproduces scipy


@pytest.mark.parametrize("dist", [stats.expon, stats.gamma, stats.lognorm,
                                  stats.weibull_min])
def test_fit_lower_bound_matches_scipy_floc(dist, expon_data):
    res = fs.fit(expon_data, dist, lowerBound=0.0)
    assert np.allclose(res.dist.args, dist.fit(expon_data, floc=0))
    assert res.dist.support()[0] == 0.0


def test_fit_upper_bound_matches_scipy_floc(wmax_data):
    res = fs.fit(wmax_data, stats.weibull_max, upperBound=0.0)
    assert np.allclose(res.dist.args, stats.weibull_max.fit(wmax_data, floc=0))
    assert res.dist.support()[1] == 0.0


def test_fit_both_bounds_match_scipy_floc_fscale(beta_data):
    res = fs.fit(beta_data, stats.beta, lowerBound=0.0, upperBound=1.0)
    ref = stats.beta.fit(beta_data, floc=0, fscale=1)
    assert np.allclose(res.dist.args, ref)
    assert res.dist.support() == (0.0, 1.0)


def test_fit_both_bounds_without_shape_parameters(beta_data):
    # nothing left to estimate: the parameters follow from the bounds alone
    res = fs.fit(beta_data, stats.uniform, lowerBound=0.0, upperBound=1.0)
    assert res.dist.args == (0.0, 1.0)
    # zero free parameters and a log-density of zero everywhere
    assert res.aic == 0.0
    assert res.bic == 0.0


def test_fit_without_bounds_is_unchanged(beta_data):
    res = fs.fit(beta_data, stats.beta)
    ref = stats.beta.fit(beta_data)
    assert np.allclose(res.dist.args, ref)
    ll = _ll(res.dist, beta_data)
    assert np.isclose(res.aic, 2.0 * 4 - 2.0 * ll)


# ---------------------------------------------------------------------------
# fit: information criteria count only the free parameters


@pytest.mark.parametrize("kwargs, nfree", [
    ({}, 3),
    ({"lowerBound": 0.0}, 2),
    ({"upperBound": 1.0}, 2),
    ({"lowerBound": 0.0, "upperBound": 1.0}, 1),
])
def test_fit_information_criteria_use_free_parameter_count(beta_data, kwargs,
                                                           nfree):
    # triangular: one shape parameter plus loc and scale
    res = fs.fit(beta_data, stats.triang, **kwargs)
    ll = _ll(res.dist, beta_data)
    assert np.isclose(res.aic, 2.0 * nfree - 2.0 * ll)
    assert np.isclose(res.bic, np.log(len(beta_data)) * nfree - 2.0 * ll)


def test_fit_result_shape_unchanged(expon_data):
    res = fs.fit(expon_data, stats.expon, lowerBound=0.0)
    assert len(res) == 7
    assert res._fields == ("bic", "aic", "ad_pval", "cvm_pval", "dist",
                           "distName", "warnings")


# ---------------------------------------------------------------------------
# fit: bounds that couple loc and scale (not a single scipy parameter)


def test_fit_coupled_lower_bound_pareto(pareto_data):
    # standard support starts at 1, so fixing the lower bound at L means
    # loc = L - scale rather than loc = L
    res = fs.fit(pareto_data, stats.pareto, lowerBound=1.0)
    lo, hi = res.dist.support()
    assert np.isclose(lo, 1.0)
    assert hi == np.inf
    # the classic Pareto MLE (loc=0, scale=1) is nested in this family
    classic = stats.pareto(len(pareto_data) / np.log(pareto_data).sum(),
                           0.0, 1.0)
    assert _ll(res.dist, pareto_data) >= _ll(classic, pareto_data) - 1e-6


def test_fit_coupled_upper_bound_beta(beta_data):
    res = fs.fit(beta_data, stats.beta, upperBound=1.0)
    both = fs.fit(beta_data, stats.beta, lowerBound=0.0, upperBound=1.0)
    lo, hi = res.dist.support()
    assert np.isclose(hi, 1.0)
    assert abs(lo) < 0.1
    # the both-fixed fit is nested in the upper-only family
    assert _ll(res.dist, beta_data) >= _ll(both.dist, beta_data) - 1e-4


def test_fit_coupled_upper_bound_without_shape_parameters(beta_data):
    # uniform on [1 - scale, 1]: the MLE puts the lower end at min(data)
    res = fs.fit(beta_data, stats.uniform, upperBound=1.0)
    lo, hi = res.dist.support()
    assert np.isclose(hi, 1.0)
    assert abs(lo - beta_data.min()) < 2e-3


@pytest.mark.parametrize("dist", [
    stats.truncexpon, stats.genhalflogistic, stats.genpareto, stats.truncnorm,
    stats.tukeylambda, stats.kappa4, stats.truncpareto,
])
def test_fit_both_bounds_with_shape_dependent_support(beta_data, dist):
    res = fs.fit(beta_data, dist, lowerBound=0.0, upperBound=1.0)
    lo, hi = res.dist.support()
    assert np.isclose(lo, 0.0)
    assert np.isclose(hi, 1.0)
    assert np.isfinite(res.bic)
    assert len(res.warnings) == 0


def test_fit_shape_dependent_lower_bound_agrees_with_scipy(expon_data):
    # the lower end of genpareto is 0 for every shape, so scipy's floc fit
    # is the same constrained problem solved by a different route
    res = fs.fit(expon_data, stats.genpareto, lowerBound=0.0)
    ref = stats.genpareto(*stats.genpareto.fit(expon_data, floc=0))
    assert res.dist.support()[0] == 0.0
    assert abs(_ll(res.dist, expon_data) - _ll(ref, expon_data)) < 0.05


def test_fit_genextreme_one_sided_bounds():
    g = stats.genextreme(-0.3).rvs(size=300,
                                   random_state=np.random.default_rng(6))
    lower = float(g.min()) - 0.5
    res = fs.fit(g, stats.genextreme, lowerBound=lower)
    assert np.isclose(res.dist.support()[0], lower)
    assert res.dist.support()[1] == np.inf
    # an upper bound needs the opposite sign of the shape parameter from the
    # one scipy's starting heuristic picks for positively skewed data
    upper = float(g.max()) + 0.5
    res = fs.fit(g, stats.genextreme, upperBound=upper)
    assert np.isclose(res.dist.support()[1], upper)
    assert res.dist.support()[0] == -np.inf


# ---------------------------------------------------------------------------
# fit: argument validation


def test_fit_rejects_bound_the_distribution_lacks(norm_data, expon_data):
    with pytest.raises(ValueError, match="no natural upper bound"):
        fs.fit(norm_data, stats.norm, upperBound=10.0)
    with pytest.raises(ValueError, match="no natural lower bound"):
        fs.fit(norm_data, stats.norm, lowerBound=-10.0)
    with pytest.raises(ValueError, match="no natural upper bound"):
        fs.fit(expon_data, stats.expon, upperBound=1e9)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_fit_rejects_non_finite_bounds(expon_data, bad):
    with pytest.raises(ValueError, match="finite"):
        fs.fit(expon_data, stats.expon, lowerBound=bad)


@pytest.mark.parametrize("bad", [True, False, np.True_, "0", [0.0]])
def test_fit_rejects_non_real_bounds(expon_data, bad):
    with pytest.raises(TypeError):
        fs.fit(expon_data, stats.expon, lowerBound=bad)


@pytest.mark.parametrize("value", [0, np.float64(0.0), np.int64(0)])
def test_fit_accepts_real_scalars(expon_data, value):
    res = fs.fit(expon_data, stats.expon, lowerBound=value)
    assert res.dist.support()[0] == 0.0


def test_fit_rejects_data_outside_bounds(expon_data, beta_data):
    with pytest.raises(ValueError, match="smallest observation"):
        fs.fit(expon_data, stats.expon,
               lowerBound=float(expon_data.min()) + 1e-3)
    with pytest.raises(ValueError, match="largest observation"):
        fs.fit(beta_data, stats.beta, lowerBound=0.0,
               upperBound=float(beta_data.max()) - 1e-3)


def test_fit_rejects_inverted_bounds(beta_data):
    with pytest.raises(ValueError, match="strictly less"):
        fs.fit(beta_data, stats.beta, lowerBound=1.0, upperBound=0.0)


def test_fit_has_no_limit_arguments(expon_data):
    with pytest.raises(TypeError):
        fs.fit(expon_data, stats.expon, lowerLimit=0.0)


# ---------------------------------------------------------------------------
# compare: legacy bool flags


def test_compare_legacy_flags_emit_one_future_warning(norm_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fs.compare(norm_data, False, False)
    fw = _future_warnings(w)
    assert len(fw) == 1
    assert "lowerLimit and upperLimit" in fw[0]
    assert "lowerBound and upperBound" in fw[0]


def test_compare_legacy_false_matches_omitted_arguments(norm_data):
    with warnings.catch_warnings(record=True) as w_legacy:
        warnings.simplefilter("always")
        legacy = fs.compare(norm_data, False, False)
    with warnings.catch_warnings(record=True) as w_new:
        warnings.simplefilter("always")
        new = fs.compare(norm_data)
    assert legacy == new
    assert len(_future_warnings(w_legacy)) == 1
    assert _future_warnings(w_new) == []


def test_compare_legacy_true_leaves_bounds_free(beta_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(beta_data, True, True)
    assert len(_future_warnings(w)) == 1
    assert "Beta" in table
    assert "Uniform" in table
    dist_list = [d for d in distfit.candidates if d[2] and d[3]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, _ = distfit._fit_all(beta_data, dist_list)
    # the bounds were estimated, not pinned at any common value
    lows = {round(float(r.dist.support()[0]), 6) for r in results}
    assert len(lows) > 1


def test_compare_legacy_flag_warning_names_only_the_flag_used(wmax_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fs.compare(wmax_data, upperLimit=True)
    fw = _future_warnings(w)
    assert len(fw) == 1
    assert "upperLimit" in fw[0]
    assert "lowerLimit" not in fw[0]


def test_compare_legacy_flags_accept_only_bools(beta_data):
    with pytest.raises(TypeError, match="lowerBound"):
        fs.compare(beta_data, 0.0, 1.0)
    with pytest.raises(TypeError, match="upperLimit must be a bool"):
        fs.compare(beta_data, True, 1.0)
    with pytest.raises(TypeError):
        fs.compare(beta_data, "yes", False)


# ---------------------------------------------------------------------------
# compare: float bounds


def test_compare_bounds_select_and_constrain(beta_data):
    dist_list = [d for d in distfit.candidates if d[2] and d[3]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, _ = distfit._fit_all(beta_data, dist_list, 0.0, 1.0)
    assert len(results) >= 15
    for r in results:
        lo, hi = r.dist.support()
        assert np.isclose(lo, 0.0), r.distName
        assert np.isclose(hi, 1.0), r.distName
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(beta_data, lowerBound=0.0, upperBound=1.0)
    assert _future_warnings(w) == []
    assert "Beta" in table
    assert "Uniform" in table


def test_compare_upper_bound_only(wmax_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(wmax_data, upperBound=0.0)
    assert _future_warnings(w) == []
    assert "Weibull Max Extreme Value" in table
    dist_list = [d for d in distfit.candidates if not d[2] and d[3]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, _ = distfit._fit_all(wmax_data, dist_list, None, 0.0)
    assert all(np.isclose(r.dist.support()[1], 0.0) for r in results)


def test_compare_lower_bound_only(expon_data, monkeypatch):
    # a short candidate list keeps this fast; the full lower-bounded group is
    # dominated by the slow studentized range fit
    monkeypatch.setattr(distfit, "candidates", [
        ("Exponential", stats.expon, True, False),
        ("Gamma", stats.gamma, True, False),
        ("Pareto", stats.pareto, True, False),
        ("Normal", stats.norm, False, False),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(expon_data, lowerBound=0.0)
    assert _future_warnings(w) == []
    names = [line.split(",")[0].strip() for line in table.splitlines()[1:]]
    assert set(names) == {"Exponential", "Gamma", "Pareto"}
    results, _ = distfit._fit_all(expon_data, distfit.candidates[:3], 0.0,
                                  None)
    assert all(np.isclose(r.dist.support()[0], 0.0) for r in results)


def test_compare_bounds_are_keyword_only(beta_data):
    with pytest.raises(TypeError):
        fs.compare(beta_data, None, None, 0.0, 1.0)


def test_compare_mixed_legacy_flag_and_bound_on_other_side(beta_data,
                                                           monkeypatch):
    # fixed upper bound, lower bound left free (legacy True)
    monkeypatch.setattr(distfit, "candidates", [
        ("Beta", stats.beta, True, True),
        ("Uniform", stats.uniform, True, True),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(beta_data, lowerLimit=True, upperBound=1.0)
    fw = _future_warnings(w)
    assert len(fw) == 1 and "lowerLimit" in fw[0]
    assert "Beta" in table and "Uniform" in table
    results, _ = distfit._fit_all(beta_data, distfit.candidates, None, 1.0)
    for r in results:
        lo, hi = r.dist.support()
        assert np.isclose(hi, 1.0)
        assert lo > -0.1 and not np.isclose(lo, 0.0, atol=1e-12)


def test_compare_rejects_flag_and_bound_on_the_same_side(beta_data):
    with pytest.raises(ValueError, match="cannot both be given"):
        fs.compare(beta_data, lowerLimit=True, lowerBound=0.0)
    with pytest.raises(ValueError, match="cannot both be given"):
        fs.compare(beta_data, upperLimit=False, upperBound=1.0)


def test_compare_rejects_invalid_bounds(beta_data):
    with pytest.raises(ValueError, match="strictly less"):
        fs.compare(beta_data, lowerBound=1.0, upperBound=0.0)
    with pytest.raises(ValueError, match="smallest observation"):
        fs.compare(beta_data, lowerBound=0.5, upperBound=1.0)
    with pytest.raises(ValueError, match="finite"):
        fs.compare(beta_data, lowerBound=0.0, upperBound=np.inf)
    with pytest.raises(TypeError):
        fs.compare(beta_data, lowerBound="0", upperBound=1.0)
    with pytest.raises(TypeError):
        fs.compare(beta_data, lowerBound=True)


# ---------------------------------------------------------------------------
# candidate table: categorization by support


def test_candidate_flags_agree_with_scipy_for_fixed_supports():
    for name, dist, lo, hi in distfit.candidates:
        if distfit._shape_dependent_support(dist):
            continue
        assert bool(np.isfinite(dist.a)) == lo, name
        assert bool(np.isfinite(dist.b)) == hi, name


def test_shape_dependent_families_listed_per_realizable_support():
    groups = {}
    for name, dist, lo, hi in distfit.candidates:
        groups.setdefault(dist, set()).add((lo, hi))
    assert groups[stats.genextreme] == {(True, False), (False, True)}
    assert groups[stats.genpareto] == {(True, False), (True, True)}
    assert groups[stats.kappa4] == {(True, True), (True, False),
                                    (False, True)}
    assert groups[stats.tukeylambda] == {(True, True), (False, False)}
    assert stats.vonmises not in groups
    assert groups[stats.vonmises_line] == {(True, True)}


def test_no_duplicate_rows_within_a_group():
    seen = set()
    for name, dist, lo, hi in distfit.candidates:
        assert (dist, lo, hi) not in seen, name
        seen.add((dist, lo, hi))


def test_compare_drops_fits_whose_support_does_not_match(norm_data,
                                                         monkeypatch):
    # a uniform fit is bounded on both sides, so it must not be reported in
    # the unbounded group even if listed there
    monkeypatch.setattr(distfit, "candidates", [
        ("Normal", stats.norm, False, False),
        ("Impostor", stats.uniform, False, False),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(norm_data)
    assert "Normal" in table
    assert "Impostor" not in table
    msgs = [str(r.message) for r in w
            if issubclass(r.category, RuntimeWarning)]
    assert any("Impostor" in m and "requested support" in m for m in msgs)


def test_compare_keeps_fits_whose_support_matches(beta_data, monkeypatch):
    # genpareto with negative shape is bounded on both sides; fixing both
    # bounds forces that region, so it belongs in the two-sided group
    monkeypatch.setattr(distfit, "candidates", [
        ("Generalized Pareto", stats.genpareto, True, True),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table = fs.compare(beta_data, lowerBound=0.0, upperBound=1.0)
    assert "Generalized Pareto" in table
    assert not any("requested support" in str(r.message) for r in w)
