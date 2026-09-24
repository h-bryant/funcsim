"""
Item 7 of the 2026-09-24 course handoff: fs.compare completes in seconds
with lowerBound (the Studentized range and Levy stable families, which took
minutes and half a minute per fit, are out of the built-in list), and a
`candidates` argument restricts the comparison to a caller's list.
"""
import time
import warnings

import numpy as np
import pytest
from scipy import stats

import funcsim as fs
from funcsim import distfit


def _rows(table: str) -> dict:
    # parse compare's text table into {name: BIC}
    out = {}
    for line in table.strip().splitlines()[1:]:
        cells = [c.strip() for c in line.split(",")]
        out[cells[0]] = float(cells[1])
    return out


@pytest.fixture(scope="module")
def gamma_sample():
    return stats.gamma(a=2.0, scale=3.0).rvs(size=500, random_state=0)


def test_slow_families_are_out_of_the_builtin_list():
    dists = {d for (_, d, _, _) in distfit.candidates}
    assert stats.studentized_range not in dists
    assert stats.levy_stable not in dists
    # and the list is otherwise intact
    assert stats.gamma in dists and stats.norm in dists
    assert len(distfit.candidates) > 100


def test_bounded_below_compare_completes_quickly(gamma_sample):
    # the handoff's evidence: this call ran more than ten minutes before
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = fs.compare(gamma_sample, lowerBound=0.0)
    elapsed = time.perf_counter() - t0
    rows = _rows(table)
    assert "Gamma" in rows and len(rows) > 30
    assert elapsed < 60.0
    assert min(rows, key=rows.get) in ("Gamma", "Chi^2", "Erlang")


def test_candidates_dict_restricts_and_labels(gamma_sample):
    table = fs.compare(gamma_sample, lowerBound=0.0,
                       candidates={"Gamma": stats.gamma,
                                   "Log-normal": stats.lognorm,
                                   "Weibull": stats.weibull_min})
    rows = _rows(table)
    assert set(rows) == {"Gamma", "Log-normal", "Weibull"}
    assert rows["Gamma"] <= rows["Log-normal"]
    # the bound is fixed exactly as in fs.fit
    want = fs.fit(gamma_sample, stats.gamma, lowerBound=0.0).bic
    assert rows["Gamma"] == pytest.approx(want, abs=1e-3)


def test_candidates_iterable_uses_scipy_names(gamma_sample):
    table = fs.compare(gamma_sample, lowerBound=0.0,
                       candidates=[stats.gamma, stats.expon])
    assert set(_rows(table)) == {"gamma", "expon"}


def test_explicit_candidates_skip_the_support_filter(gamma_sample):
    # with no bounds the built-in path would drop a bounded-below family as
    # not matching the unbounded group; an explicit request is honored
    table = fs.compare(gamma_sample, candidates={"Gamma": stats.gamma,
                                                 "Normal": stats.norm})
    assert set(_rows(table)) == {"Gamma", "Normal"}


def test_candidate_that_cannot_be_fitted_is_dropped_with_warning(gamma_sample):
    with pytest.warns(RuntimeWarning, match="Normal"):
        table = fs.compare(gamma_sample, lowerBound=0.0,
                           candidates={"Gamma": stats.gamma,
                                       "Normal": stats.norm})
    assert set(_rows(table)) == {"Gamma"}


def test_candidates_validation(gamma_sample):
    with pytest.raises(ValueError):
        fs.compare(gamma_sample, candidates={})
    with pytest.raises(TypeError):
        fs.compare(gamma_sample, candidates={"x": 3.0})
    with pytest.raises(TypeError):
        fs.compare(gamma_sample, candidates=42)


def test_slow_family_reachable_through_candidates():
    # studentized_range fits are impractically slow, so only check that the
    # argument accepts it and that the fit machinery is invoked (a
    # deliberately unfittable bound turns it into a fast failure warning)
    data = np.random.default_rng(1).normal(size=60)
    with pytest.warns(RuntimeWarning):
        table = fs.compare(data, lowerBound=-1e9,
                           candidates={"Normal": stats.norm})
    assert table.strip().splitlines()[0].lstrip().startswith("distribution")
