"""
Item 8 of the 2026-09-24 course handoff: the draw-allocation guarantee.
Draws are allocated one column per consumed uniform, in consumption order,
from one generator seeded with `seed`, so the j-th uniform of trial r does
not depend on how many uniforms f consumes in total.  Exercises 11 and 12 of
the course rely on this to compare step functions consuming K and K + 1
uniforms under one seed.
"""
import numpy as np
import pytest
from scipy import stats

import funcsim as fs


def _two(ugen):
    return {"u1": next(ugen), "u2": next(ugen)}


def _three(ugen):
    return {"u1": next(ugen), "u2": next(ugen), "u3": next(ugen)}


def _one_per_step(ugen):
    return {"a": next(ugen)}


def _two_per_step(ugen):
    return {"a": next(ugen), "b": next(ugen)}


@pytest.mark.parametrize("sampling", ["lh", "mc"])
def test_first_k_uniforms_identical_for_k_and_k_plus_1_consumers(sampling):
    kw = dict(ntrials=300, seed=11, sampling=sampling)
    a = fs.simulate(f=_two, **kw).sel(steps=0)
    b = fs.simulate(f=_three, **kw).sel(steps=0)
    np.testing.assert_array_equal(a.sel(variables=["u1", "u2"]).values,
                                  b.sel(variables=["u1", "u2"]).values)
    # the extra column is a fresh draw, not a copy
    assert not np.allclose(b.sel(variables="u3").values,
                           b.sel(variables="u1").values)
    # and a different seed gives different draws
    c = fs.simulate(f=_two, ntrials=300, seed=12, sampling=sampling).sel(steps=0)
    assert not np.allclose(a.values, c.values)


def test_prefix_does_not_depend_on_what_f_computes():
    def transformed(ugen):
        return {"z": stats.norm.ppf(next(ugen)), "e": -np.log(next(ugen))}

    a = fs.simulate(f=_two, ntrials=100, seed=3).sel(steps=0)
    b = fs.simulate(f=transformed, ntrials=100, seed=3).sel(steps=0)
    np.testing.assert_allclose(stats.norm.cdf(b.sel(variables="z").values),
                               a.sel(variables="u1").values, atol=1e-12)
    np.testing.assert_allclose(np.exp(-b.sel(variables="e").values),
                               a.sel(variables="u2").values, atol=1e-12)


def test_stdnorm_draws_share_the_prefix():
    kw = dict(ntrials=100, seed=5, stdnorm=True)
    a = fs.simulate(f=_two, **kw).sel(steps=0)
    b = fs.simulate(f=_three, **kw).sel(steps=0)
    np.testing.assert_array_equal(a.sel(variables=["u1", "u2"]).values,
                                  b.sel(variables=["u1", "u2"]).values)


def test_multi_step_stream_order():
    # trial stream: step s of a function consuming K per step gets uniforms
    # sK + 1 .. (s + 1)K, so with one versus two per step only step 0 matches,
    # and the two-per-step function's second draw at step 0 is the
    # one-per-step function's draw at step 1
    kw = dict(ntrials=50, nsteps=3, seed=7)
    a = fs.simulate(f=_one_per_step, **kw)
    b = fs.simulate(f=_two_per_step, **kw)
    np.testing.assert_array_equal(a.sel(variables="a", steps=0).values,
                                  b.sel(variables="a", steps=0).values)
    np.testing.assert_array_equal(a.sel(variables="a", steps=1).values,
                                  b.sel(variables="b", steps=0).values)
    assert not np.allclose(a.sel(variables="a", steps=1).values,
                           b.sel(variables="a", steps=1).values)


def test_copula_scenarios_share_uniforms():
    # the course's use: Gaussian (K draws) versus Student's t (K + 1 draws)
    # copulas fed the same first K uniforms in every trial under one seed
    rho = [[1.0, 0.6], [0.6, 1.0]]
    cg = fs.CopulaGauss.from_params(rho)
    ct = fs.CopulaStudent.from_params(rho, nu=4.0)

    def raw(ugen):
        return {"u1": next(ugen), "u2": next(ugen), "u3": next(ugen)}

    def gauss(ugen):
        d = cg.draw(ugen)
        return {"x": d.v0, "y": d.v1}

    def student(ugen):
        d = ct.draw(ugen)
        return {"x": d.v0, "y": d.v1}

    kw = dict(ntrials=200, seed=21)
    u = fs.simulate(f=raw, **kw).sel(steps=0)
    g = fs.simulate(f=gauss, **kw).sel(steps=0)
    s = fs.simulate(f=student, **kw).sel(steps=0)
    # both copulas map the same uniforms; the Gaussian result is the
    # explicit transform of the first two columns of the raw stream
    z = np.linalg.cholesky(rho) @ stats.norm.ppf(
        u.sel(variables=["u1", "u2"]).values.T)
    np.testing.assert_allclose(g.sel(variables=["x", "y"]).values,
                               stats.norm.cdf(z).T, atol=1e-12)
    # and the Student's t result uses the third column as its mixing draw
    chi2 = stats.chi2.ppf(u.sel(variables="u3").values, df=4.0)
    want = stats.t.cdf(np.sqrt(4.0 / chi2) * z, df=4.0).T
    np.testing.assert_allclose(s.sel(variables=["x", "y"]).values, want,
                               atol=1e-12)


def test_multi_does_not_change_the_allocation():
    kw = dict(ntrials=64, seed=9)
    a = fs.simulate(f=_two, multi=False, **kw)
    b = fs.simulate(f=_two, multi=True, **kw)
    np.testing.assert_array_equal(a.values, b.values)
