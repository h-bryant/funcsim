"""
Regression tests for the Critical (#1-#10), Important (#11-#31), and
Minor (#32-#50) items in code-review-2026-07-21.md.  Each test is named
for the review item it guards.
"""

import builtins
import io
import pathlib
import sys
import tokenize
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import scipy.stats as stats

import funcsim as fs
from funcsim import copfit, dependence, distfit, ecdfgof, multicore
from funcsim import rdarrays, shrinkage
from funcsim.conversions import vlCoords, vlValidate
from funcsim.screen import (adf_test_const_trend, bartels_test, ljung_box,
                            white_test)
from funcsim.testsim import simulator
from funcsim.vect import varange, vectorize


# ---------------------------------------------------------------------------
# item 1: package must not contain PEP 701 f-strings (Python 3.12+ only),
# so that "import funcsim" works on Python 3.10/3.11


def test_01_no_multiline_fstrings():
    # a single-quoted f-string whose replacement field spans a newline is
    # a SyntaxError before Python 3.12; scan every module in the package
    pkg = pathlib.Path(fs.__file__).parent
    offenders = []
    for py in sorted(pkg.glob("*.py")):
        toks = tokenize.generate_tokens(io.StringIO(py.read_text()).readline)
        stack = []
        for t in toks:
            if t.type == tokenize.FSTRING_START:
                stack.append((t.start[0], t.string))
            elif t.type == tokenize.FSTRING_END:
                line0, opener = stack.pop()
                triple = opener.endswith('"""') or opener.endswith("'''")
                if not triple and t.end[0] != line0:
                    offenders.append(f"{py.name}:{line0}")
    assert offenders == []


# ---------------------------------------------------------------------------
# item 2: hist0 with dims ordered ("variables", "steps") must not be
# silently transposed


def _step_rw(draw, data):
    ua, ub = next(draw), next(draw)
    return {"a": data.recall("a", lag=1) + ua,
            "b": data.recall("b", lag=1) + ub}


def test_02_hist0_dim_order():
    # non-square history (5 steps x 2 variables), so a transposition
    # cannot masquerade as a reshape
    hist = np.random.default_rng(0).normal(size=(5, 2))
    h_sv = xr.DataArray(hist, dims=("steps", "variables"),
                        coords={"steps": np.arange(5),
                                "variables": ["a", "b"]})
    h_vs = h_sv.transpose("variables", "steps")

    r1 = fs.simulate(_step_rw, ntrials=7, nsteps=3, hist0=h_sv, seed=11)
    r2 = fs.simulate(_step_rw, ntrials=7, nsteps=3, hist0=h_vs, seed=11)

    # both orderings give identical simulations
    assert np.allclose(r1.values, r2.values)

    # historical rows appear unchanged in the output
    assert np.allclose(r1.sel(variables="a").values[:, :5], hist[:, 0])
    assert np.allclose(r1.sel(variables="b").values[:, :5], hist[:, 1])


# ---------------------------------------------------------------------------
# item 3: White test p-values


def test_03_white_df_ignores_redundant_const():
    # rank-based df: passing a constant column must not change the p-value
    rng = np.random.default_rng(1)
    n = 200
    obsnum = np.arange(n)
    y = rng.normal(size=n)
    X = np.column_stack([np.ones(n), obsnum])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    exog1 = pd.DataFrame({"obsnum": obsnum})
    exog2 = pd.DataFrame({"obsnum": obsnum, "const": np.ones(n)})
    assert np.isclose(white_test(resid, exog1), white_test(resid, exog2))


def test_03_white_null_calibration():
    # under homoskedastic null, p-values should be roughly uniform
    rng = np.random.default_rng(1)
    n = 200
    obsnum = np.arange(n)
    X = np.column_stack([np.ones(n), obsnum])
    exog = pd.DataFrame({"obsnum": obsnum})
    pvals = []
    for _ in range(500):
        y = rng.normal(size=n)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pvals.append(white_test(y - X @ beta, exog))
    pvals = np.array(pvals)
    assert 0.02 <= np.mean(pvals < 0.05) <= 0.09
    assert 0.35 <= np.median(pvals) <= 0.65


def test_03_white_power():
    # clear heteroskedasticity should be rejected
    rng = np.random.default_rng(2)
    n = 200
    obsnum = np.arange(n)
    X = np.column_stack([np.ones(n), obsnum])
    y = rng.normal(size=n) * np.linspace(0.2, 3.0, n)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    assert white_test(resid, pd.DataFrame({"obsnum": obsnum})) < 0.01


# ---------------------------------------------------------------------------
# item 4: shrink() with non-square DataFrame / DataArray input


def test_04_shrink_dataframe_nonsquare():
    rng = np.random.default_rng(3)
    x = rng.multivariate_normal([0, 0, 0], np.diag([1, 2, 3]), size=40)
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    out = fs.shrink(df, "D")
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 3)
    assert list(out.index) == ["a", "b", "c"]
    assert list(out.columns) == ["a", "b", "c"]


def test_04_shrink_dataarray_nonsquare():
    rng = np.random.default_rng(3)
    x = rng.multivariate_normal([0, 0, 0], np.diag([1, 2, 3]), size=40)
    da = xr.DataArray(x, dims=("obs", "var"),
                      coords={"obs": np.arange(40), "var": ["a", "b", "c"]})
    out = fs.shrink(da, "D")
    assert isinstance(out, xr.DataArray)
    assert out.shape == (3, 3)
    assert list(np.asarray(out.coords["var"].values)) == ["a", "b", "c"]
    # same numbers as the ndarray path
    assert np.allclose(out.values, fs.shrink(x, "D"))


# ---------------------------------------------------------------------------
# item 5: shrinkage target F must shrink toward the sample correlation
# (the old corrcoef-of-the-covariance bug gave rbar = -1 for p = 2)


def test_05_target_f_rbar():
    rng = np.random.default_rng(4)
    x = rng.multivariate_normal([0, 0], [[1.0, 0.5], [0.5, 1.0]], size=500)
    s = np.cov(x, rowvar=False)
    vcv = fs.shrink(x, "F")
    # with correct rbar the p=2 target off-diagonal equals s01 exactly,
    # so the estimate stays at s01; with rbar = -1 it was pulled negative
    assert abs(vcv[0, 1] - s[0, 1]) < 0.02
    assert vcv[0, 1] > 0.3


def test_05_target_f_degenerate_denominator():
    # data whose sample covariance has exactly constant correlation used
    # to hit a 0/0 in the shrinkage intensity; result must still equal s
    rho = np.array([[1.0, 0.9, 0.9],
                    [0.9, 1.0, 0.9],
                    [0.9, 0.9, 1.0]])
    vcv = fs.shrink(rho, "F")
    assert np.all(np.isfinite(vcv))
    assert np.allclose(vcv, np.cov(rho, rowvar=False))


# ---------------------------------------------------------------------------
# item 6: imanconover with tied values


def test_06_imanconover_ties():
    rng = np.random.default_rng(5)
    v1 = rng.integers(0, 5, size=200).astype(float)  # many ties
    v2 = rng.normal(size=200)
    rho_s = np.array([[1.0, 0.7], [0.7, 1.0]])
    ic = fs.imanconover(rho_s, [v1, v2], seed=6)  # used to raise KeyError
    rho_hat = stats.spearmanr(ic["v1"], ic["v2"]).statistic
    assert abs(rho_hat - 0.7) < 0.15
    # marginals must be preserved exactly
    assert sorted(ic["v1"]) == sorted(v1.tolist())
    assert sorted(ic["v2"]) == sorted(v2.tolist())


# ---------------------------------------------------------------------------
# item 7: MvKde must apply bandwidth covariance H, not H @ H.T


def test_07_mvkde_chol():
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.normal(size=(50, 2)), columns=["x", "y"])
    kde = fs.MvKde(data)
    assert np.allclose(kde._chol @ kde._chol.T, kde._bw)


def test_07_mvkde_draw_variance():
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.normal(size=(50, 2)), columns=["x", "y"])
    kde = fs.MvKde(data)
    ugen = iter(np.random.default_rng(8).random(3 * 10000))
    draws = np.array([kde.draw(ugen)["x"] for _ in range(10000)])
    # in standardized units: var(draw) = var(data) + H[0, 0]
    var_std = np.var((draws - kde._means[0]) / kde._stds[0])
    expected = np.var(kde._data[:, 0]) + kde._bw[0, 0]
    undersmoothed = np.var(kde._data[:, 0]) + kde._bw[0, 0] ** 2
    assert abs(var_std - expected) < 0.06
    assert abs(var_std - undersmoothed) > 0.1  # old bug is distinguishable


# ---------------------------------------------------------------------------
# item 8: Frank copula frailty sampling


def test_08_logser_sampler_pmf():
    # the Kemp sampler must reproduce the log-series pmf
    theta = 5.0
    p = -np.expm1(-theta)
    rng = np.random.default_rng(9)
    ks = np.array([dependence._logser_draw(theta, rng.random(), rng.random())
                   for _ in range(100000)])
    emp = np.array([np.mean(ks == k) for k in range(1, 8)])
    theo = stats.logser.pmf(np.arange(1, 8), p)
    assert np.max(np.abs(emp - theo)) < 0.006


def test_08_frank_draws_large_theta():
    # theta values near copfit._THETA_MAX used to hang (logser.ppf O(k)
    # scan) and then return all-NaN once 1 - exp(-theta) rounded to 1.0
    rng = np.random.default_rng(10)
    u = np.column_stack([np.linspace(0.01, 0.99, 60)] * 2)
    u[:, 1] = np.clip(u[:, 1] + rng.normal(0.0, 0.01, 60), 0.001, 0.999)
    cf = fs.CopulaFrank(pd.DataFrame(u, columns=["a", "b"]))
    for theta in (50.0, 100.0):
        cf._theta = theta
        ugen = iter(np.random.default_rng(11).random(4 * 20000))
        out = np.array([cf.draw(ugen).values for _ in range(20000)])
        assert np.all(np.isfinite(out))
        assert np.all((out > 0.0) & (out < 1.0))
        # Kendall's tau must match tau(theta) = 1 - (4/theta)(1 - D1(theta))
        tau_hat = stats.kendalltau(out[:, 0], out[:, 1]).statistic
        tau_theo = 1.0 - (4.0 / theta) * (1.0 - copfit._debye1(theta))
        assert abs(tau_hat - tau_theo) < 0.01


# ---------------------------------------------------------------------------
# item 9: GoF tests with a distribution class plus args


def test_09_gof_args_unpacking():
    rng = np.random.default_rng(12)
    data = rng.normal(2.0, 3.0, size=100)
    for fn in (fs.kstest, fs.adtest, fs.cvmtest):
        ra = fn(data, stats.norm, args=(2.0, 3.0))  # used to raise TypeError
        rb = fn(data, stats.norm(2.0, 3.0))
        assert np.isclose(ra.statistic, rb.statistic)
        assert np.isclose(ra.pvalue, rb.pvalue)


# ---------------------------------------------------------------------------
# item 10: compare() columns relabeled as heuristic scores


def test_10_compare_header_relabeled():
    header = distfit._result_line(None, header=True)
    assert "AD_score" in header
    assert "CvM_score" in header
    assert "p-val" not in header


# ---------------------------------------------------------------------------
# item 11: CopulaGauss must fit a zero-mean, unit-diagonal correlation


def test_11_copula_gauss_correlation_only():
    rng = np.random.default_rng(20)
    # pseudo-observations with distorted marginals: their normal scores
    # have nonzero mean and non-unit variance, which the old full-MvNorm
    # fit absorbed, wrecking the uniform marginals of draws
    u_raw = np.clip(rng.beta(4.0, 2.0, size=(400, 2)), 1e-6, 1.0 - 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Shapiro warnings must not fire
        cg = fs.CopulaGauss(u_raw)
    assert np.allclose(np.diag(cg._rho), 1.0)
    gen = iter(rng.random(2 * 6000))
    draws = np.array([cg.draw(gen).values for _ in range(6000)])
    for k in range(2):
        assert stats.kstest(draws[:, k], "uniform").pvalue > 0.01


# ---------------------------------------------------------------------------
# item 12: Iman-Conover variance-reduction step (step 6)


def test_12_imanconover_variance_reduction():
    target = 0.5
    errs = []
    for s in range(120):
        rng = np.random.default_rng(1000 + s)
        v1, v2 = rng.normal(size=30), rng.normal(size=30)
        ic = fs.imanconover(np.array([[1.0, target], [target, 1.0]]),
                            [v1, v2], seed=2000 + s)
        errs.append(stats.spearmanr(ic["v1"], ic["v2"]).statistic - target)
    # review: error sd ~0.037 with step 6, ~0.085 without (N=30)
    assert np.std(errs) < 0.06


def test_12_imanconover_seed_reproducible():
    rng = np.random.default_rng(21)
    vecs = [rng.normal(size=25), rng.normal(size=25)]
    rho = np.array([[1.0, 0.6], [0.6, 1.0]])
    a = fs.imanconover(rho, vecs, seed=3)
    b = fs.imanconover(rho, vecs, seed=3)
    assert np.allclose(a.values, b.values)


# ---------------------------------------------------------------------------
# items 13 & 14: Kde bandwidth semantics and CDF integration


def test_13_kde_float_bw_is_bandwidth():
    data = np.random.default_rng(22).normal(0.0, 10.0, size=300)
    kde = fs.Kde(data, bw=0.5)
    applied_sd = float(np.sqrt(kde.gkde.covariance[0, 0]))
    assert abs(applied_sd - 0.5) < 1e-9  # not 0.5 * sd(data) ~ 5


def test_14_kde_cdf_from_minus_inf():
    kde = fs.Kde([1.0, 2.0, 3.0, 4.0, 5.0])  # tiny n: fat kernels
    assert kde.cdf(-50.0) == 0.0  # was negative with a finite cutoff
    assert kde.cdf(50.0) == 1.0   # was < 1 (lost tail mass)
    assert kde.ppf(0.995) > 5.0   # high quantiles solvable again


# ---------------------------------------------------------------------------
# item 15: ad_stat clips at float resolution, not 1e-4


def test_15_ad_stat_tail_sensitivity():
    with_tail = ecdfgof.ad_stat(np.sort([1e-8, 0.3, 0.5, 0.7, 0.9]))
    old_clip = ecdfgof.ad_stat(np.sort([1e-4, 0.3, 0.5, 0.7, 0.9]))
    assert with_tail > old_clip + 1.0  # 1e-8 must register, not clip to 1e-4


# ---------------------------------------------------------------------------
# item 16: utests CGR term at boundary values


def test_16_utests_boundary_values():
    samp = np.concatenate([np.random.default_rng(23).random(998),
                           [0.0, 1.0]])
    res = fs.utests(samp)
    assert np.isfinite(res["cook_gelman_rubin_pval"])
    assert res["cook_gelman_rubin_pval"] > 0.0  # was exactly 0.0


# ---------------------------------------------------------------------------
# items 17 & 18: cpt validation


def test_17_cpt_probability_sum():
    with pytest.raises(ValueError):
        fs.cpt(utilFunc=lambda x: fs.utilPower(x),
               weightFuncGains=lambda p: fs.weightTK(p, 0.61),
               weightFuncLosses=lambda p: fs.weightTK(p, 0.69),
               outcomes=[10.0, 20.0, 30.0], refOutcome=0.0,
               probabilities=[0.2, 0.2, 0.2])  # sums to 0.6


def test_18_utilpower_rejects_zero_exponents():
    with pytest.raises(ValueError):
        fs.utilPower(1.0, alpha=0.0)
    with pytest.raises(ValueError):
        fs.utilPower(-1.0, beta=0.0)


# ---------------------------------------------------------------------------
# item 19: frank_theta0 bracket direction; K < 2 gate


def test_19_frank_theta0_directions():
    # near-zero tau must give the weakest, not the strongest, dependence
    assert copfit.frank_theta0(1e-9) == copfit.THETA_MIN_FRANK
    # interior tau still solved numerically
    assert 5.0 < copfit.frank_theta0(0.5) < 6.0
    # large tau capped inside the admissible range
    assert 90.0 < copfit.frank_theta0(0.99) <= copfit._THETA_MAX


def test_19_taubar_rejects_single_column():
    with pytest.raises(ValueError):
        copfit.taubar(np.random.default_rng(24).random((50, 1)))


# ---------------------------------------------------------------------------
# item 20: numpy bools honored for stdnorm/multi


def _trial_one_draw(draw):
    return {"z": next(draw)}


def test_20_stdnorm_numpy_bool():
    out = fs.simulate(_trial_one_draw, ntrials=200, stdnorm=np.True_, seed=1)
    zs = out.sel(variables="z", steps=0).values
    # normal draws, not silently-uniform ones
    assert zs.min() < -0.5 and zs.max() > 0.5


# ---------------------------------------------------------------------------
# item 21: RDdata.recall bound check


def test_21_recall_bounds():
    d = rdarrays.RDdata(np.ones((2, 1)), 1, {"x": 0})
    with pytest.raises(rdarrays.MissingValue):
        d.recall("x", lag=10)  # used to wrap around via negative indexing


# ---------------------------------------------------------------------------
# item 22: parmap raises for failed jobs


def _boom(x):
    raise RuntimeError("boom")


def test_22_parmap_raises_on_worker_failure():
    with pytest.raises(RuntimeError, match="boom|jobs"):
        multicore.parmap(_boom, range(4), nprocs=2)


# ---------------------------------------------------------------------------
# item 23: no global warnings.showwarning hijack


def test_23_showwarning_not_hijacked():
    assert "custom_warning_format" not in repr(warnings.showwarning)


# ---------------------------------------------------------------------------
# item 24: no global RNG mutation


def test_24_global_rng_untouched():
    np.random.seed(12345)
    before = np.random.get_state()[1].copy()
    fs.simulate(_trial_one_draw, ntrials=50, seed=7)
    fs.imanconover(np.array([[1.0, 0.3], [0.3, 1.0]]),
                   [np.arange(20.0), np.arange(20.0)[::-1]], seed=1)
    adf_test_const_trend(np.cumsum(np.random.default_rng(8).normal(size=60)),
                         max_lag=1, n_sim=50, seed=42)
    assert np.array_equal(before, np.random.get_state()[1])


# ---------------------------------------------------------------------------
# item 25: real package imports (no sys.path hack, single module identity)


def test_25_single_module_identity():
    assert "funcsim.core" in sys.modules
    # the old sys.path hack gave every submodule a second, top-level
    # identity, so exceptions raised inside simulate could not be caught
    # by their funcsim.-qualified names
    assert sys.modules.get("rdarrays") is None


def _step_needs_lag(draw, data):
    return {"x": data.recall("x", lag=5)}


def test_25_missingvalue_catchable():
    hist = xr.DataArray(np.ones((2, 1)), dims=("steps", "variables"),
                        coords={"steps": np.arange(2), "variables": ["x"]})
    with pytest.raises(rdarrays.MissingValue):
        fs.simulate(_step_needs_lag, ntrials=3, nsteps=1, hist0=hist)


# ---------------------------------------------------------------------------
# item 26: eut fixes


def test_26_eut_row_vector():
    outc = np.array([[2.0, 3.0, 4.0, 5.0]])  # (1, N): len() == 1
    r1 = fs.eut(lambda y: np.log(y), outc)
    r2 = fs.eut(lambda y: np.log(y), outc.flatten())
    assert np.isclose(r1.ExpectedValue, r2.ExpectedValue)
    assert np.isclose(r1.CertaintyEquiv, r2.CertaintyEquiv)


def test_26_eutisoelastic_small_outcomes():
    res = fs.eutIsoelastic(2.0, [0.5, 0.8, 1.5, 2.0])  # values < 1.0 legal
    assert np.isfinite(res.ExpectedValue)
    assert 0.5 < res.CertaintyEquiv < 2.0


def test_26_utilisoelastic_int_and_errors():
    assert np.isclose(fs.utilIsoelastic(2, 1.0), np.log(2.0))
    with pytest.raises(ValueError):
        fs.utilIsoelastic(-1.0, 1.0)


# ---------------------------------------------------------------------------
# item 27: vectorize binds per instance


def test_27_vectorize_per_instance():
    class Adder:
        def __init__(self, k):
            self.k = k

        @vectorize
        def add(self, x):
            return x + self.k

    a1, a2 = Adder(1), Adder(100)
    assert list(a1.add((1, 2))) == [2, 3]
    assert list(a2.add((1, 2))) == [101, 102]  # not [2, 3]
    assert list(a1.add((1, 2))) == [2, 3]      # a1 unaffected by a2


# ---------------------------------------------------------------------------
# item 28: vlCoords on lists and 1-column DataFrames


def test_28_vlcoords():
    assert list(vlCoords([10.0, 20.0, 30.0])) == [0, 1, 2]  # was NameError
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=["x", "y", "z"])
    assert list(vlCoords(df)) == ["x", "y", "z"]  # was silent None


# ---------------------------------------------------------------------------
# item 29: repackage on the xarray path


def test_29_nearestpd_dataarray():
    bad = np.array([[1.0, 0.99, 0.0],
                    [0.99, 1.0, 0.99],
                    [0.0, 0.99, 1.0]])  # not positive definite
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = fs.nearestpd(xr.DataArray(bad, dims=("r", "c")))
    assert isinstance(fixed, xr.DataArray)  # was NameError
    assert np.all(np.linalg.eigvalsh(fixed.values) > 0)


# ---------------------------------------------------------------------------
# item 30: fan() raises ImportError when plotly is absent


def test_30_fan_importerror(monkeypatch):
    real_import = builtins.__import__

    def no_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("hidden for test")
        return real_import(name, *args, **kwargs)

    da = fs.simulate(_trial_one_draw, ntrials=5, nsteps=3)
    monkeypatch.setattr(builtins, "__import__", no_plotly)
    with pytest.raises(ImportError):  # was NameError ('importerror')
        fs.fan(da, "z")


# ---------------------------------------------------------------------------
# item 31: screen's line B is a genuine trend-stationarity test


def test_31_adf_const_trend():
    rng = np.random.default_rng(25)
    n = 200
    # stationary around a linear trend: must reject the unit-root H0
    trend_stationary = 0.5 * np.arange(n) + rng.normal(size=n)
    p_ts = adf_test_const_trend(trend_stationary, max_lag=2, n_sim=500,
                                seed=42)[1]
    assert p_ts < 0.05
    # random walk with drift: must not reject
    rw_drift = np.cumsum(0.5 + rng.normal(size=n))
    p_rw = adf_test_const_trend(rw_drift, max_lag=2, n_sim=500, seed=42)[1]
    assert p_rw > 0.10


# ---------------------------------------------------------------------------
# items 32 & 33: _rand_int clamp; dead memoization code


def test_32_rand_int_clamp():
    assert dependence._rand_int(1.0, 10) == 9  # not 10


def test_33_dead_code_removed():
    assert not hasattr(dependence, "_memoize")
    assert not hasattr(dependence, "_makeA")


# ---------------------------------------------------------------------------
# item 34: MvKde bandwidth validation


def test_34_mvkde_bw_validation():
    data = pd.DataFrame(np.random.default_rng(26).normal(size=(50, 2)),
                        columns=["x", "y"])
    with pytest.raises(ValueError):  # was an AttributeError much later
        fs.MvKde(data, bw="gauss")
    with pytest.raises(ValueError):  # wrong shape for K=2
        fs.MvKde(data, bw=np.eye(3))
    # bw=None used to TypeError in the matrix branch
    assert np.allclose(fs.MvKde(data, bw=None)._bw,
                       fs.MvKde(data, bw="scott")._bw)


# ---------------------------------------------------------------------------
# item 35: Clayton frailty underflow at extreme theta


def test_35_clayton_frailty_underflow():
    rng = np.random.default_rng(27)
    u = np.column_stack([np.linspace(0.01, 0.99, 60)] * 2)
    u[:, 1] = np.clip(u[:, 1] + rng.normal(0.0, 0.01, 60), 0.001, 0.999)
    cc = fs.CopulaClayton(pd.DataFrame(u, columns=["a", "b"]))
    cc._theta = 100.0
    # the first draw drives the gamma frailty; ppf(1e-12, shape=0.01)
    # underflows to 0.0, which used to raise ZeroDivisionError
    out = cc.draw(iter([1e-12, 0.5, 0.5]))
    assert np.all(np.isfinite(out.values))
    assert np.all((out.values >= 0.0) & (out.values <= 1.0))


# ---------------------------------------------------------------------------
# item 36: six dependency removed


def test_36_no_six():
    src = pathlib.Path(ecdfgof.__file__).read_text()
    assert "six" not in src
    r = fs.kstest(np.random.default_rng(28).normal(size=50), "norm")
    assert 0.0 <= r.pvalue <= 1.0  # string dist names still work


# ---------------------------------------------------------------------------
# item 37: Kde.ppf contract and resolution


def test_37_kde_ppf_valueerror():
    kde = fs.Kde(np.random.default_rng(29).normal(size=200))
    with pytest.raises(ValueError):  # was an assert
        kde.ppf(1.5)


def test_37_kde_ppf_small_scale():
    # an absolute xtol of 1e-4 made ppf meaningless on scales << 1e-3
    tiny = fs.Kde(np.random.default_rng(30).normal(0.0, 1e-5, size=200))
    assert float(tiny.ppf(0.7)) > float(tiny.ppf(0.3))
    assert abs(float(tiny.cdf(tiny.ppf(0.5))) - 0.5) < 1e-3


# ---------------------------------------------------------------------------
# item 38: fit docstring names the tests actually computed


def test_38_fit_docstring():
    assert "Cramer-von Mises" in fs.fit.__doc__
    assert "Kolmogorov" not in fs.fit.__doc__


# ---------------------------------------------------------------------------
# item 39: warning-category filtering and per-candidate failure isolation


class _DeprecatedButFine:
    # emits deprecation chatter during fit, like some third-party dists
    def fit(self, data):
        warnings.warn("old API", DeprecationWarning)
        return stats.norm.fit(data)

    def __call__(self, *params):
        return stats.norm(*params)


class _Boom:
    def fit(self, data):
        raise RuntimeError("fit exploded")

    def __call__(self, *params):
        raise AssertionError("unreachable")


def test_39_deprecation_does_not_disqualify():
    data = np.random.default_rng(31).normal(size=80)
    res = fs.fit(data, _DeprecatedButFine(), "depr")
    assert len(res.warnings) == 0


def test_39_failure_isolated():
    data = np.random.default_rng(31).normal(size=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, failures = distfit._fit_all(
            data, [("Boom", _Boom()), ("Normal", stats.norm)])
    assert [r.distName for r in results] == ["Normal"]
    assert failures[0][0] == "Boom"


# ---------------------------------------------------------------------------
# items 40-43: core validation and draw-count protocol


def _step_lag1(draw, data):
    return {"x": data.recall("x", lag=1) + next(draw)}


def test_40_int32_steps_index():
    h = xr.DataArray(np.ones((3, 1)), dims=("steps", "variables"),
                     coords={"steps": np.arange(3, dtype=np.int32),
                             "variables": ["x"]})
    out = fs.simulate(_step_lag1, ntrials=4, nsteps=2, hist0=h)
    assert out.shape == (4, 1, 5)


def _step_extra_vars(draw, data):
    return {"zeta": next(draw), "x": data.recall("x", lag=1), "alpha": 1.0}


def test_41_variable_order_deterministic():
    h = xr.DataArray(np.ones((2, 1)), dims=("steps", "variables"),
                     coords={"steps": np.arange(2), "variables": ["x"]})
    out = fs.simulate(_step_extra_vars, ntrials=3, nsteps=1, hist0=h)
    # hist0 variables first, then new stepf variables in stepf order
    assert list(out.coords["variables"].values) == ["x", "zeta", "alpha"]


def test_42_empty_steps_hist0():
    h = xr.DataArray(np.empty((0, 1)), dims=("steps", "variables"),
                     coords={"steps": np.array([], dtype=int),
                             "variables": ["z"]})
    out = fs.simulate(_trial_one_draw, ntrials=4, nsteps=1, hist0=h)
    assert out.shape == (4, 1, 1)  # was a raw IndexError


def _greedy_trial(draw):
    u = next(draw)
    if u > 0.6:  # data-dependent extra draw
        next(draw)
    return {"x": u}


def test_43_variable_draw_count_raises():
    with pytest.raises(RuntimeError, match="fixed"):
        fs.simulate(_greedy_trial, ntrials=100, seed=1)


# ---------------------------------------------------------------------------
# item 44: nested-list validation consistent with ndarray rules


def test_44_vlvalidate_nested_lists():
    assert vlValidate([[1.0, 2.0], [3.0, 4.0]]) is False  # was True
    assert vlValidate(np.array([[1.0, 2.0], [3.0, 4.0]])) is False
    assert vlValidate([1.0, 2.0, 3.0]) is True
    assert vlValidate([[1.0, 2.0, 3.0]]) is True  # 1xN row, like ndarray


# ---------------------------------------------------------------------------
# item 45: fan() dim order; show() imports


def test_45_fan_transposed_input():
    pytest.importorskip("plotly")
    da = fs.simulate(_trial_one_draw, ntrials=6, nsteps=4, seed=2)
    f1 = fs.fan(da, "z")
    f2 = fs.fan(da.transpose("steps", "trials", "variables"), "z")
    for t1, t2 in zip(f1.data, f2.data):
        assert np.array_equal(np.asarray(t1.y, dtype=float),
                              np.asarray(t2.y, dtype=float))


def test_45_show_no_ipython_import():
    import funcsim.plotting as plotting_mod
    src = pathlib.Path(plotting_mod.__file__).read_text()
    assert "from IPython" not in src and "import IPython" not in src


# ---------------------------------------------------------------------------
# item 47: eut certainty equivalent is a float


def test_47_eut_ce_is_float():
    res = fs.eut(lambda y: np.log(y), [2.0, 3.0, 4.0])
    assert isinstance(res.CertaintyEquiv, float)


# ---------------------------------------------------------------------------
# item 48: Bartels midranks; Ljung-Box standard ACF; alpha display


def test_48_bartels_midranks():
    x = np.array([1.0, 3.0, 3.0, 2.0, 5.0, 4.0, 4.0, 6.0, 2.0, 7.0])
    res = bartels_test(x)
    ranks = stats.rankdata(x)
    num = np.sum((ranks[:-1] - ranks[1:]) ** 2)
    den = np.sum((ranks - (len(x) + 1) / 2.0) ** 2)
    assert np.isclose(res["RVN"], num / den)


def test_48_ljung_box_standard_acf():
    resid = np.random.default_rng(32).normal(size=100)
    xd = resid - resid.mean()
    r1 = np.sum(xd[1:] * xd[:-1]) / np.sum(xd ** 2)
    Q1 = 100 * 102 * (r1 ** 2 / 99)
    expected = 1 - stats.chi2.cdf(Q1, 1)
    assert np.isclose(ljung_box(resid, lags=3)[0], expected)


def test_48_alpha_formatting():
    txt = fs.screen(np.random.default_rng(33).normal(size=60), alpha=0.025)
    assert "2.5%" in txt  # int() truncation printed "2%"


# ---------------------------------------------------------------------------
# item 49: varange returns 1-D for scalar starts


def test_49_varange_dims():
    assert np.array_equal(varange(1, 5), np.array([1, 2, 3, 4, 5]))
    assert varange(1, 5).ndim == 1
    assert varange((1, 3), 5).shape == (2, 5)


# ---------------------------------------------------------------------------
# item 50: simulator validates rounds/precision divisibility


def test_50_simulator_divisibility():
    with pytest.raises(ValueError):
        simulator(lambda d: d.max(), 3, 10, 95)  # 95 % 10 != 0
