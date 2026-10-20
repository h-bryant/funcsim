import functools
import itertools
import warnings
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats
import funcsim as fs


# reference example: two scenarios on common states of nature, simulated at
# the funcsim defaults (500 trials, seed 6, Latin hypercube sampling)
SCENARIOS = {"s0": stats.norm(loc=50.0, scale=15.0),
             "s1": stats.chi2(df=47.0)}


def _one_source(ugen, dist):
    return {"w": float(dist.ppf(next(ugen)))}


def _two_source(ugen, dist):
    u = next(ugen)
    shock = next(ugen)
    return {"w": float(dist.ppf(u)) + 10.0 * shock}


def _wealth(f):
    return {name: fs.simulate(f=functools.partial(f, dist=dist))
                    .sel(variables="w", steps=0).to_numpy()
            for name, dist in SCENARIOS.items()}


G12 = functools.partial(fs.regretPower, exponent=1.2)
G2 = functools.partial(fs.regretPower, exponent=2.0)
W1 = _wealth(_one_source)
BEST1 = np.maximum(W1["s0"], W1["s1"])
W2 = _wealth(_two_source)
BEST2 = np.maximum(W2["s0"], W2["s1"])


def _linear(y):
    return y


# item 1: hand-computable case
def test_01_hand_case():
    # per-state index [0, 2]: ExpectedValue 1.0; the certainty equivalent
    # solves c - (2 - c) ** 2 = 1, so c = (5 - 5 ** 0.5) / 2
    target = (5.0 - 5.0 ** 0.5) / 2.0  # 1.3819660112501051
    for outcomes in ([1.0, 2.0], [2.0, 1.0]):
        r = fs.rteu(_linear, G2, 1.0, outcomes, [2.0, 2.0])
        assert r.ExpectedValue == 1.0
        np.testing.assert_allclose(r.CertaintyEquiv, target, rtol=1e-9)
    r0 = fs.rteu(_linear, G2, 0.0, [1.0, 2.0], [2.0, 2.0])
    np.testing.assert_allclose(r0.CertaintyEquiv, 1.5, rtol=1e-9)


# item 2: reduction to expected utility at k = 0
@pytest.mark.parametrize("name", ["s0", "s1"])
def test_02_k_zero_is_expected_utility(name):
    e = fs.eutIsoelastic(1.0, W1[name])
    for best in (BEST1, W1[name]):
        r = fs.rteu(np.log, G12, 0.0, W1[name], best)
        np.testing.assert_allclose(r.ExpectedValue, e.ExpectedValue, rtol=1e-8)
        np.testing.assert_allclose(r.CertaintyEquiv, e.CertaintyEquiv, rtol=1e-8)


# item 3: reference regression, one source of randomness per scenario
def test_03_reference_one_source():
    np.testing.assert_allclose(W1["s0"].mean(), 50.01, atol=0.005)
    np.testing.assert_allclose(W1["s1"].mean(), 47.01, atol=0.005)
    r = fs.rteu(np.log, G12, 1.5, W1["s0"], BEST1)
    # RteuResult(ExpectedValue=3.807724736360835, CertaintyEquiv=47.0236865245643)
    np.testing.assert_allclose(r.ExpectedValue, 3.807724736360835, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 47.0236865245643, rtol=1e-9)
    r = fs.rteu(np.log, G12, 1.5, W1["s1"], BEST1)
    # RteuResult(ExpectedValue=3.7602361689566735, CertaintyEquiv=46.387593488897636)
    np.testing.assert_allclose(r.ExpectedValue, 3.7602361689566735, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 46.387593488897636, rtol=1e-9)
    r = fs.rteu(np.log, G12, 0.0, W1["s0"], BEST1)
    # RteuResult(ExpectedValue=3.8579123527584698, CertaintyEquiv=47.36636382049248)
    np.testing.assert_allclose(r.ExpectedValue, 3.8579123527584698, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 47.36636382049248, rtol=1e-9)
    r = fs.rteu(np.log, G12, 0.0, W1["s1"], BEST1)
    # RteuResult(ExpectedValue=3.828876417845734, CertaintyEquiv=46.01081225291071)
    np.testing.assert_allclose(r.ExpectedValue, 3.828876417845734, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 46.01081225291071, rtol=1e-9)


# item 3: reference regression, two sources of randomness per scenario
def test_03_reference_two_sources():
    np.testing.assert_allclose(W2["s0"].mean(), 55.01, atol=0.005)
    np.testing.assert_allclose(W2["s1"].mean(), 52.01, atol=0.005)
    r = fs.rteu(np.log, G12, 1.5, W2["s0"], BEST2)
    # RteuResult(ExpectedValue=3.9232998514539013, CertaintyEquiv=52.30505329424999)
    np.testing.assert_allclose(r.ExpectedValue, 3.9232998514539013, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 52.30505329424999, rtol=1e-9)
    r = fs.rteu(np.log, G12, 1.5, W2["s1"], BEST2)
    # RteuResult(ExpectedValue=3.8701337138950938, CertaintyEquiv=51.396885253631034)
    np.testing.assert_allclose(r.ExpectedValue, 3.8701337138950938, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 51.396885253631034, rtol=1e-9)
    r = fs.rteu(np.log, G12, 0.0, W2["s0"], BEST2)
    # RteuResult(ExpectedValue=3.9629715130789855, CertaintyEquiv=52.61343540502131)
    np.testing.assert_allclose(r.ExpectedValue, 3.9629715130789855, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 52.61343540502131, rtol=1e-9)
    r = fs.rteu(np.log, G12, 0.0, W2["s1"], BEST2)
    # RteuResult(ExpectedValue=3.932453153362854, CertaintyEquiv=51.03201359611055)
    np.testing.assert_allclose(r.ExpectedValue, 3.932453153362854, rtol=1e-9)
    np.testing.assert_allclose(r.CertaintyEquiv, 51.03201359611055, rtol=1e-9)


# item 4: best equal to outcomes (the two-element menu of scenario and sure
# amount)
def test_04_best_equal_to_outcomes():
    for name in ("s0", "s1"):
        r = fs.rteu(np.log, G12, 1.5, W1[name], W1[name])
        np.testing.assert_allclose(r.ExpectedValue, np.log(W1[name]).mean(),
                                   rtol=1e-12)
        assert r.CertaintyEquiv < fs.eutIsoelastic(1.0, W1[name]).CertaintyEquiv
    # for s1 the menu best (s0's upper tail) exceeds the sure amount in many
    # states, so the two-element and menu-based certainty equivalents differ
    own = fs.rteu(np.log, G12, 1.5, W1["s1"], W1["s1"])
    menu = fs.rteu(np.log, G12, 1.5, W1["s1"], BEST1)
    # own.CertaintyEquiv = 45.97618347925898; menu.CertaintyEquiv = 46.387593488897636
    np.testing.assert_allclose(own.CertaintyEquiv, 45.97618347925898, rtol=1e-9)
    assert own.CertaintyEquiv < menu.CertaintyEquiv


# item 5: monotonicity in best
def test_05_monotone_in_best():
    menu = fs.rteu(np.log, G12, 1.5, W1["s0"], BEST1)
    own = fs.rteu(np.log, G12, 1.5, W1["s0"], W1["s0"])
    assert menu.ExpectedValue <= own.ExpectedValue


# item 6: a sure scenario is its own certainty equivalent
def test_06_sure_scenario():
    const = np.full_like(W1["s0"], 40.0)
    r = fs.rteu(np.log, G12, 1.5, const, np.maximum(const, W1["s0"]))
    assert r.CertaintyEquiv == 40.0
    # RteuResult(ExpectedValue=3.3844491757962007, CertaintyEquiv=40.0)
    np.testing.assert_allclose(r.ExpectedValue, 3.3844491757962007, rtol=1e-9)
    assert r.ExpectedValue < np.log(40.0)


# item 7: monotonicity in the regret coefficient
def test_07_monotone_in_k():
    ces = [fs.rteu(_linear, G2, k, [1.0, 2.0], [2.0, 2.0]).CertaintyEquiv
           for k in (0.0, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in itertools.pairwise(ces))


# item 8: ExpectedValue matches the formula
@pytest.mark.parametrize("name", ["s0", "s1"])
def test_08_expected_value_formula(name):
    y = W1[name]
    hand = np.mean(np.log(y) - 1.5 * (np.log(BEST1) - np.log(y)) ** 1.2)
    r = fs.rteu(np.log, G12, 1.5, y, BEST1)
    np.testing.assert_allclose(r.ExpectedValue, hand, rtol=1e-12)


# item 9: input handling
def test_09_vectorlike_inputs():
    y = [1.0, 2.0, 3.0, 2.5]
    b = [2.0, 2.0, 3.0, 4.0]
    ref = fs.rteu(np.log, G12, 1.5, y, b)
    assert isinstance(ref.ExpectedValue, float)
    assert isinstance(ref.CertaintyEquiv, float)
    ya = np.array(y)
    ba = np.array(b)
    forms = [
        (tuple(y), tuple(b)),
        (ya, ba),
        (ya.reshape(1, -1), ba.reshape(1, -1)),
        (ya.reshape(-1, 1), ba.reshape(-1, 1)),
        (pd.Series(y), pd.Series(b)),
        (pd.DataFrame({"w": y}), pd.DataFrame({"w": b})),
        (xr.DataArray(ya), xr.DataArray(ba)),
    ]
    for yy, bb in forms:
        r = fs.rteu(np.log, G12, 1.5, yy, bb)
        assert isinstance(r.ExpectedValue, float)
        assert isinstance(r.CertaintyEquiv, float)
        np.testing.assert_allclose(r.ExpectedValue, ref.ExpectedValue, rtol=1e-12)
        np.testing.assert_allclose(r.CertaintyEquiv, ref.CertaintyEquiv, rtol=1e-12)


def test_09_errors():
    with pytest.raises(ValueError, match="same length"):
        fs.rteu(np.log, G12, 1.0, [1.0, 2.0], [2.0, 2.0, 2.0])
    with pytest.raises(ValueError,
                       match="regretCoeff .* must be non-negative"):
        fs.rteu(np.log, G12, -1.0, [1.0, 2.0], [2.0, 2.0])
    with pytest.raises(ValueError, match=r"regretFunc\(0.0\) must be zero"):
        fs.rteu(np.log, lambda x: x + 1.0, 1.5, [1.0, 2.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="at position 1 best=2.0 is below"):
        fs.rteu(np.log, G12, 1.5, [1.0, 3.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="not a vector-like"):
        fs.rteu(np.log, G12, 1.5, np.ones((2, 2)), [2.0, 2.0])
    # a non-positive outcome under np.log raises a ValueError naming the
    # outcome, and no RuntimeWarning escapes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="not finite at outcome -1.0"):
            fs.rteu(np.log, G12, 1.5, [-1.0, 2.0], [2.0, 2.0])
        with pytest.raises(ValueError, match="not finite at outcome 0.0"):
            fs.rteu(np.log, G12, 1.5, [0.0, 2.0], [2.0, 2.0])


# item 10: regretPower
def test_10_regret_power():
    v = fs.regretPower(2.0, 2.0)
    assert isinstance(v, float)
    assert v == 4.0
    a = fs.regretPower(np.array([0.0, 1.0, 2.0]), 1.5)
    assert isinstance(a, np.ndarray)
    np.testing.assert_allclose(a, [0.0, 1.0, 2.0 ** 1.5], rtol=1e-12)
    with pytest.raises(ValueError, match="must be non-negative"):
        fs.regretPower(-1.0, 2.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        fs.regretPower(np.array([0.0, -1.0, 2.0]), 2.0)
    with pytest.raises(ValueError, match="exponent .* must be at least 1.0"):
        fs.regretPower(2.0, 0.5)


def test_10_linear_regret_preserves_eut_ranking():
    g1 = functools.partial(fs.regretPower, exponent=1.0)
    by_rteu = sorted(W1, key=lambda n: fs.rteu(np.log, g1, 1.5, W1[n], BEST1)
                     .ExpectedValue)
    by_eut = sorted(W1, key=lambda n: fs.eutIsoelastic(1.0, W1[n])
                    .ExpectedValue)
    assert by_rteu == by_eut
