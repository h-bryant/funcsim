"""
Item 6 of the 2026-09-24 course handoff: simulate decides from the signature
of f whether to pass the history, so a step function with a defaulted second
parameter (a setting such as scale=1.0) runs as a static simulation.
"""
import functools

import numpy as np
import pytest
import xarray as xr
from scipy import stats

import funcsim as fs
from funcsim import core


def _hist0():
    return xr.DataArray(data=np.array([[1.0], [2.0], [3.0]]),
                        dims=("steps", "variables"),
                        coords={"steps": [0, 1, 2], "variables": ["p"]})


def test_defaulted_second_parameter_runs_static_with_its_default():
    def plain(ugen):
        return {"x": next(ugen)}

    def scaled(ugen, scale=2.0):
        return {"x": scale * next(ugen)}

    a = fs.simulate(f=plain, ntrials=50, seed=1).sel(steps=0, variables="x")
    b = fs.simulate(f=scaled, ntrials=50, seed=1).sel(steps=0, variables="x")
    np.testing.assert_allclose(b.values, 2.0 * a.values)


def test_documented_partial_scenario_example():
    # the overview's common-random-numbers example: binding by keyword makes
    # 'prob' keyword-only with a default, which used to be counted as a
    # second positional argument
    def trial(ugen, prob):
        eps = stats.norm.ppf(next(ugen))
        b = stats.bernoulli.ppf(next(ugen), prob)
        return {"eps": eps, "b": b}

    low = fs.simulate(f=functools.partial(trial, prob=0.25), ntrials=400)
    high = fs.simulate(f=functools.partial(trial, prob=0.50), ntrials=400)
    np.testing.assert_allclose(low.sel(variables="eps").values,
                               high.sel(variables="eps").values)
    assert float(low.sel(variables="b").mean()) < float(
        high.sel(variables="b").mean())


def test_two_required_parameters_receive_history():
    def step(ugen, history):
        return {"p": history.recall("p", lag=1) + 1.0 + 0.0 * next(ugen)}

    out = fs.simulate(f=step, hist0=_hist0(), nsteps=2, ntrials=3)
    np.testing.assert_allclose(out.sel(variables="p", trials=0).values,
                               [1.0, 2.0, 3.0, 4.0, 5.0])


def test_second_parameter_named_hist_with_default_receives_history():
    def step(ugen, hist=None):
        assert hist is not None
        return {"p": hist.recall("p", lag=1) * 2.0 + 0.0 * next(ugen)}

    out = fs.simulate(f=step, hist0=_hist0(), nsteps=2, ntrials=2)
    np.testing.assert_allclose(out.sel(variables="p", trials=1).values,
                               [1.0, 2.0, 3.0, 6.0, 12.0])


def test_history_plus_defaulted_setting():
    def step(ugen, hist, growth=0.5):
        return {"p": hist.recall("p", lag=1) + growth + 0.0 * next(ugen)}

    out = fs.simulate(f=step, hist0=_hist0(), nsteps=2, ntrials=2)
    np.testing.assert_allclose(out.sel(variables="p", trials=0).values,
                               [1.0, 2.0, 3.0, 3.5, 4.0])


def test_lambda_and_callable_object():
    out = fs.simulate(f=lambda ugen: {"u": next(ugen)}, ntrials=10)
    assert out.sizes["trials"] == 10

    class Trial:
        def __call__(self, ugen, scale=3.0):
            return {"u": scale * next(ugen)}

    out = fs.simulate(f=Trial(), ntrials=10)
    assert float(out.sel(variables="u").max()) > 1.0


def test_arity_helper_directly():
    assert core._stepf_arity(lambda ugen: None) == 1
    assert core._stepf_arity(lambda ugen, scale=1.0: None) == 1
    assert core._stepf_arity(lambda ugen, hist: None) == 2
    assert core._stepf_arity(lambda ugen, hist=None: None) == 2
    assert core._stepf_arity(lambda ugen, h, s=1.0: None) == 2
    assert core._stepf_arity(lambda ugen, *args: None) == 1
    assert core._stepf_arity(lambda ugen, hist, **kw: None) == 2


def test_ambiguous_or_unsupported_signatures_raise():
    def hist_third(ugen, scale=1.0, hist=None):
        return {"x": 1.0}

    def three_required(ugen, hist, extra):
        return {"x": 1.0}

    def kwonly_required(ugen, *, k):
        return {"x": 1.0}

    def no_params():
        return {"x": 1.0}

    with pytest.raises(ValueError, match="hist"):
        fs.simulate(f=hist_third, ntrials=2)
    with pytest.raises(ValueError, match="without defaults"):
        fs.simulate(f=three_required, ntrials=2)
    with pytest.raises(ValueError, match="keyword-only"):
        fs.simulate(f=kwonly_required, ntrials=2)
    with pytest.raises(ValueError, match="first parameter"):
        fs.simulate(f=no_params, ntrials=2)
