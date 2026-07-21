import numpy as np
import pandas as pd
import pytest
import xarray as xr

import funcsim as fs


def _hist0():
    # historical data with a pandas PeriodIndex on the 'steps' dimension
    steps = pd.period_range("2020Q1", periods=3, freq="Q")
    a = np.array([[1.0], [1.01], [0.99]])
    return xr.DataArray(data=a, coords=(("steps", steps),
                                        ("variables", ["p"])))


def _step(draw, data):
    prev = data.recall("p", lag=1)
    return {"p": prev + 0.1 * (next(draw) - 0.5)}


def test_hist0_periodindex():
    # recursive-dynamic simulation with a PeriodIndex 'steps' index
    out = fs.simulate(f=_step, hist0=_hist0(), nsteps=2, ntrials=8)
    assert out.shape == (8, 1, 5)
    steps = list(out.indexes["steps"])
    assert steps[0] == pd.Period("2020Q1", freq="Q")
    assert steps[3] == pd.Period("2020Q4", freq="Q")
    assert steps[4] == pd.Period("2021Q1", freq="Q")


def test_fan_periodindex():
    # fan chart over a Period-labeled 'steps' axis
    go = pytest.importorskip("plotly.graph_objects")
    out = fs.simulate(f=_step, hist0=_hist0(), nsteps=2, ntrials=8)
    fig = fs.fan(out, "p")
    assert isinstance(fig, go.Figure)
