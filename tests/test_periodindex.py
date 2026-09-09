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


def _kaleido_like_dumps(fig) -> str:
    # kaleido >= 1.0 serializes the figure spec with orjson, falling back to
    # ``obj.tolist()`` for anything orjson cannot handle natively.  pandas
    # Timestamp has no ``tolist``, so it raises TypeError there, and does the
    # same under this stdlib emulation.
    import json
    return json.dumps(fig.to_dict(), default=lambda o: o.tolist())


def test_fan_periodindex_x_values_are_iso_strings():
    # every trace's x values must be plain ISO 8601 strings so that static
    # image export (kaleido/orjson) can serialize the figure
    pytest.importorskip("plotly.graph_objects")
    out = fs.simulate(f=_step, hist0=_hist0(), nsteps=2, ntrials=8)
    fig = fs.fan(out, "p")
    assert len(fig.data) == 6  # five quantile bands plus the mean line
    for trace in fig.data:
        assert all(isinstance(v, str) for v in trace.x)
    assert fig.data[-1].x[0] == "2020-01-01T00:00:00"   # 2020Q1 start
    assert fig.data[-1].x[-1] == "2021-01-01T00:00:00"  # 2021Q1 start
    _kaleido_like_dumps(fig)


def test_fan_monthly_periodindex_serializes():
    # monthly PeriodIndex derived from a DatetimeIndex, as in agec-643 ch. 10
    pytest.importorskip("plotly.graph_objects")
    steps = pd.date_range("2020-01-01", periods=3, freq="MS").to_period()
    a = np.array([[1.0], [1.01], [0.99]])
    hist0 = xr.DataArray(data=a, coords=(("steps", steps),
                                         ("variables", ["p"])))
    out = fs.simulate(f=_step, hist0=hist0, nsteps=3, ntrials=8)
    fig = fs.fan(out, "p")
    assert fig.data[-1].x[-1] == "2020-06-01T00:00:00"
    _kaleido_like_dumps(fig)


def test_fan_integer_index_unchanged():
    # integer 'steps' labels pass through untouched and still serialize
    pytest.importorskip("plotly.graph_objects")
    a = np.array([[1.0], [1.01], [0.99]])
    hist0 = xr.DataArray(data=a, coords=(("steps", [0, 1, 2]),
                                         ("variables", ["p"])))
    out = fs.simulate(f=_step, hist0=hist0, nsteps=2, ntrials=8)
    fig = fs.fan(out, "p")
    assert list(fig.data[-1].x) == [0, 1, 2, 3, 4]
    _kaleido_like_dumps(fig)


def test_xvalues_object_index_with_periods_and_timestamps():
    # object-dtype index mixing Periods, Timestamps, and other labels
    from funcsim.plotting import _xvalues
    idx = pd.Index([pd.Period("2020Q1", freq="Q"),
                    pd.Timestamp("2021-06-15"), 7], dtype=object)
    assert _xvalues(idx) == ["2020-01-01T00:00:00", "2021-06-15T00:00:00", 7]
    assert _xvalues(pd.Index([0, 1, 2])) == [0, 1, 2]
