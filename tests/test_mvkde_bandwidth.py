"""
MvKde bandwidth handling: the forms a bandwidth matrix may take, the two
rules of thumb, and the ``bandwidth`` property (handoff of 2026-09-29,
items 1 to 3).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import funcsim as fs


M = 272  # the observation count of book chapter 15's example


def _data(K: int, seed: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"x{k}" for k in range(K)]
    return pd.DataFrame(rng.normal(size=(M, K)) * (1.0 + np.arange(K)),
                        columns=cols)


def _draws(kde, n: int, seed: int = 3) -> np.ndarray:
    ugen = iter(np.random.default_rng(seed).random(n * (kde._K + 1)))
    return np.array([kde.draw(ugen).to_numpy() for _ in range(n)])


H2 = [[0.052, 0.510], [0.510, 8.882]]  # chapter 15's plug-in matrix


# ---------------------------------------------------------------------------
# item 1: a bandwidth matrix given as a NumPy array


def test_ndarray_bandwidth_accepted():
    data = _data(2)
    fs.MvKde(data, bw=np.array(H2))  # raised ValueError before 0.2.8


def test_ndarray_and_nested_list_give_identical_draws():
    data = _data(2)
    from_list = fs.MvKde(data, bw=H2)
    from_array = fs.MvKde(data, bw=np.array(H2))
    np.testing.assert_array_equal(_draws(from_list, 50),
                                  _draws(from_array, 50))


def test_dataframe_bandwidth_gives_identical_draws():
    data = _data(2)
    from_list = fs.MvKde(data, bw=H2)
    from_frame = fs.MvKde(data, bw=pd.DataFrame(H2, index=data.columns,
                                                columns=data.columns))
    np.testing.assert_array_equal(_draws(from_list, 50),
                                  _draws(from_frame, 50))


def test_wrong_shaped_array_raises_shape_error():
    data = _data(2)
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        fs.MvKde(data, bw=np.eye(3))
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        fs.MvKde(data, bw=np.array([0.1, 0.2]))


def test_misspelled_method_raises_options_error():
    data = _data(2)
    with pytest.raises(ValueError, match="unknown bandwidth method"):
        fs.MvKde(data, bw="silvermann")


def test_none_is_scott():
    data = _data(2)
    np.testing.assert_array_equal(_draws(fs.MvKde(data, bw=None), 20),
                                  _draws(fs.MvKde(data, bw="scott"), 20))
