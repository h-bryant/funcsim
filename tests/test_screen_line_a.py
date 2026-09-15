import numpy as np
import funcsim as fs
from funcsim.screen import adf_test_with_const


def _line_a(txt):
    return txt.split("A) H0")[1].split("B) H0")[0]


def test_line_a_accepts_iid_series_with_nonzero_mean():
    # an i.i.d. series with a nonzero mean is stationary, so line A must
    # reject the unit-root null (no warning); a no-constant ADF equation
    # cannot do this because its alternative is zero-mean stationarity
    rng = np.random.default_rng(7)
    assert "WARNING" not in _line_a(fs.screen(rng.random(60) + 5.0))


def test_line_a_flags_random_walk_with_nonzero_level():
    rng = np.random.default_rng(11)
    rw = np.cumsum(rng.normal(size=120)) + 5.0
    assert "WARNING" in _line_a(fs.screen(rw))


def test_adf_with_const_rejects_for_iid_nonzero_mean():
    rng = np.random.default_rng(3)
    p = adf_test_with_const(rng.random(60) + 5.0, max_lag=2, n_sim=500,
                            seed=42)[1]
    assert p < 0.05
