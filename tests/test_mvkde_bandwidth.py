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


# ---------------------------------------------------------------------------
# item 2: Silverman's rule of thumb


def _factor(kde) -> float:
    # the common factor on the data's standard deviation, from the
    # standardized-unit bandwidth matrix (a diagonal of factor squared)
    return float(np.sqrt(kde._bw[0, 0]))


def test_silverman_one_column_matches_scipy_and_kde():
    x = _data(1)["x0"]
    mv = fs.MvKde(x.to_frame(), bw="silverman")
    uni = fs.Kde(x, bw="silverman")
    scipy_factor = stats.gaussian_kde(x.to_numpy()).silverman_factor()
    assert _factor(mv) == pytest.approx(scipy_factor, rel=1e-12)
    assert uni.gkde.factor == pytest.approx(scipy_factor, rel=1e-12)
    # in the units of the data the two classes differ only through the
    # standard deviation each applies the factor to (ddof=0 in MvKde,
    # ddof=1 in Kde through scipy)
    mv_h2 = mv._stds[0] ** 2 * mv._bw[0, 0]
    uni_h2 = uni.gkde.covariance[0, 0]
    assert mv_h2 / x.var(ddof=0) == pytest.approx(uni_h2 / x.var(ddof=1),
                                                  rel=1e-12)


def test_silverman_k2_unchanged():
    # chapter 15's example has two variables, where Silverman's and Scott's
    # rules coincide: the factor and the draws must not move
    data = _data(2)
    silverman = fs.MvKde(data, bw="silverman")
    scott = fs.MvKde(data, bw="scott")
    assert _factor(silverman) == pytest.approx(M ** (-1.0 / 6.0), rel=1e-14)
    assert _factor(silverman) == pytest.approx(0.3928606365489575, rel=1e-14)
    np.testing.assert_array_equal(silverman._bw, scott._bw)
    np.testing.assert_array_equal(_draws(silverman, 50), _draws(scott, 50))


@pytest.mark.parametrize("K", [1, 3, 4])
def test_silverman_factor_formula(K):
    kde = fs.MvKde(_data(K), bw="silverman")
    expected = ((K + 2.0) * M / 4.0) ** (-1.0 / (K + 4.0))
    inverted = (4.0 * M / (K + 2.0)) ** (-1.0 / (K + 4.0))  # pre-0.2.8
    assert _factor(kde) == pytest.approx(expected, rel=1e-14)
    assert abs(_factor(kde) - inverted) > 1e-3
    assert np.allclose(kde._bw, np.eye(K) * expected ** 2)


def test_scott_factor_formula():
    for K in (1, 2, 3):
        kde = fs.MvKde(_data(K), bw="scott")
        assert _factor(kde) == pytest.approx(M ** (-1.0 / (K + 4.0)),
                                             rel=1e-14)


# ---------------------------------------------------------------------------
# item 3: the bandwidth property


def test_kde_bandwidth_scott():
    x = _data(1)["x0"]
    kde = fs.Kde(x)
    assert isinstance(kde.bandwidth, float)
    assert kde.bandwidth == pytest.approx(kde.gkde.factor * x.std(ddof=1),
                                          rel=1e-12)
    assert kde.bandwidth == pytest.approx(M ** (-0.2) * x.std(ddof=1),
                                          rel=1e-12)


def test_kde_bandwidth_silverman_and_float():
    x = _data(1)["x0"]
    silverman = fs.Kde(x, bw="silverman")
    assert silverman.bandwidth == pytest.approx(
        (3.0 * M / 4.0) ** (-0.2) * x.std(ddof=1), rel=1e-12)
    assert fs.Kde(x, bw=0.5).bandwidth == pytest.approx(0.5, rel=1e-12)


def test_mvkde_bandwidth_scott():
    data = _data(3)
    kde = fs.MvKde(data)
    H = kde.bandwidth
    assert isinstance(H, pd.DataFrame)
    assert list(H.index) == list(data.columns)
    assert list(H.columns) == list(data.columns)
    expected = np.diag(data.var(ddof=0).to_numpy()) * M ** (-2.0 / 7.0)
    np.testing.assert_allclose(H.to_numpy(), expected, rtol=1e-12)


def test_mvkde_bandwidth_silverman():
    data = _data(3)
    H = fs.MvKde(data, bw="silverman").bandwidth
    expected = np.diag(data.var(ddof=0).to_numpy()) * \
        (5.0 * M / 4.0) ** (-2.0 / 7.0)
    np.testing.assert_allclose(H.to_numpy(), expected, rtol=1e-12)


def test_mvkde_bandwidth_round_trip():
    data = _data(2)
    for bw in (H2, np.array(H2)):
        H = fs.MvKde(data, bw=bw).bandwidth
        np.testing.assert_allclose(H.to_numpy(), np.array(H2), rtol=1e-12)


def test_mvkde_bandwidth_names_from_array_data():
    H = fs.MvKde(_data(2).to_numpy()).bandwidth
    assert list(H.index) == ["v0", "v1"]
    assert list(H.columns) == ["v0", "v1"]


def test_mvkde_bandwidth_reports_repaired_matrix():
    data = _data(2)
    singular = [[1.0, 1.0], [1.0, 1.0]]
    with pytest.warns(UserWarning):
        kde = fs.MvKde(data, bw=singular)
    H = kde.bandwidth.to_numpy()
    assert np.all(np.linalg.eigvalsh(H) > 0.0)
    # the reported matrix is the one the draws apply
    D = np.diag(kde._stds)
    np.testing.assert_allclose(H, D @ kde._chol @ kde._chol.T @ D,
                               rtol=1e-10, atol=1e-12)


def test_one_column_classes_differ_only_by_ddof():
    x = _data(1)["x0"]
    mv = fs.MvKde(x.to_frame()).bandwidth.iloc[0, 0]
    uni = fs.Kde(x).bandwidth
    assert mv == pytest.approx(uni ** 2 * (M - 1) / M, rel=1e-12)


def test_bandwidth_is_read_only():
    x = _data(1)["x0"]
    with pytest.raises(AttributeError):
        fs.Kde(x).bandwidth = 1.0
    with pytest.raises(AttributeError):
        fs.MvKde(x.to_frame()).bandwidth = np.eye(1)
