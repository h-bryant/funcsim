import numpy as np
import pandas as pd
import xarray as xr
import pytest

from funcsim.conversions import vlValidate, vlToArray

def test_vlValidate_list():
    assert vlValidate([1, 2, 3]) is True

def test_vlValidate_tuple():
    assert vlValidate((1, 2, 3)) is True

def test_vlValidate_1d_array():
    assert vlValidate(np.array([1, 2, 3])) is True

def test_vlValidate_2d_array_column():
    assert vlValidate(np.array([[1], [2], [3]])) is True

def test_vlValidate_2d_array_row():
    assert vlValidate(np.array([[1, 2, 3]])) is True

def test_vlValidate_2d_array_invalid():
    assert vlValidate(np.array([[1, 2], [3, 4]])) is False

def test_vlValidate_series():
    assert vlValidate(pd.Series([1, 2, 3])) is True

def test_vlValidate_dataframe_column():
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert vlValidate(df) is True

def test_vlValidate_dataframe_invalid():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert vlValidate(df) is False

def test_vlValidate_xarray_1d():
    arr = xr.DataArray([1, 2, 3])
    assert vlValidate(arr) is True

def test_vlValidate_xarray_2d_column():
    arr = xr.DataArray([[1], [2], [3]])
    assert vlValidate(arr) is True

def test_vlValidate_xarray_2d_invalid():
    arr = xr.DataArray([[1, 2], [3, 4]])
    assert vlValidate(arr) is False

def test_vlToArray_list():
    arr = vlToArray([1, 2, 3])
    assert np.allclose(arr, np.array([1, 2, 3]))
    assert arr.ndim == 1

def test_vlToArray_tuple():
    arr = vlToArray((1, 2, 3))
    assert np.allclose(arr, np.array([1, 2, 3]))
    assert arr.ndim == 1

def test_vlToArray_series():
    arr = vlToArray(pd.Series([1, 2, 3]))
    assert np.allclose(arr, np.array([1, 2, 3]))
    assert arr.ndim == 1

def test_vlToArray_xarray():
    arr = vlToArray(xr.DataArray([1, 2, 3]))
    assert np.allclose(arr, np.array([1, 2, 3]))
    assert arr.ndim == 1

def test_vlToArray_2d_array_column():
    arr = vlToArray(np.array([[1], [2], [3]]))
    assert np.allclose(arr, np.array([1, 2, 3]))
    assert arr.ndim == 1

def test_vlToArray_2d_array_row():
    arr = vlToArray(np.array([[1, 2, 3]]))
    assert np.allclose(arr, np.array([1, 2, 3]))
    assert arr.ndim == 1

def test_vlToArray_invalid():
    with pytest.raises(ValueError):
        vlToArray(np.array([[1, 2], [3, 4]]))