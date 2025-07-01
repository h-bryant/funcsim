from typing       import TypeVar, Union, Sequence, Any
import numpy      as np
import pandas     as pd
import xarray     as xr
from numpy.typing import NDArray

T = TypeVar("T")

VectorLike = Union[
    Sequence[T],         # list[T], tuple[T], etc.
    NDArray[Any],        # np.NDArray of any dtype (runtime shape not enforced)
    pd.Series,           # 1‑D pandas Series
    pd.DataFrame,        # 1‑D pandas DataFrame
    xr.DataArray         # 1‑D xarray DataArray
]

ArrayLike = Union[
    NDArray[Any],        # np.NDArray of any dtype (runtime shape not enforced)
    pd.DataFrame,        # 2‑D pandas Series
    xr.DataArray         # 2‑D xarray DataArray
]


def vlValidate(vl: VectorLike) -> bool:
    """
    Validate if the input is a vector-like object.

    Parameters
    ----------
    vl : VectorLike
        The object to validate.

    Returns
    -------
    bool
        True if the input is a vector-like object, False otherwise.
    """
    if isinstance(vl, (list, tuple, pd.Series)):
        return True
    elif isinstance(vl, np.ndarray):
        a = np.asarray(vl)
        if a.ndim > 2:
            return False
        if a.ndim == 2:
            if not (a.shape[0] == 1 or a.shape[1] == 1):
                return False
        return True
    elif isinstance(vl, xr.DataArray):
        a = np.asarray(vl)
        if a.ndim > 2:
            return False
        if a.ndim == 2:
            if not (a.shape[0] == 1 or a.shape[1] == 1):
                return False
        return True
    elif isinstance(vl, pd.DataFrame):  
        a = np.asarray(vl)
        if a.ndim > 2:
            return False
        if a.ndim == 2:
            if not (a.shape[0] == 1 or a.shape[1] == 1):
                return False
        return True            
    return False
    

def vlToArray(vl: VectorLike) -> NDArray:
    """
    Convert a vector-like object to a 1-D NumPy array.

    Parameters
    ----------
    vl : VectorLike
        The vector-like object to convert.

    Returns
    -------
    NDArray
        The converted 1-D NumPy array.
    """
    if not vlValidate(vl):
        raise ValueError("argument passed is not a vector-like object")
    if isinstance(vl, (list, tuple)):
        a = np.array(vl)
    elif isinstance(vl, pd.Series):
        a = vl.to_numpy()
    elif isinstance(vl, xr.DataArray):
        a = vl.values
    else:
        a = np.asarray(vl)
    return a.flatten()  # Ensure the array is 1-D


def alValidate(al: ArrayLike) -> bool:
    """
    Validate if the input is a 2-D array-like object.

    Parameters
    ----------
    al : ArrayLike
        The object to validate.

    Returns
    -------
    bool
        True if the input is a array-like object, False otherwise.
    """
    if isinstance(al, np.ndarray):
        a = np.asarray(al)
    elif isinstance(al, xr.DataArray):
        a = np.asarray(al)
    elif isinstance(al, pd.DataFrame):  
        a = np.asarray(al)
    else:
        return False
    if a.ndim != 2:
        return False
    return True


def alToArray(al: ArrayLike) -> NDArray:
    """
    Convert an array-like object to a 2-D NumPy array.

    Parameters
    ----------
    al : ArrayLike
        The array-like object to convert.

    Returns
    -------
    NDArray
        The converted 2-D NumPy array.
    """
    if not alValidate(al):
        raise ValueError("argument passed is not an array-like object")
    if isinstance(al, pd.DataFrame):
        a = al.to_numpy()
    elif isinstance(al, xr.DataArray):
        a = al.values
    else:  # Assume it's a NumPy array
        a = np.asarray(al)
    return a


def vlCoords(vl: VectorLike) -> pd.Index:
    """
    Get or create coordinates for a vector-like object.

    Parameters
    ----------
    vl : VectorLike
        The vector-like object to convert.

    Returns
    -------
    List[str]
        A list of coordinates corresponding to the vector-like object.
    """
    if not vlValidate(vl):
        raise ValueError("argument passed is not a vector-like object")
    if isinstance(vl, (list, tuple)):
        return pd.Index(list(range(max(shape(vl)))))
    if isinstance(vl, (np.ndarray)):
        if vl.ndim == 1:
            return pd.Index(list(range(len(vl))))
        if vl.ndim == 2:
            if vl.shape[0] == 1:
                return pd.Index(list(range(vl.shape[1])))
            if vl.shape[1] == 1:
                return pd.Index(list(range(vl.shape[0])))
    if isinstance(vl, pd.Series):
        return vl.index
    if isinstance(vl, xr.DataArray):
        dim_size_to_name = dict(zip(vl.shape, vl.dims))
        longest_dim = max(dim_size_to_name.keys())
        ret = vl.coords[dim_size_to_name[longest_dim]]
        return pd.Index(ret)
