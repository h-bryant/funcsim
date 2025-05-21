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
        raise ValueError("vlToArray: vl is not a vector-like object")
    if isinstance(vl, (list, tuple)):
        a = np.array(vl)
    elif isinstance(vl, pd.Series):
        a = vl.to_numpy()
    elif isinstance(vl, xr.DataArray):
        a = vl.values
    else:
        a = np.asarray(vl)
    return a.flatten()  # Ensure the array is 1-D