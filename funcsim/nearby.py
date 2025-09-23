import warnings
import numpy as np
import pandas as pd
import xarray as xr
import conversions


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

warnings.showwarning = custom_warning_format


def is_positive_definite(A):
    """Checks whether a matrix is positive definite"""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def repackage(orig, newA):
    # return new array with a type and any indexing that
    # match 'orig', but data from 'newA'
    if isinstance(orig, pd.DataFrame):
        df = pd.DataFrame(data=newA, index=orig.index, columns=orig.columns)
        return df
    elif isinstance(orig, xr.DataArray):
        da = data.copy(data=newA)
        return da
    else:
        return newA


def nearestpd(array):
    """
    Return the nearest positive definite matrix to the input matrix.

    This function computes the nearest symmetric positive definite matrix
    (in the Frobenius norm) to the given matrix using an algorithm based on
    Higham (1988).

    Parameters
    ----------
    array : ArrayLike
        Input matrix to be converted to the nearest positive definite matrix.

    Returns
    -------
    ArrayLike
        The nearest symmetric positive definite matrix, with a type matching
        the passed array.

    References
    ----------
    Higham, N.J. (1988). Computing a nearest symmetric positive semidefinite
    matrix. Linear Algebra and its Applications, 103, 103-118.
    https://doi.org/10.1016/0024-3795(88)90223-6

    Notes
    -----
    This is a Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code.
    """
    # convert to numpy array
    A = conversions.alToArray(array)

    if is_positive_definite(A):
        return array

    # original matrix was not positive definite; try Higham's 1st method
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        msg = ("Higham's (1988) first method was employed to compute "
               "the pos. def. matrix nearest to the sample covariance matrix.")
        warnings.warn(msg, UserWarning)
        return repackage(array, A3)

    # matrix is still not positive definite; try Higham's 2nd method
    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    Id = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += Id * (-mineig * k ** 2 + spacing)
        k += 1
    msg = ("Higham's (1988) second method was employed to compute "
            "the pos. def. matrix nearest to the sample covariance matrix.")
    warnings.warn(msg, UserWarning)
    return repackage(array, A3)
