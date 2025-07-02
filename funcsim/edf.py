import conversions
from typing import Callable


def edf(data: conversions.VectorLike) -> Callable[[float], float]:
    """
    Create an empirical distribution function (EDF) from a data sample.

    Parameters
    ----------
    data : VectorLike
        1-D data sample (list, np.array, pd.Series, or 1-D xr.DataArray).

    Returns
    -------
    Callable[[float], float]
        A function that computes the EDF at a given value.
    """
    sampleVec = conversions.vlToArray(data)
    M = float(len(sampleVec))

    def edf(v):
        return sum(map(lambda a: 1 if a <= v else 0, sampleVec)) / M

    return edf