import conversions
from typing import Callable


def edf(sample: conversions.VectorLike) -> Callable[[float], float]:
    """
    Create an empirical distribution function (EDF) from a data sample.

    Parameters
    ----------
    sample : VectorLike
        1-D data sample (list, np.array, pd.Series, or 1-D xr.DataArray).

    Returns
    -------
    Callable[[float], float]
        A function that computes the EDF at a given value.
    """
    sampleVec = conversions.vlToArray(sample)
    M = float(len(sampleVec))

    def edf(v):
        return sum(map(lambda a: 1 if a <= v else 0, sample)) / M

    return edf