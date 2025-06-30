import scipy.stats as stats
import conversions


def swtest(data: conversions.VectorLike) -> tuple:
    """
    Shapiro-Wilk test statistic for normality.

    Parameters
    ----------
    data : VectorLike
        The input data to test for normality.

    Returns
    -------
    ShapiroResult
        A named tuple with fields:

        statistic : float
            The Shapiro-Wilk test statistic.
        pvalue : float
            The p-value of the test.
    """
    dataA = conversions.vlToArray(data)
    return stats.shapiro(data)