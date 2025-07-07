"""Iman & Conover (1982) dependence induction"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import conversions


def _vdw(i, N):
    # calculate a single van der Waerden score
    assert isinstance(N, int) and N > 1
    assert isinstance(i, int) and i > 0 and i <= N
    return stats.norm().ppf(i / (N + 1.0))


def _vdw_row(N):
    # calculate a length N sequence of van der Waerden scores
    assert isinstance(N, int) and N > 1
    return np.array(list(map(lambda i: _vdw(i, N), range(1, N+1))))


def _shuffle(a):
    # return a shuffle of the one-dimensional np.array "v" without mutating
    assert isinstance(a, np.ndarray) and len(a.shape) == 1
    return np.array(list(map(lambda t: t[1],
                             sorted(zip(np.random.random(len(a)), a)))))


def _arrange(example, values):
    # arrange the items in "values" in the same rank order
    # as the items in "example". Basically, this IC step 7 for a single row
    assert len(example) == len(values)

    # ranks of the items in "example"
    desired_ranks = stats.rankdata(example)

    # map so we can look up the values in "values" by their corresponding rank
    value_map = dict(zip(stats.rankdata(values), values))

    return np.array(list(map(lambda rank: value_map[rank], desired_ranks)))


def imanconover(spear: conversions.ArrayLike,
                vectors: List[conversions.VectorLike],
                names: List[str] = [],
                )-> pd.DataFrame:

    """
    Induce a Spearman correlation stucture using the Iman & Conover method.

    Parameters
    ----------
    spear : ArrayLike
        Desired Spearman correlation matrix (must be symetric and
        positive definite).
    vectors : list of VectorLike
        List of vectors, each representing draws for one variable.
    names : list of str, optional
        Column names for the output DataFrame. Defaults to v1, v2, ...

    Returns
    -------
    pandas.DataFrame
        DataFrame with one column for each vector/variable, where the
        data refelct the desired Spearman correlation structure.

    References
    ----------
    Iman, R. L., & Conover, W. J. (1980). Small sample sensitivity analysis
    techniques for computer models, with an application to risk assessment.
    Communications in Statistics - Theory and Methods, 9(17), 1749–1842.
    """
    K = len(vectors)
    N = len(vectors[0])

    if names != [] and len(vectors) != len(names):
        raise ValueError("The number of vectors must match"
                         "the number of names.") 
    if len(vectors) != spear.shape[0]:
        raise ValueError("The number of vectors must match"
                         "the number of rows in the Spearman matrix.")
    for v in vectors:
        if len(v) != N:
            raise ValueError("All vectors must have the same length.")

    # IC step 2
    L = np.linalg.cholesky(spear)

    # IC step 3
    vdw_scores = _vdw_row(N)

    # IC step 4
    R_ind = np.array(list(map(lambda i: _shuffle(vdw_scores), range(K))))

    # IC step 5
    R = L.dot(R_ind)

    # IC step 7
    array = np.array(list(map(_arrange, R, vectors))).transpose()

    # pack into a DataFrame
    if names == []:
        names = [f"v{i+1}" for i in range(K)]
    index = conversions.vlCoords(vectors[0])
    return pd.DataFrame(array, columns=names, index=index)


if __name__ == "__main__":
    # usage example
    v1 = stats.norm(4.0, 0.2).rvs(30)
    v2 = stats.norm(0.0, 3.2).rvs(30)
    rho_s = np.array([[1.0, 0.89], [0.89, 1.0]])
    ic = imanconover(rho_s, [v1, v2])
    print(ic)
