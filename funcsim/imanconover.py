"""Iman & Conover (1982) dependence induction"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from . import conversions


def _vdw(i, N):
    # calculate a single van der Waerden score
    assert isinstance(N, int) and N > 1
    assert isinstance(i, int) and i > 0 and i <= N
    return stats.norm().ppf(i / (N + 1.0))


def _vdw_row(N):
    # calculate a length N sequence of van der Waerden scores
    assert isinstance(N, int) and N > 1
    return np.array(list(map(lambda i: _vdw(i, N), range(1, N+1))))


def _arrange(example, values):
    # arrange the items in "values" in the same rank order
    # as the items in "example". Basically, this IC step 7 for a single row
    assert len(example) == len(values)

    # ordinal ranks of the items in "example" (ordinal, rather than the
    # default average, so that tied values cannot collapse the ranks)
    desired_ranks = stats.rankdata(example, method="ordinal").astype(int)

    # the value holding rank r is element r-1 of the sorted values
    return np.sort(np.asarray(values))[desired_ranks - 1]


def imanconover(spear: conversions.ArrayLike,
                vectors: List[conversions.VectorLike],
                names: List[str] = [],
                seed: Optional[int] = None,
                )-> pd.DataFrame:

    """
    Induce a Spearman correlation structure using the Iman & Conover method.

    Parameters
    ----------
    spear : ArrayLike
        Desired Spearman correlation matrix (must be symmetric and
        positive definite).
    vectors : list of VectorLike
        List of vectors, each representing draws for one variable.
    names : list of str, optional
        Column names for the output DataFrame. Defaults to v1, v2, ...
    seed : int, optional
        Seed for the pseudo-random shuffling of scores.  If None (the
        default), fresh entropy is used and results vary across calls.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one column for each vector/variable, where the
        data reflect the desired Spearman correlation structure.

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
    rng = np.random.default_rng(seed)
    R_ind = np.array([rng.permutation(vdw_scores) for _ in range(K)])

    # IC steps 5 & 6: remove the incidental sample correlation of the
    # shuffled score matrix (variance-reduction step 6 of Iman & Conover),
    # then impose the target correlation
    E = np.corrcoef(R_ind)
    R = L @ np.linalg.inv(np.linalg.cholesky(E)) @ R_ind

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
