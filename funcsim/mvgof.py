"""
Two-sample multivariate goodness-of-fit tests: energy and MMD.

Both tests ask whether two samples with the same variables were drawn from
the same joint distribution, which is the question a simulation modeler asks
of simulated output against historical data.  Both are built from the
Euclidean distances between observations, so each variable's scale matters;
by default the variables are standardized by their pooled mean and standard
deviation before anything is computed.

The permutation p-values here are exact in the usual finite-sample sense,
and the optional chi-square approximation is that of Shen, Panda, and
Vogelstein (2022), applied through the equivalence of the two-sample tests
to distance correlation (Shen and Vogelstein, 2021).  The reported
statistics and p-values agree with those of the ``hyppo`` package
(Panda et al., 2021) up to permutation noise, except that ``hyppo``'s MMD
statistic is a rescaled distance correlation rather than the squared MMD
itself; this module reports the textbook statistics.
"""

from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2

from . import conversions
from .ecdfgof import GofResult


Standardize = Union[bool, str]


def _as_matrix(a: conversions.ArrayLike, name: str) -> NDArray:
    # one observation per row; a vector is one variable
    if conversions.alValidate(a):
        arr = np.asarray(conversions.alToArray(a), dtype=float)
    elif conversions.vlValidate(a):
        arr = np.asarray(conversions.vlToArray(a), dtype=float)[:, np.newaxis]
    else:
        raise ValueError(f"{name} must be a one- or two-dimensional "
                         "array-like object")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must have at least two observations")
    return arr


def _prepare(x: conversions.ArrayLike,
             y: conversions.ArrayLike,
             standardize: Standardize) -> tuple[NDArray, int]:
    """
    Validate, standardize, and pool the two samples.

    Returns the pooled matrix (sample ``x`` first) and the size of ``x``.
    """
    xa = _as_matrix(x, "x")
    ya = _as_matrix(y, "y")
    if xa.shape[1] != ya.shape[1]:
        raise ValueError("x and y must have the same number of variables")
    pooled = np.vstack([xa, ya])
    m = xa.shape[0]

    if standardize is True or standardize == "pooled":
        ref = pooled
    elif standardize == "first":
        ref = xa
    elif standardize is False or standardize is None:
        ref = None
    else:
        raise ValueError("standardize must be True, 'pooled', 'first', "
                         "or False")

    if ref is not None:
        sd = ref.std(axis=0, ddof=1)
        if np.any(sd == 0.0):
            raise ValueError("a variable has zero variance in the "
                             "standardizing sample; pass "
                             "standardize=False or drop it")
        pooled = (pooled - ref.mean(axis=0)) / sd

    return pooled, m


def _block_sums(A: NDArray, ix: NDArray, iy: NDArray,
                total: float) -> tuple[float, float, float]:
    """
    Sums of a symmetric matrix over the x-x, y-y, and x-y blocks implied by
    the index sets ``ix`` and ``iy`` (a partition of the rows).
    """
    r = A[:, ix].sum(axis=1)
    sxx = float(r[ix].sum())
    sxy = float(r[iy].sum())
    syy = total - sxx - 2.0 * sxy
    return sxx, syy, sxy


def _energy_from_sums(sxx: float, syy: float, sxy: float, m: int, n: int,
                      unbiased: bool) -> float:
    # within-sample sums exclude the (zero) diagonal in either case
    if unbiased:
        return 2.0 * sxy / (m * n) - sxx / (m * (m - 1)) - syy / (n * (n - 1))
    return 2.0 * sxy / (m * n) - sxx / (m * m) - syy / (n * n)


def _mmd_from_sums(sxx: float, syy: float, sxy: float, dxx: float, dyy: float,
                   m: int, n: int, unbiased: bool) -> float:
    # dxx, dyy are the diagonal (self-similarity) sums within each sample
    if unbiased:
        return ((sxx - dxx) / (m * (m - 1)) + (syy - dyy) / (n * (n - 1))
                - 2.0 * sxy / (m * n))
    return sxx / (m * m) + syy / (n * n) - 2.0 * sxy / (m * n)


def _perm_pvalue(stat_of: Callable[[NDArray, NDArray], float],
                 N: int, m: int, reps: int,
                 seed: Optional[int]) -> tuple[float, float]:
    """
    Observed statistic and permutation p-value, relabeling the pooled
    observations ``reps`` times.  The p-value counts the observed
    relabeling among the replications (Phipson and Smyth, 2010).
    """
    if reps < 1:
        raise ValueError("reps must be a positive integer")
    order = np.arange(N)
    obs = stat_of(order[:m], order[m:])
    rng = np.random.default_rng(seed)
    null = np.fromiter(
        (stat_of(p[:m], p[m:]) for p in (rng.permutation(N)
                                         for _ in range(reps))),
        dtype=float, count=reps)
    pvalue = (1.0 + float((null >= obs).sum())) / (1.0 + reps)
    return obs, pvalue


def _ucenter(A: NDArray) -> NDArray:
    # U-centering of Székely and Rizzo (2014)
    N = A.shape[0]
    rows = A.sum(axis=1, keepdims=True) / (N - 2)
    cols = A.sum(axis=0, keepdims=True) / (N - 2)
    grand = A.sum() / ((N - 1) * (N - 2))
    B = A - rows - cols + grand
    np.fill_diagonal(B, 0.0)
    return B


def _chi2_pvalue(A: NDArray, m: int) -> float:
    """
    Chi-square approximation of Shen, Panda, and Vogelstein (2022) applied to
    the unbiased distance correlation between the pooled distance matrix ``A``
    and the group-label distance (Shen and Vogelstein, 2021).
    """
    N = A.shape[0]
    if N < 4:
        raise ValueError("the chi-square approximation needs at least "
                         "four observations in total")
    label = np.zeros(N)
    label[m:] = 1.0
    L = np.abs(label[:, None] - label[None, :])
    Ac, Lc = _ucenter(A), _ucenter(L)
    dcov = (Ac * Lc).sum()
    varx = (Ac * Ac).sum()
    vary = (Lc * Lc).sum()
    if varx <= 0.0 or vary <= 0.0:
        dcor = 0.0
    else:
        dcor = dcov / np.sqrt(varx * vary)
    return float(chi2.sf(dcor * N + 1.0, 1))


def _check_method(method: str) -> None:
    if method not in ("perm", "chi2"):
        raise ValueError("method must be 'perm' or 'chi2'")


def energy(x: conversions.ArrayLike,
               y: conversions.ArrayLike,
               reps: int = 1000,
               seed: Optional[int] = None,
               method: str = "perm",
               standardize: Standardize = True,
               unbiased: bool = True) -> GofResult:
    """
    Two-sample energy test that `x` and `y` share a joint distribution.

    The energy distance of Székely and Rizzo (2004) is twice the mean
    Euclidean distance between observations in different samples, less the
    mean distance within each sample.  Its population value is zero when
    the two distributions are the same and positive otherwise, so the sample
    value is a measure of the distance between the two distributions and can
    be compared across candidate specifications of a joint distribution.

    Parameters
    ----------
    x, y : ArrayLike
        The two samples, with variables in columns and observations in rows
        (a one-dimensional input is a single variable).  They must have the
        same variables, in the same order, but may differ in size.
    reps : int, optional
        Number of relabelings for the permutation p-value.  Defaults to
        1000.  Ignored when `method` is ``"chi2"``.
    seed : int, optional
        Seed for the permutation p-value, which is then reproducible.
    method : {"perm", "chi2"}, optional
        ``"perm"`` (the default) obtains the p-value from a permutation
        test: the pooled observations are relabeled `reps` times and the
        p-value is the share of relabelings whose statistic is at least the
        observed one, counting the observed labeling itself.  ``"chi2"``
        uses the chi-square approximation of Shen, Panda, and Vogelstein
        (2022), which is nearly instantaneous and slightly conservative.
    standardize : {True, "pooled", "first", False}, optional
        Whether to put the variables on a common scale before computing
        distances, so that a variable with a large spread does not dominate
        the test.  ``True`` (the default, equivalently ``"pooled"``) centers
        and scales each variable by the mean and standard deviation of the
        pooled sample; ``"first"`` uses the mean and standard deviation of
        `x`, say a historical sample that the simulated `y` is compared
        with; ``False`` leaves the data as given.  Either standardization is
        one affine map applied to both samples, so it preserves the null
        hypothesis.  Centering does not affect either test (the distances
        are shift-invariant); only the scaling matters.
    unbiased : bool, optional
        If True (the default), the within-sample mean distances are taken
        over distinct pairs, which makes the statistic an unbiased estimate
        of the population energy distance at the price of being slightly
        negative when the two samples are indistinguishable.  If False, the
        within-sample means are over all pairs including each observation
        with itself, and the statistic is nonnegative.

    Returns
    -------
    GofResult
        A named tuple with the test statistic and p-value.

    Notes
    -----
    The ``"first"`` standardization fixes the scaling from the original
    labeling, which the permutation test then holds fixed across
    relabelings; the pooled standardization is label-invariant and is
    therefore the exact choice.  The chi-square p-value is computed from the
    unbiased distance correlation whatever the value of `unbiased`, and it
    equals 1 whenever that correlation is negative, so it is uninformative
    for samples that are indistinguishable.  The p-values agree with those
    of ``hyppo.ksample.Energy`` up to permutation noise; that package's
    statistic is a rescaled unbiased distance covariance, which differs from
    the distinct-pairs energy distance reported here by a relative amount
    of order one over the pooled sample size.

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2004). Testing for equal distributions
    in high dimension. InterStat, 5.

    Shen, C., & Vogelstein, J. T. (2021). The exact equivalence of distance
    and kernel methods in hypothesis testing. AStA Advances in Statistical
    Analysis, 105, 385-403.

    Shen, C., Panda, S., & Vogelstein, J. T. (2022). The chi-square test of
    distance correlation. Journal of Computational and Graphical
    Statistics, 31(1), 254-262.

    Examples
    --------
    >>> import numpy as np
    >>> import funcsim as fs
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(100, 2))
    >>> y = rng.normal(size=(300, 2)) + [1.0, 0.0]
    >>> res = fs.energy(x, y, seed=0)
    >>> res.pvalue < 0.01
    True
    """
    _check_method(method)
    pooled, m = _prepare(x, y, standardize)
    N = pooled.shape[0]
    n = N - m
    D = squareform(pdist(pooled))
    total = float(D.sum())

    def stat_of(ix: NDArray, iy: NDArray) -> float:
        sxx, syy, sxy = _block_sums(D, ix, iy, total)
        return _energy_from_sums(sxx, syy, sxy, m, n, unbiased)

    if method == "chi2":
        order = np.arange(N)
        stat = stat_of(order[:m], order[m:])
        pvalue = _chi2_pvalue(D, m)
    else:
        stat, pvalue = _perm_pvalue(stat_of, N, m, reps, seed)
    return GofResult(float(stat), float(pvalue))


def mmd(x: conversions.ArrayLike,
            y: conversions.ArrayLike,
            reps: int = 1000,
            seed: Optional[int] = None,
            method: str = "perm",
            standardize: Standardize = True,
            unbiased: bool = True,
            bandwidth: Optional[float] = None) -> GofResult:
    """
    Two-sample maximum mean discrepancy (MMD) test that `x` and `y` share a
    joint distribution.

    The MMD of Gretton et al. (2012) is the largest difference between the
    two samples' mean values of a function in the unit ball of a reproducing
    kernel Hilbert space.  With a Gaussian kernel
    :math:`k(u, v) = \\exp(-\\|u - v\\|^2 / (2 h^2))` the squared MMD is the
    mean similarity within each sample less twice the mean similarity
    between them, the mirror image of the energy distance with a kernel in
    place of a distance.  The statistic reported is the squared MMD.

    Parameters
    ----------
    x, y : ArrayLike
        The two samples, with variables in columns and observations in rows
        (a one-dimensional input is a single variable).  They must have the
        same variables, in the same order, but may differ in size.
    reps : int, optional
        Number of relabelings for the permutation p-value.  Defaults to
        1000.  Ignored when `method` is ``"chi2"``.
    seed : int, optional
        Seed for the permutation p-value, which is then reproducible.
    method : {"perm", "chi2"}, optional
        ``"perm"`` (the default) obtains the p-value from a permutation
        test; ``"chi2"`` uses the chi-square approximation of Shen, Panda,
        and Vogelstein (2022).  See :func:`funcsim.energy`.
    standardize : {True, "pooled", "first", False}, optional
        Scaling of the variables before the kernel is evaluated; see
        :func:`funcsim.energy`.  The median bandwidth below adapts to
        the overall size of the distances but not to the balance among the
        variables, so standardization matters for this test as well.
    unbiased : bool, optional
        If True (the default), the within-sample mean similarities are taken
        over distinct pairs, giving the unbiased estimate of the squared
        MMD, which is slightly negative when the two samples are
        indistinguishable.  If False, the biased (V-statistic) estimate,
        which is nonnegative.
    bandwidth : float, optional
        The Gaussian kernel bandwidth `h`, in units of the (standardized)
        variables.  Defaults to the median of the pairwise distances in the
        pooled sample.

    Returns
    -------
    GofResult
        A named tuple with the test statistic and p-value.

    Notes
    -----
    The p-values agree with those of ``hyppo.ksample.MMD`` up to
    permutation noise.  That package's reported statistic is not the squared
    MMD but a rescaled distance correlation of the kernel-induced distance
    against the group labels; it has the same permutation distribution and
    therefore the same p-value.  The chi-square p-value is computed from the
    unbiased distance correlation of the kernel-induced distance
    :math:`1 - k(u, v)` whatever the value of `unbiased`.

    References
    ----------
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
    (2012). A kernel two-sample test. Journal of Machine Learning Research,
    13, 723-773.

    Shen, C., & Vogelstein, J. T. (2021). The exact equivalence of distance
    and kernel methods in hypothesis testing. AStA Advances in Statistical
    Analysis, 105, 385-403.

    Shen, C., Panda, S., & Vogelstein, J. T. (2022). The chi-square test of
    distance correlation. Journal of Computational and Graphical
    Statistics, 31(1), 254-262.

    Examples
    --------
    >>> import numpy as np
    >>> import funcsim as fs
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(100, 2))
    >>> y = rng.normal(size=(300, 2)) + [1.0, 0.0]
    >>> res = fs.mmd(x, y, seed=0)
    >>> res.pvalue < 0.01
    True
    """
    _check_method(method)
    pooled, m = _prepare(x, y, standardize)
    N = pooled.shape[0]
    n = N - m
    D = squareform(pdist(pooled))
    if bandwidth is None:
        h = float(np.median(D[np.triu_indices(N, 1)]))
        if h == 0.0:
            h = 1.0
    else:
        h = float(bandwidth)
        if h <= 0.0:
            raise ValueError("bandwidth must be positive")
    K = np.exp(-(D * D) / (2.0 * h * h))
    total = float(K.sum())
    diag = np.diag(K)

    def stat_of(ix: NDArray, iy: NDArray) -> float:
        sxx, syy, sxy = _block_sums(K, ix, iy, total)
        return _mmd_from_sums(sxx, syy, sxy, float(diag[ix].sum()),
                              float(diag[iy].sum()), m, n, unbiased)

    if method == "chi2":
        order = np.arange(N)
        stat = stat_of(order[:m], order[m:])
        pvalue = _chi2_pvalue(1.0 - K, m)
    else:
        stat, pvalue = _perm_pvalue(stat_of, N, m, reps, seed)
    return GofResult(float(stat), float(pvalue))
