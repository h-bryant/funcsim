"""
Sample-size requirements of the kernel density estimators, by bandwidth
family and matrix structure: a hard minimum below which the bandwidth
cannot be computed (ValueError) and a soft minimum below which it can be
computed but the estimate is unreliable (UserWarning).
"""
import warnings
from typing import Optional

# Silverman (1986, Table 4.2): sample size required for the relative mean
# squared error at the center of a standard K-variate normal density to
# fall below 0.1 with the optimal bandwidth, for K = 1, ..., 10
SILVERMAN_TABLE = {1: 4, 2: 19, 3: 67, 4: 223, 5: 768, 6: 2790, 7: 10_700,
                   8: 43_700, 9: 187_000, 10: 842_000}

# bandwidth families: the rules of thumb, the data-driven selectors, and a
# bandwidth supplied by the user
RULES = ('scott', 'silverman')
SELECTORS = ('lscv', 'sj', 'scv')
FIXED = 'fixed'

# factor applied to the number of bandwidth parameters for the soft
# minimum of a data-driven selector, and to the number of variables for
# the soft minimum of a full-matrix rule (covariance estimation)
_PER_PARAMETER = 10


def n_parameters(K: int, full: bool) -> int:
    """Number of free entries of a diagonal or full K-by-K bandwidth."""
    return K * (K + 1) // 2 if full else K


def check_sample_size(M: int, K: int, family: str, full: bool) -> None:
    """
    Raise ValueError when M observations of K variables are too few to
    compute a bandwidth of the given family and structure, and warn when
    they are enough to compute it but too few to trust it.

    Parameters
    ----------
    M, K : int
        Numbers of observations and variables.
    family : str
        'scott', 'silverman', 'lscv', 'sj', 'scv', or 'fixed' (a bandwidth
        supplied by the user).
    full : bool
        Whether the bandwidth is a full matrix (False: diagonal, or a
        scalar when K = 1).

    Notes
    -----
    Hard minimums follow from what the computation needs to exist: two
    observations for a variance; more observations than variables for a
    positive definite sample covariance (full matrices); and more
    observations than bandwidth parameters plus one for a data-driven
    selector.  Soft minimums are Silverman's (1986, Table 4.2) sample
    sizes for a usable estimate of a K-variate normal density (4, 19, 67,
    223, 768, 2790, 10,700, 43,700, 187,000, and 842,000 for K = 1 to 10;
    always below the requirement beyond K = 10), raised to ten
    observations per variable for a full-matrix rule and to ten per
    bandwidth parameter for a data-driven selector.
    """
    what = f"{M} observation{'s' if M != 1 else ''} of {K} variable" \
           f"{'s' if K != 1 else ''}"
    label = family if family != FIXED else "fixed-bandwidth"
    structure = "full-matrix " if (full and K > 1) else ""
    p = n_parameters(K, full)

    # hard minimums
    if M < 2:
        raise ValueError(f"{what}: at least two observations are needed "
                         f"for a kernel density estimate")
    if full and M <= K:
        raise ValueError(f"{what}: a full bandwidth matrix needs more "
                         f"observations than variables (the sample "
                         f"covariance is singular)")
    if family in SELECTORS and M < max(3, p + 2):
        raise ValueError(f"{what}: bw='{family}' with {p} bandwidth "
                         f"parameter{'s' if p != 1 else ''} needs at "
                         f"least {max(3, p + 2)} observations")

    # soft minimums
    table: Optional[int] = SILVERMAN_TABLE.get(K)
    needed = table if table is not None else None
    if family in SELECTORS:
        needed = max(needed or 0, _PER_PARAMETER * p)
    elif full and K > 1:
        needed = max(needed or 0, _PER_PARAMETER * K)
    if needed is None:
        warnings.warn(f"{what}: Silverman's (1986, Table 4.2) sample-size "
                      f"requirement for a kernel density estimate exceeds "
                      f"842,000 beyond 10 variables; the "
                      f"{structure}{label} estimate is unreliable",
                      UserWarning, stacklevel=3)
    elif M < needed:
        warnings.warn(f"{what}: fewer than the {needed} recommended for "
                      f"a {structure}{label} kernel density estimate in "
                      f"{K} dimension{'s' if K != 1 else ''} (Silverman, "
                      f"1986, Table 4.2, and ten observations per "
                      f"{'bandwidth parameter' if family in SELECTORS else 'variable'}); "
                      f"the estimate is unreliable",
                      UserWarning, stacklevel=3)
