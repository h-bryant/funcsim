"""
Information-criteria comparison of funcsim's copula families.
"""

import math

import numpy as np
import pandas as pd

from . import conversions
from . import copfit
from .dependence import (CopulaGauss, CopulaStudent, CopulaClayton,
                         CopulaGumbel, CopulaFrank)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _rho_label(rho: np.ndarray) -> str:
    # a scalar for two variables, otherwise just the matrix size
    K = rho.shape[0]
    if K == 2:
        return f"rho={_fmt(rho[0, 1])}"
    return f"rho: {K}x{K} matrix"


def copcompare(udata: conversions.ArrayLike) -> pd.DataFrame:
    """
    Compare fits of funcsim's copula families to pseudo-observations.

    Each family (Gaussian, Student's t, Clayton, Gumbel, Frank) is fitted
    to `udata` exactly as the corresponding copula class would fit it, its
    copula log-likelihood (the pseudo-likelihood of Genest, Ghoudi, and
    Rivest, 1995) is evaluated at those fitted parameters, and the Akaike
    and Bayesian information criteria are computed from it.

    Parameters
    ----------
    udata : ArrayLike
        Pseudo-observations (marginal CDF values) with variables in columns
        and observations in rows; every value must lie in (0, 1).

    Returns
    -------
    pd.DataFrame
        One row per family, indexed by family name and sorted by BIC from
        best (smallest) to worst, with columns ``parameters`` (the fitted
        values as text), ``n_params``, ``loglik``, ``AIC``, and ``BIC``.

    Notes
    -----
    The parameters are the ones the copula classes use: normal-scores
    correlation for the Gaussian copula, Kendall's-tau inversion plus a
    profile likelihood for the degrees of freedom for the Student's t copula,
    and maximum pseudo-likelihood for the Archimedean families.  The
    Gaussian and Student's t log-likelihoods are therefore evaluated at
    moment-type estimates rather than at the exact maximum, so their
    criteria are conservative by a small amount.  A Student's t row with
    ``nu=inf`` means the profile likelihood peaked at the ceiling of the
    degrees-of-freedom search (200), so the fitted copula is the Gaussian
    copula with the Kendall's-tau correlation matrix; it still pays for the
    extra parameter in the criteria.  Standard information
    criteria applied to pseudo-likelihoods are approximate (Grønneberg and
    Hjort, 2014); treat differences of a few points as ties.  Warnings
    issued while fitting a family (for example, negative dependence in data
    handed to an Archimedean family that represents only positive
    dependence) are passed through.

    References
    ----------
    Genest, C., Ghoudi, K., & Rivest, L.-P. (1995). A semiparametric
    estimation procedure of dependence parameters in multivariate families
    of distributions. Biometrika, 82(3), 543-552.

    Grønneberg, S., & Hjort, N. L. (2014). The copula information criteria.
    Scandinavian Journal of Statistics, 41(2), 436-459.

    Examples
    --------
    >>> table = fs.copcompare(uh)
    >>> print(table.round(2))
    """
    u = conversions.alToArray(udata)
    if u.ndim != 2 or u.shape[1] < 2:
        raise ValueError("udata must have at least two columns (variables)")
    (M, K) = u.shape
    npair = K * (K - 1) // 2

    def row(name, parameters, n_params, loglik):
        return {"copula": name,
                "parameters": parameters,
                "n_params": n_params,
                "loglik": loglik,
                "AIC": 2.0 * n_params - 2.0 * loglik,
                "BIC": n_params * math.log(M) - 2.0 * loglik}

    cg = CopulaGauss(u)
    rho_g = cg.rho.to_numpy()
    ct = CopulaStudent(u)
    rho_t = ct.rho.to_numpy()
    cc = CopulaClayton(u)
    cgu = CopulaGumbel(u)
    cf = CopulaFrank(u)
    uc = copfit._clipu(u)

    rows = [
        row("Gaussian", _rho_label(rho_g), npair,
            copfit.gauss_loglik(rho_g, u)),
        row("Student", f"{_rho_label(rho_t)}, nu={_fmt(ct.nu)}", npair + 1,
            copfit.student_loglik(rho_t, ct.nu, u)),
        row("Clayton", f"theta={_fmt(cc.theta)}", 1,
            copfit.clayton_loglik(cc.theta, uc)),
        row("Gumbel", f"theta={_fmt(cgu.theta)}", 1,
            copfit.gumbel_loglik(cgu.theta, uc)),
        row("Frank", f"theta={_fmt(cf.theta)}", 1,
            copfit.frank_loglik(cf.theta, uc)),
    ]
    table = pd.DataFrame(rows).set_index("copula")
    return table.sort_values("BIC")
