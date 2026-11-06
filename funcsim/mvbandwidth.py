"""
Bandwidth matrices for the multivariate Gaussian KDE (:class:`MvKde`):
the full-matrix normal-reference rules, least-squares cross-validation
(LSCV), and smoothed cross-validation (SCV), diagonal or full.

Everything works in the standardized coordinates MvKde uses (each column
divided by its ddof=0 standard deviation), where the sample covariance is
the correlation matrix.  The criteria are affine equivariant, so a
bandwidth chosen there maps back to the data's units by the same
scaling.  The formulas and conventions follow R's ks package (Duong,
2007; Chacón and Duong, 2018), against which they are tested.

References
----------
Chacón, J. E., & Duong, T. (2018). *Multivariate Kernel Smoothing and
Its Applications*. Chapman and Hall/CRC.

Duong, T. (2007). ks: Kernel density estimation and kernel discriminant
analysis for multivariate data in R. *Journal of Statistical Software*,
21(7), 1-16.

Duong, T., & Hazelton, M. L. (2005). Cross-validation bandwidth matrices
for multivariate kernel density estimation. *Scandinavian Journal of
Statistics*, 32(3), 485-506.

Hall, P., Marron, J. S., & Park, B. U. (1992). Smoothed cross-validation.
*Probability Theory and Related Fields*, 92, 1-20.

Sain, S. R., Baggerly, K. A., & Scott, D. W. (1994). Cross-validation of
multivariate densities. *Journal of the American Statistical
Association*, 89(427), 807-817.
"""
import math
import warnings
from typing import Callable, List, Tuple
import numpy as np
import scipy.optimize

# pairwise differences are held in memory when the (pairs by K) array has
# at most this many elements, and recomputed in row chunks otherwise
_PAIR_ELEMENTS = 10_000_000
# the optimizer is a plateau outside this range of generalized bandwidth
# scales, (det H)^(1 / 2K), relative to the starting matrix (as ks's trunc
# guard), and a warning is issued when the result lies outside a narrower
# range (as the univariate grid-edge warnings)
_PLATEAU = 1e5
_WARN_RANGE = (1.0 / 20.0, 4.0)


# ---------------------------------------------------------------------------
# rules of thumb

def rule_factor(rule: str, M: int, K: int) -> float:
    """
    The scalar multiplying the (co)variance in the normal-reference rules:
    Scott's ``M ** (-2 / (K + 4))`` and Silverman's
    ``((K + 2) M / 4) ** (-2 / (K + 4))``.
    """
    if rule == 'scott':
        return M ** (-2.0 / (K + 4.0))
    if rule == 'silverman':
        return ((K + 2.0) * M / 4.0) ** (-2.0 / (K + 4.0))
    raise ValueError(f"unknown rule '{rule}'")


# ---------------------------------------------------------------------------
# pairwise sums of Gaussian densities

class _Pairs:
    # the differences x_i - x_j over pairs i < j, held or recomputed
    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        M, K = data.shape
        self.M = M
        self.K = K
        self.n_pairs = M * (M - 1) // 2
        self._held = None
        if self.n_pairs * K <= _PAIR_ELEMENTS:
            iu = np.triu_indices(M, 1)
            self._held = data[iu[0]] - data[iu[1]]

    def chunks(self):
        # yields (pairs by K) arrays covering every pair i < j once
        if self._held is not None:
            yield self._held
            return
        M, K = self.M, self.K
        step = max(1, _PAIR_ELEMENTS // (M * K))
        for i in range(0, M - 1, step):
            rows = self._data[i:i + step]
            d = rows[:, None, :] - self._data[None, :, :]
            # keep j > i for each row i in the chunk
            jj = np.arange(M)[None, :]
            ii = np.arange(i, i + rows.shape[0])[:, None]
            yield d[jj > ii]

    def gaussian_sum(self, S: np.ndarray) -> float:
        # sum over pairs i < j of the K-variate normal density with
        # covariance S at x_i - x_j
        Sinv = np.linalg.inv(S)
        norm = math.sqrt((2.0 * math.pi) ** self.K * np.linalg.det(S))
        total = 0.0
        for d in self.chunks():
            q = np.einsum('pk,pk->p', d @ Sinv, d)
            total += float(np.exp(-0.5 * q).sum())
        return total / norm


def _phi0(S: np.ndarray, K: int) -> float:
    # K-variate normal density with covariance S at the origin
    return 1.0 / math.sqrt((2.0 * math.pi) ** K * np.linalg.det(S))


def lscv(pairs: _Pairs, H: np.ndarray) -> float:
    """
    Least-squares cross-validation criterion of the Gaussian KDE with
    bandwidth matrix H: ``int f^2 - (2 / M) sum_i f_{-i}(x_i)``, as the
    pairwise sums ``(1 / M^2) sum_{i,j} phi_{2H}(x_i - x_j)
    - (2 / (M (M - 1))) sum_{i != j} phi_H(x_i - x_j)``.
    """
    M, K = pairs.M, pairs.K
    sq = (2.0 * pairs.gaussian_sum(2.0 * H) + M * _phi0(2.0 * H, K)) / (M * M)
    loo = 2.0 * pairs.gaussian_sum(H) / (M * (M - 1))
    return sq - 2.0 * loo


def scv(pairs: _Pairs, H: np.ndarray, G: np.ndarray) -> float:
    """
    Smoothed cross-validation criterion (Hall, Marron, and Park, 1992;
    Duong and Hazelton, 2005) with pilot matrix G:
    ``(4 pi)^(-K/2) |H|^(-1/2) / M + (1 / M^2) sum_{i,j}
    [phi_{2H+2G} - 2 phi_{H+2G} + phi_{2G}](x_i - x_j)``, the second term
    floored at zero.
    """
    M, K = pairs.M, pairs.K

    def full_sum(S):
        return (2.0 * pairs.gaussian_sum(S) + M * _phi0(S, K)) / (M * M)

    bias2 = (full_sum(2.0 * H + 2.0 * G) - 2.0 * full_sum(H + 2.0 * G)
             + full_sum(2.0 * G))
    variance = (4.0 * math.pi) ** (-K / 2.0) / (M * math.sqrt(np.linalg.det(H)))
    return variance + max(0.0, bias2)


# ---------------------------------------------------------------------------
# SCV pilot

def _odd_factorial(m: int) -> float:
    # (m - 1)!! for even m, as ks's OF
    return math.factorial(m) / (2.0 ** (m / 2) * math.factorial(m // 2))


def _nu(r: int, A: np.ndarray) -> float:
    # E[(Z' A Z)^r] for Z standard normal, by the cumulant recursion of
    # ks's nu
    ei = np.linalg.eigvalsh(A)
    tr = [float(np.sum(ei ** p)) for p in range(1, r + 1)]
    val = [1.0]
    for p in range(1, r + 1):
        val.append(sum(tr[q - 1] * val[p - q] for q in range(1, p + 1))
                   / (2.0 * p))
    return math.factorial(r) * 2.0 ** r * val[r]


def scv_pilot(S: np.ndarray, M: int) -> float:
    """
    The scalar pilot bandwidth ``g`` of the SCV criterion (pilot matrix
    ``g^2 I``) from the normal-reference AMSE-optimal rule, for data with
    covariance S: the one-stage "dscalar" pilot of R's ks (Chacón and
    Duong, 2010, 2018).  The density functional of order six it needs is
    that of the normal density with covariance 2S at the origin, whose
    derivative tensor follows from Isserlis' theorem.
    """
    K = S.shape[0]
    Sinv = np.linalg.inv(S)
    A = 0.5 * Sinv                       # inverse covariance of phi_{2S}
    c = -_phi0(2.0 * S, K)               # (-1)^3 phi_{2S}(0)
    A2 = A @ A
    trA = float(np.trace(A))
    # T_ijkl = sum_a (D^6 phi_{2S}(0))_aaijkl, from the 15 pairings of
    # (a, a, i, j, k, l): the pair (a, a) gives tr(A) times the three
    # pairings of (i, j, k, l); a paired with two of (i, j, k, l) gives
    # (A^2) on that pair, twice, times A on the other pair
    T = trA * (np.einsum('ij,kl->ijkl', A, A) + np.einsum('ik,jl->ijkl', A, A)
               + np.einsum('il,jk->ijkl', A, A))
    T = T + 2.0 * (np.einsum('ij,kl->ijkl', A2, A)
                   + np.einsum('ij,kl->ijkl', A, A2)
                   + np.einsum('ik,jl->ijkl', A2, A)
                   + np.einsum('ik,jl->ijkl', A, A2)
                   + np.einsum('il,jk->ijkl', A2, A)
                   + np.einsum('il,jk->ijkl', A, A2))
    T = c * T
    B3 = 4.0 * float(np.sum(T * T))
    B1 = (2.0 ** -3 * (4.0 * math.pi) ** -K * _odd_factorial(6)
          * float(np.prod([K + 2.0 * j for j in range(3)])))
    B2 = (-(K + 2.0) * 2.0 ** -4 * (4.0 * math.pi) ** -K
          * np.linalg.det(S) ** -0.5 * _odd_factorial(4) * _nu(2, Sinv))
    return ((2.0 * B1 / (-B2 + math.sqrt(B2 * B2 + 4.0 * B1 * B3)))
            ** (1.0 / (K + 8.0)) * M ** (-1.0 / (K + 8.0)))


# ---------------------------------------------------------------------------
# minimization over diagonal or full matrices

def _unpack(theta: np.ndarray, K: int, full: bool) -> np.ndarray:
    # H from the parameter vector: log standard deviations of a diagonal
    # matrix, or a Cholesky factor with log diagonal
    if not full:
        return np.diag(np.exp(2.0 * theta))
    L = np.zeros((K, K))
    L[np.diag_indices(K)] = np.exp(theta[:K])
    L[np.tril_indices(K, -1)] = theta[K:]
    return L @ L.T


def _pack(H: np.ndarray, K: int, full: bool) -> np.ndarray:
    if not full:
        return 0.5 * np.log(np.diag(H))
    L = np.linalg.cholesky(H)
    return np.concatenate([np.log(np.diag(L)), L[np.tril_indices(K, -1)]])


def _scale(H: np.ndarray, K: int) -> float:
    # generalized bandwidth scale, (det H)^(1 / 2K)
    return float(np.linalg.det(H)) ** (0.5 / K)


def minimize_criterion(criterion: Callable[[np.ndarray], float],
                       H0: np.ndarray, full: bool, name: str) -> np.ndarray:
    """
    Minimize a bandwidth criterion over diagonal (``full=False``) or
    positive definite (``full=True``) matrices by Nelder-Mead from H0, with
    a plateau outside a wide range of scales around H0 and a warning when
    the result lies outside a narrower one.
    """
    K = H0.shape[0]
    f0 = criterion(H0)
    s0 = _scale(H0, K)

    def objective(theta):
        H = _unpack(theta, K, full)
        s = _scale(H, K)
        if not (s0 / _PLATEAU < s < s0 * _PLATEAU):
            return f0
        val = criterion(H)
        if not np.isfinite(val) or abs(val) > _PLATEAU * abs(f0):
            return f0
        return val

    theta0 = _pack(H0, K, full)
    res = scipy.optimize.minimize(
        objective, theta0, method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-12 * max(abs(f0), 1e-300),
                 'maxiter': 4000 * theta0.size, 'maxfev': 8000 * theta0.size,
                 'adaptive': theta0.size > 2})
    H = _unpack(res.x, K, full)
    if criterion(H) > f0:
        H = H0
    ratio = _scale(H, K) / s0
    if not (_WARN_RANGE[0] <= ratio <= _WARN_RANGE[1]):
        warnings.warn(f"the {name} bandwidth matrix has generalized scale "
                      f"{ratio:.3g} times the normal-reference matrix's "
                      f"(outside {_WARN_RANGE[0]:.3g} to "
                      f"{_WARN_RANGE[1]:.3g}); the criterion may have "
                      f"collapsed (tied or rounded data) or the sample "
                      f"may be nearly uniform", UserWarning, stacklevel=4)
    return H


def select(family: str, data: np.ndarray, full: bool,
           H0: np.ndarray) -> np.ndarray:
    """
    Bandwidth matrix for `data` (standardized, observations in rows) by
    'lscv' or 'scv', diagonal or full, starting the search at H0 (the
    normal-reference matrix of the same structure).
    """
    M, K = data.shape
    if np.unique(data, axis=0).shape[0] < 0.9 * M:
        warnings.warn("more than a tenth of the observations are tied; "
                      "cross-validation is unreliable for rounded data "
                      "and may undersmooth", UserWarning, stacklevel=3)
    pairs = _Pairs(data)
    if family == 'lscv':
        return minimize_criterion(lambda H: lscv(pairs, H), H0, full, 'lscv')
    if family == 'scv':
        S = np.cov(data, rowvar=False, bias=True).reshape(K, K)
        if full:
            # pilot g^2 I in pre-sphered coordinates is g^2 S here
            g = scv_pilot(np.eye(K), M)
            G = g * g * S
        else:
            g = scv_pilot(S, M)
            G = g * g * np.eye(K)
        return minimize_criterion(lambda H: scv(pairs, H, G), H0, full, 'scv')
    raise ValueError(f"unknown bandwidth family '{family}'")
