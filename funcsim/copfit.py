"""
Native copula parameter fitting.

Internal module: estimates copula parameters from pseudo-observations
(marginal CDF values) without any third-party copula package.

Archimedean families (Clayton, Gumbel, Frank) are fit by maximum
pseudo-likelihood using exact K-dimensional copula log-densities, with
Kendall's-tau inversion supplying the starting value and a sign gate.
The Student's t copula is fit in two stages: pairwise Kendall's-tau
inversion for the correlation matrix, then a profile maximum
pseudo-likelihood search for the degrees of freedom.

References
----------
Hofert, M., Machler, M., & McNeil, A. J. (2012). Likelihood inference for
Archimedean copulas in high dimensions under known margins. Journal of
Multivariate Analysis, 110, 133-150.
"""

import math
import warnings
from typing import Callable, Tuple

import numpy as np
from scipy import integrate, optimize, special, stats

from . import nearby


# clip pseudo-observations away from 0 and 1 to keep quantile transforms
# and log terms finite
_UEPS = 1e-10

# smallest admissible theta for each family (strictly inside the domain
# that the corresponding draw() machinery can handle)
THETA_MIN_CLAYTON = 1e-6
THETA_MIN_GUMBEL = 1.0000038089  # theta == 1.0 breaks the stable sampler
THETA_MIN_FRANK = 1e-6

# largest admissible theta (tau approx. 0.96-0.99 depending on family)
_THETA_MAX = 100.0

# search bounds for the Student's t degrees of freedom
_NU_MIN = 2.0
_NU_MAX = 200.0


def _clipu(u: np.ndarray) -> np.ndarray:
    # clip pseudo-observations into (0, 1), strictly
    return np.clip(np.asarray(u, dtype=float), _UEPS, 1.0 - _UEPS)


def taumatrix(u: np.ndarray) -> np.ndarray:
    # matrix of pairwise Kendall's tau (tau-b) values
    K = u.shape[1]
    tm = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            tau = stats.kendalltau(u[:, i], u[:, j]).statistic
            if not np.isfinite(tau):
                raise ValueError(f"cannot compute Kendall's tau for column "
                                 f"pair ({i}, {j}); a column may be constant")
            tm[i, j] = tm[j, i] = tau
    return tm


def taubar(u: np.ndarray) -> float:
    # mean pairwise Kendall's tau
    if u.ndim != 2 or u.shape[1] < 2:
        # with a single column there are no pairs: the mean of an empty
        # set is NaN, which would slip past the negative-dependence gates
        raise ValueError("copula fitting requires at least two variables "
                         "(columns) in the pseudo-observation array")
    tm = taumatrix(u)
    iu = np.triu_indices(tm.shape[0], k=1)
    return float(tm[iu].mean())


def clayton_loglik(theta: float, u: np.ndarray) -> float:
    # total Clayton copula log-density over the rows of u, for theta > 0.
    # log c = sum_k log(1 + k*theta) - (d + 1/theta) log S - (1+theta) sum log u
    # with S = sum_j u_j^(-theta) - (d - 1), evaluated in log space
    (M, d) = u.shape
    logu = np.log(u)
    t = -theta * logu  # positive
    a = np.concatenate([t, np.zeros((M, 1))], axis=1)
    b = np.concatenate([np.ones((M, d)), np.full((M, 1), -(d - 1.0))], axis=1)
    logS = special.logsumexp(a, b=b, axis=1)  # S > 0 for u in (0,1)^d
    const = np.sum(np.log1p(theta * np.arange(1, d)))
    return float(M * const - (d + 1.0 / theta) * np.sum(logS)
                 - (1.0 + theta) * np.sum(logu))


def _gumbel_logcoefs(d: int, alpha: float) -> np.ndarray:
    # log of the polynomial coefficients a_{dk}(alpha), k = 1..d, in the
    # Gumbel generator derivative (-1)^d psi^(d)(t) = psi(t)/t^d * P_d(t^alpha)
    # computed via the cancellation-free recurrence
    #   P_1(x) = alpha x;  a_{d+1,k} = alpha a_{d,k-1} + (d - alpha k) a_{d,k}
    # (all coefficients are strictly positive for alpha in (0, 1))
    a = np.array([alpha])
    for dd in range(1, d):
        nxt = np.zeros(dd + 1)
        nxt[1:] += alpha * a
        nxt[:dd] += (dd - alpha * np.arange(1, dd + 1)) * a
        a = nxt
    return np.log(a)


def gumbel_loglik(theta: float, u: np.ndarray) -> float:
    # total Gumbel copula log-density over the rows of u, for theta > 1,
    # via the Hofert-Machler-McNeil (2012) formula, evaluated in log space
    (M, d) = u.shape
    alpha = 1.0 / theta
    logu = np.log(u)
    L = np.log(-logu)  # log(-log u)
    logt = special.logsumexp(theta * L, axis=1)  # log sum_j (-log u_j)^theta
    logx = alpha * logt
    logcoef = _gumbel_logcoefs(d, alpha)
    ks = np.arange(1, d + 1)
    logP = special.logsumexp(logcoef[None, :] + np.outer(logx, ks), axis=1)
    return float(np.sum(-np.exp(logx) + logP - d * logt)
                 + M * d * math.log(theta)
                 + (theta - 1.0) * np.sum(L) - np.sum(logu))


def _eulerian(n: int) -> list:
    # exact Eulerian numbers <n, k> for k = 0..n-1 (n >= 1), via
    # <n,k> = (k+1)<n-1,k> + (n-k)<n-1,k-1>
    row = [1]
    for m in range(2, n + 1):
        row = [(k + 1) * (row[k] if k < len(row) else 0)
               + (m - k) * (row[k - 1] if k >= 1 else 0)
               for k in range(m)]
    return row


def frank_loglik(theta: float, u: np.ndarray) -> float:
    # total Frank copula log-density over the rows of u, for theta > 0:
    # log c = (d-1) log(theta) + log Li_{-(d-1)}(z)
    #         + sum_j [-theta u_j - log(1 - exp(-theta u_j))]
    # with log z = sum_j log(1 - exp(-theta u_j)) - (d-1) log(1 - exp(-theta))
    # and Li_{-n}(z) evaluated via Eulerian numbers, in log space
    (M, d) = u.shape
    n = d - 1
    log1me_u = np.log(-np.expm1(-theta * u))  # log(1 - exp(-theta u))
    log1me_t = math.log(-math.expm1(-theta))
    logz = np.sum(log1me_u, axis=1) - n * log1me_t  # z in (0, 1)
    logeul = np.log(np.array(_eulerian(n), dtype=float))
    powers = n - np.arange(n)
    log1mz = np.log(-np.expm1(logz))
    logli = (special.logsumexp(logeul[None, :] + np.outer(logz, powers),
                               axis=1)
             - (n + 1) * log1mz)
    return float(M * n * math.log(theta) + np.sum(logli)
                 + np.sum(-theta * u - log1me_u))


def _debye1(x: float) -> float:
    # Debye function of order 1: (1/x) * integral_0^x t/(e^t - 1) dt
    if x < 1e-8:
        return 1.0 - x / 4.0
    val = integrate.quad(lambda t: t / math.expm1(t) if t > 0.0 else 1.0,
                         0.0, x)[0]
    return val / x


def clayton_theta0(tau: float) -> float:
    # Kendall's-tau inversion for Clayton: theta = 2 tau / (1 - tau)
    t = min(tau, 0.97)
    return min(max(2.0 * t / (1.0 - t), THETA_MIN_CLAYTON), _THETA_MAX)


def gumbel_theta0(tau: float) -> float:
    # Kendall's-tau inversion for Gumbel: theta = 1 / (1 - tau)
    t = min(tau, 0.97)
    return min(max(1.0 / (1.0 - t), THETA_MIN_GUMBEL), _THETA_MAX)


def frank_theta0(tau: float) -> float:
    # Kendall's-tau inversion for Frank: solve
    # tau(theta) = 1 - (4/theta) (1 - D_1(theta)) numerically.
    # tau(theta) is increasing in theta, so test which side of the
    # bracket tau falls outside of before calling brentq (assuming the
    # failure direction, as this once did, returned theta = _THETA_MAX
    # -- maximal dependence -- for near-zero tau)
    t = min(tau, 0.96)
    f = lambda th: 1.0 - (4.0 / th) * (1.0 - _debye1(th)) - t
    if f(1e-6) >= 0.0:    # tau at or below tau(1e-6): weakest dependence
        return THETA_MIN_FRANK
    if f(200.0) <= 0.0:   # tau at or above tau(200): strongest dependence
        return _THETA_MAX
    theta = optimize.brentq(f, 1e-6, 200.0)
    return min(max(theta, THETA_MIN_FRANK), _THETA_MAX)


def _fit_theta(loglik: Callable[[float, np.ndarray], float],
               u: np.ndarray,
               theta0: float,
               lo: float,
               hi: float) -> float:
    # maximize a one-parameter copula pseudo-log-likelihood via bounded
    # Brent search, falling back to the tau-inversion value theta0 if the
    # optimizer fails or is beaten by theta0.  The search is bracketed
    # around theta0: tau inversion and pseudo-ML are both consistent for
    # the same theta, so a generous multiplicative window contains the
    # ML optimum while keeping the search away from regions where the
    # log-density underflows (e.g., Frank with large theta)
    def negll(th):
        try:
            with np.errstate(divide="ignore", invalid="ignore",
                             over="ignore"):
                val = loglik(th, u)
        except (ValueError, FloatingPointError, OverflowError):
            return np.inf
        return -val if np.isfinite(val) else np.inf

    try:
        res = optimize.minimize_scalar(negll,
                                       bounds=(max(lo, 0.25 * theta0),
                                               min(hi, 4.0 * theta0)),
                                       method="bounded")
        thetahat = float(res.x) if res.success else None
    except Exception:
        thetahat = None

    if thetahat is None or negll(thetahat) > negll(theta0):
        warnings.warn("maximum-likelihood estimation of the copula "
                      "parameter did not converge; using the Kendall's-tau "
                      "inversion estimate instead", UserWarning)
        return theta0
    return thetahat


def fit_clayton(u: np.ndarray) -> Tuple[float, float, bool]:
    # fit Clayton theta; returns (theta, mean pairwise tau, clamped flag)
    uc = _clipu(u)
    tb = taubar(uc)
    if tb <= 0.0:
        return (THETA_MIN_CLAYTON, tb, True)
    theta0 = clayton_theta0(tb)
    theta = _fit_theta(clayton_loglik, uc, theta0, THETA_MIN_CLAYTON,
                       _THETA_MAX)
    return (theta, tb, False)


def fit_gumbel(u: np.ndarray) -> Tuple[float, float, bool]:
    # fit Gumbel theta; returns (theta, mean pairwise tau, clamped flag)
    uc = _clipu(u)
    tb = taubar(uc)
    if tb <= 0.0:
        return (THETA_MIN_GUMBEL, tb, True)
    theta0 = gumbel_theta0(tb)
    theta = _fit_theta(gumbel_loglik, uc, theta0, THETA_MIN_GUMBEL,
                       _THETA_MAX)
    return (theta, tb, False)


def fit_frank(u: np.ndarray) -> Tuple[float, float, bool]:
    # fit Frank theta; returns (theta, mean pairwise tau, clamped flag)
    uc = _clipu(u)
    tb = taubar(uc)
    if tb <= 0.0:
        return (THETA_MIN_FRANK, tb, True)
    theta0 = frank_theta0(tb)
    theta = _fit_theta(frank_loglik, uc, theta0, THETA_MIN_FRANK, _THETA_MAX)
    return (theta, tb, False)


def fit_student(u: np.ndarray) -> Tuple[np.ndarray, float]:
    # fit a Student's t copula; returns (correlation matrix, deg. of freedom).
    # stage 1: pairwise Kendall's tau -> rho_ij = sin(pi tau_ij / 2),
    # corrected to the nearest positive definite correlation matrix.
    # stage 2: profile maximum pseudo-likelihood for nu, searching on a
    # log scale (the likelihood is flat in nu near the Gaussian limit)
    uc = _clipu(u)
    R = np.sin(0.5 * np.pi * taumatrix(uc))
    np.fill_diagonal(R, 1.0)
    R = nearby.nearestpd(0.5 * (R + R.T))
    s = np.sqrt(np.diag(R))
    R = R / np.outer(s, s)  # renormalize to unit diagonal
    np.fill_diagonal(R, 1.0)

    mvt_dim = R.shape[0]

    def negll(lognu):
        nu = math.exp(lognu)
        try:
            with np.errstate(divide="ignore", invalid="ignore",
                             over="ignore"):
                x = stats.t.ppf(uc, df=nu)
                ll = (stats.multivariate_t(loc=np.zeros(mvt_dim), shape=R,
                                           df=nu).logpdf(x).sum()
                      - stats.t.logpdf(x, df=nu).sum())
        except Exception:
            return np.inf
        return -ll if np.isfinite(ll) else np.inf

    res = optimize.minimize_scalar(negll,
                                   bounds=(math.log(_NU_MIN),
                                           math.log(_NU_MAX)),
                                   method="bounded")
    nu = float(math.exp(res.x)) if res.success else _NU_MAX
    return (R, nu)
