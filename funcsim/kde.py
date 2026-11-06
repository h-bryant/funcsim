import math
import warnings
from typing import Callable, Optional, Tuple, Union
import numpy as np
import scipy.optimize
import scipy.stats as stats
from scipy import interpolate, special
from . import conversions
from .bounds import validate_bound, check_bounds_against_data
from .samplesize import check_sample_size, FIXED


# inverse-CDF table: number of grid points, and the margin beyond the data
# range in kernel standard deviations (the CDF is within 6e-16 of 0 or 1 at
# eight standard deviations from every observation)
_PPF_GRID = 4096
_PPF_MARGIN = 8.0
# at a hard bound the CDF goes to 0 or 1 linearly, so a uniform grid is
# sparse in normal-score space there.  Next to each bound the first
# _BOUND_CELLS grid cells are subdivided _BOUND_SUBDIV times, and points at
# distances R * 2 ** -k from the bound (k over _BOUND_K, R the grid width)
# carry the table down to probabilities of order 1e-12 of the way to it
_BOUND_CELLS = 16
_BOUND_SUBDIV = 8
_BOUND_K = range(13, 41)
# Fritsch-Carlson limit on a node derivative relative to the adjacent
# secants, which keeps a cubic Hermite interpolant monotone
_HERMITE_LIMIT = 3.0

# least-squares cross-validation: the bandwidth is searched on a grid of
# _LSCV_GRID log-spaced points between _LSCV_RANGE[0] and _LSCV_RANGE[1]
# times Scott's bandwidth, then refined between the neighbors of the
# chosen grid point.  Pairwise sums are evaluated in chunks of at most
# _PAIR_CHUNK elements
_LSCV_GRID = 25
_LSCV_RANGE = (1.0 / 20.0, 4.0)
_PAIR_CHUNK = 4_000_000

# Sheather-Jones plug-in: the root of the solve-the-equation rule is
# bracketed in [0.1, 1] times a normal-reference bandwidth, and the bracket
# is widened by the factor 1.2 (alternately above and below) at most
# _SJ_MAX_WIDEN times before giving up, as R's bw.SJ does
_SJ_MAX_WIDEN = 99

# a pair of arrays (CDF, survival function) evaluated on an x array
_CdfPair = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _gaussian_cdf_pair(data: np.ndarray, weights: np.ndarray, h: float,
                       x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # CDF and survival function of the unbounded Gaussian KDE with centers
    # `data`, weights `weights`, and kernel standard deviation h, on x.
    # Chunked so the (x by data) intermediate stays modest
    n = x.size
    F = np.empty(n)
    S = np.empty(n)
    step = max(1, 2_000_000 // max(1, data.size))
    for i in range(0, n, step):
        z = (x[i:i + step, None] - data[None, :]) / h
        F[i:i + step] = special.ndtr(z) @ weights
        S[i:i + step] = special.ndtr(-z) @ weights
    return F, S


def _centers(data: np.ndarray, lower: Optional[float],
             upper: Optional[float]) -> np.ndarray:
    # kernel centers of the reflected estimator: the data followed by their
    # images in each bound, so that the first data.size entries are the data
    parts = [data]
    if lower is not None:
        parts.append(2.0 * lower - data)
    if upper is not None:
        parts.append(2.0 * upper - data)
    return np.concatenate(parts)


def _lscv_criterion(h: float, data: np.ndarray, centers: np.ndarray,
                    lower: Optional[float], upper: Optional[float]) -> float:
    # least-squares cross-validation criterion (Rudemo, 1982; Bowman, 1984)
    #     LSCV(h) = int f_R^2 - (2 / M) sum_i f_{R,-i}(x_i)
    # for the reflected estimator with kernel standard deviation h,
    # unnormalized (Z taken as 1).  Both terms are closed-form pairwise
    # sums over the centers c_k (data and images; see kde-plan-2026-11-06.md,
    # step 2):
    #     int_lo^hi f_R^2 = (1 / M^2) sum_{k,l} phi_{h sqrt2}(c_k - c_l) W_kl
    #     W_kl = Phi((hi - m) / (h / sqrt2)) - Phi((lo - m) / (h / sqrt2)),
    #     m = (c_k + c_l) / 2
    # (the product of two Gaussian kernels is a Gaussian in the difference
    # times a normal density in x with mean m and variance h^2 / 2), and
    # the leave-one-out estimate at x_i sums phi_h(x_i - c_l) over the
    # centers of every other observation.  With no bound W = 1; with one
    # bound the integral is half the unweighted integral over the line
    M = data.size
    n_img = centers.size // M
    N = centers.size
    # with a single bound the reflected density extended below (above)
    # the bound is symmetric about it, so the integral over the support is
    # half the integral over the line and no weight W is needed; with two
    # bounds W is computed from the midpoints
    both = lower is not None and upper is not None
    half = 0.5 if (n_img == 2) else 1.0
    root2_h = math.sqrt(2.0) / h
    # squared-integral term, by symmetry over pairs k <= l: for each chunk
    # of rows the rectangle of columns from the chunk's first index on is
    # summed, and the strictly lower triangle of its diagonal block (which
    # the rectangle includes) is removed
    upper_tri = 0.0     # sum over k < l
    diag = 0.0          # sum over k == l
    step = max(1, _PAIR_CHUNK // N)
    for i in range(0, N, step):
        rows = centers[i:i + step, None]
        cols = centers[None, i:]
        d = rows - cols
        g2 = np.exp(-d * d / (4.0 * h * h))
        if both:
            m = 0.5 * (rows + cols)
            g2 *= (special.ndtr((upper - m) * root2_h)
                   - special.ndtr((lower - m) * root2_h))
        k = g2.shape[0]
        block = g2[:, :k]
        diag += float(np.trace(block))
        upper_tri += float(g2.sum()) - float(np.tril(block).sum())
    sq_int = half * (2.0 * upper_tri + diag)
    # leave-one-out term: data rows against every center, minus each
    # observation's own centers
    loo = 0.0
    step = max(1, _PAIR_CHUNK // N)
    for i in range(0, M, step):
        d = data[i:i + step, None] - centers[None, :]
        loo += float(np.exp(-d * d / (2.0 * h * h)).sum())
    own = float(M)                                  # phi_h(0) terms
    for k in range(1, n_img):
        dk = data - centers[k * M:(k + 1) * M]
        own += float(np.exp(-dk * dk / (2.0 * h * h)).sum())
    loo -= own
    sq_int /= (h * math.sqrt(4.0 * math.pi) * M * M)
    loo /= (h * math.sqrt(2.0 * math.pi) * M * (M - 1))
    return sq_int - 2.0 * loo


def _lscv_bandwidth(data: np.ndarray, lower: Optional[float],
                    upper: Optional[float]) -> float:
    # bandwidth minimizing the least-squares cross-validation criterion:
    # the largest local minimizer on a log-spaced grid (Hall and Marron,
    # 1991), refined between its grid neighbors
    M = data.size
    if M < 3:
        raise ValueError("bw='lscv' needs at least three observations")
    if np.unique(data).size < 0.9 * M:
        warnings.warn("more than a tenth of the observations are tied; "
                      "least-squares cross-validation is unreliable for "
                      "rounded data and may undersmooth", stacklevel=3)
    h_scott = M ** (-0.2) * float(np.std(data, ddof=1))
    centers = _centers(data, lower, upper)
    crit = lambda h: _lscv_criterion(h, data, centers, lower, upper)
    grid = np.geomspace(_LSCV_RANGE[0] * h_scott, _LSCV_RANGE[1] * h_scott,
                        _LSCV_GRID)
    vals = np.array([crit(h) for h in grid])
    interior = np.flatnonzero((vals[1:-1] < vals[:-2])
                              & (vals[1:-1] <= vals[2:])) + 1
    if interior.size == 0:
        i = int(np.argmin(vals))
        warnings.warn("the least-squares cross-validation criterion has no "
                      "interior minimum between %.3g and %.3g times "
                      "Scott's bandwidth; using the %s end of that range"
                      % (_LSCV_RANGE[0], _LSCV_RANGE[1],
                         "lower" if i == 0 else "upper"), stacklevel=3)
        return float(grid[i])
    i = int(interior.max())
    res = scipy.optimize.minimize_scalar(
        crit, bounds=(grid[i - 1], grid[i + 1]), method="bounded",
        options={"xatol": 1e-5 * grid[i]})
    return float(res.x) if res.fun <= vals[i] else float(grid[i])


def _psi(r: int, g: float, centers: np.ndarray) -> float:
    # kernel estimate of the density functional psi_r = int f^(r) f, for
    # r = 4 or 6, with the Gaussian kernel at pilot bandwidth g:
    #     psi_r(g) = (1 / (N (N - 1) g^(r+1))) sum_{k,l} phi^(r)((c_k - c_l) / g)
    # over all N centers including the diagonal, with
    #     phi^(4)(z) = (z^4 - 6 z^2 + 3) phi(z)
    #     phi^(6)(z) = (z^6 - 15 z^4 + 45 z^2 - 15) phi(z)
    # (the conventions of R's bw.SJ).  Summed once over k < l by symmetry
    N = centers.size
    upper_tri = 0.0
    step = max(1, _PAIR_CHUNK // N)
    for i in range(0, N, step):
        z = (centers[i:i + step, None] - centers[None, i:]) / g
        z2 = z * z
        if r == 4:
            term = (z2 * z2 - 6.0 * z2 + 3.0) * np.exp(-0.5 * z2)
        else:
            term = ((z2 * z2 * z2 - 15.0 * z2 * z2 + 45.0 * z2 - 15.0)
                    * np.exp(-0.5 * z2))
        k = term.shape[0]
        upper_tri += float(term.sum()) - float(np.tril(term[:, :k]).sum())
    diag = N * (3.0 if r == 4 else -15.0)
    return (2.0 * upper_tri + diag) / (N * (N - 1) * g ** (r + 1)
                                       * math.sqrt(2.0 * math.pi))


def _sj_bandwidth(data: np.ndarray, lower: Optional[float],
                  upper: Optional[float], method: str = "ste") -> float:
    # Sheather and Jones (1991) plug-in bandwidth, in the form of R's
    # bw.SJ: "ste" solves the equation
    #     h = (c1 / psi_4(alpha_2(h)))^(1/5),
    #     alpha_2(h) = 1.357 (psi_4(a) / TD)^(1/7) h^(5/7)
    # with pilots a = 1.24 lam M^(-1/7) and b = 1.23 lam M^(-1/9),
    # lam = min(s, IQR / 1.349), TD = -psi_6(b), c1 = 1 / (2 sqrt(pi) M);
    # "dpi" is the two-stage direct plug-in h = (c1 / psi_4(g))^(1/5) with
    # g = (2.394 / (M TD))^(1/7).
    #
    # With a bound (a heuristic; see kde-plan-2026-11-06.md, step 3b) the
    # functionals are estimated from the data together with their images,
    # as a sample from the symmetric extension of f, and multiplied by the
    # number of centers per observation (int (f'')^2 over the support is
    # that multiple of its value for the extension over the line); M, lam,
    # and the pilots are those of the data
    M = data.size
    if M < 3:
        raise ValueError("bw='sj' needs at least three observations")
    s = float(np.std(data, ddof=1))
    q75, q25 = np.percentile(data, [75.0, 25.0])
    lam = min(s, float(q75 - q25) / 1.349)
    if not lam > 0.0:
        raise ValueError("bw='sj' needs a sample with a positive "
                         "interquartile range; use another bw")
    centers = _centers(data, lower, upper)
    n_img = centers.size // M
    psi4 = lambda g: n_img * _psi(4, g, centers)
    psi6 = lambda g: n_img * _psi(6, g, centers)
    a = 1.24 * lam * M ** (-1.0 / 7.0)
    b = 1.23 * lam * M ** (-1.0 / 9.0)
    c1 = 1.0 / (2.0 * math.sqrt(math.pi) * M)
    TD = -psi6(b)
    if not (np.isfinite(TD) and TD > 0.0):
        raise ValueError("bw='sj': the sample is too sparse to estimate "
                         "the density functional TD; use another bw")
    if method == "dpi":
        S = psi4((2.394 / (M * TD)) ** (1.0 / 7.0))
        if not (np.isfinite(S) and S > 0.0):
            raise ValueError("bw='sj': the sample is too sparse to estimate "
                             "the density functional SD; use another bw")
        return (c1 / S) ** 0.2
    S_a = psi4(a)
    if not (np.isfinite(S_a) and S_a > 0.0):
        raise ValueError("bw='sj': the sample is too sparse to estimate "
                         "the density functional SD; use another bw")
    alph2 = 1.357 * (S_a / TD) ** (1.0 / 7.0)

    def fSD(h):
        S = psi4(alph2 * h ** (5.0 / 7.0))
        if not (np.isfinite(S) and S > 0.0):
            return np.nan
        return (c1 / S) ** 0.2 - h

    hmax = 1.144 * lam * M ** (-0.2)
    lo, hi = 0.1 * hmax, hmax
    f_lo, f_hi = fSD(lo), fSD(hi)
    tries = 0
    while not (f_lo * f_hi < 0.0):
        if tries >= _SJ_MAX_WIDEN or not (np.isfinite(f_lo)
                                          and np.isfinite(f_hi)):
            raise ValueError("bw='sj': no solution of the Sheather-Jones "
                             "equation was found in the range of "
                             "bandwidths searched; use bw='lscv' or a "
                             "float")
        if tries % 2 == 0:
            hi *= 1.2
            f_hi = fSD(hi)
        else:
            lo /= 1.2
            f_lo = fSD(lo)
        tries += 1
    return float(scipy.optimize.brentq(fSD, lo, hi, xtol=1e-9 * hmax))


class _PpfTable:
    # monotone inverse of a KDE's CDF.  The CDF F and survival function S
    # are evaluated on a grid spanning [lo, hi], and x is interpolated as a
    # function of the normal score q = Phi^-1(F) (computed from S in the
    # upper half, where 1 - F loses precision).  On that scale the inverse
    # CDF is smooth and nearly linear in the tails of an unbounded
    # estimate, so a shape-preserving cubic (PCHIP) on a uniform grid of
    # 4096 points is accurate to about 1e-9 of the data range and monotone
    # in u (see tests/test_kde_ppf_table.py).
    #
    # At a hard bound x(q) is strongly curved (F goes to 0 linearly in x,
    # so x - a is about Phi(q) / f(a)), and PCHIP's secant-based
    # derivatives limit it to about 1e-6 of the range there.  With a bound
    # the table therefore uses a cubic Hermite interpolant with the exact
    # derivative dx/dq = phi(q) / f(x), clipped where needed to keep it
    # monotone (Fritsch and Carlson, 1980), on the uniform grid plus points
    # crowded toward each bound; that is accurate to about 1e-8 of the
    # range (tests/test_kde_bounds.py).  Unbounded tables keep PCHIP so
    # that results are unchanged from versions 0.2.7 to 0.2.9
    def __init__(self, cdf_pair: _CdfPair, pdf: Callable[[np.ndarray],
                                                         np.ndarray],
                 lo: float, hi: float,
                 lower_is_bound: bool, upper_is_bound: bool) -> None:
        xg = np.linspace(lo, hi, _PPF_GRID)
        bounded = lower_is_bound or upper_is_bound
        if bounded:
            R = hi - lo
            cell = R / (_PPF_GRID - 1)
            near = np.concatenate([
                np.linspace(0.0, _BOUND_CELLS * cell,
                            _BOUND_CELLS * _BOUND_SUBDIV + 1)[1:],
                R * 2.0 ** -np.array(_BOUND_K, dtype=float)])
            parts = [xg]
            if lower_is_bound:
                parts.append(lo + near)
            if upper_is_bound:
                parts.append(hi - near)
            xg = np.unique(np.concatenate(parts))
            xg = xg[(xg >= lo) & (xg <= hi)]
        F, S = cdf_pair(xg)
        with np.errstate(divide="ignore"):
            q = np.where(F <= 0.5, special.ndtri(F), -special.ndtri(S))
        # the interpolant needs strictly increasing abscissae; drop grid
        # points where the CDF is flat to machine precision (a gap between
        # data clusters, or a hard bound) or the score is not finite
        keep = np.isfinite(q)
        keep[1:] &= np.diff(q) > 0.0
        self._q = q[keep]
        self._x = xg[keep]
        if not bounded:
            self._interp = interpolate.PchipInterpolator(self._q, self._x,
                                                         extrapolate=False)
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            d = stats.norm.pdf(self._q) / pdf(self._x)
        secant = np.diff(self._x) / np.diff(self._q)
        limit = np.empty_like(d)
        limit[0] = _HERMITE_LIMIT * secant[0]
        limit[-1] = _HERMITE_LIMIT * secant[-1]
        limit[1:-1] = _HERMITE_LIMIT * np.minimum(secant[:-1], secant[1:])
        d = np.where(np.isfinite(d), np.minimum(d, limit), limit)
        self._interp = interpolate.CubicHermiteSpline(self._q, self._x, d,
                                                      extrapolate=False)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        # inverse CDF for probabilities inside the table's range; NaN outside
        with np.errstate(divide="ignore", invalid="ignore"):
            q = special.ndtri(u)
        return self._interp(q)


class Kde():
    """
    Univariate kernel density estimator (KDE).

    Parameters
    ----------
    data : conversions.VectorLike
        1-D data vector (list, tuple, np.ndarray, xr.DataArray, or pd.Series).
    bw : str or float, optional
        Bandwidth selection method ('scott', 'silverman', 'lscv' for
        least-squares cross-validation, or 'sj' for the Sheather-Jones
        plug-in rule; see Notes) or a positive float to use as the
        bandwidth (the standard deviation of the Gaussian kernel), in the
        same units as the data.  Default is 'scott'.  The bandwidth in
        use is available as :attr:`bandwidth`.
    lowerBound : float, optional
        Lower bound of the support.  If given, the estimate is confined to
        values at or above it by reflection; see Notes.  Default is None
        (no lower bound).
    upperBound : float, optional
        Upper bound of the support, likewise.  Default is None.
    exact_ppf : bool, optional
        If True, :meth:`ppf` finds every value by root finding on
        :meth:`cdf`, as funcsim did before version 0.2.7 (about 0.3 ms per
        value).  If False (the default), :meth:`ppf` is served from an
        inverse-CDF table built once, on the first call; see Notes.

    Raises
    ------
    TypeError
        If a bound is a bool or not a real number, or `bw` is neither a
        string nor a real number.
    ValueError
        If a bound is not finite, ``lowerBound >= upperBound``, an
        observation lies outside the bounds, `bw` is an unknown method
        name, a numeric `bw` is not positive, or there are too few
        observations: fewer than two, or fewer than three for ``'lscv'``
        and ``'sj'``.

    Warns
    -----
    UserWarning
        If there are fewer observations than Silverman's (1986, Table
        4.2) requirement of four for a usable univariate estimate, or
        fewer than ten for ``'lscv'`` and ``'sj'``; and in the
        circumstances described under each of those methods in Notes.

    Notes
    -----
    ``pdf``, ``cdf``, and ``ppf`` return a Python float for a scalar
    argument and a NumPy array of the same shape for an array-like argument
    (a pandas Series or a list gives an array).

    The two rules of thumb set the bandwidth to a factor times the sample
    standard deviation ``s`` of the M observations, computed with
    ``ddof=1`` (dividing by M - 1, as :class:`scipy.stats.gaussian_kde`
    does): Scott's rule gives ``M ** (-1 / 5) * s`` and Silverman's rule
    ``(3 * M / 4) ** (-1 / 5) * s``.  (:class:`funcsim.MvKde` applies the
    same factors to a ``ddof=0`` standard deviation, so on the same single
    column its bandwidth is smaller by the factor ``sqrt((M - 1) / M)``.)
    The rules are computed from the data as given; bounds do not change
    the bandwidth.

    **Cross-validated bandwidth.**  ``bw='lscv'`` chooses the bandwidth
    that minimizes the least-squares cross-validation criterion (Rudemo,
    1982; Bowman, 1984), an unbiased estimate, up to a constant, of the
    integrated squared error of the estimate: ``int f^2 - (2 / M) sum_i
    f_{-i}(x_i)``, where ``f_{-i}`` is the estimate without observation
    ``i``.  With a bound the criterion is evaluated for the reflected
    estimate (ignoring the two-bound normalization), not for the
    unbounded estimate of the raw data, which the boundary bias would
    pull toward too small a bandwidth.  Both terms are closed-form
    pairwise sums for the Gaussian kernel, costing of order ``M ** 2``
    operations per evaluation; the search takes well under a second for
    a few hundred observations and a few seconds for a few thousand.
    The criterion is minimized on a grid of 25 log-spaced bandwidths
    from 1/20 to 4 times Scott's bandwidth, taking the largest local
    minimizer when there are several (Hall and Marron, 1991), refined
    between its grid neighbors.  A warning is issued when the minimum
    lies at an end of the grid (at the lower end usually because of tied
    or rounded data, which the criterion rewards with ever smaller
    bandwidths; at the upper end when the sample is close to uniform
    under reflection), and when more than a tenth of the observations
    are tied.  Cross-validated bandwidths are considerably more variable
    from sample to sample than the rules of thumb.

    **Plug-in bandwidth.**  ``bw='sj'`` is the solve-the-equation plug-in
    rule of Sheather and Jones (1991), in the form of R's ``bw.SJ``: the
    bandwidth minimizing the asymptotic mean integrated squared error,
    with the unknown curvature functional of the density estimated from
    the data at a pilot bandwidth that itself depends on the bandwidth
    sought (Wand and Jones, 1995, section 3.6).  It is far less variable
    than cross-validation and is the usual recommendation for smooth
    densities on the whole line; it needs a sample with a positive
    interquartile range and raises ``ValueError`` when the sample is too
    sparse or rounded to estimate the functionals, or when the equation
    has no solution in the range searched.  Evaluating a functional
    costs of order ``M ** 2`` operations.

    With a bound the plug-in rule is a heuristic, because the expansion
    behind it assumes a density that is smooth on the whole line: the
    functionals are estimated from the data together with their
    reflections, as a sample from the symmetric extension of the density
    across the bound, and scaled to the support.  That is exact in spirit
    when the density is flat at the bound (its symmetric extension is
    then smooth) and remains reasonable in tests when it is not (for an
    exponential sample bounded at zero it comes within a few percent of
    the cross-validated choice), but ``'lscv'`` is the choice with a
    guarantee in the bounded case.

    **Bounded support.**  With a bound, the estimate is the reflection
    estimator (Schuster, 1985; Silverman, 1986, section 2.10): the
    unbounded Gaussian KDE ``f`` of the data is folded back across each
    bound, so that with a lower bound ``a`` the density on the support is
    ``f(x) + f(2a - x)``, and with an upper bound ``b`` it is
    ``f(x) + f(2b - x)``.  With both bounds the sum of the three terms is
    divided by its total mass on ``[a, b]``, which is within machine
    precision of one unless the interval is narrower than about eight
    bandwidths.  The result is a proper density: it is nonnegative,
    integrates to one on the support, and is zero outside it, so
    ``cdf(a) == 0``, ``cdf(b) == 1``, ``ppf(0) == a``, and
    ``ppf(1) == b``.  Observations exactly on a bound are allowed.

    Reflection forces the estimated density to have zero slope at each
    bound.  It is therefore most accurate when the true density is flat
    at the bound (a half-normal at zero, say) and biased, by an amount of
    order the bandwidth, when the true density has a nonzero slope there
    (an exponential at zero).  Corrections that remove that bias can
    produce negative densities, which would make ``ppf`` ill defined, so
    they are not offered.

    The inverse-CDF table evaluates the CDF on 4096 points spanning the data
    range plus eight kernel standard deviations on each side (truncated at
    the bounds, with extra points crowded toward each bound), and
    interpolates x against the normal score of the CDF with a
    shape-preserving cubic (PCHIP).  The interpolant is monotone, so ``ppf``
    is nondecreasing in ``u``, and in tests over samples of 20 to 1000
    observations (unimodal, bimodal, bounded, and on scales from 1e-6
    upward) it agrees with the root finder to better than 1e-8 of the data
    range; the documented accuracy target is 1e-6 of the data range.
    Probabilities outside the table's range (below about 1e-15 or above
    1 - 1e-15 at an unbounded end) fall back to root finding.  Because the
    table and the root finder differ in the last digits, simulations that
    pass draws through ``ppf`` change slightly at that level when
    upgrading; pass ``exact_ppf=True`` to reproduce earlier results
    exactly.

    References
    ----------
    Bowman, A. W. (1984). An alternative method of cross-validation for
    the smoothing of density estimates. *Biometrika*, 71(2), 353-360.

    Hall, P., & Marron, J. S. (1991). Local minima in cross-validation
    functions. *Journal of the Royal Statistical Society B*, 53(1),
    245-252.

    Rudemo, M. (1982). Empirical choice of histograms and kernel density
    estimators. *Scandinavian Journal of Statistics*, 9(2), 65-78.

    Schuster, E. F. (1985). Incorporating support constraints into
    nonparametric estimators of densities. *Communications in Statistics,
    Theory and Methods*, 14(5), 1123-1136.

    Sheather, S. J., & Jones, M. C. (1991). A reliable data-based
    bandwidth selection method for kernel density estimation. *Journal
    of the Royal Statistical Society B*, 53(3), 683-690.

    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
    Analysis*. Chapman and Hall.

    Wand, M. P., & Jones, M. C. (1995). *Kernel Smoothing*. Chapman and
    Hall.
    """
    def __init__(self,
                 data : conversions.VectorLike,
                 bw : Union[str, float] = 'scott',
                 lowerBound : Optional[float] = None,
                 upperBound : Optional[float] = None,
                 exact_ppf : bool = False):

        sampleA = np.asarray(conversions.vlToArray(data), dtype=float)
        self._data = sampleA

        self._lower = validate_bound(lowerBound, "lowerBound")
        self._upper = validate_bound(upperBound, "upperBound")
        check_bounds_against_data(sampleA, self._lower, self._upper)

        # raw kde object: the unbounded Gaussian KDE of the data.  a
        # numeric bw is the kernel standard deviation in data units;
        # scipy's float bw_method is instead a factor multiplied by the
        # sample std. dev., so convert
        if isinstance(bw, (bool, np.bool_)):
            raise TypeError("'bw' must be a method name or a positive "
                            "number, not a bool")
        # too few observations for the method: an error if the bandwidth
        # cannot be computed, a warning if the estimate is unreliable
        if isinstance(bw, str) and bw in ('scott', 'silverman', 'lscv',
                                          'sj'):
            check_sample_size(sampleA.size, 1, bw, False)
        elif not isinstance(bw, str):
            check_sample_size(sampleA.size, 1, FIXED, False)
        if isinstance(bw, str):
            if bw in ('scott', 'silverman'):
                bw_method = bw
            elif bw == 'lscv':
                h = _lscv_bandwidth(sampleA, self._lower, self._upper)
                bw_method = h / float(np.std(sampleA, ddof=1))
            elif bw == 'sj':
                h = _sj_bandwidth(sampleA, self._lower, self._upper)
                bw_method = h / float(np.std(sampleA, ddof=1))
            else:
                raise ValueError(f"unknown bandwidth method '{bw}'; use "
                                 f"'scott', 'silverman', 'lscv', 'sj', or "
                                 f"a positive float")
        elif isinstance(bw, (int, float, np.integer, np.floating)):
            if bw <= 0.0:
                raise ValueError("a numeric 'bw' must be positive")
            bw_method = float(bw) / float(np.std(sampleA, ddof=1))
        else:
            raise TypeError("'bw' must be a method name or a positive "
                            f"number, not {type(bw).__name__}")
        self.gkde = stats.gaussian_kde(sampleA, bw_method)
        self._h = math.sqrt(float(self.gkde.covariance[0, 0]))
        self._weights = np.asarray(self.gkde.weights, dtype=float)

        # initial guess for PPF optimization: the sample mean
        self.ppf_x0 = float(sum(sampleA)) / float(len(sampleA))

        # "bracket" for PPF root finding: the support where bounded, else
        # the data range extended by three times its width on each side
        width = float(max(sampleA) - min(sampleA))
        self.ppf_low = (self._lower if self._lower is not None
                        else float(min(sampleA)) - 3.0 * width)
        self.ppf_high = (self._upper if self._upper is not None
                         else float(max(sampleA)) + 3.0 * width)

        # x tolerance for PPF root finding, proportional to the data
        # scale (an absolute tolerance would make ppf meaningless for
        # data on scales much smaller than the tolerance)
        self.ppf_xtol = max((self.ppf_high - self.ppf_low) * 1e-9, 1e-300)

        # constants of the reflected CDF (see _cdf_pair): the unnormalized
        # CDF G at the ends of the support, and the mass between them
        if self._lower is not None and self._upper is not None:
            a, b = self._lower, self._upper
            Fab, Sab = self._F_S(np.array([2.0 * a - b, 2.0 * b - a]))
            self._G_lo = float(Sab[1])          # G(a) = S(2b - a)
            self._H_hi = float(Fab[0])          # 1 - G(b) = F(2a - b)
        else:
            self._G_lo = 0.0
            self._H_hi = 0.0
        self._Z = 1.0 - self._H_hi - self._G_lo

        self._exact_ppf = bool(exact_ppf)
        self._table = None  # inverse-CDF table, built on first use

    @property
    def bandwidth(self) -> float:
        """
        The bandwidth in use, the standard deviation of the Gaussian
        kernel, in the units of the data.

        For ``bw='scott'`` or ``bw='silverman'`` this is the rule's factor
        (``gkde.factor``) times the sample standard deviation of the data
        with ``ddof=1``; for a float ``bw`` it is that value.
        """
        return self._h

    @property
    def lowerBound(self) -> Optional[float]:
        """The lower bound of the support, or None."""
        return self._lower

    @property
    def upperBound(self) -> Optional[float]:
        """The upper bound of the support, or None."""
        return self._upper

    # ------------------------------------------------------------------
    # unbounded and reflected CDFs

    def _F_S(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # CDF and survival function of the unbounded KDE
        return _gaussian_cdf_pair(self._data, self._weights, self._h, x)

    def _cdf_pair(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # CDF and survival function of the (possibly reflected) estimate,
        # for x inside the support.  With f the unbounded KDE, F its CDF,
        # and S = 1 - F, the reflected density is
        #     f_R(x) = [f(x) + [a] f(2a - x) + [b] f(2b - x)] / Z
        # and its unnormalized CDF is
        #     G(x) = F(x) - [a] F(2a - x) + [b] S(2b - x)
        # so that F_R = (G(x) - G(lo)) / Z and S_R = (G(hi) - G(x)) / Z
        # with Z = G(hi) - G(lo).  G(lo) is S(2b - a) when both bounds are
        # given and 0 otherwise; 1 - G(hi) is F(2a - b) when both are
        # given and 0 otherwise.  The survival side is accumulated from
        # survival terms so that it keeps its precision near 0
        a, b = self._lower, self._upper
        F, S = self._F_S(x)
        G = F.copy()
        H = S.copy()                        # H(x) = 1 - G(x)
        if a is not None:
            Fa, Sa = self._F_S(2.0 * a - x)
            G -= Fa
            H += Fa
        if b is not None:
            Fb, Sb = self._F_S(2.0 * b - x)
            G += Sb
            H -= Sb
        if a is None and b is None:
            return G, H
        cdf = np.clip((G - self._G_lo) / self._Z, 0.0, 1.0)
        sf = np.clip((H - self._H_hi) / self._Z, 0.0, 1.0)
        # exact at and beyond the bounds (the ratio can be off by rounding)
        if a is not None:
            at = x <= a
            cdf[at] = 0.0
            sf[at] = 1.0
        if b is not None:
            at = x >= b
            cdf[at] = 1.0
            sf[at] = 0.0
        return cdf, sf

    def _in_support(self, x: np.ndarray) -> np.ndarray:
        inside = np.ones(x.shape, dtype=bool)
        if self._lower is not None:
            inside &= x >= self._lower
        if self._upper is not None:
            inside &= x <= self._upper
        return inside

    @staticmethod
    def _apply(func, x):
        # evaluate a 1-D-array function on scalar or array-like input:
        # scalar in, Python float out; array-like in, array of that shape out
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return float(func(arr.reshape(1))[0])
        return func(arr.ravel()).reshape(arr.shape)

    def _pdf_array(self, v: np.ndarray) -> np.ndarray:
        a, b = self._lower, self._upper
        out = np.asarray(self.gkde(v), dtype=float)
        if a is None and b is None:
            return out
        if a is not None:
            out = out + self.gkde(2.0 * a - v)
        if b is not None:
            out = out + self.gkde(2.0 * b - v)
        out = out / self._Z
        out[~self._in_support(v)] = 0.0
        return out

    def pdf(self,
            v: Union[float, conversions.VectorLike]
           ) -> Union[float, np.ndarray]:
        """
        Probability density function of the KDE at value v.

        Parameters
        ----------
        v : float or array-like
            Value(s) at which to evaluate the PDF.

        Returns
        -------
        float or numpy.ndarray
            The estimated probability density at v: a float for a scalar
            `v`, an array of the same shape otherwise.  Zero outside the
            support when a bound is given.
        """
        return self._apply(self._pdf_array, v)

    def _cdf_array(self, v: np.ndarray) -> np.ndarray:
        # clip to [0, 1] against numerical error; exactly 0 and 1 at and
        # beyond any bound (handled in _cdf_pair)
        return np.clip(self._cdf_pair(v)[0], 0.0, 1.0)

    def cdf(self,
            v: Union[float, conversions.VectorLike]
           ) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function of the KDE at value v.

        Parameters
        ----------
        v : float or array-like
            Value(s) at which to evaluate the CDF.

        Returns
        -------
        float or numpy.ndarray
            The estimated cumulative probability at v: a float for a scalar
            `v`, an array of the same shape otherwise.  Exactly 0 below a
            lower bound and 1 above an upper bound.
        """
        return self._apply(self._cdf_array, v)

    def _ppf_root(self, u: float) -> float:
        # inverse CDF of one probability by bracketed root finding on the
        # CDF (Brent's method), the method used for every value before 0.2.7
        try:
            return float(scipy.optimize.brentq(
                lambda x: self._cdf_array(np.array([x]))[0] - u,
                self.ppf_low, self.ppf_high, xtol=self.ppf_xtol))
        except ValueError:
            # the bracket does not contain a root: u is at or beyond the
            # CDF's value at an end of the bracket
            if u > 0.98:
                return self.ppf_high
            elif u < 0.02:
                return self.ppf_low
            raise ValueError("kde ppf could not find solution")

    def _ppf_array(self, u: np.ndarray) -> np.ndarray:
        if np.any(np.isnan(u)) or np.any((u < 0.0) | (u > 1.0)):
            raise ValueError("u must be within the range [0.0, 1.0]")
        out = np.full(u.shape, np.nan)
        # at a hard bound the inverse CDF of 0 or 1 is the bound itself
        done = np.zeros(u.shape, dtype=bool)
        if self._lower is not None:
            at = u == 0.0
            out[at] = self._lower
            done |= at
        if self._upper is not None:
            at = u == 1.0
            out[at] = self._upper
            done |= at
        todo = np.flatnonzero(~done)
        if todo.size == 0:
            return out
        if self._exact_ppf:
            out[todo] = [self._ppf_root(x) for x in u[todo]]
            return out
        if self._table is None:
            self._table = _PpfTable(self._cdf_pair, self._pdf_array,
                                    self._table_lo(), self._table_hi(),
                                    self._lower is not None,
                                    self._upper is not None)
        vals = self._table(u[todo])
        for i in np.flatnonzero(~np.isfinite(vals)):
            vals[i] = self._ppf_root(u[todo[i]])  # outside the table's range
        out[todo] = vals
        return out

    def _table_lo(self) -> float:
        lo = float(self._data.min()) - _PPF_MARGIN * self._h
        return lo if self._lower is None else max(lo, self._lower)

    def _table_hi(self) -> float:
        hi = float(self._data.max()) + _PPF_MARGIN * self._h
        return hi if self._upper is None else min(hi, self._upper)

    def ppf(self,
            u: Union[float, conversions.VectorLike]
           ) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF) of the KDE.

        Parameters
        ----------
        u : float or array-like
            Probability value(s) in the range [0.0, 1.0].

        Returns
        -------
        float or numpy.ndarray
            The value x such that CDF(x) = u: a float for a scalar `u`, an
            array of the same shape otherwise.

        Raises
        ------
        ValueError
            If a value of u is not in [0.0, 1.0], or root finding (used
            when `exact_ppf` is True, or for a probability outside the
            table's range) fails.

        Notes
        -----
        By default values are read from a monotone inverse-CDF table built
        on the first call (see the class notes for its accuracy); with
        ``exact_ppf=True`` each value is found by root finding on
        :meth:`cdf`.  A value of exactly 0 or 1 returns the corresponding
        bound when one is given; at an unbounded end it lies outside the
        table and returns the lower or upper end of the root-finding
        bracket (the data range extended by three times its width on each
        side), as in earlier versions.
        """
        return self._apply(self._ppf_array, u)
