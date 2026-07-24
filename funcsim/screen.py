import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, beta, rankdata
from . import conversions
import warnings


def runs_test(data, cutoff='median'):
    """
    Perform a two-sided Wald-Wolfowitz runs test for randomness.

    Parameters:
    - data: list or 1D numpy array of numerical data
    - cutoff: 'median' (default), or a numeric threshold to determine 
              binary classification

    Returns:
    - z_stat: z-score
    - p_value: two-sided p-value
    """
    data = np.asarray(data)

    # Step 1: Convert to binary sequence
    if cutoff == 'median':
        median = np.median(data)
    else:
        median = float(cutoff)
    signs = data > median  # True/False

    # Step 2: Count runs
    runs = 1  # At least one run
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1

    # Step 3: Count number of positives and negatives
    n1 = np.sum(signs)       # number of positives
    n2 = len(signs) - n1     # number of negatives

    if n1 == 0 or n2 == 0:
        raise ValueError("All values are on one side of the cutoff — "
                         "test is undefined.")

    # Step 4: Calculate expected runs and variance under null
    expected_runs = 1 + (2 * n1 * n2) / (n1 + n2)
    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
        (((n1 + n2)**2) * (n1 + n2 - 1))

    # Step 5: Compute z statistic
    z = (runs - expected_runs) / np.sqrt(variance_runs)

    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return z, p_value


def bartels_test(x, alternative="two.sided", pvalue_method="normal"):
    """
    Performs the Bartels Rank Test for Randomness.

    This test checks if a numeric sequence is random against a specified
    alternative hypothesis.

    Args:
        x (list or np.array): A numeric vector containing the observations.
        alternative (str): The alternative hypothesis. Must be one of
                           "two.sided" (default), "left.sided" (for trend),
                           or "right.sided" (for oscillation).
        pvalue_method (str): The method used to compute the p-value.
                             Must be one of "normal" (default), "beta", or
                             "exact". "exact" is recommended only for small
                             sample sizes (n <= 10).

    Returns:
        dict: A dictionary containing the test results:
              - 'statistic': The value of the normalized test stat (Z-score).
              - 'n': The sample size.
              - 'p_value': The p-value of the test.
              - 'alternative': The alternative hypothesis tested.
              - 'method': The method used for p-value calculation.
              - 'RVN': The value of the RVN statistic.
              - 'mean_RVN': The theoretical mean of the RVN statistic.
              - 'var_RVN': The theoretical variance of the RVN statistic.
    """

    x = np.array(x).flatten()
    n = len(x)

    if n < 4:
        raise ValueError("Sample size must be at least 4 for Bartels test.")

    # Remove NaN values (if any)
    x = x[~np.isnan(x)]
    if len(x) != n:
        msg = (f"Removed {n - len(x)} NaN values from the input data.")
        warnings.warn(msg, UserWarning)
        n = len(x)
        if n < 4:
            raise ValueError("After removing NaNs, sample size is too small.")


    # Calculate ranks (midranks for ties, so the p-value does not depend
    # on the input order of tied values)
    ranks = rankdata(x)

    # Calculate the RVN statistic (Rank von Neumann Ratio)
    numerator = np.sum((ranks[:-1] - ranks[1:])**2)
    denominator = np.sum((ranks - (n + 1) / 2)**2)

    if denominator == 0:
        # This can happen if all ranks are the same, indicating no variability.
        # In such a case, the test is not meaningful.
        msg = ("Denominator of RVN statistic is zero (all values are "
               "identical). Test not applicable.")
        return {
            'statistic': np.nan,
            'n': n,
            'p_value': np.nan,
            'alternative': alternative,
            'method': pvalue_method,
            'RVN': np.nan,
            'mean_RVN': np.nan,
            'var_RVN': np.nan,
            'message': msg
        }

    RVN = numerator / denominator

    # Theoretical mean and variance of RVN
    mean_RVN = 2
    var_RVN = (4 * (n - 2) * (5 * n**2 - 2 * n - 9)) / \
        (5 * n * (n + 1) * (n - 1)**2)

    # Standard error for normalized statistic
    std_RVN = np.sqrt(var_RVN)

    # Normalized test statistic (Z-score)
    if std_RVN == 0:
        statistic = np.nan # Avoid division by zero if variance is zero
    else:
        statistic = (RVN - mean_RVN) / std_RVN

    p_value = np.nan

    if pvalue_method == "normal":
        if alternative == "two.sided":
            p_value = 2 * norm.cdf(-np.abs(statistic))
        elif alternative == "left.sided":  # Testing against a trend (small RVN)
            p_value = norm.cdf(statistic)
        elif alternative == "right.sided": # Test against oscillation (lrg RVN)
            p_value = norm.sf(statistic)  # Survival function (1 - cdf)
        else:
            raise ValueError("Invalid alternative. Choose 'two.sided', "
                             "'left.sided', or 'right.sided'.")
    elif pvalue_method == "beta":
        # Parameters for Beta approximation
        # This approximation is generally for 10 <= n < 100
        a_beta = (5 * n * (n + 1) * (n - 1)**2) / \
            (2 * (n - 2) * (5 * n**2 - 2 * n - 9)) - 0.5
        b_beta = a_beta # a = b for this specific beta distribution

        # The Beta distribution is for RVN directly, not the normalized stat
        if alternative == "two.sided":
            p_value = 2 * min(beta.cdf(RVN / 4, a_beta, b_beta),
                              beta.sf(RVN / 4, a_beta, b_beta))
        elif alternative == "left.sided":
            p_value = beta.cdf(RVN / 4, a_beta, b_beta)
        elif alternative == "right.sided":
            p_value = beta.sf(RVN / 4, a_beta, b_beta)
        else:
            raise ValueError("Invalid alternative. Choose 'two.sided', "
                             "'left.sided', or 'right.sided'.")
    elif pvalue_method == "exact":
        # Exact calculation is complex and computationally intensive.
        # For simplicity, this example will raise an error,
        # but in a real-world scenario, you would need pre-computed tables
        # or more complex algorithms for small n.
        if n > 10:
            msg = ("Exact p-value calculation is only "
                   "recommended for n <= 10." \
                   "Using normal approximation instead.")
            warnings.warn(msg, UserWarning)
            return bartels_test(x, alternative, pvalue_method="normal")
        else:
            raise NotImplementedError("Exact p-value calculation is "
                                      "not implemented in "
                                      "this example due to complexity.")
    else:
        raise ValueError("Invalid pvalue_method. Choose 'normal', 'beta', or 'exact'.")

    return {
        'statistic': statistic,
        'n': n,
        'p_value': p_value,
        'alternative': alternative,
        'method': pvalue_method,
        'RVN': RVN,
        'mean_RVN': mean_RVN,
        'var_RVN': var_RVN
    }


def _adf_tstat(y, max_lag, det):
    # ADF regression t-statistic on the lagged level.  'det' selects the
    # deterministic terms: "n" none, "c" constant, "ct" constant + trend
    dy = np.diff(y)
    y_lag = y[:-1]
    X = y_lag.reshape(-1, 1)
    # Add lagged differences if max_lag > 0
    if max_lag > 0:
        for i in range(1, max_lag+1):
            lagged = np.roll(dy, i)
            lagged[:i] = 0
            X = np.column_stack((X, lagged))
    # Add deterministic terms ahead of the stochastic regressors
    if det == "c":
        X = np.column_stack((np.ones(len(X)), X))
    elif det == "ct":
        X = np.column_stack((np.ones(len(X)),
                             np.arange(len(X), dtype=float), X))
    elif det != "n":
        raise ValueError('det must be one of "n", "c", or "ct"')
    dy = dy[max_lag:]
    X = X[max_lag:]
    # OLS regression
    beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    y_pred = X @ beta
    se = np.sqrt(np.sum((dy - y_pred) ** 2) / (len(dy) - X.shape[1]))
    var_beta = se ** 2 * np.linalg.inv(X.T @ X)
    k = {"n": 0, "c": 1, "ct": 2}[det]  # position of the lagged level
    return beta[k] / np.sqrt(var_beta[k, k])


def _adf_test(series, det, max_lag=0, n_sim=1000, seed=42):
    # simulated-null ADF test; returns (t_stat, p_value).  a local
    # generator is used so the caller's global random state is untouched
    rng = np.random.default_rng(seed)
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    t_stat = _adf_tstat(y, max_lag, det)

    # simulate the null distribution (driftless random walk, fit with the
    # same deterministic terms as the test regression)
    n = len(y)
    sim_stats = np.array([_adf_tstat(np.cumsum(rng.normal(size=n)),
                                     max_lag, det)
                          for _ in range(n_sim)])
    p_value = float(np.mean(sim_stats < t_stat))
    return t_stat, p_value


def adf_test_no_const(series, max_lag=0, n_sim=1000, seed=42):
    """
    Augmented Dickey-Fuller test with no deterministic terms.
    H0: series has a unit root; alternative: zero-mean stationarity.
    Returns (t_stat, p_value).
    """
    return _adf_test(series, "n", max_lag, n_sim, seed)


def adf_test_with_const(series, max_lag=0, n_sim=1000, seed=42):
    """
    Augmented Dickey-Fuller test with a constant.
    H0: series has a unit root; alternative: level stationarity
    (stationary around a constant).  Returns (t_stat, p_value).
    """
    return _adf_test(series, "c", max_lag, n_sim, seed)


def adf_test_const_trend(series, max_lag=0, n_sim=1000, seed=42):
    """
    Augmented Dickey-Fuller test with a constant and a linear trend.
    H0: series has a unit root; alternative: trend stationarity
    (stationary around a linear trend).  Returns (t_stat, p_value).
    """
    return _adf_test(series, "ct", max_lag, n_sim, seed)


def white_test(residuals, exog):
    n = len(residuals)
    res2 = residuals ** 2
    X = exog.copy()
    for col in exog.columns:
        X[f"{col}^2"] = exog[col] ** 2
    if len(exog.columns) > 1:
        for i in range(len(exog.columns)):
            for j in range(i+1, len(exog.columns)):
                X[f"{exog.columns[i]}*{exog.columns[j]}"] = \
                    exog.iloc[:, i] * exog.iloc[:, j]
    X = np.asarray(X)
    X = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X, res2, rcond=None)
    y_hat = X @ beta
    SSR = ((y_hat - res2.mean()) ** 2).sum()
    LM = n * SSR / ((res2 - res2.mean()) ** 2).sum()
    # degrees of freedom: number of linearly independent regressors in the
    # auxiliary regression, excluding the constant (rank counts the constant
    # once, no matter how many redundant/constant columns were generated)
    df = np.linalg.matrix_rank(X) - 1
    pval = 1 - chi2.cdf(LM, df)
    return pval


def ljung_box(residuals, lags=3):
    x = np.asarray(residuals, dtype=float)
    n = len(x)
    # standard ACF estimator: full-sample mean and variance in the
    # denominator (trimmed-segment Pearson correlations are not the
    # statistic the Ljung-Box null distribution assumes)
    xd = x - x.mean()
    denom = np.sum(xd ** 2)
    acfs = [1.0] + [float(np.sum(xd[k:] * xd[:-k]) / denom)
                    for k in range(1, lags + 1)]
    Qs = []
    for h in range(1, lags+1):
        Q = n * (n+2) * np.sum([acfs[k]**2 / (n-k) for k in range(1, h+1)])
        pval = 1 - chi2.cdf(Q, h)
        Qs.append(pval)
    return Qs


def _frmt(p_value: float,
          alpha: float,
          reject_is_good: bool = False
          ) -> str:
    if p_value < alpha:
        problem = not reject_is_good
    else:
        problem = reject_is_good
    if problem:
        msg = "WARNING: possible non-i.i.d. properties"
        return f"{p_value:.3f} ({msg})"
    else:
        return f"{p_value:.3f}"


def screen(data: conversions.VectorLike,
           alpha: float = 0.05
           ) -> str:
    """
    Screen a time series for non-i.i.d. properties using several tests.

    Parameters
    ----------
    data : VectorLike
        Input time series data (array-like or pandas Series).
    alpha : float, optional
        Significance level for hypothesis tests (default is 0.05).

    Returns
    -------
    str
        A formatted string summarizing the results of stationarity, 
        heteroskedasticity, autocorrelation, and randomness tests.
    """

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in the range (0, 1)")

    seriesA = conversions.vlToArray(data)

    df = pd.DataFrame({"obsnum": range(len(seriesA)),
                       "series": seriesA})

    X = np.column_stack([np.ones(len(seriesA)), df["obsnum"]])
    y = df["series"].values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    # ADF tests: line A has no deterministic terms; line B includes a
    # constant and linear trend, so it is a genuine trend-stationarity test
    p_adf = adf_test_no_const(seriesA, max_lag=2, n_sim=1000, seed=42)[1]
    p_adf_t = adf_test_const_trend(seriesA, max_lag=2, n_sim=1000, seed=42)[1]

    # White test for heteroskedasticity (constant is added inside
    # white_test; passing it as an exog column would inflate the df)
    p_white = white_test(resid, df[["obsnum"]])

    # Ljung-Box test for autocorrelation
    p_lb = ljung_box(resid, lags=3)

    # overall tests for i.i.d. properties
    p_ww = runs_test(seriesA, cutoff='median')[1]
    p_bartel = bartels_test(seriesA)["p_value"]

    # formatted ADF p-values (hoisted: a replacement field spanning a
    # newline inside an f-string requires PEP 701, i.e. Python >= 3.12)
    adf_frmt = _frmt(p_adf, alpha, reject_is_good=True)
    adf_t_frmt = _frmt(p_adf_t, alpha, reject_is_good=True)

    ret = (f'Screening series at the {100.0 * alpha:g}% level:\n\n'
           f"  A) H0: series is not stationary (ADF)\n"
           f"           p-value: {adf_frmt}\n\n"
           f"  B) H0: series is not trend stationary (ADF)\n"
           f"           p-value: {adf_t_frmt}\n\n"
           f"  C) H0: homoskedasticity (White's)\n"
           f"           p-value: {_frmt(p_white, alpha)}\n\n"
           f"  D) H0: no first-order autocorrelation (Ljung-Box)\n"
           f"           p-value: {_frmt(p_lb[0], alpha)}\n"
           f"     H0: no second-order autocorrelation (Ljung-Box)\n"
           f"           p-value: {_frmt(p_lb[1], alpha)}\n"
           f"     H0: no third-order autocorrelation (Ljung-Box)\n"
           f"           p-value: {_frmt(p_lb[2], alpha)}\n\n"
           f"  E) H0: series is i.i.d. (Wald-Wolfowitz)\n"
           f"           p-value: {_frmt(p_ww, alpha)}\n\n"
           f"  F) H0: series is i.i.d. (Bartels)\n"
           f"           p-value: {_frmt(p_bartel, alpha)}\n")
    return ret