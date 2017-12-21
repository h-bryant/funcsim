"""
Six covariance shrinkage estimators, as presented in Shafer & Strimmer,
"A shrinkage approach to large-scale covariance matrix estimation and
implications for Functional Genomics." //Statistical Applications in
Genetics and Molecular Biology//, vol 4(2005), issue 1.
"""

import itertools
import numpy as np


def _wtilde(x, i, j):
    # make a centered "w" series
    xtilde = x - x.mean(0)
    w = xtilde[:, i] * xtilde[:, j]
    return w - w.mean()


def _cov_s(x, i, j, l, m):
    # unbiased estimates of covar between two individual sij
    n = float(len(x[:, 0]))
    thesum = sum(_wtilde(x, i, j) * _wtilde(x, l, m))
    thefrac = n / ((n - 1.0)**3.0)
    return thefrac * thesum


def _var_s(x, i, j):
    return _cov_s(x, i, j, i, j)


def _f(x, i, j):
    s = np.cov(x, rowvar=False)
    part0 = (s[j, j] / s[i, i])**0.5 * _cov_s(x, i, i, i, j)
    part1 = (s[i, i] / s[j, j])**0.5 * _cov_s(x, j, j, i, j)
    return 0.5 * (part0 + part1)


def _target_a(x):
    # diagonal, unit variance shrinkage estimator
    s = np.cov(x, rowvar=False)
    idx = range(len(s))

    # create target mat
    t = np.identity(len(s))

    # calculate lambda
    lam_num = 0.0
    lam_denom = 0.0
    for i, j in itertools.product(idx, idx):
        lam_num += _var_s(x, i, j)
        if i != j:
            lam_denom += s[i, j]**2.0
        else:  # i == j
            lam_denom += (s[i, i] - 1.0)**2.0

    # final calcs & return
    lam = max(0.0, min(1.0, lam_num / lam_denom))
    vcv = lam * t + (1-lam) * s
    return vcv, lam


def _target_b(x):
    # diagonal, common variance shrinkage estimator.  From Ledoit & Wolf, "A
    # well conditioned estimator for large dimension covariance matrices."
    # //Journal of Multivariate Analysis//, 88(2004):365-411
    s = np.cov(x, rowvar=False)
    idx = range(len(s))
    v = s.trace() / float(len(s))

    # create target mat
    t = v * np.identity(len(s))

    # calculate lambda
    lam_num = 0.0
    lam_denom = 0.0
    for i, j in itertools.product(idx, idx):
        lam_num += _var_s(x, i, j)
        if i != j:
            lam_denom += s[i, j]**2.0
        else:  # i == j
            lam_denom += (s[i, i] - v)**2.0

    # final calcs & return
    lam = max(0.0, min(1.0, lam_num / lam_denom))
    vcv = lam * t + (1-lam) * s
    return vcv, lam


def _target_c(x):
    # common variance and covariance shrinkage estimator.
    s = np.cov(x, rowvar=False)
    idx = range(len(s))

    # create target mat
    v = s.trace() / float(len(s))
    off_diag = [s[i, j] for i in range(len(s)) for j in range(len(s)) if i != j]
    c = sum(off_diag) / float(len(off_diag))
    t = np.empty(s.shape)
    for i, j in itertools.product(idx, idx):
        t[i, j] = v if i == j else c

    # calculate lambda
    lam_num = 0.0
    lam_denom = 0.0
    for i, j in itertools.product(idx, idx):
        lam_num += _var_s(x, i, j)
        if i == j:
            lam_denom += (s[i, i] - v)**2.0
        else:
            lam_denom += (s[i, j] - c)**2.0

    # final calcs & return
    lam = max(0.0, min(1.0, lam_num / lam_denom))
    vcv = lam * t + (1-lam) * s
    return vcv, lam


def _target_d(x):
    # diagonal, unequal variance shrinkage estimator.  From Shafer & Strimmer,
    # "A shrinkage approach to large-scale covariance matrix estimation and
    # implications for Functional Genomics." //Statistical Applications in
    # Genetics and Molecular Biology//, vol 4(2005), issue 1.
    s = np.cov(x, rowvar=False)
    idx = range(len(s))

    # create target mat
    t = np.diag(s) * np.identity(len(s))

    # calculate lambda
    lam_num = 0.0
    lam_denom = 0.0
    for i, j in itertools.product(idx, idx):
        if i != j:
            lam_num += _var_s(x, i, j)
            lam_denom += s[i, j]**2.0
    lam = max(0.0, min(1.0, lam_num / lam_denom))

    # return combination of 's' and target mat
    vcv = lam * t + (1 - lam) * s
    return vcv, lam


def _target_e(x):
    # perfect positive correlation shrinkage estimator.  From Ledoit & Wolf,
    # "Improved estimation of the covariance matrix of stock returns with an
    # application to portfolio selection," //Journal of Empirical Finance//,
    # 10(2003): 603-621.
    s = np.cov(x, rowvar=False)
    idx = range(len(s))

    # create target mat
    t = np.empty(s.shape)
    for i, j in itertools.product(idx, idx):
        t[i, j] = s[i, j] if i == j else (s[i, i] * s[j, j])**0.5

    # calculate lambda
    lam_num = 0.0
    lam_denom = 0.0
    for i, j in itertools.product(idx, idx):
        if i != j:
            lam_num += _var_s(x, i, j) - _f(x, i, j)
            lam_denom += (s[i, j] - (s[i, i] * s[j, j])**0.5)**2.0
    lam = max(0, min(1.0, lam_num / lam_denom))

    # return combination of 's' and target mat
    vcv = lam * t + (1-lam) * s
    return vcv, lam


def _target_f(x):
    # constant correlation shrinkage estimator.  From Ledoit & Wolf,
    # "Honey, I shrunk the sample covariance matrix"
    # //Portfolio Management//, 30(2004): 110-119
    s = np.cov(x, rowvar=False)
    idx = range(len(s))
    corr = np.corrcoef(s)
    if len(corr) < len(s):  # calculate corr mat by hand from sample cov mat
        d = np.sqrt(np.diag(np.diag(s)))
        dinv = np.linalg.inv(d)
        corr = np.dot(np.dot(dinv, s), dinv)
    off_diag = [corr[i, j] for i, j in itertools.product(idx, idx) if i != j]
    rbar = sum(off_diag) / float(len(off_diag))

    # create target mat
    t = np.empty(s.shape)
    for i, j in itertools.product(idx, idx):
        t[i, j] = s[i, j] if i == j else rbar * (s[i, i] * s[j, j])**0.5

    # calculate lambda
    lam_num = 0.0
    lam_denom = 0.0
    for i, j in itertools.product(idx, idx):
        if i != j:
            lam_num += _var_s(x, i, j) - rbar * _f(x, i, j)
            lam_denom += (s[i, j] - rbar * (s[i, i] * s[j, j])**0.5)**2.0
    lam = max(0, min(1.0, lam_num / lam_denom))

    # return combination of 's' and target mat
    vcv = lam * t + (1-lam) * s
    return vcv, lam


def shrink(a):
    # shrinkage cov estimator.
    # 'a' is a numpy array, with vars in cols & obs in rows
    #
    # First, try constant correlation shrinkage estimator, from Ledoit & Wolf,
    # "Honey, I shrunk the sample covariance matrix"
    # //Portfolio Management//, 30(2004): 110-119
    #
    # If that result is not positive definite, return the (guaranteed P.D.)
    # diagonal, unequal variance shrinkage estimate.  From Shafer & Strimmer,
    # "A shrinkage approach to large-scale covariance matrix estimation and
    # implications for Functional Genomics." //Statistical Applications in
    # Genetics and Molecular Biology//, vol 4(2005), issue 1.
    sig = _target_f(a)[0]
    if np.all(np.linalg.eigvals(sig) > 0):
        return sig
    else:
        return _target_d(a)[0]


if __name__ == '__main__':

    targets = [_target_a, _target_b, _target_c, _target_d, _target_f]

    def calc_all_norms(seed, mu, r, n):
        np.random.seed(seed)
        x = np.random.multivariate_normal(mu, r, size=n)
        return np.array([np.linalg.norm(tgt(x)[0] - r) for tgt in targets])

    def test_0():
        mu = np.array([10.0, 5.0, 0.0])

        rho = np.array([
            [1, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]])

        variances = ([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 5.0]])

        r = np.dot(np.dot(variances, rho), variances)

        all_norms = calc_all_norms(seed=1, mu=mu, r=r, n=20)
        print(all_norms)
        print("sum: %s" % sum(all_norms))
        assert abs(sum(all_norms) - 27.5470609894) < 0.01
        print("test_0 passed")

    test_0()
