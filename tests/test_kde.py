import numpy as np
import funcsim as fs


def test_kde():
    sample = np.random.rand(1000)
    kde = fs.Kde(sample)
    assert abs(kde.cdf(0.5) - 0.5) < 0.01
    assert abs(kde.ppf(0.5) - 0.5) < 0.01


def test_kde_pdf():
    # regression test: scipy's gaussian_kde returns a length-1 array for a
    # scalar input, and NumPy >= 2.5 raises TypeError on float() of any
    # array that is not 0-dimensional, which broke Kde.pdf in 0.2.0
    sample = np.random.default_rng(0).uniform(size=1000)
    kde = fs.Kde(sample)

    # scalar input: a positive density close to the uniform density of 1.0
    single = float(kde.pdf(0.5))
    assert single > 0.0
    assert abs(single - 1.0) < 0.15

    # array input: vectorized, and identical to the underlying scipy KDE
    x = np.array([0.25, 0.5, 0.75])
    many = kde.pdf(x)
    assert many.shape == x.shape
    np.testing.assert_allclose(many, kde.gkde(x))
