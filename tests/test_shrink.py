import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))
import shrinkage


def test_shrink():
    targets = [shrinkage._target_a, shrinkage._target_b,
               shrinkage._target_c, shrinkage._target_d,
               shrinkage._target_f]

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
        assert abs(sum(all_norms) - 27.5470609894) < 0.01
