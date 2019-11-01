#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))

from scipy import stats
import funcsim as fs


def trial(draw, prob):
    # function to perform one trial.
    # simulate one std. norm variable, and one Bernoilli variable that is 1
    # with probability 'prob'

    # independent uniform draws
    u1 = next(draw)
    u2 = next(draw)

    # inverse CDF transformations
    eps = stats.norm.ppf(u1)
    b = stats.bernoulli.ppf(u2, prob)

    # return dict with var names and values
    return {"eps": eps, "b": b}


# simulate for first scenario, with 'prob' = 0.25
out = fs.static(trial=lambda dr: trial(dr, 0.25), trials=15, multi=True)
print("\nMeans of the simulated variables:\n%s" % out.mean(dim='trials'))

# simulate for second scenario, with 'prob' = 0.5
out = fs.static(trial=lambda dr: trial(dr, 0.5), trials=15, multi=True)
print("\nMeans of the simulated variables:\n%s" % out.mean(dim='trials'))
