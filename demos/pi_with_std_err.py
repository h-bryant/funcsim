#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))

import funcsim as fs
import numpy as np

# https://en.wikipedia.org/wiki/Monte_Carlo_integration


def h(x, y):
    return 1.0 if x**2.0 + y**2.0 < 1.0 else 0.0


def trial(draw):
    x = 2.0 * next(draw) - 1.0  # uniform over [-1, 1]
    y = 2.0 * next(draw) - 1.0
    return {"h": h(x, y)}

"""
out = fs.crosssec(trial=trial, trials=5000, multi=False)
area = float(out.mean())  # area of a quarter circle
sigf = float(out.std())
n = len(out)
print("n: %s" % n)
print("value of pi/4 is approximately %s" % area)
print("sig_f: %s" % sigf)
print("sig_area: %s" % (sigf / n**0.5))
"""

for s in range(100):
    out = fs.static(trial=trial, trials=5000, multi=False, seed=s)
    area = float(out.mean())  # area of a quarter circle
    print(area)