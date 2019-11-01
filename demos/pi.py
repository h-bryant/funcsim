#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))

import funcsim as fs

# https://en.wikipedia.org/wiki/Monte_Carlo_integration


def h(x, y):
    return 1.0 if x**2.0 + y**2.0 < 1.0 else 0.0


def trial(draw):
    x = 2.0 * next(draw) - 1.0  # uniform over [-1, 1]
    y = 2.0 * next(draw) - 1.0
    return {"h": h(x, y)}


out = fs.static(trial=trial, trials=500)
area = float(out.mean())  # area of a quarter circle
print("value of pi is approximately %s" % (4.0 * area))