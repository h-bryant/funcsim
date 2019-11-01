#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))

import numpy as np
import funcsim as fs


def trial(draw):
    # function to perform one trial.
    # simulate one std. norm variable, and one Bernoilli variable

    # indep. uniform draw using built-in stratified/Latin hypercube sampling
    u_LHS = next(draw)

    # indep. uniform draw using non-stratified sampling
    u_no_LHS = np.random.uniform(0.0, 1.0)

    # return dict with var names and values
    return {"u_LHS": u_LHS, "u_no_LHS": u_no_LHS}


# simulate
out = fs.static(trial=trial, trials=100)

# sort simulated u values, send to screen
srtd_u_LHS = np.sort(out[:, 0])
srtd_u_no_LHS = np.sort(out[:, 1])
print("\nu_LHS, u_no_LHS")
for i in range(len(srtd_u_LHS)):
    print("%s, %s" % (srtd_u_LHS[i], srtd_u_no_LHS[i]))
