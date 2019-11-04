from copy import deepcopy as copy
import numpy as np
import pandas as pd
import xarray as xr
import multicore
import tailcall
import rdarrays
from scipy import stats


def _recurse(f, x0, S):
    # wrap f in tail call recursive function g
    @tailcall.TailCaller
    def g(n, x):
        if n == 0:
            return x
        x1 = f(x)
        return tailcall.TailCall(g, n-1, x1)

    return g(S, x0)


def _checkdata0(data0):
    # check that user's data0 seems sane.  Return sorted list of variable names
    if isinstance(data0, xr.DataArray) is False:
        raise ValueError('"data0" must be an xarray.DataArray')
    data0Coords = data0.coords
    if not ("variables" in data0Coords.keys() and
            "steps" in data0Coords.keys()):
        raise ValueError('"data0" must have dimensions "variables" and "steps"')
    return sorted(list(data0Coords["variables"]))


def _checkf(f, data0=None):
    # Count number of times 'f', and any code 'f' invokes, calls 'next(draw)'
    # If 'data0' is None, infer that 'f' is the 'trial' func for a cross-sec sim
    # If 'data0' is a xr.DataArray, infer that 'f' is 'step' for rec. dyn. sim
    # Also, check that f returns something that makes sense.
    fakeugen = _countgen()
    if data0 is None:
        if type(f(fakeugen)) != dict:
            raise ValueError('"trial" function must return a dict')
    else:
        out = f(data0, fakeugen)
        if isinstance(out, rdarrays.RDdata) is False:
            msg = '"step" function must return the result of funcsim.chron()'
            raise ValueError(msg)

    calls = int(round((next(fakeugen) - 0.5) * 10**4))
    return calls


def _countgen():
    # dummy generator for counting calls but always returning approximately 0.5.
    i = 0
    while i < int(10000):
        yield 0.5 + float(i) * 10**-4
        i += 1


def _makewgen(w, r):
    # given an array 'w' of indep. draws, where rows reflect variables
    # and columns reflect trials, make a generator for trial 'r' tha emits
    # a number of draws equal to the number of RVs
    i = 0
    while i < w.shape[0]:
        yield w[i, r]
        i += 1


def _strat(R):
    # stratified sampling for a single uniformly distributed random variable.
    # 'R' (an int) is the number of draws to perform
    # returns a numpy array of floats, each in the interval [0, 1).
    draws = (np.arange(0, R) + np.random.uniform(0.0, 1.0, R)) / float(R)
    np.random.shuffle(draws)  # warning: mutating 'draws'
    return draws


def _lhs(K, R):
    # Latin hypercube sampling.  For each of K independent uniform (over the
    # unit interval) random variables, create a stratified sample of length R.
    # 'K' (an int) is the number of variables
    # 'R' (an int) is the number of trials
    # returns a KxR numpy array containing draws
    return np.concatenate([[_strat(R)] for i in range(K)], axis=0)


def _extendIndex(idx, nNewSteps):
    # extend a 'steps' index; should work for ints or pd.Period
    if len(idx) == 0:  # no previous index; just use integers for the new index
        return list(range(nNewSteps))
    newIdx = list(idx)
    [newIdx.append(newIdx[-1] + 1) for i in range(nNewSteps)]
    return newIdx


def static(trial, trials, multi=False, seed=6, stdnorm=False):
    """
    Cross sectional simulation
    """
    # static simulation
    # 'trial' is a function that takes argument 'draw'

    # infer number of random vars reflected in 'trial' fucntion
    rvs = _checkf(trial)

    # draws for all RVs, w/ sampling stratified across trials
    if rvs > 0:
        np.random.seed(seed)
        u = _lhs(rvs, trials)  # np.array, dimensions rvs x trials
        w = stats.norm.ppf(u) if stdnorm is True else u

    def tryl(r):
        # closure that binds to 'trial' a 'u' generator for trial number 'r'
        # and coerces the output of 'trial' into an xarray.DataArray
        wgen = _makewgen(w, r) if rvs > 0 else None
        return xr.DataArray(pd.Series(trial(wgen)), dims=['variables'])

    # create and return a 2-D DataArray with new dimension 'trials'
    if multi is True:
        out = multicore.parmap(tryl, range(trials))
    else:
        out = [tryl(r) for r in range(trials)]
    return xr.concat(out, pd.Index(list(range(trials)), name='trials'))


def recdyn(step, data0, steps, trials, multi=False, seed=6, stdnorm=False):
    # recursive dynamic simulation

    _checkdata0(data0)

    # check that we know how to cope with the types for the 'steps' index
    sidx = data0.indexes['steps']
    if len(sidx) > 0:
        assert type(sidx[0]) in [pd.Period, np.int64]

    # indexes for the final output xr.DataArray
    varNames = data0.indexes['variables']
    namePositions = {nm: i for i, nm in enumerate(varNames)}
    stepLabels = _extendIndex(sidx, steps)

    # create example data object in which data for one trail can accumulate
    data = rdarrays.RDdata(data0.to_masked_array(), steps, namePositions)

    # infer number of random vars reflected in 'step' fucntion
    rvs = _checkf(step, copy(data))

    # draws for all RVs in all time steps, w/ sampling stratified across trials
    if rvs > 0:
        np.random.seed(seed)
        u = _lhs(rvs * steps, trials)  # np.array dimension (rvs*steps) x trials
        w = stats.norm.ppf(u) if stdnorm is True else u

    def trial(r):
        wgen = _makewgen(w, r) if rvs > 0 else None  # 'w' gener. for trial 'r'
        # perform all time steps for one trial
        return _recurse(f=lambda x: step(x, wgen), x0=copy(data), S=steps)

    # create and return 3-D output DataArray, with new dimension 'trials'
    if multi is True:
        out = multicore.parmap(lambda r: trial(r)._a, range(trials))
    else:
        out = [trial(r)._a for r in range(trials)]

    prelim = xr.DataArray(out, coords=[('trials', list(range(trials))),
                                       ('variables', varNames),
                                       ('steps', stepLabels)])
    return prelim.transpose('trials', 'variables', 'steps')
