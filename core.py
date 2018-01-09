import numpy as np
import pandas as pd
import xarray as xr
import multicore
import tailcall


def _recurse(f, x0, S):
    # wrap f in tail call recursive function g
    @tailcall.TailCaller
    def g(n, x):
        if n == 0:
            return x
        x1 = f(x)
        return tailcall.TailCall(g, n-1, x1)

    return g(S, x0)


def _countrv(f, data0=None):
    # count number of times 'f', and any code 'f' invokes, calls 'next(draw)'
    # if 'data0' is None, infer that 'f' is the 'trial' func for a cross-sec sim
    # if 'data0' is a xr.DataArray, infer that 'f' is 'step' for rec. dyn. sim
    maxcalls = 10000.0
    fakeugen = _countgen(maxcalls)
    if data0 is None:
        f(fakeugen)
    elif type(data0) == xr.DataArray:
        f(data0, fakeugen)
    else:
        raise ValueError
    calls = int(next(fakeugen) * maxcalls)
    return calls


def _countgen(scale):
    # dummy generator for counting calls but always returning a number in [0, 1)
    # 'scale' should be a float such that you will never call more this
    # generator more than 'scale' times
    i = 0
    while i < int(scale):
        yield float(i) / scale
        i += 1


def _makeugen(u, r):
    # given an array 'u' of indep. uniform draws, where rows reflect variables
    # and columns reflect trials, make a generator for trial 'r' tha emits
    # a number of draws equal to the number of RVs
    i = 0
    while i < u.shape[0]:
        yield u[i, r]
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


def crosssec(trial, trials, multi=False, seed=6):
    """
    Cross sectional simulation
    """
    # cross-sectional simulation
    # 'trial' is a function that takes argument 'draw'

    # infer number of random vars reflected in 'trial' fucntion
    rvs = _countrv(trial)

    # draws for all RVs, w/ sampling stratified across trials
    np.random.seed(seed)
    u = _lhs(rvs, trials)  # np.array, dimensions rvs x trials

    def tryl(r):
        # closure that binds to 'trial' a 'u' generator for trial number 'r'
        # and coerces the output of 'trial' into an xarray.DataArray
        return xr.DataArray(pd.Series(trial(_makeugen(u, r))),
                            dims=['variables'])

    # create and return a 2-D DataArray with new dimension 'trials'
    if multi is True:
        out = multi.parmap(tryl, range(trials))
    else:
        out = [tryl(r) for r in range(trials)]
    return xr.concat(out, pd.Index(list(range(trials)), name='trials'))


def recdyn(step, data0, steps, trials, multi=False, seed=6):
    # recursive dynamic simulation

    # infer number of random vars reflected in 'step' fucntion
    rvs = _countrv(step, data0)

    # draws for all RVs in all time steps, w/ sampling stratified across trials
    np.random.seed(seed)
    u = _lhs(rvs * steps, trials)  # np.array, dimensions (rvs*steps) x trials

    def trial(r):
        ugen = _makeugen(u, r)  # 'u' generator for trial number 'r'
        # perform all time steps for one trial
        return _recurse(f=lambda x: step(x, ugen), x0=data0, S=steps)

    # create and return 3-D output DataArray, with new dimension 'trials'
    if multi is True:
        out = multicore.parmap(trial, range(trials))
    else:
        out = [trial(r) for r in range(trials)]
    prelim = xr.concat(out, pd.Index(list(range(trials)), name='trials'))
    return prelim.transpose('trials', 'variables', 'steps')
