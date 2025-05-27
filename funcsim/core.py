import sys
from copy import deepcopy as copy
import numpy as np
import pandas as pd
import xarray as xr
import multicore
import rdarrays
from scipy import stats
from collections.abc import Callable, Generator


def _checkdata0(data0):
    # check that user's data0 seems sane.  Return list of variable names
    # in data0 order.

    if isinstance(data0, xr.DataArray) is False:
        raise ValueError('"data0" must be an xarray.DataArray')

    # check that data0 has the right dimension names 
    data0Coords = data0.coords
    if not len(data0Coords) == 2:
        raise ValueError('"data0" must have exactly two dimensions')
    if not ("variables" in data0Coords.keys() and
            "steps" in data0Coords.keys()):
        raise ValueError('"data0" must have dimensions "variables" and "steps"')
    
    # Check for an appropriate index for the 'variables' dimension
    stepsCoords = data0Coords["steps"]
    if not (np.issubdtype(stepsCoords.dtype, np.integer) or
            isinstance(stepsCoords.to_index(), pd.PeriodIndex)):
        raise ValueError('"data0" must have either an integer index or a '
                         'pandas PeriodIndex for the "steps" dimension')

    return list(data0Coords["variables"])


def _checkf(f, data0=None):
    # Count number of times 'f', and any code 'f' invokes, calls 'next(draw)'
    # If 'data0' is None, infer that 'f' is the 'trial' func for a cross-sec sim
    # If 'data0' is a xr.DataArray, infer that 'f' is 'step' for rec. dyn. sim
    # Also, check that f returns something that makes sense.
    fakeugen = _countgen()
    if data0 is None:
        out = f(fakeugen)
        if type(out) != dict:
            raise ValueError('"trialf" function must return a dict')
    else:
        out = f(fakeugen, data0)
        if type(out) != dict:
            raise ValueError('"stepf" function must return a dict')

    # check that the dict returned by 'f' has variable names as keys
    varnames = list(out.keys())

    calls = int(round((next(fakeugen) - 0.5) * 10**4))
    return calls, varnames


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


def _mcs(K, R):
    # Monte Carlo sampling.  For each of K independent uniform (over the
    # unit interval) random variables, create a sample of length R.
    # 'K' (an int) is the number of variables
    # 'R' (an int) is the number of trials
    # returns a KxR numpy array containing draws
    return np.concatenate([[np.random.uniform(0.0, 1.0, R)]
                           for i in range(K)], axis=0)


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


def static(trialf: Callable[[Generator[int, float, None]], dict[str, float]],
           ntrials: int,
           multi: bool = False,
           seed: int = 6,
           stdnorm: bool = False,
           sampling: str = 'lh'
           ) -> xr.DataArray:
    """
    Static stochastic simulation.

    Parameters
    ----------
    trialf : function
        Function that performs a single trial.  Should take 'draw' as an
        argument, where 'draw' will be a generator that emits random draws
        that will be provided by ``static``.  This fucntion should return
        a dict with variable names (as strings) as keys and values for those
        variables (as floats) as values.
    ntrials : int
        The number of trials to perform.
    multi : bool, optional
        Use multiple processes/cores for the simulation. Default is False.
    seed : int, optional
        Seed for pseudo-random number generation. Default is 6.
    stdnorm : book, optional
        If False, ``next(draw)`` within `trialf` will return standard uniform
        random draws. If True, ``next(draw)`` will return standard normal draws.
        Default is False.
    sampling : {'lh', 'mc'}, optional
        If 'lh', Latin Hypercube sampling is employed.  If 'mc', simple
        Monte Carlo sampling is employed.  Default is 'lh'.

    Returns
    -------
    xarray.DataArray
        2-D xarray.DataArray with dimensions 'trials' and 'variables'.

    """
    # infer number of random vars reflected in 'trial' fucntion
    rvs, _ = _checkf(trialf)

    # draws for all RVs, w/ sampling stratified across trials
    if rvs > 0:
        np.random.seed(seed)
        if sampling == 'lh':
            u = _lhs(rvs, ntrials)  # np.array, dimensions rvs x trials
        elif sampling == 'mc':
            u = _mcs(rvs, ntrials)  # monte carlo, not latin hypercube
        else:
            raise ValueError('sampling must be "lh" or "mc"')
        w = stats.norm.ppf(u) if stdnorm is True else u

    def tryl(r):
        # closure that binds to 'trial' a 'u' generator for trial number 'r'
        # and coerces the output of 'trial' into an xarray.DataArray
        wgen = _makewgen(w, r) if rvs > 0 else None
        return xr.DataArray(pd.Series(trialf(wgen)), dims=['variables'])

    # create and return a 2-D DataArray with new dimension 'trials'
    if multi is True:
        out = multicore.parmap(tryl, range(ntrials))
    else:
        out = [tryl(r) for r in range(ntrials)]
    return xr.concat(out, pd.Index(list(range(ntrials)), name='trials'))


def recdyn(stepf: Callable[[Generator[int, float, None], rdarrays.RDdata],
                           dict[str, float]],
           data0: xr.DataArray,
           nsteps: int,
           ntrials: int,
           multi: bool = False,
           seed: int = 6,
           stdnorm: bool = False,
           sampling: str = 'lh'
           ) -> xr.DataArray:
    """
    Recursive dynamic stochastic simulation.

    Parameters
    ----------
    stepf : function
        Function that performs a single step through time.  Should take 'draw'
        as a first argument, where 'draw' will be a generator that emits random
        draws that will be provided by ``recdyn``.  Should take 'data' as a
        second argument, where this will be a type of array that is also
        provided by ``recdyn.`` This fucntion should return a dict with variable
        names (as strings) as keys and values for those variables
        (as floats) as values.
    data0 : xarray.DataArray
        Initial and/or historical data relevant to the simulation.
        Should have dimensions 'variables' and 'steps'.  Any lagged values
        needed by `stepf` must be in `data0`.  The 'steps' dimension should
        have either an integer index or a pandas Period index.
    nsteps : int
        The number of steps to perform in each trial.
    ntrials : int
        The number of trials to perform.
    multi : bool, optional
        Use multiple processes/cores for the simulation. Default is False.
    seed : int, optional
        Seed for pseudo-random number generation. Default is 6.
    stdnorm : book, optional
        If False, ``next(draw)`` within `trialf` will return standard uniform
        random draws. If True, ``next(draw)`` will return standard normal draws.
        Default is False.
    sampling : {'lh', 'mc'}, optional
        If 'lh', Latin Hypercube sampling is employed.  If 'mc', simple
        Monte Carlo sampling is employed.  Default is 'lh'.

    Returns
    -------
    xarray.DataArray
        3-D xarray.DataArray with dimensions 'trials', 'variables', and 'steps'.
    """

    _checkdata0(data0)

    # check for 'stepf'
    if not isinstance(stepf, Callable):
        raise ValueError('"stepf" must be a callable function')

    # check that we know how to cope with the types for the 'steps' index
    sidx = data0.indexes['steps']
    if len(sidx) > 0:
        assert type(sidx[0]) in [pd.Period, np.int64]

    # indexes for the final output xr.DataArray
    varNames = data0.indexes['variables']
    namePositionsPrelim = {nm: i for i, nm in enumerate(varNames)}
    stepLabels = _extendIndex(sidx, nsteps)

    # create example data object in which data for one trail can accumulate
    dataPrelim = rdarrays.RDdata(data0.to_masked_array(),
                                 nsteps, namePositionsPrelim)

    # infer number of random vars reflected in 'step' fucntion
    # and the variable names being returned by 'step' and their order
    rvs, stepfNames = _checkf(stepf, copy(dataPrelim))

    # specify that the data objects that accumulate data for each trial
    # will reflect the union of the variables in 'data0' and the variables
    # returned by 'step'
    # breakpoint()
    varNamesList = list(varNames)
    if stepfNames == varNamesList:
        finalNames = varNamesList
        finalData0 = data0
    else:
        finalNames = list(set(varNamesList).union(set(stepfNames)))
        finalData0 = data0.copy()
        finalData0 = finalData0.reindex(
            variables=finalNames, fill_value=np.nan)

    # create a DataArray to hold the data for one trial
    # create example data object in which data for one trail can accumulate
    namePositions = {nm: i for i, nm in enumerate(finalNames)}
    data = rdarrays.RDdata(finalData0.to_masked_array(), nsteps, namePositions)

    # draws for all RVs in all time steps, w/ sampling stratified across trials
    if rvs > 0:
        np.random.seed(seed)
        if sampling == 'lh':    
            u = _lhs(rvs * nsteps, ntrials)  # np.array: (rvs*steps) x trials
        elif sampling == 'mc':  
            u = _mcs(rvs * nsteps, ntrials)  # monte carlo
        else:
            raise ValueError('sampling must be "lh" or "mc"')
        w = stats.norm.ppf(u) if stdnorm is True else u

    def trial(r):
        wgen = _makewgen(w, r) if rvs > 0 else None  # 'w' gener. for trial 'r'
        # perform all time steps for one trial
        # return _recurse(f=lambda x: step(x, wgen), x0=copy(data), S=steps)
        dataWorking = copy(data)
        for s in range(nsteps):
            # step 's' of trial 'r'
            dataWorking.append(stepf(wgen, dataWorking))
        return dataWorking


    # create and return 3-D output DataArray, with new dimension 'trials'
    if multi is True:
        out = multicore.parmap(lambda r: trial(r)._a, range(ntrials))
    else:
        out = [trial(r)._a for r in range(ntrials)]

    prelim = xr.DataArray(out, coords=[('trials', list(range(ntrials))),
                                       ('variables', finalNames),
                                       ('steps', stepLabels)])
    return prelim.transpose('trials', 'variables', 'steps')
