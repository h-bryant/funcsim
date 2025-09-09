import sys
from copy import deepcopy as copy
import numpy as np
import pandas as pd
import xarray as xr
import multicore
import rdarrays
from scipy import stats
from collections.abc import Callable, Generator
from typing import Optional
import inspect


def _get_arg_count(func):
    sig = inspect.signature(func)
    params = sig.parameters.values()

    # Count only parameters that are positional or keyword
    # (excluding *args and **kwargs)
    return sum(
        1 for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    )


def _checkhist0(hist0):
    # check that user's hist0 seems sane.  Return list of variable names
    # in hist0 order.

    if isinstance(hist0, xr.DataArray) is False:
        raise ValueError('"hist0" must be an xarray.DataArray')

    # check that hist0 has the right dimension names 
    hist0Coords = hist0.coords
    if not len(hist0Coords) == 2:
        raise ValueError('"hist0" must have exactly two dimensions')
    if not ("variables" in hist0Coords.keys() and
            "steps" in hist0Coords.keys()):
        raise ValueError('"hist0" must have dimensions "variables" and "steps"')
    
    # Check for an appropriate index for the 'variables' dimension
    stepsCoords = hist0Coords["steps"]
    if not (np.issubdtype(stepsCoords[0], np.integer) or
            isinstance(stepsCoords.to_index(), pd.PeriodIndex)):
        raise ValueError('"hist0" must have either an integer index or a '
                         'pandas PeriodIndex for the "steps" dimension')

    return list(hist0Coords["variables"])


def _checkf(f, data0):
    # Count number of times 'f', and any code 'f' invokes, calls 'next(draw)'
    # If 'data0' is None, infer that 'f' is the 'trial' func for a cross-sec sim
    # If 'data0' is a xr.DataArray, infer that 'f' is 'step' for rec. dyn. sim
    # Also, check that f returns something that makes sense.
    fakeugen = _countgen()

    # check that 'f' returns a dict
    out = f(fakeugen, data0)
    if type(out) != dict:
        raise ValueError('"f" function must return a dict')

    # check that the dict returned by 'f' has variable names as keys
    varnames = list(out.keys())
    if sum(1 if type(k) is str else 0 for k in varnames) < len(varnames):
        raise ValueError('The keys of the dictionary returned by "f" must be '
                         'variable names as strings')

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


def simulate(f: Callable[[Generator[int, float, None],
                          Optional[rdarrays.RDdata]],
                       dict[str, float]],
             ntrials: Optional[int] = 500,
             nsteps: Optional[int] = 1,
             hist0: Optional[xr.DataArray] = None,
             multi: Optional[bool] = False,
             seed: Optional[int] = 6,
             stdnorm: Optional[bool] = False,
             sampling: Optional[str] = 'lh'
             ) -> xr.DataArray:
    """
    Stochastic simulation.

    Parameters
    ----------
    f : function
        Function that performs a single trial in a static simulation or a s
        single step through time in a recursive dynamic simulation.  Should take
        'ugen' as a first argument in either case, where 'ugen' will be a
        generator that emits standard uniform draws (or standard normal draws,
        if `stdnorm` is True) that will be passed to f by ``simulate``.
        In the case of a recursive dynamic simulation that employs past values,
        f should take 'data' as a second argument, where this will be a type of
        array that is also provided by ``simulate.`` This function should return
        a dict with variable names (as strings) as keys and values for those
        variables (as floats) as values.
    ntrials : int, optional
        The number of trials to perform.  Default is 500.
    nsteps : int, optional
        The number of steps to perform in each trial.  Default is 1.
    hist0 : xarray.DataArray, optional
        Initial and/or historical data relevant to the simulation.
        Should have dimensions 'variables' and 'steps'.  Any lagged values
        needed (recalled) by `f` must be in `hist0`.  The 'steps' dimension
        should have either an integer index or a pandas PeriodIndex.
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
    if hist0 is not None:
        _checkhist0(hist0)

        # check that we know how to cope with the types for the 'steps' index
        sidx = hist0.indexes['steps']
        if len(sidx) > 0:
            if not type(sidx[0]) in [pd.Period, np.int64]:
                raise ValueError("'hist0' should have either an integer index"
                                 " or a pandas.PeriodIndex for the 'steps' "
                                 "dimension.")
    else:
        sidx = []
        # create an empty hist0.
        variables = np.array([], dtype=str)
        steps = np.array([], dtype=int)
        hist0 = xr.DataArray(
            data=np.empty((0, 0)),
            dims=("steps", "variables"),
            coords={"steps": steps, "variables": variables}
        )

    # check for 'f'
    if not isinstance(f, Callable):
        raise ValueError('"f" must be a callable function')

    # infer number of arguments in 'f'.  If it takes only a single arg, wrap it
    # in an outer func that takes "hist" as a second arg
    numb_f_args = _get_arg_count(f)
    if numb_f_args == 1:
        stepf = lambda draw, hist: f(draw)
    elif numb_f_args == 2:
        stepf = f
    else:
        raise ValueError('"f" should take two arguments at most')
     
    # indexes for the final output xr.DataArray
    varNames = hist0.indexes['variables']
    namePositionsPrelim = {nm: i for i, nm in enumerate(varNames)}
    stepLabels = _extendIndex(sidx, nsteps)

    # create example data object in which data for one trail can accumulate
    dataPrelim = rdarrays.RDdata(hist0.to_masked_array(),
                                 nsteps, namePositionsPrelim)

    # infer number of random vars reflected in 'step' fucntion
    # and the variable names being returned by 'step' and their order
    rvs, stepfNames = _checkf(stepf, copy(dataPrelim))

    # specify that the data objects that accumulate data for each trial
    # will reflect the union of the variables in 'hist0' and the variables
    # returned by 'step'
    # breakpoint()
    varNamesList = list(varNames)
    if stepfNames == varNamesList:
        finalNames = varNamesList
        finalHist0 = hist0
    else:
        finalNames = list(set(varNamesList).union(set(stepfNames)))
        finalHist0 = hist0.copy()
        finalHist0 = finalHist0.reindex(variables=finalNames, fill_value=np.nan)

    # create a DataArray to hold the data for one trial
    # create example data object in which data for one trail can accumulate
    namePositions = {nm: i for i, nm in enumerate(finalNames)}
    data = rdarrays.RDdata(finalHist0.to_masked_array(), nsteps, namePositions)

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
