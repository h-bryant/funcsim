import numpy as np
import pandas as pd
import xarray as xr

# convenience functions for non-mutating operations on
# xarray.DataArray instances with indexes 'steps' and 'variables'


def fromcsv(path):
    # create a 2-D xr.DataArray from a csv file
    #
    # The csv file is expected to have:
    # 1) variables in columns and observations in rows
    # 2) variable names in the first row
    # 3) dates in a the first column
    # 4) float-like data in the fields
    #
    # The output DataArray will have
    # a) a first dimension named 'steps' with a pd.PeriodIndex index
    # b) a second dimension named 'variables'

    pdframe = pd.read_csv(path, parse_dates=True, infer_datetime_format=True,
                          index_col=0)
    return xr.DataArray(pdframe.to_period(copy=False),
                        dims=('steps', 'variables'))


def chron(da, new):
    # chronicle 'new' data in the historical record 'da' (prior to 'new')
    #
    # 'da' is an xarray.DataArray w/indexes ['steps', 'variables']
    # 'new' is a dict with var names as keys and values as values

    # check types of inputs
    assert type(da) == xr.DataArray and type(new) == dict

    # get indices from 'da'
    assert sorted(da.indexes.keys()) == ['steps', 'variables']
    sidx = da.indexes['steps']
    vidx = da.indexes['variables']

    # check that we know how to cope with the types for the 'steps' index
    if len(sidx) > 0:
        assert type(sidx[0]) in [pd.Period, np.int64]

    # check that 'new' reflects all vars in 'da'
    assert sorted(list(vidx)) == sorted(new.keys())

    # extended steps index
    nextStep = sidx[-1] + 1 if len(sidx) > 0 else 0
    if type(nextStep) == pd.Period:
        newSidx = sidx.append(pd.PeriodIndex([nextStep]))
    else:
        newSidx = sidx.append(pd.Index([nextStep]))

    # actual work
    newA = np.array([new[v] for v in vidx])  # preserve variable order
    newA.shape = (1, len(vidx))
    return xr.DataArray(np.concatenate((da.values, newA)),
                        coords=[('steps', newSidx), ('variables', vidx)])


def recall(da, var, lag):
    # recall a value for a specific variable with the specified lag
    # relative to the last observation
    assert type(lag) == int and lag >= 1
    return float(da.sel(variables=var)[-lag])
