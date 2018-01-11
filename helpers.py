import pandas as pd
import xarray as xr


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
