import numpy as np
import xarray as xr
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))
import rdarrays as rd
import conversions


def test_rda():
    # set up existing/historical data
    steps = [0, 1, 2]
    variables = ["p", "c"]
    a = np.array([[1.0, np.nan], [1.01, np.nan], [0.99, np.nan]])
    x = xr.DataArray(data=a, coords=(('steps', steps),
                                     ('variables', variables)))

    varNames = x.indexes['variables']
    namePos = {nm: i for i, nm in enumerate(varNames)}
    a = np.array(x.values)

    # steps index including all steps
    if a.shape[0] > 0:
        stepLabels = list(x.indexes['steps'])
        for i in range(len(steps)):
            stepLabels.append(stepLabels[-1] + 1)
    else:
        stepLabels = list(range(len(steps)))

    d = rd.RDdata(x, steps=3, namePositions=namePos)

    # add values for a step
    d.append({"p": 998.0, "c": 13.0})
    assert d.recall("p", 1) == 998.0
    assert d.recall("c", 1) == 13.0
    assert d.recall("p", 2) == 0.99

    # add values for another step
    d.append({"p": 999.0, "c": 14.0})
    assert d.recall("p", 1) == 999.0
    assert d.recall("c", 1) == 14.0
    assert d.recall("p", 2) == 998.0
    assert d.recall("c", 2) == 13.0
    assert d.recall("p", 3) == 0.99
