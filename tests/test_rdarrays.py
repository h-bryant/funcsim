import numpy as np
import xarray as xr
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../"))
import rdarrays as rd


def test_rda():
    # set up existing/historical data
    steps = [0, 1, 2]
    variables = ["p", "c"]
    a = np.array([[1.0, np.nan], [1.01, np.nan], [0.99, np.nan]])
    x = xr.DataArray(data=a, coords=(('steps', steps),
                                     ('variables', variables)))

    varNames = x.indexes['variables']
    namePos = {nm: i for i, nm in enumerate(varNames)}
    a = x.to_masked_array()

    # steps index inclusing all steps
    if a.shape[0] > 0:
        stepLabels = list(x.indexes['steps'])
        for i in range(len(steps)):
            stepLabels.append(stepLabels[-1] + 1)
    else:
        stepLabels = list(range(steps))

    d = rd.RDdata(x, steps=3, namePositions=namePos)

    # add values for a step
    d = rd.chron(d, {"p": 998.0, "c": 13.0})
    assert rd.recall(d, "p", 1) == 998.0
    assert rd.recall(d, "c", 1) == 13.0
    assert rd.recall(d, "p", 2) == 0.99

    # add values for another step
    d = rd.chron(d, {"p": 999.0, "c": 14.0})
    assert rd.recall(d, "p", 1) == 999.0
    assert rd.recall(d, "c", 1) == 14.0
    assert rd.recall(d, "p", 2) == 998.0
    assert rd.recall(d, "c", 2) == 13.0
    assert rd.recall(d, "p", 3) == 0.99
