import numpy as np
import xarray as xr


class RDdata():
    # container for an np array with pre-allocated memory for future values
    # for a single trial in an RD sim
    def __init__(self, a, steps, namePositions):
        # 'a' is hist data as np.array, with vars as columns and steps as rows
        # 'steps' is integer number of new/future steps
        # 'namePositions' is a dict with var names as keys and integer column
        #     number of that var within 'a'

        self._namePos = namePositions
        self._histSteps = a.shape[0]
        self._totSteps = self._histSteps + steps
        self._varNames = set(namePositions.keys())

        # set up values container for all variables & time steps
        self._a = np.empty((len(self._namePos.keys()), self._totSteps))
        self._a[:, 0:self._histSteps] = a.transpose()

        self._currStep = 0

    def incrStep(self):
        self._currStep += 1

    def append(self, varname, value):
        self._a[self._namePos[varname], self._histSteps+self._currStep] = value

    def recall(self, varname, lag=0):
        return self._a[self._namePos[varname],
                       self._histSteps + self._currStep - lag]

    @property
    def varNameSet(self):
        return self._varNames

    @property
    def array(self):
        return self._a


def chron(dobj, new):
    # append data for all vars with a dict with var names as keys

    # check types of inputs; check that 'new' reflects all vars in 'da'
    assert isinstance(dobj, RDdata)
    assert type(new) == dict
    if not dobj.varNameSet == set(new.keys()):
        msg = 'variable names passed to "chron" do not match the '
        msg2 = 'variable names in "data0"'
        raise ValueError(msg + msg2)

    # actual work
    [dobj.append(i[0], i[1]) for i in new.items()]
    dobj.incrStep()
    return dobj


def recall(dobj, var, lag):
    # retrieve a previous value
    return dobj.recall(var, lag)
