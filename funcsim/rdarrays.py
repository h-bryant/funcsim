import numpy as np
import xarray as xr


class MissingValue(Exception):
    """Raised by :meth:`RDdata.recall` when a requested value is unavailable."""
    pass


class RDdata():
    """
    Per-trial history for a recursive-dynamic simulation.

    ``simulate`` passes an instance of this class as the second argument
    (conventionally ``hist``) to a step function that accepts two
    arguments.  It holds the historical values supplied via ``hist0``
    followed by the values the step function has returned so far in the
    current trial.  Step functions should only call :meth:`recall`; the
    class is not meant to be instantiated by users.
    """
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
        self._a = np.full(shape=(len(self._namePos.keys()), self._totSteps),
                          fill_value=np.nan, dtype=float)
        self._a[:, 0:self._histSteps] = np.copy(a).transpose()

        self._currStep = 0

    def incrStep(self):
        self._currStep += 1

    def append(self, valueDict):
        for varname, value in valueDict.items():
            self._a[self._namePos[varname],
                    self._histSteps + self._currStep] = value   
        self._currStep += 1

    def recall(self,
               varname: str,
               lag: int = 0
              ) -> float:
        """
        Recall the value of a variable at the current step minus lag.

        Parameters
        ----------
        varname : str
            Name of the variable to recall.
        lag : int, optional
            Number of steps to look back.  ``lag=1`` is the previous step
            (the most recent value available while a step is being
            computed).  Default is 0.

        Returns
        -------
        float
            The recalled value.

        Raises
        ------
        MissingValue
            If the requested value is not available.
        """
        # bound-check the position explicitly: a negative index would
        # silently wrap around and return a wrong value
        pos = self._histSteps + self._currStep - lag
        if varname not in self._namePos or pos < 0 or pos >= self._totSteps:
            ret = np.nan
        else:
            ret = self._a[self._namePos[varname], pos]
        if np.isnan(ret):
            if self._currStep == 0:
                raise MissingValue(f'In the 1st time step in the '
                                   f'simulation, no (non-NaN) value for the '
                                   f'variable "{varname}" with {lag} lag(s) is '
                                   f'available. Be sure that you passed a '
                                   f'"hist0" array to "simulate" and that it '
                                   f'contains this value.')
            else:
                raise MissingValue(f'In time step {self._currStep} in the '
                                   f'simulation, no value for the variable '
                                   f'"{varname}" with {lag} lag(s) is '
                                   f'available. Be sure that your "f" is '
                                   f'including non-NaN values for this '
                                   f'variable in the dictionary it returns.')
        return ret

    @property
    def varNameSet(self):
        return self._varNames

    @property
    def array(self):
        return self._a