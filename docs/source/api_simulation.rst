Simulation
==========

:func:`funcsim.simulate` is the single entry point for running a simulation.
It accepts a user-written trial or step function and returns a labeled
:class:`xarray.DataArray`.  See :doc:`overview` for a narrative
introduction with examples.

.. automodule:: funcsim
   :no-index:
   :members: simulate
   :undoc-members:
   :show-inheritance:


History object passed to step functions
---------------------------------------

In a recursive-dynamic simulation, a step function that accepts two
arguments receives an :class:`~funcsim.rdarrays.RDdata` instance as its
second argument.  Step functions interact with it only through its
:meth:`~funcsim.rdarrays.RDdata.recall` method.

.. autoclass:: funcsim.rdarrays.RDdata
   :members: recall

.. autoexception:: funcsim.rdarrays.MissingValue
