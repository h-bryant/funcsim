Overview
========

The basic idea is that all an end user can run a simulation after
simply writing a specially-crafted function that performs a single
trial (for a cross-sectional simulation) or that takes one step
through time or space (for a recursive-dynamic simulation).


Simple cross-sectional example
------------------------------

For a cross-sectional simulation, the trial function has the
following form:

.. code-block:: python

     # function to perform one trial
     def trial(draw):
         # 'draw' argument is required and must be the only argument

         # independent draws from U(0, 1)
         u1 = next(draw)
         u2 = next(draw)

         # do some kind of work
         x = 1.5 * u1
         y = 34.2 + 0.95 * x + u2

         # return variables of interest as a dictionary
         return {"x": x, "y": y}

Any number of independent uniform draws can be taken.

Then a simulation is performed by invoking the ``crosssec`` function:

.. code-block:: python

    import funcsim as fs
  
    da = fs.crossec(trial=trial, trials=500)

Here, the value passed as the ``trials`` argument is the number of
trials that will be performed.  The returned object is a
2-dimensional `xarray <http://xarray.pydata.org/>`_
``DataArray`` instance, with a first dimension named ``trails``
and a second dimension named ``variables``.


Simple recursive-dynamic example
--------------------------------

For problems that reflect movement through time or space,
a recursive-dynamic simulation is appropriate.  The user specifies a
function describing one step forward, such as the one below.
The ``data`` argument here reflects historical observations. It is a 
2-dimensional ``DataArray`` instance, with a first dimension named 
``steps`` and a second dimension named ``variables``.

.. code-block:: python

    from scipy import stats

    def step(data, draw):
        # take one step through time

        # value of "p" in the previous period
        plag = fs.recall(data, "p", lag=1)

        # do some kind of work
        pnew = 0.25 + 0.9 * plag + stats.norm.ppf(next(draw))

        # chronicle the new value of "p" in the historical record
        datanew = fs.chron(data, {"p": pnew})

        return datanew

After specifying a ``DataArray`` containing historical data (here, called
``data0``), the simulation is invoked thusly:

.. code-block:: python

    out = fs.recdyn(step=step, data0=data0, steps=10, trials=500)

The output is a 3-dimensional ``DataArray`` instance, with dimensions named
``trials``, ``variables``, and ``steps``.
