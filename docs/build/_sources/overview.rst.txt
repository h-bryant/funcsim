Overview
========

This is a tool that allows an end user to run a simulation after
simply writing a specially-crafted function that performs a single
trial (for a static simulation) or that takes one step
through time (for a recursive-dynamic simulation).


Simple static simulation example
--------------------------------

For a static simulation, the trial function has the
following form:

.. code-block:: python

     # function to perform one trial
     def trial(draw):
         # 'draw' argument is required and must be the first argumnet

         # independent draws from U(0, 1)
         u1 = next(draw)
         u2 = next(draw)

         # do some kind of work
         x = 1.5 * u1
         y = 34.2 + 0.95 * x + u2

         # return variables of interest as a dictionary
         return {"x": x, "y": y}

Any number of independent uniform draws can be taken.

Then a simulation is performed by invoking the ``simulate`` function:

.. code-block:: python

    import funcsim as fs
  
    da = fs.simulate(f=trial, ntrials=500)

Here, the value passed as the ``ntrials`` argument is the number of
trials that will be performed.  
The output is a 3-dimensional ``xarray.DataArray`` instance, with dimensions
named ``trials``, ``variables``, and ``steps``.
For this simple static simulation, the ``steps`` dimension is unimportant,
as the simulation does not reflect movement through time.


Simple recursive-dynamic example
--------------------------------

For problems that reflect movement through time,
a recursive-dynamic simulation is appropriate.  The user specifies a
function describing one step forward, as in the example below.

.. code-block:: python

    from scipy import stats

    def step(draw, data):
        # Function to take one step through time. 
        # The 'draw' argument is first and is
        # required as with the static simulation.  The 'data' argument is needed
        # if 'f' refers to lagged values of some sort, as in the example below).

        # value of "p" in the previous period
        plag = data.recall("p", lag=1)

        # do some kind of work
        pnew = 0.25 + 0.9 * plag + stats.norm.ppf(next(draw))

        # return new value(s) for this step/time period
        return {"p": pnew}

The ``step`` function above is relying on lagged values of the variable ``p``.
To accommodate this, we specify a 2-D ``xarray.DataArray`` containing historical
data (here, called ``myhist``), with a first dimension named 
``steps`` and a second dimension named ``variables``, that we will pass to
``simulate`` via an argument named ``hist0``.
Then, a simulation reflecting 10 time steps can be performed thusly:


.. code-block:: python

    out = fs.recdyn(f=step, hist0=myhist, nsteps=10)


The output is again a 3-dimensional ``xarray.DataArray`` instance, with
dimensions named ``trials``, ``variables``, and ``steps``.
However, it will now reflect multiple ``steps``: ten simulated time steps plus
however many historical observations were reflected in ``data0``.
