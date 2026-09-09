Plotting
========

funcsim's plotting functions build interactive `plotly
<https://plotly.com/python/>`_ figures.  They require the optional
``plotly`` dependency:

.. code-block:: bash

   pip install "funcsim[plotting]"

Every function returns a :class:`plotly.graph_objects.Figure`, which you
can display, customize further with plotly's own methods, or export.


Fan charts of simulation output
-------------------------------

:func:`funcsim.fan` plots one variable from the output of
:func:`funcsim.simulate` across the ``steps`` dimension, with the mean
across trials as a line and shaded bands containing the central 10%, 30%,
50%, 70%, and 90% of trials at each step.  Historical steps from ``hist0``
appear at the left, where all trials coincide.

.. code-block:: python

   fig = fs.fan(out, "p", title="Price")
   fig.show()

If the ``steps`` coordinate is a :class:`pandas.PeriodIndex`, the x axis
is a date axis.  The ``filepath`` argument saves the chart in one call:
a ``.html`` extension writes an interactive page, and ``.png``, ``.jpg``,
``.webp``, ``.svg``, or ``.pdf`` write a static image, which requires the
``kaleido`` package (``pip install kaleido``).  PDF export is the usual
route to including a fan chart in a LaTeX document.

.. code-block:: python

   fs.fan(out, "p", filepath="price_fan.pdf")


Distribution diagnostics
------------------------

:func:`funcsim.histpdf`
   A histogram of a data sample with a probability density function
   overlaid, for visually checking a fitted or kernel density:

   .. code-block:: python

      kde = fs.Kde(sample)
      fs.histpdf(sample, kde.pdf).show()

      fitted = fs.fit(sample, stats.gamma, lowerBound=0.0).dist
      fs.histpdf(sample, fitted.pdf, nbins=30).show()

:func:`funcsim.qqplot`
   A quantile-quantile plot of a data sample against a hypothesized
   distribution given by its ``ppf``:

   .. code-block:: python

      fs.qqplot(sample, fitted.ppf).show()

:func:`funcsim.dblscat`
   Two scatter plots of paired values on the same axes, for comparing the
   joint distribution of two simulated variables with the data they were
   fitted to:

   .. code-block:: python

      sim = da.sel(steps=0).to_pandas()[["a", "b"]]
      fs.dblscat(data[["a", "b"]], sim, "data", "simulated").show()

:func:`funcsim.twofuncs`
   Two functions of one variable on the same axes, for example two
   utility or probability weighting functions:

   .. code-block:: python

      fs.twofuncs(fs.utilPower, fs.utilNormLog, -50.0, 50.0,
                  name0="power", name1="normalized log").show()


Displaying figures
------------------

Plotly figures display in Jupyter and most IDE notebooks with
``fig.show()``.  Where the default renderer does not work (a remote
Jupyter server, VS Code, a headless machine), :func:`funcsim.show` calls
``fig.show()`` with the renderer named in the environment variable
``FUNCSIM_PLOTLY_RENDERER``, so that the choice of renderer can live in
your environment rather than in your code:

.. code-block:: bash

   export FUNCSIM_PLOTLY_RENDERER=vscode     # or notebook, browser, png, svg, ...

.. code-block:: python

   fs.show(fig)

If the variable is not set, plotly's default renderer is used.
