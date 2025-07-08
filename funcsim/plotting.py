
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from typing import Callable, Optional
import conversions


def fan(da: xr.DataArray,
        varname: str,
        filepath: str = None,
        line_color: str = 'blue',
        title: str = "",
        width: int = 750,
        height: int = 400):
    """
    Create a fan chart for a variable from a simulation output xr.DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Simulation results with dimensions "trials", "variables", "steps".
    varname : str
        Name of the variable to plot from the "variables" dimension.
    filepath : str, opti
        Path to save the chart. Supports HTML and image formats. Default None.
    line_color : str, optional
        Color of the mean line. Default is 'blue'.
    title : str, optional
        Title for the figure. Default is "" (no title).
    width : int, optional
        Width of the chart in pixels. Default is 800.
    height : int, optional
        Height of the chart in pixels. Default is 500.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object for the fan chart.
    """
    # conditional import of plotly
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Optional dependency 'plotly' is required for all "
                            "funcsim plotting fucntions. Install with "
                            "`pip install plotly`.") from e
  
    # Validate input DataArray dimensions
    if not all(dim in da.dims for dim in ["trials", "variables", "steps"]):
        raise ValueError("Input DataArray 'da' must have dimensions 'trials',"
                         "'variables', and 'steps'.")
    
    # Validate if varname exists in the DataArray
    if varname not in da["variables"].values:
        raise ValueError(f"Variable '{varname}' not found in DataArray's "
                         f"'variables' coordinate.")

    # Select the specified variable and convert to a pandas DataFrame
    # Transpose so that 'steps' becomes the index and 'trials' become columns
    dt = da.sel(variables=varname).to_pandas().transpose()
    
    if dt.empty:
        warnings.warn(f"Warning: Data for variable '{varname}' is empty after "
                      f"selection and processing.", UserWarning)
        # Return an empty figure or handle as appropriate
        fig = go.Figure()
        fig.update_layout(title_text=f"{varname} (No data)")
        return fig

    # 'steps' values from the index of the transposed DataFrame
    idx_original = dt.index # Keep original for potential messages or fallback
    idx = dt.index 
    
    # Ensure index is JSON serializable by converting to Timestamps if it
    # contains Periods. This handles both PeriodIndex and object-dtype Index
    # containing Period objects.
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
        # print("DEBUG: Converted PeriodIndex to DatetimeIndex.") # Optional
    elif hasattr(idx, 'dtype') and pd.api.types.is_object_dtype(idx.dtype):
        # If it's an object-dtype idx, check if it might contain Period objects.
        if len(idx) > 0:
            try:
                converted_values = []
                needs_conversion = False
                for item in idx:
                    if isinstance(item, pd.Period):
                        converted_values.append(item.to_timestamp())
                        needs_conversion = True
                    else:
                        # Pass through other types (e.g., pd.NaT, None, or
                        # other data if mixed)
                        converted_values.append(item) 
                
                if needs_conversion:
                    idx = pd.DatetimeIndex(converted_values)
                    # print("DEBUG: Converted object-dtype index with Period "
                    #       "objects to DatetimeIndex.") # Optional debug
                # If no Period objects were found and converted, idx remains the
                # original object-dtype index.
                    
            except Exception as e:
                # This might happen if conversion of an item fails unexpectedly.
                warnings.warn(f"Warning: Attempted to convert an object-dtype "
                              f"index that might contain Periods, but an error "
                              f"occurred: {e}. Using original index for x-axis,"
                              f" which may lead to Plotly errors.", UserWarning)
                idx = idx_original # Revert to original index
        
    # Calculate the mean across trials for each step
    mean_values = dt.mean(axis=1)

    # Initialize a Plotly Figure
    fig = go.Figure()

    # Define the quantile band widths to plot (from widest to narrowest
    # for layering)
    band_widths = [90, 70, 50, 30, 10] 
    
    # Add quantile bands as filled areas
    for band_width in band_widths:
        # Calculate upper and lower percentile values
        upper_percentile_val = 100.0 - (100.0 - float(band_width)) / 2.0
        lower_percentile_val = (100.0 - float(band_width)) / 2.0
        
        # Calculate percentile values across trials for each step
        y_lower = np.percentile(dt, lower_percentile_val, axis=1)
        y_upper = np.percentile(dt, upper_percentile_val, axis=1)

        # Add a scatter trace for the filled area
        # X-coordinates go from start to end for the upper bound, then end to
        # start for the lower bound
        # Y-coordinates match this path to create a closed shape
        fig.add_trace(go.Scatter(
            x=list(idx) + list(idx[::-1]), 
            y=list(y_upper) + list(y_lower[::-1]), 
            fill='toself',  # Fill the area enclosed by the trace
            fillcolor='rgba(173, 216, 230, 0.4)',  # Light blue with 20% opacity
            line=dict(color='rgba(255,255,255,0)'),  # No visible band edges
            hoverinfo="skip",  # Don't show hover tooltips for the band shapes
            showlegend=False,  # Don't show these bands in the legend
            name=f'{band_width}% Quantile' 
        ))

    # Add the mean line (plotted on top of the bands)
    fig.add_trace(go.Scatter(
        x=idx,
        y=mean_values,
        mode='lines',
        line=dict(color=line_color), # Use the specified line color
        name='Mean'  # Name for the legend
    ))

    # Customize the layout of the figure
    fig.update_layout(
        xaxis_title='',      # Remove the x-axis label (originally "steps")
        yaxis_title=varname, # Set y-axis label to variable name
        showlegend=True      # Display the legend (will show the 'Mean' trace)
    )

    # Save the figure to a file if a filepath is provided
    if filepath is not None:
        try:
            if filepath.endswith(".html"):
                fig.write_html(filepath)
                print(f"Chart saved to {filepath}")
            elif filepath.endswith((".png", ".jpeg", ".jpg", ".webp",
                                    ".svg", ".pdf")):
                # Static image export requires the 'kaleido' package
                # You can install it via: pip install -U kaleido
                fig.write_image(filepath)
                print(f"Chart saved to {filepath}")
            else:
                # Default to HTML if format is unrecognized
                html_filepath = filepath.rsplit('.', 1)[0] + ".html"
                fig.write_html(html_filepath)
                print(f"Unsupported file format for '{filepath}'. "
                      f"Saved as HTML: {html_filepath}")
        except Exception as e:
            print(f"Error saving file to '{filepath}': {e}")
            # Fallback to HTML if image saving fails (e.g. Kaleido missing)
            if not filepath.endswith(".html"):
                try:
                    html_fallback_path = \
                        filepath.rsplit('.', 1)[0] + "_fallback.html"
                    fig.write_html(html_fallback_path)
                    print(f"As a fallback, chart saved to {html_fallback_path}")
                except Exception as html_e:
                    print(f"Could not save HTML fallback: {html_e}")

    # make the background grid subtle
    fig.update_layout(
        title=varname,
        xaxis_title='',    
        yaxis_title=varname,
        legend=dict(borderwidth=0),
        template='simple_white',
        plot_bgcolor='white'         # explicit white background
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(200,200,200,0.4)',
        gridwidth=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(200,200,200,0.4)',
        gridwidth=1
    )
    if title != "":
        fig.update_layout(
            title=title,
        )

    # resize for fitting in a jupyter window without scrolling
    fig.update_layout(width=width, height=height)
    
    return fig # Return the Plotly figure object


def twofuncs(f0: Callable[[float], float],
             f1: Callable[[float], float],
             xmin: float,
             xmax: float,
             name0: str = "f0",
             name1: str = "f1",
             title: str = "",
             width: int = 750,
             height: int = 400
             ):
    """
    Plot two functions on the same axes using Plotly.

    Parameters
    ----------
    f0 : callable
        First function to plot. Should accept a float and return a float.
    f1 : callable
        Second function to plot. Should accept a float and return a float.
    xmin : float
        Minimum value of the x-axis.
    xmax : float
        Maximum value of the x-axis.
    name0 : str, optional
        Label for the first function. Default is "f0".
    name1 : str, optional
        Label for the second function. Default is "f1".
    title : str, optinal
        Title for the top of the figure.  Defaults to None.
    width : int, optional
        Width of the chart in pixels. Default is 800.
    height : int, optional
        Height of the chart in pixels. Default is 500.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object with both functions plotted.
    """
    # conditional import of plotly
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Optional dependency 'plotly' is required for all "
                            "funcsim plotting fucntions. Install with "
                            "`pip install plotly`.") from e
 
    x = np.linspace(xmin, xmax, 200)
    y0 = [f0(val) for val in x]
    y1 = [f1(val) for val in x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y0, mode='lines', name=name0))
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=name1))
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        template="simple_white"
    )
    if title != "":
        fig.update_layout(
            title=title,
        )

    # resize for fitting in a jupyter window without scrolling
    fig.update_layout(width=width, height=height)
 
    return fig


def histpdf(data: conversions.VectorLike,
            pdf: Callable[[float], float],
            nbins: int = None,
            title: str = "",
            width: int = 750,
            height: int = 400
            ):
    """
    Plot a histogram of data and a probability density function (PDF).

    Parameters
    ----------
    data : VectorLike
        Input data to plot as a histogram.
    pdf : Callable[[float], float]
        Probability density function to plot. Should accept a float and
        return a float.
    nbins : int, optional
        Number of bins for the histogram. If None, uses 'auto' binning.
    title : str, optional
        Title for the figure. Default is "" (no title).
    width : int, optional
        Width of the chart in pixels. Default is 800.
    height : int, optional
        Height of the chart in pixels. Default is 500.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object reflecting the histogram and PDF.
    """
    # conditional import of plotly
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Optional dependency 'plotly' is required for all "
                            "funcsim plotting fucntions. Install with "
                            "`pip install plotly`.") from e
 
    dataA = conversions.vlToArray(data)
    x = np.linspace(np.min(dataA), np.max(dataA), 200)
    y = [pdf(val) for val in x]

    if nbins is None:
        bins = np.histogram_bin_edges(dataA, bins='auto')
        nbinsx = len(bins) - 1
    else:
        nbinsx = nbins

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dataA, nbinsx=nbinsx, histnorm='probability density', opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', line=dict(color='red')
    ))
    fig.update_layout(
        yaxis_title="Density",
        template="simple_white"
    )
    if title:
        fig.update_layout(title=title)

    fig.update_layout(showlegend=False)
 
    if title != "":
        fig.update_layout(
            title=title,
        )

    # resize for fitting in a jupyter window without scrolling
    fig.update_layout(width=width, height=height)
 
    return fig


def dblscat(a0: conversions.ArrayLike,
            a1: conversions.ArrayLike,
            name0: Optional[str] = None,
            name1: Optional[str] = None,
            title: str = "",
            width: int = 500,
            height: int = 500
            ):
    """
    Create a scatter plot reflecting two sets of paired values.

    Parameters
    ----------
    a0 : conversions.ArrayLike
        First array of paired values, with shape (M, 2), with variables
        in columns, obs in rows.
    a1 : conversions.ArrayLike
        Second array of paired values, with shape (N, 2), variables
        in columns, obs in rows.
    name0 : str, optional
        Label for the first array of paired values. Default is "a0".
    name1 : str, optional
        Label for the second array of paired values. Default is "a1".
    title : str, optional
        Title for the figure. Default is "" (no title).
    width : int, optional
        Width of the chart in pixels. Default is 800.
    height : int, optional
        Height of the chart in pixels. Default is 500.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object with both scatter plots.

    Raises
    ------
    ValueError
        If input arrays are not 2D with exactly two columns.
    """
    # conditional import of plotly
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Optional dependency 'plotly' is required for all "
                            "funcsim plotting fucntions. Install with "
                            "`pip install plotly`.") from e

    a0A = conversions.alToArray(a0)
    a1A = conversions.alToArray(a1)

    # names
    varNames = conversions.alColNames(a0)
    n0 = name0 if name0 is not None else "a0"
    n1 = name1 if name1 is not None else "a1"

    if a0A.ndim != 2:
        raise ValueError("a0 must have exactly two dimensions")
    if a1A.ndim != 2:
        raise ValueError("a1 must have exactly two dimensions")
    if a0A.shape[1] != 2:
        raise ValueError("a0 must have exactly two columns")
    if a1A.shape[1] != 2:
        raise ValueError("a1 must have exactly two columns")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a0A[:, 0], y=a0A[:, 1], mode='markers',
        marker=dict(color='red', symbol='circle', size=8),
        name=n0
    ))
    fig.add_trace(go.Scatter(
        x=a1A[:, 0], y=a1A[:, 1], mode='markers',
        marker=dict(color='grey', symbol='x', size=8),
        name=n1
    ))
    fig.update_layout(
        xaxis_title=varNames[0],
        yaxis_title=varNames[1],
        template="simple_white",
        legend=dict(x=0.01, y=0.99)
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    if title != "":
        fig.update_layout(
            title=title,
        )

    # resize for fitting in a jupyter window without scrolling
    fig.update_layout(width=width, height=height)
 
    return fig