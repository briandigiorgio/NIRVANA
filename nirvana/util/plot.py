"""
Plotting utilities.

.. include:: ../include/links.rst
"""

import numpy as np

from matplotlib import ticker

def get_logformatter():
    """
    Return a matplotlib label formatter that uses decimals instead of exponentials.

    This is a hack that I pulled off of stackoverflow (since lost the url).
    It breaks some imshow things, I think. Probably better to get this
    functionality by subclassing from ticker.LogFormatter...
    """
    # TODO: Introduce an upper/lower limit to switch back to exponential form ;
    # i.e., make label go something like 10^4, 1000, ..., 0.01, 0.001, 10^{-4}
    return ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(
                                        int(np.maximum(-np.log10(y),0)))).format(y))


def init_ax(fig, pos, facecolor='0.85', tickdir='in', top=True, right=True, majlen=4, minlen=2,
            grid=True, gridcolor='0.75'):
    """
    Convenience method for initializing a `matplotlib.axes.Axes`_ object.

    Args:
        fig (`matplotlib.figure.Figure`_):
            Figure in which to place the Axes object.
        pos (:obj:`list`):
            The rectangle outlining the locating of the axes in the Figure
            object, specifying the left, bottom, width, and height
            dimensions. See the ``rect`` argument of
            `matplotlib.figure.Figure.add_axes`_.
        facecolor (:obj:`str`, optional):
            Color for the axis background.
        tickdir (:obj:`str`, optional):
            Direction for the axis tick marks.
        top (:obj:`bool`, optional):
            Add ticks to the top axis.
        right (:obj:`bool`, optional):
            Add ticks to the right axis.
        majlen (:obj:`int`, optional):
            Length for the major tick marks.
        minlen (:obj:`int`, optional):
            Length for the minor tick marks.
        grid (:obj:`bool`, optional):
            Include a major-axis grid.
        gridcolor (:obj:`str`, optional):
            Color for the grid lines.

    Returns:
        `matplotlib.axes.Axes`_: Axes object
    """
    ax = fig.add_axes(pos, facecolor=facecolor)
    ax.minorticks_on()
    ax.tick_params(which='major', length=majlen, direction=tickdir, top=top, right=right)
    ax.tick_params(which='minor', length=minlen, direction=tickdir, top=top, right=right)
    if grid:
        ax.grid(True, which='major', color=gridcolor, zorder=0, linestyle='-')
    return ax


def get_twin(ax, axis, tickdir='in', majlen=4, minlen=2):
    """
    Construct the "twin" axes of the provided Axes object.

    Args:
        ax (`matplotlib.axes.Axes`_):
            Original Axes object.
        axis (:obj:`str`):
            The axis to *mirror*. E.g., to get the right ordinate that
            mirrors the left ordinate, use ``axis='y'``.
        tickdir (:obj:`str`, optional):
            Direction for the axis tick marks.
        majlen (:obj:`int`, optional):
            Length for the major tick marks.
        minlen (:obj:`int`, optional):
            Length for the minor tick marks.

    Returns:
        `matplotlib.axes.Axes`_: Axes object selecting the requested axis
        twin.
    """
    axt = ax.twinx() if axis == 'y' else ax.twiny()
    axt.minorticks_on()
    axt.tick_params(which='major', length=majlen, direction=tickdir)
    axt.tick_params(which='minor', length=minlen, direction=tickdir)
    return axt


def rotate_y_ticks(ax, rotation, va):
    """
    Rotate all the existing y tick labels by the provided rotation angle
    (deg) and reset the vertical alignment.

    Args:
        ax (`matplotlib.axes.Axes`_):
            Rotate the tick labels for this Axes object. **The object is
            edited in place.**
        rotation (:obj:`float`):
            Rotation angle in degrees
        va (:obj:`str`):
            Vertical alignment for the tick labels.
    """
    for tick in ax.get_yticklabels():
        tick.set_rotation(rotation)
        tick.set_verticalalignment(va)


