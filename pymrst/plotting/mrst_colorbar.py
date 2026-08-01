"""Modern MRST colorbar/histogram wrapper (mrstColorbar.m)."""

from __future__ import annotations

import numpy as np

from pymrst.plotting.colorbar_hist import colorbar_hist


def _axes_values(axes):
    values = []
    for artist in (*axes.collections, *axes.images):
        array = getattr(artist, "get_array", lambda: None)()
        if array is not None and np.asarray(array).size:
            values.append(np.asarray(array, dtype=float).ravel())
    if not values:
        raise ValueError("no plotted scalar data found; provide values")
    return np.concatenate(values)


def mrst_colorbar(
    axes=None,
    values=None,
    location: str = "southoutside",
    logarithmic: bool = False,
    limits=None,
    *,
    bins: int = 50,
    cmap: str = "viridis",
):
    """Add a colorbar and histogram to an axes or supplied values."""
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt.gca()
    data = _axes_values(axes) if values is None else np.asarray(values)
    if limits is None:
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("values contain no finite data")
        limits = (float(finite.min()), float(finite.max()))
    loc = location.lower()
    return colorbar_hist(
        data,
        limits,
        loc,
        bins,
        logarithmic,
        axes=axes,
        cmap=cmap,
    )


mrstColorbar = mrst_colorbar

__all__ = ["mrst_colorbar", "mrstColorbar"]
