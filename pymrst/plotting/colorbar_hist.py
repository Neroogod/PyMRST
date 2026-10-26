"""Colorbar with an adjacent histogram (MRST colorbarHist.m)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HistogramColorbarResult:
    counts: np.ndarray
    bins: np.ndarray
    limits: tuple[float, float]
    location: str
    logarithmic: bool
    colorbar: Any = None
    histogram_axes: Any = None
    histogram_artist: Any = None


def hist_at_centers(values, centers):
    """MATLAB hist-compatible counts for explicit bin centers."""
    data = np.asarray(values, dtype=float).ravel()
    center = np.asarray(centers, dtype=float).ravel()
    if center.size == 0 or not np.all(np.diff(center) > 0):
        raise ValueError("histogram centers must be strictly increasing")
    if center.size == 1:
        return np.asarray([data.size], dtype=np.int64)
    boundaries = (center[:-1]+center[1:])/2.0
    indices = np.searchsorted(boundaries, data, side="right")
    return np.bincount(indices, minlength=center.size)


def histogram_colorbar_data(values, limits, *, bins=50, logarithmic=False):
    data = np.asarray(values, dtype=float).ravel()
    limits_array = np.asarray(limits, dtype=float).ravel()
    if limits_array.size != 2 or not np.all(np.isfinite(limits_array)):
        raise ValueError("limits must contain two finite values")
    if limits_array[1] <= limits_array[0]:
        raise ValueError("limits must be strictly increasing")
    if logarithmic:
        if np.any(data <= 0) or np.any(limits_array <= 0):
            raise ValueError("logarithmic colorbars require positive values")
        data = np.log10(data)
        limits_array = np.log10(limits_array)
    if np.isscalar(bins):
        count = int(bins)+1
        if count < 2:
            raise ValueError("bins must be a positive integer")
        centers = np.linspace(limits_array[0], limits_array[1], count)
    else:
        centers = np.asarray(bins, dtype=float).ravel()
    counts = hist_at_centers(data, centers)
    return counts, centers, tuple(map(float, limits_array))


def colorbar_hist(
    values,
    limits,
    location: str = "south",
    bins=50,
    logarithmic=None,
    *,
    axes=None,
    cmap: str = "viridis",
) -> HistogramColorbarResult:
    """Render an MRST-style colorbar/histogram pair.

    As in MRST, `bins` is used only when `logarithmic` is not given (MRST's
    `nbins = varargin{2}+1` applies only for `nargin == 4`): passing the
    log flag, true or false, gives MRST's default 51 bin centres.
    """
    if logarithmic is not None:
        bins = 50
    logarithmic = bool(logarithmic)
    loc = location.lower().replace("outside", "")
    if loc not in {"south", "north", "east", "west"}:
        raise ValueError("location must be North, South, East, or West")
    counts, centers, transformed_limits = histogram_colorbar_data(
        values, limits, bins=bins, logarithmic=logarithmic
    )
    try:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "colorbar_hist requires Matplotlib; install pymrst[plotting]"
        ) from error
    target = axes
    if target is None:
        _, target = plt.subplots()
    figure = target.figure
    bbox = target.get_position()
    horizontal = loc in {"south", "north"}
    if loc == "south":
        hist_pos = [bbox.x0, max(0.02, bbox.y0-0.16), bbox.width, 0.08]
        color_pos = [bbox.x0, max(0.005, bbox.y0-0.205), bbox.width, 0.025]
    elif loc == "north":
        # MRST places the histogram next to the axes and the colorbar farther
        # outward. Keep both inside the figure even for Matplotlib's default
        # axes rectangle.
        available = max(0.04, 1.0-bbox.y1)
        gap = min(0.012, 0.10*available)
        color_height = min(0.025, 0.22*available)
        histogram_height = min(0.08, 0.55*available)
        hist_y = bbox.y1 + gap
        color_y = hist_y + histogram_height + gap
        hist_pos = [bbox.x0, hist_y, bbox.width, histogram_height]
        color_pos = [bbox.x0, color_y, bbox.width, color_height]
    elif loc == "east":
        hist_pos = [min(0.88, bbox.x1+0.02), bbox.y0, 0.08, bbox.height]
        color_pos = [min(0.97, bbox.x1+0.105), bbox.y0, 0.02, bbox.height]
    else:
        hist_pos = [max(0.02, bbox.x0-0.10), bbox.y0, 0.08, bbox.height]
        color_pos = [max(0.005, bbox.x0-0.14), bbox.y0, 0.02, bbox.height]
    histogram_axes = figure.add_axes(hist_pos)
    width = np.min(np.diff(centers)) if centers.size > 1 else 1.0
    if horizontal:
        artist = histogram_axes.bar(
            centers, counts, width=width, facecolor="none", edgecolor="0.2"
        )
        histogram_axes.set_xlim(transformed_limits)
        if loc == "north":
            # Equivalent to MRST's view(0, -90), opposite the south view.
            histogram_axes.invert_yaxis()
    else:
        signed = -counts if loc == "west" else counts
        artist = histogram_axes.barh(
            centers, signed, height=width,
            facecolor="none", edgecolor="0.2",
        )
        histogram_axes.set_ylim(transformed_limits)
    histogram_axes.set_axis_off()
    color_axes = figure.add_axes(color_pos)
    mappable = ScalarMappable(
        norm=Normalize(*transformed_limits), cmap=cmap
    )
    colorbar = figure.colorbar(
        mappable,
        cax=color_axes,
        orientation="horizontal" if horizontal else "vertical",
    )
    if loc == "north":
        colorbar.ax.xaxis.set_ticks_position("top")
        colorbar.ax.xaxis.set_label_position("top")
    if logarithmic:
        ticks = colorbar.get_ticks()
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"1e{tick:g}" for tick in ticks])
    return HistogramColorbarResult(
        counts=counts,
        bins=centers,
        limits=transformed_limits,
        location=loc,
        logarithmic=bool(logarithmic),
        colorbar=colorbar,
        histogram_axes=histogram_axes,
        histogram_artist=artist,
    )


colorbarHist = colorbar_hist

__all__ = [
    "HistogramColorbarResult",
    "hist_at_centers",
    "histogram_colorbar_data",
    "colorbar_hist",
    "colorbarHist",
]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VALUES = np.geomspace(0.1, 1000.0, 40)

    def test_the_bin_count_is_used_without_the_log_flag():
        """MRST's `nbins = varargin{2}+1` for `nargin == 4`."""
        result = colorbar_hist(VALUES, [0.1, 1000.0], "South", 31)
        assert result.counts.size == 32 and result.counts.sum() == VALUES.size
        plt.close("all")
        print("ok test_the_bin_count_is_used_without_the_log_flag")

    def test_with_the_log_flag_mrst_uses_51_centres():
        """`colorbarHist(K, lim, 'South', 51, true)` histograms on 51 centres
        of log10(lim), whatever the count passed."""
        for flag in (True, False):
            result = colorbar_hist(VALUES, [0.1, 1000.0], "South", 7, flag)
            assert result.counts.size == 51, result.counts.size
        np.testing.assert_allclose(result.limits, [0.1, 1000.0])
        result = colorbar_hist(VALUES, [0.1, 1000.0], "South", 7, True)
        np.testing.assert_allclose(result.limits, [-1.0, 3.0])
        plt.close("all")
        print("ok test_with_the_log_flag_mrst_uses_51_centres")

    def test_hist_counts_by_nearest_centre():
        np.testing.assert_array_equal(hist_at_centers([0.1, 0.9, 1.6, 2.0], [0, 1, 2]), [1, 1, 2])
        print("ok test_hist_counts_by_nearest_centre")

    test_the_bin_count_is_used_without_the_log_flag()
    test_with_the_log_flag_mrst_uses_51_centres()
    test_hist_counts_by_nearest_centre()
    print("\nAll tests passed!")
