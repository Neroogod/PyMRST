"""MRST ``ad-core/plotting/simpleUIbar.m``.

A progress bar in a figure: an axes in the upper half of the band
`[start, start + height]` with a filled bar for `data` (a fraction), an outline
and the percentage, and the text `txt` below it. Extra keyword arguments are
MRST's `rectangle` properties for the filled bar (`FaceColor="r"` by default),
passed to matplotlib's `Rectangle` with lower-case names.
"""

from __future__ import annotations

from typing import Any


def simple_ui_bar(parent: Any, data: float, start: float, height: float, txt: str,
                  **varargin: Any) -> None:
    """MRST `simpleUIbar`."""
    h = height/2
    ax = parent.add_axes((0.1, start + h, 0.8, h))
    _plot_progress(ax, data, **varargin)
    parent.text(0.5, start + h*0.45, txt, ha="center", va="center")


def _plot_progress(ax: Any, data: float, **varargin: Any) -> None:
    from matplotlib.patches import Rectangle
    props = {"facecolor": "r", **{key.lower(): value for key, value in varargin.items()}}
    ax.add_patch(Rectangle((0, 0.01), data, 0.99, **props))
    ax.add_patch(Rectangle((0, 0.01), 1, 0.99, facecolor="none", edgecolor="k"))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{100*data:2.1f}%")


simpleUIbar = simple_ui_bar

__all__ = ["simpleUIbar", "simple_ui_bar"]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def test_bar_shows_fraction_colour_and_caption():
        fig = plt.figure()
        simple_ui_bar(fig, 0.25, 0.1, 0.2, "case 1", FaceColor=(0.3, 0.3, 1))
        ax = fig.axes[0]
        bar, outline = ax.patches
        assert bar.get_width() == 0.25 and outline.get_width() == 1
        assert tuple(bar.get_facecolor()[:3]) == (0.3, 0.3, 1)
        assert ax.texts[0].get_text() == "25.0%"
        assert fig.texts[0].get_text() == "case 1"
        assert ax.get_position().bounds[1] == 0.2
        plt.close(fig)
        print("ok test_bar_shows_fraction_colour_and_caption")

    def test_default_colour_is_red():
        fig = plt.figure()
        simple_ui_bar(fig, 1.0, 0.0, 0.5, "")
        assert tuple(fig.axes[0].patches[0].get_facecolor()) == (1.0, 0.0, 0.0, 1.0)
        plt.close(fig)
        print("ok test_default_colour_is_red")

    test_bar_shows_fraction_colour_and_caption()
    test_default_colour_is_red()
    print("\nAll tests passed!")
