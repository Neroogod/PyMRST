"""MRST ``ad-blackoil/examples/utils/surfWithOutline.m``: `surf(x, y, z)` in
the current axes, outlined in black along its first and last columns and
rows.

Matplotlib draws a surface only in 3-D axes, so a current 2-D axes is
replaced by a 3-D one in the same place, as MATLAB's `surf` turns the
current axes 3-D. Matplotlib always adds to an axes, so MATLAB's
`hold on`/`hold off` bookkeeping has nothing to do.

`surf_with_outline_ad` is the earlier version, which returned the arrays
for the caller to plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _current_3d_axes() -> Any:
    import matplotlib.pyplot as plt
    ax = plt.gca()
    if getattr(ax, "name", "") == "3d":
        return ax
    fig, position = ax.figure, ax.get_subplotspec()
    ax.remove()
    return fig.add_subplot(position, projection="3d")


def surf_with_outline(x: Any, y: Any, z: Any) -> None:
    """The surface `(x, y, z)` with a black outline, in the current axes."""
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    ax = _current_3d_axes()
    ax.plot_surface(x, y, z)
    for j in (0, -1):
        ax.plot(x[:, j], y[:, j], z[:, j], "-k", linewidth=2)
    for i in (0, -1):
        ax.plot(x[i, :], y[i, :], z[i, :], "-k", linewidth=2)


def surf_with_outline_ad(x, y, z):
    """Return surface arrays and outline polylines for plotting elsewhere."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z must have identical shapes")
    outline = {
        "bottom": (x[0, :], y[0, :], z[0, :]),
        "top": (x[-1, :], y[-1, :], z[-1, :]),
        "left": (x[:, 0], y[:, 0], z[:, 0]),
        "right": (x[:, -1], y[:, -1], z[:, -1]),
    }
    return {"surface": (x, y, z), "outline": outline}


surfWithOutline = surf_with_outline
__all__ = ["surf_with_outline", "surfWithOutline", "surf_with_outline_ad"]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(np.arange(1.0, 5.0), np.arange(1.0, 4.0))
    Z = X*Y

    def test_surface_and_four_black_outlines():
        plt.figure()
        surf_with_outline(X, Y, Z)
        ax = plt.gca()
        assert ax.name == "3d" and len(ax.collections) == 1
        lines = ax.get_lines()
        assert len(lines) == 4 and all(l.get_color() == "k" for l in lines)
        xs, ys, zs = lines[1].get_data_3d()
        np.testing.assert_array_equal(zs, Z[:, -1])
        xs, ys, zs = lines[2].get_data_3d()
        np.testing.assert_array_equal(xs, X[0, :])
        plt.close("all")
        print("ok test_surface_and_four_black_outlines")

    def test_an_existing_3d_axes_is_drawn_into():
        ax = plt.figure().add_subplot(projection="3d")
        surf_with_outline(X, Y, Z)
        surf_with_outline(X, Y, 2*Z)
        assert plt.gca() is ax and len(ax.collections) == 2 and len(ax.get_lines()) == 8
        plt.close("all")
        print("ok test_an_existing_3d_axes_is_drawn_into")

    def test_legacy_version_returns_the_outline():
        out = surf_with_outline_ad(X, Y, Z)
        np.testing.assert_array_equal(out["outline"]["right"][2], Z[:, -1])
        print("ok test_legacy_version_returns_the_outline")

    test_surface_and_four_black_outlines()
    test_an_existing_3d_axes_is_drawn_into()
    test_legacy_version_returns_the_outline()
    print("\nAll tests passed!")
