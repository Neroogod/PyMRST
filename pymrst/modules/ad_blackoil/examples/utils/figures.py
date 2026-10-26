"""MATLAB-style 3-D figures for the ad-blackoil example ports.

MRST's examples draw cell data with `plotCellData`, set the camera with
`view(az, el)`, and animate a simulation by redrawing each state with a
short `pause`. These helpers do the same with PyVista:
- `view_vector` is MATLAB's `view([az, el])` as a camera direction;
- `cell_data_figure` is `figure; plotCellData(G, data, cells); view(v);
  title(...); colorbar` with an optional `caxis`;
- `animate_cell_data` redraws a sequence of cell data, one frame per state,
  with MATLAB's `drawnow, pause(t)`.

Under `pyvista.OFF_SCREEN` nothing opens and nothing waits.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np


def view_vector(view: Sequence[float]) -> Tuple[float, float, float]:
    """MATLAB `view([az, el])` as the direction from the target to the camera."""
    az, el = np.deg2rad(np.asarray(view, dtype=float).ravel()[:2])
    return (float(np.sin(az)*np.cos(el)), float(-np.cos(az)*np.cos(el)), float(np.sin(el)))


def set_view(plotter: Any, view: Optional[Sequence[float]]) -> None:
    if view is not None:
        plotter.view_vector(view_vector(view), viewup=(0, 0, 1))
        plotter.reset_camera()


def cell_data_figure(G: Any, data: Any, title: str = "", *, cells: Any = None,
                     view: Optional[Sequence[float]] = None, clim: Any = None,
                     plotter: Any = None, show: bool = True, **options: Any) -> Any:
    """`plotCellData(G, data, cells)` in a figure of its own, with `view`,
    `title` and `caxis(clim)`. Returns the plotter."""
    import pyvista as pv
    from pymrst.plotting.plot_cell_data import plot_cell_data
    plotter = pv.Plotter() if plotter is None else plotter
    plot_cell_data(G, data, cells, plotter=plotter, clim=clim, show=False,
                   label=title, **options)
    if title:
        plotter.add_title(title, font_size=10)
    set_view(plotter, view)
    if show:
        plotter.show()
    return plotter


def animate_cell_data(G: Any, frames: Sequence[Any], titles: Sequence[str], *,
                      view: Optional[Sequence[float]] = None, clim: Any = None,
                      pause: float = 0.2, initial: Any = None,
                      initial_title: str = "") -> Any:
    """Redraw `frames` one after the other (`drawnow, pause(pause)`). With
    `initial`, it stays on the left, as MRST's `subplot(1,2,1)`."""
    import pyvista as pv
    from pymrst.plotting.plot_cell_data import plot_cell_data
    plotter = pv.Plotter(shape=(1, 2) if initial is not None else (1, 1))
    if initial is not None:
        plotter.subplot(0, 0)
        plot_cell_data(G, initial, plotter=plotter, clim=clim, show=False)
        plotter.add_title(initial_title, font_size=10)
        set_view(plotter, view)
        plotter.subplot(0, 1)
    interactive = not pv.OFF_SCREEN
    if interactive:
        plotter.show(interactive_update=True, auto_close=False)
    for data, title in zip(frames, titles):
        plotter.clear_actors()
        plot_cell_data(G, data, plotter=plotter, clim=clim, show=False)
        plotter.add_title(title, font_size=10)
        set_view(plotter, view)
        if interactive:
            plotter.update()
            time.sleep(pause)
    if interactive:
        plotter.show()
    return plotter


def animate_panels(G: Any, panels: Sequence[Sequence[Any]], titles: Sequence[str], *,
                   view: Optional[Sequence[float]] = None, clim: Any = None,
                   cmap: str = "viridis", pause: float = 0.1, **options: Any) -> Any:
    """Several `plotCellData` panels side by side whose data are replaced
    together, frame by frame (MRST's `set(hp, 'CData', ...); drawnow,
    pause(t)`). `panels[k][i]` is panel k's data in frame i."""
    import pyvista as pv
    from pymrst.plotting.plot_cell_data import plot_cell_data
    plotter = pv.Plotter(shape=(1, len(panels)))
    interactive = not pv.OFF_SCREEN
    if interactive:
        plotter.show(interactive_update=True, auto_close=False)
    for frame in range(len(panels[0])):
        for k, panel in enumerate(panels):
            plotter.subplot(0, k)
            plotter.clear_actors()
            plot_cell_data(G, panel[frame], plotter=plotter, clim=clim, cmap=cmap,
                           show=False, **options)
            plotter.add_title(titles[k], font_size=10)
            set_view(plotter, view)
        if interactive:
            plotter.update()
            time.sleep(pause)
    if interactive:
        plotter.show()
    return plotter


def matlab_colormap(name: str, n: int) -> Any:
    """MATLAB's `name(n)`, e.g. `jet(32)`, as an n-colour matplotlib map."""
    from matplotlib import colormaps
    return colormaps[name].resampled(n)


__all__ = ["animate_cell_data", "animate_panels", "cell_data_figure", "matlab_colormap",
           "set_view", "view_vector"]


if __name__ == "__main__":
    import pyvista as pv

    from pymrst.gridprocessing import cart_grid, compute_geometry

    pv.OFF_SCREEN = True
    G = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))

    def test_the_view_is_matlabs():
        """`view(0, 90)` looks straight down, `view(0, 0)` along +y and
        `view(90, 0)` along -x."""
        np.testing.assert_allclose(view_vector((0, 90)), (0, 0, 1), atol=1e-15)
        np.testing.assert_allclose(view_vector((0, 0)), (0, -1, 0), atol=1e-15)
        np.testing.assert_allclose(view_vector((90, 0)), (1, 0, 0), atol=1e-15)
        print("ok test_the_view_is_matlabs")

    def test_figures_render_off_screen():
        cell_data_figure(G, np.arange(G.cells.num), "cells", view=(50, 20), clim=[0, 11])
        animate_cell_data(G, [np.zeros(12), np.ones(12)], ["t=1", "t=2"],
                          view=(50, 20), clim=[0, 1], initial=np.ones(12),
                          initial_title="T=0")
        animate_panels(G, [[np.zeros(12), np.ones(12)], [np.ones(12), np.zeros(12)]],
                       ["open", "closed"], clim=[0, 1], cmap=matlab_colormap("jet", 32))
        print("ok test_figures_render_off_screen")

    test_the_view_is_matlabs()
    test_figures_render_off_screen()
    print("\nAll tests passed!")
