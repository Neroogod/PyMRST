"""MRST ``ad-core/simulators/sim_runner/plotPackedProblem.m``.

Plot a packed problem's stored output. The states go to `plotToolbar`, with
the first control's wells drawn when the grid has nodes. The well solutions
go to `plotWellSols` against report time. For several problems, their well
solutions share one `plotWellSols` and each gets its own state viewer.
Returns `(well-solution figure, state viewer)`, with `None` for a part not
plotted (MRST returns `nan`).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

WELL_ARG = dict(color=(1.0, 0.2, 0.2), color2=(0.2, 0.2, 1.0), fontsize=12)


def plot_packed_problem(problem: Any, *, readFromDisk: bool = False,
                        plotStates: bool = True, view: Optional[Sequence[float]] = None,
                        name: Optional[str] = None, wellArg: Optional[dict] = None,
                        gridArg: Optional[dict] = None, wsArg: Optional[dict] = None,
                        time: Optional[np.ndarray] = None, plotWell: Optional[bool] = None,
                        plotWellSols: bool = True):
    """MRST `plotPackedProblem`."""
    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    first = problem[0] if isinstance(problem, (list, tuple)) else problem
    opt = dict(readFromDisk=readFromDisk, plotStates=plotStates, view=view,
               name=first.BaseName if name is None else name,
               wellArg=dict(WELL_ARG) if wellArg is None else wellArg,
               gridArg={} if gridArg is None else gridArg,
               wsArg={} if wsArg is None else wsArg, time=time,
               plotWell=(getattr(first.SimulatorSetup.model.G, "nodes", None) is not None
                         if plotWell is None else plotWell))
    if isinstance(problem, (list, tuple)):
        wh = plot_well_sols(list(problem), **opt["wsArg"]) if plotWellSols else None
        gh = None
        if opt["plotStates"]:
            gh = [plot_packed_problem(p, readFromDisk=readFromDisk, plotStates=True,
                                      view=view, wellArg=wellArg, gridArg=gridArg,
                                      wsArg=wsArg, time=time, plotWell=plotWell,
                                      plotWellSols=False,
                                      name=f"{p.BaseName} ({p.Name})")[1]
                  for p in problem]
        return wh, gh
    if opt["time"] is None:
        opt["time"] = np.cumsum(problem.SimulatorSetup.schedule.step_val)
    gh = _plot_reservoir(problem, opt) if opt["plotStates"] else None
    wh = None
    if plotWellSols:
        handler = problem.OutputHandlers.wellSols
        wsd = [handler[i] for i in handler.get_valid_ids()]
        wh, _ = plot_well_sols(wsd, opt["time"][:len(wsd)], **opt["wsArg"])
        if wh.canvas.manager is not None:
            wh.canvas.manager.set_window_title(opt["name"])
    return wh, gh


def _plot_reservoir(problem: Any, opt: dict):
    import pyvista as pv

    from pymrst.plotting.extensions.plot_toolbar import plot_toolbar
    from pymrst.plotting.plot_well import plot_well
    setup = problem.SimulatorSetup
    G = setup.model.G
    handler = problem.OutputHandlers.states
    states = [handler[i] for i in handler.get_valid_ids()]
    h = plot_toolbar(G, states, show=False, times=opt["time"][:len(states)],
                     title=opt["name"], off_screen=pv.OFF_SCREEN, **opt["gridArg"])
    view = opt["view"]
    if view is None:
        view = (45, 45) if G.griddim == 3 else (0, 90)
    h.pl.camera.azimuth, h.pl.camera.elevation = view
    ctrl = setup.schedule.controls[0]
    if getattr(ctrl, "W", None) and opt["plotWell"]:
        plot_well(G, ctrl.W, h.pl, **opt["wellArg"])
    if not h.off_screen:
        h.show()
    return h


plotPackedProblem = plot_packed_problem

__all__ = ["plotPackedProblem", "plot_packed_problem"]
