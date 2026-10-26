"""MRST ``ad-core/plotting/simulationRuntimePanel.m``.

A progress panel for a running simulation, redrawn after every control
step: the model class, a bar for control steps solved and one for simulated
time, the elapsed and estimated remaining wall time, grid statistics, the
iterations per control step and their total. Its buttons:
- **Abort** makes the next call return `ok = False`, which stops the run;
- **Pause** holds the simulation until pressed again;
- **dtMax*2**/**dtMax/2** change the solver's `timeStepSelector.maxTimestep`
  (an infinite one becomes 30 days);
- **Dump to workspace** stores `states`, `simtime` and `reports` as
  `simdata` in `__main__` (MATLAB's base workspace);
- **Debug** stops in the debugger (`breakpoint()`, for MATLAB's `dbstop`);
- **Play sound when done** rings the terminal bell at the end (MATLAB plays
  `gong.mat`).

The panel is a matplotlib figure with the label 'mrst-simpanel'; the button
states persist between calls, as MATLAB's `persistent` variables do, and
are reset when the panel is closed.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.plotting.simple_ui_bar import simple_ui_bar
from pymrst.utils.format_time_range import format_time_range
from pymrst.utils.units import DAY

PANEL_TAG = "mrst-simpanel"

_FLAGS: Dict[str, bool] = {"abort": False, "debug": False, "sound": False,
                           "dump": False, "pause": False}
_WIDGETS: list = []


def simulation_runtime_panel(model: Any, states: Sequence[Any], ctrl_reports: Sequence[Any],
                             solver: Any, schedule: Any, simtime: Sequence[float], *,
                             figure: Any = None) -> bool:
    """MRST `ok = simulationRuntimePanel(model, states, reports, solver,
    schedule, simtime)`."""
    import matplotlib.pyplot as plt
    plt.pause(0.01)
    ctrl_reports = [r for r in ctrl_reports if r is not None]
    step_no = len(ctrl_reports)
    if not plt.fignum_exists(PANEL_TAG):
        for key in _FLAGS:
            _FLAGS[key] = False
    ok = True
    fig = _get_panel()
    fig.text(0.5, 0.95, f"Simulating {type(model).__name__} ", fontsize=18, ha="center")
    step_val = np.asarray(schedule.step_val, dtype=float)
    step_total = step_val.size
    completed = step_no == step_total
    if completed:
        text = f"Simulation done: {step_total} control steps solved"
    else:
        text = f"Solving control step number {step_no + 1} of {step_total}"
    simple_ui_bar(fig, step_no/step_total, 0.75, 0.15, text)
    t, T = float(step_val[:step_no].sum()), float(step_val.sum())
    tm = lambda x: format_time_range(x).lower()
    simple_ui_bar(fig, t/T, 0.60, 0.15, f"Simulated {tm(t)} of {tm(T)} in schedule")
    sim_t = max(float(np.sum(simtime)), 1.0)
    end_t = max((step_total - step_no)*sim_t/max(step_no, 1), 1.0)
    text = f"Simulation has been running for {tm(sim_t)} "
    text += "and is done" if completed else f"and will be completed in about {tm(end_t)}"
    fig.text(0.5, 0.53, text + "\n" + _get_stats(model), ha="center", va="center")
    ax = fig.add_axes((0.1, 0.25, 0.8, 0.22))
    its = np.array([r["Iterations"] for r in ctrl_reports], dtype=float)
    tmp = np.full(step_total, np.nan)
    tmp[:step_no] = its
    ax.step(np.arange(1, step_total + 1), tmp, where="post")
    ax.set_ylim(1, max(np.nanmax(tmp) + 0.1 if step_no else 1.01, 1.01))
    ax.set_xlim(1, max(step_total, 1 + 1e-9))
    ax.grid(True)
    fig.text(0.5, 0.15, f"Total number of iterations {int(its.sum())} with an average of "
             f"{its.sum()/max(step_no, 1):1.2f} iterations per control step", ha="center")
    _add_buttons(fig, solver)
    if _FLAGS["sound"] and completed:
        sys.stdout.write("\a")
        sys.stdout.flush()
    if _FLAGS["abort"]:
        _FLAGS["abort"] = False
        return False
    if _FLAGS["dump"]:
        setattr(sys.modules["__main__"], "simdata",
                {"states": states, "simtime": simtime, "reports": ctrl_reports})
        _FLAGS["dump"] = False
    while _FLAGS["pause"] and plt.fignum_exists(PANEL_TAG):
        plt.pause(0.1)
    if _FLAGS["debug"]:
        _FLAGS["debug"] = False
        breakpoint()
    fig.canvas.draw_idle()
    return ok


def _add_buttons(fig: Any, solver: Any) -> None:
    from matplotlib.widgets import Button, CheckButtons

    def flag(key: str, value: bool = True):
        def handler(_event):
            _FLAGS[key] = value
        return handler

    def timestep(factor: float):
        def handler(_event):
            selector = solver.timeStepSelector
            dt0 = selector.maxTimestep
            dt_new = 30*DAY if np.isinf(dt0) else dt0*factor
            print(f"**** Adjusted timestep interactively (dt: {format_time_range(dt0)} "
                  f"-> {format_time_range(dt_new)})")
            selector.maxTimestep = dt_new
        return handler

    specs = [("Abort", (0.05, 0.005, 0.3, 0.04), flag("abort")),
             ("Debug", (0.05, 0.055, 0.3, 0.04), flag("debug")),
             ("Dump to workspace", (0.35, 0.055, 0.3, 0.04), flag("dump")),
             ("dtMax*2", (0.65, 0.055, 0.15, 0.04), timestep(2)),
             ("dtMax/2", (0.80, 0.055, 0.15, 0.04), timestep(1/2))]
    for label, rect, handler in specs:
        button = Button(fig.add_axes(rect), label)
        button.on_clicked(handler)
        _WIDGETS.append(button)
    toggles = CheckButtons(fig.add_axes((0.35, 0.005, 0.6, 0.04)),
                           ["Play sound when done", "Pause"],
                           [_FLAGS["sound"], _FLAGS["pause"]])

    def toggled(label: str) -> None:
        sound, pause = toggles.get_status()
        _FLAGS["sound"], _FLAGS["pause"] = sound, pause

    toggles.on_clicked(toggled)
    _WIDGETS.append(toggles)


def _get_panel() -> Any:
    import matplotlib.pyplot as plt
    fig = plt.figure(PANEL_TAG, figsize=(6.4, 7.2))
    for widget in _WIDGETS:
        widget.disconnect_events()
    _WIDGETS.clear()
    fig.clf()
    return fig


def _get_stats(model: Any) -> str:
    txt = ""
    G = getattr(model, "G", None)
    if G is None:
        return txt
    txt += f"Grid contains {G.cells.num} active cells. "
    rock = getattr(model, "rock", None)
    if rock is not None:
        perm = np.asarray(rock.perm)
        nr = 1 if perm.ndim == 1 else perm.shape[1]
        dim = G.griddim
        if nr == 1:
            kind = "scalar"
        elif (nr == 2 and dim == 2) or (nr == 3 and dim == 3):
            kind = "diagonal tensor"
        elif (nr == 3 and dim == 2) or (nr == 6 and dim == 3):
            kind = "full tensor"
        else:
            kind = "unknown"
        txt += f"Model uses {kind} permeability. "
    return txt


simulationRuntimePanel = simulation_runtime_panel

__all__ = ["simulationRuntimePanel", "simulation_runtime_panel"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _case():
        G = SimpleNamespace(cells=SimpleNamespace(num=12), griddim=3)
        model = SimpleNamespace(G=G, rock=SimpleNamespace(perm=np.ones((12, 3))))
        schedule = SimpleNamespace(step_val=np.full(4, DAY))
        solver = SimpleNamespace(timeStepSelector=SimpleNamespace(maxTimestep=np.inf))
        return model, schedule, solver

    def test_the_panel_lets_the_run_continue():
        model, schedule, solver = _case()
        ok = simulation_runtime_panel(model, [{}], [{"Iterations": 3}], solver,
                                      schedule, [2.0])
        assert ok is True and plt.fignum_exists(PANEL_TAG)
        print("ok test_the_panel_lets_the_run_continue")

    def test_abort_stops_the_run_once():
        model, schedule, solver = _case()
        simulation_runtime_panel(model, [{}], [{"Iterations": 3}], solver, schedule, [2.0])
        _FLAGS["abort"] = True
        assert simulation_runtime_panel(model, [{}], [{"Iterations": 3}], solver,
                                        schedule, [2.0]) is False
        assert simulation_runtime_panel(model, [{}], [{"Iterations": 3}], solver,
                                        schedule, [2.0]) is True
        print("ok test_abort_stops_the_run_once")

    def test_dtmax_buttons_change_the_selector():
        model, schedule, solver = _case()
        simulation_runtime_panel(model, [{}], [{"Iterations": 3}], solver, schedule, [2.0])
        double, halve = _WIDGETS[3], _WIDGETS[4]
        double._observers.process("clicked", None)
        assert solver.timeStepSelector.maxTimestep == 30*DAY
        halve._observers.process("clicked", None)
        assert solver.timeStepSelector.maxTimestep == 15*DAY
        print("ok test_dtmax_buttons_change_the_selector")

    def test_dump_goes_to_main():
        model, schedule, solver = _case()
        _FLAGS["dump"] = True
        simulation_runtime_panel(model, ["s1"], [{"Iterations": 2}], solver, schedule, [1.0])
        assert sys.modules["__main__"].simdata["states"] == ["s1"]
        print("ok test_dump_goes_to_main")

    def test_stats_name_the_permeability_type():
        model, _, _ = _case()
        assert _get_stats(model) == ("Grid contains 12 active cells. "
                                     "Model uses diagonal tensor permeability. ")
        print("ok test_stats_name_the_permeability_type")

    test_the_panel_lets_the_run_continue()
    test_abort_stops_the_run_once()
    test_dtmax_buttons_change_the_selector()
    test_dump_goes_to_main()
    test_stats_name_the_permeability_type()
    plt.close("all")
    print("\nAll tests passed!")
