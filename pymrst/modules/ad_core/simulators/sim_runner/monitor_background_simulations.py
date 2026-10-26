"""MRST ``ad-core/simulators/sim_runner/monitorBackgroundSimulations.m``.

Show the progress of packed problems simulating elsewhere, read from their
state output handlers, until none is still running (or once, with
`singleUpdate`). Each problem gets a bar for the number of states stored out
of the number of schedule steps, under a "Total progress" line. In text mode
the previous display is erased with backspaces (`dynamicText`) and redrawn;
with `useFigure` the bars are `simpleUIbar`s in one figure. Returns the handle
`h` (`figure`, `text`, `iteration`) to pass back as `handle`.

MRST defaults `useFigure` to whether MATLAB runs with its desktop; a Python
session has none, so the default is text. MRST's quirks are kept: the total
line counts `totalProgress` plus problems marked done in the same sweep, and
each problem's `done` flag is set from the counts of the line before it, so the
total shows `totalProgress`.

`monitor_background_simulations_ad` (in `background.py`) is PyMRST's earlier
version, which reads process status, kept for its callers.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

BAR_WIDTH = 50
SPINNER = "|/-\\"


def monitor_background_simulations(problems: Any | Sequence[Any], *, pause: float = 0.25,
                                   useFigure: Optional[bool] = None,
                                   indices: Optional[Sequence[int]] = None,
                                   dynamicText: bool = True,
                                   totalNumberOfCases: Optional[int] = None,
                                   totalProgress: int = 0, handle: Any = None,
                                   singleUpdate: bool = False) -> SimpleNamespace:
    """MRST `monitorBackgroundSimulations`."""
    count = len(problems) if isinstance(problems, (list, tuple)) else 1
    indices = list(range(1, count + 1)) if indices is None else list(indices)
    totalNumberOfCases = count if totalNumberOfCases is None else totalNumberOfCases
    h = SimpleNamespace(figure=None, text="", iteration=0) if handle is None else handle
    if not isinstance(problems, (list, tuple)):
        problems = [problems]
    n_problems = len(problems)
    simulating = True
    fn = "Simulating: " + ",".join(sorted({p.BaseName for p in problems}))
    use_figure = bool(useFigure)
    if use_figure:
        import matplotlib.pyplot as plt
        if h.figure is None:
            h.figure = plt.figure(fn)
        use_figure = plt.fignum_exists(h.figure.number)
    dispstr = ""
    if not use_figure:
        print()
        if h.text:
            dispstr = h.text
    height = 0.9/(n_problems + 1)
    descriptions = [_get_description(p) for p in problems]
    while simulating:
        h.iteration += 1
        n_prev = len(dispstr)
        dispstr = ""
        if use_figure:
            h.figure.clf()
        active = np.zeros(n_problems, dtype=bool)
        done = np.zeros(n_problems, dtype=bool)
        for i in range(n_problems + 1):
            if i == 0:
                num = totalProgress + int(done.sum())
                n = totalNumberOfCases
                descr = f"Total progress: {num} of {n} cases simulated."
                case_str = "Total"
                color = (0.3, 0.3, 1)
                dispstr = descr
            else:
                done[i - 1] = num >= n
                n, num, active[i - 1] = _get_status(problems[i - 1])
                descr = descriptions[i - 1]
                case_str = f"{indices[i - 1]:3d})"
                color = (1, 0.4, 0.4)
            if use_figure:
                from pymrst.modules.ad_core.plotting.simple_ui_bar import simple_ui_bar
                barpos = n_problems*height if i == 0 else (i - 1)*height
                simple_ui_bar(h.figure, num/n, barpos, height, descr, FaceColor=color)
                continue
            nextstr = _progress_line(case_str, descr, num, n, h.iteration)
            if i > 0:
                dispstr = f"{dispstr}\n{nextstr}"
        simulating = bool(active.any())
        if not use_figure:
            remstr = "\b"*(n_prev + 1) if dynamicText else ""
            h.text = dispstr
            print(remstr + dispstr)
        if singleUpdate:
            break
        if use_figure:
            import matplotlib.pyplot as plt
            plt.pause(pause)
        else:
            time.sleep(pause)
    return h


def _progress_line(case_str: str, descr: str, num: int, n: int, iteration: int) -> str:
    perc = min(num/n, 1) if n else 1.0
    if n == num:
        mystr = "Complete!"
    elif num > n:
        mystr = "Complete (??)"
    elif num > 0:
        mystr = f"{100*perc:1.2f}% {_get_spinner(iteration)}"
    else:
        mystr = "Not started."
    pbar = "#"*math.ceil(BAR_WIDTH*perc) + " "*math.floor(BAR_WIDTH*(1 - perc))
    return (f"{case_str} {descr}\n    -> {num:3d} of {n:3d} steps simulated "
            f"[{pbar}] {mystr}")


def _get_status(p: Any) -> tuple[int, int, bool]:
    n = int(np.asarray(p.SimulatorSetup.schedule.step_val).size)
    num = p.OutputHandlers.states.numel_data()
    return n, num, num < n


def _get_description(p: Any) -> str:
    return p.Name if not p.Description else f"{p.Name}: {p.Description}"


def _get_spinner(it: int) -> str:
    return f"[{SPINNER[it % 4]}]"


monitorBackgroundSimulations = monitor_background_simulations

__all__ = ["monitorBackgroundSimulations", "monitor_background_simulations"]


if __name__ == "__main__":
    import contextlib
    import io

    def _problem(name: str, stored: int, steps: int = 4, description: str = ""):
        states = SimpleNamespace(numel_data=lambda: stored)
        return SimpleNamespace(
            BaseName="base", Name=name, Description=description,
            SimulatorSetup=SimpleNamespace(schedule=SimpleNamespace(step_val=np.ones(steps))),
            OutputHandlers=SimpleNamespace(states=states))

    def _run(problems, **kwargs) -> tuple[SimpleNamespace, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            h = monitor_background_simulations(problems, singleUpdate=True, **kwargs)
        return h, out.getvalue()

    def test_text_lines_match_mrst_format():
        h, text = _run([_problem("a", 2, description="first"), _problem("b", 0)])
        lines = h.text.split("\n")
        assert lines[0] == "Total progress: 0 of 2 cases simulated."
        assert lines[1] == "  1) a: first"
        assert lines[2] == ("    -> " + "  2 of   4 steps simulated ["
                            + "#"*25 + " "*25 + "] 50.00% [/]")
        assert lines[3] == "  2) b"
        assert lines[4].endswith("[" + " "*50 + "] Not started.")
        assert h.iteration == 1 and text.startswith("\n\b")
        print("ok test_text_lines_match_mrst_format")

    def test_complete_and_overcomplete():
        h, _ = _run([_problem("a", 4), _problem("b", 5)])
        lines = h.text.split("\n")
        assert lines[2].endswith("#"*50 + "] Complete!")
        assert lines[4].endswith("] Complete (??)")
        print("ok test_complete_and_overcomplete")

    def test_handle_carries_text_and_iteration():
        problems = [_problem("a", 1)]
        h, _ = _run(problems)
        h, text = _run(problems, handle=h, indices=[7])
        assert h.iteration == 2 and "  7) a" in h.text
        assert text.startswith("\n" + "\b"*(len(h.text) + 1))
        print("ok test_handle_carries_text_and_iteration")

    def test_loop_ends_when_nothing_is_active():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            h = monitor_background_simulations(_problem("a", 4), pause=0.0)
        assert h.iteration == 1
        print("ok test_loop_ends_when_nothing_is_active")

    def test_figure_mode_draws_one_bar_per_line():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = monitor_background_simulations([_problem("a", 1), _problem("b", 2)],
                                           useFigure=True, singleUpdate=True)
        assert len(h.figure.axes) == 3
        plt.close(h.figure)
        print("ok test_figure_mode_draws_one_bar_per_line")

    test_text_lines_match_mrst_format()
    test_complete_and_overcomplete()
    test_handle_carries_text_and_iteration()
    test_loop_ends_when_nothing_is_active()
    test_figure_mode_draws_one_bar_per_line()
    print("\nAll tests passed!")
