"""MRST ``ad-core/simulators/sim_runner/simulatePackedProblemBackground.m``.

Store a packed problem in `<workdir>/<BaseName>_<Name>` and simulate it in a
separate, detached session that runs `simulatePackedProblemStandalone` there,
its output going to `simulation.log`. Returns at once with `info.path`; the
output handlers are how progress is read (`monitorBackgroundSimulations`).

MRST starts a new MATLAB (`matlab <options> -sd ROOTDIR -logfile ... -r ...`)
after saving the problem with its module list and options, and the MRST module
paths to `pathmap.mat` for the new session to re-register. PyMRST starts a new
Python interpreter the same way: in the PyMRST root, with the problem in
`problem.pkl`, and `sys.path` in `pathmap.pkl` for the new interpreter to
restore. `linux_arg`/`win_arg` and `extra_arg` are interpreter options as in
MRST; `python_arg` takes the place of `matlab_arg`, and defaults to `-u` so the
log is written as the run goes, as MATLAB's is.

MATLAB's `save` stores function handles; Python's `pickle` cannot store the
closures in a model's operators and fluid, so the problem is written with
`cloudpickle` when it is installed (the new interpreter needs it too;
`pip install pymrst[background]`).

`simulate_packed_problem_background_ad` (in `background.py`) is PyMRST's
earlier version for the legacy models, kept for its callers.
"""

from __future__ import annotations

import math
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pymrst
from pymrst.utils.settings_manager.mrst_settings import mrst_settings

CHILD_SCRIPT = (
    "import pickle, sys\n"
    "with open({pathmap!r}, 'rb') as stream:\n"
    "    sys.path[:] = pickle.load(stream)\n"
    "from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem_standalone "
    "import simulate_packed_problem_standalone\n"
    "simulate_packed_problem_standalone({path!r})\n"
)


def simulate_packed_problem_background(problem: Any, *, workdir: str | Path | None = None,
                                       linux_arg: str = "", win_arg: str = "",
                                       python_arg: str = "-u", extra_arg: str = "",
                                       verbose: bool = False,
                                       maxNumCompThreads: float = math.inf) -> SimpleNamespace:
    """MRST `simulatePackedProblemBackground`."""
    if workdir is None:
        workdir = Path(mrst_settings("get", "outputDirectory"))/"sim_runner"
    opt = dict(workdir=str(workdir), linux_arg=linux_arg, win_arg=win_arg,
               python_arg=python_arg, extra_arg=extra_arg, verbose=verbose,
               maxNumCompThreads=maxNumCompThreads)
    basepath = Path(workdir)/f"{problem.BaseName}_{problem.Name}"
    if verbose:
        print(f"Storing problem {problem.BaseName}: {problem.Name} to {basepath}...", end="")
    basepath.mkdir(parents=True, exist_ok=True)
    logfile = basepath/"simulation.log"
    pathmap_file = basepath/"pathmap.pkl"
    with pathmap_file.open("wb") as stream:
        pickle.dump(list(sys.path), stream)
    with (basepath/"problem.pkl").open("wb") as stream:
        _dump({"problem": problem, "modlist": list(problem.Modules), "opt": opt}, stream)
    if verbose:
        print("Ok!")
        print("Initializing background Python session...", end="")
    script = CHILD_SCRIPT.format(pathmap=str(pathmap_file), path=str(basepath))
    command = [sys.executable, *shlex.split(build_python_options(opt)), "-c", script]
    with logfile.open("w") as log:
        subprocess.Popen(command, cwd=_rootdir(), stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    if verbose:
        print(" Ok! Simulation launched.")
    return SimpleNamespace(path=basepath)


def build_python_options(opt: dict) -> str:
    """MRST's `build_matlab_options`: platform options, then the rest."""
    base = opt["win_arg"] if os.name == "nt" else opt["linux_arg"]
    return f"{base} {opt['python_arg']} {opt['extra_arg']}"


def _rootdir() -> Path:
    """MRST's `ROOTDIR`: the PyMRST root, above the package."""
    return Path(pymrst.__file__).resolve().parents[1]


def _dump(payload: dict, stream: Any) -> None:
    try:
        import cloudpickle
    except ImportError:
        try:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, AttributeError) as error:
            raise RuntimeError("the problem holds functions pickle cannot store; install "
                               "cloudpickle (pip install pymrst[background]) to run it "
                               "in the background") from error
    else:
        cloudpickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


simulatePackedProblemBackground = simulate_packed_problem_background

__all__ = ["simulatePackedProblemBackground", "simulate_packed_problem_background"]


if __name__ == "__main__":
    import tempfile
    from unittest import mock

    def _problem() -> SimpleNamespace:
        return SimpleNamespace(BaseName="base", Name="case", Modules=["ad-core"])

    def test_stores_problem_and_launches_the_standalone_run():
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("subprocess.Popen") as popen:
                info = simulate_packed_problem_background(_problem(), workdir=tmp,
                                                          maxNumCompThreads=2)
            assert info.path == Path(tmp)/"base_case"
            with (info.path/"problem.pkl").open("rb") as stream:
                stored = pickle.load(stream)
            assert stored["problem"].Name == "case"
            assert stored["modlist"] == ["ad-core"]
            assert stored["opt"]["maxNumCompThreads"] == 2
            command = popen.call_args.args[0]
            assert command[0] == sys.executable and command[1] == "-u"
            assert "simulate_packed_problem_standalone(" in command[-1]
            assert popen.call_args.kwargs["cwd"] == _rootdir()
            assert (info.path/"pathmap.pkl").exists()
        print("ok test_stores_problem_and_launches_the_standalone_run")

    def test_options_follow_the_platform_arguments():
        opt = dict(linux_arg="-X dev", win_arg="-E", python_arg="-u", extra_arg="-O")
        expected = "-E -u -O" if os.name == "nt" else "-X dev -u -O"
        assert build_python_options(opt) == expected
        print("ok test_options_follow_the_platform_arguments")

    def test_rootdir_holds_the_package():
        assert (_rootdir()/"pymrst"/"__init__.py").exists()
        print("ok test_rootdir_holds_the_package")

    test_stores_problem_and_launches_the_standalone_run()
    test_options_follow_the_platform_arguments()
    test_rootdir_holds_the_package()
    print("\nAll tests passed!")
