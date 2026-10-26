"""MRST ``ad-core/simulators/sim_runner/packSimulationProblem.m``.

A simulation problem as one object that can be simulated, resumed and read
back: the initial state, model, schedule and nonlinear solver, plus one
`ResultHandler` each for states, reports and well solutions. Results go to
`<Directory>/<Name>`, where `Directory` defaults to
`<mrstOutputDirectory>/<BaseName>`. With `useHash` (MRST's default setting),
the name gets a hash of the state, model (without `inputdata`) and
schedule, so a changed problem gets its own results folder. The hash is
PyMRST's `obj2hash`, so it differs from MATLAB's. MRST also records the
loaded modules; PyMRST has no module system, so `Modules` is what the
caller passes.

`pack_simulation_problem_ad` (in `packed_problem.py`) is PyMRST's earlier
version, kept for its callers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from pymrst.modules.ad_core.utils.obj2hash import obj2hash
from pymrst.modules.ad_core.utils.result_handler import ResultHandler
from pymrst.utils.settings_manager.mrst_settings import mrst_settings


def pack_simulation_problem(state0: Any, model: Any, schedule: Any, BaseName: str, *,
                            NonLinearSolver: Any = None, Directory: str = "",
                            Name: str = "", useHash: Optional[bool] = None,
                            Modules: Sequence[str] = (), Description: str = "",
                            ExtraArguments: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """MRST `packSimulationProblem`."""
    if useHash is None:
        useHash = bool(mrst_settings("get", "useHash"))
    no_name = not Name
    if no_name:
        Name = type(model).__name__
    name = Name
    if useHash:
        digest = " ".join([obj2hash(state0), obj2hash(model, skip=["inputdata"]),
                           obj2hash(schedule)])
        Name = f"{Name}_{obj2hash(digest)}"
    if not Description:
        Description = name if no_name else f"{name}_{type(model).__name__}"
        if NonLinearSolver is not None:
            ident = NonLinearSolver.get_id()
            if ident:
                Description = f"{Description}_{ident}"
    if not Directory:
        Directory = str(Path(mrst_settings("get", "outputDirectory"))/BaseName)
    extra = dict(ExtraArguments or {})
    output_ministeps = bool(extra.pop("OutputMinisteps", False))
    setup = SimpleNamespace(state0=state0, model=model, schedule=schedule,
                            OutputMinisteps=output_ministeps,
                            NonLinearSolver=NonLinearSolver, ExtraArguments=extra)
    simprefix = "sub" if output_ministeps else ""

    def make_handler(prefix: str) -> ResultHandler:
        return ResultHandler(data_prefix=simprefix + prefix, write_to_disk=True,
                             store_in_memory=False, data_directory=Directory,
                             data_folder=Name, cleardir=False)

    return SimpleNamespace(
        BaseName=BaseName, Name=Name, Description=Description, SimulatorSetup=setup,
        Modules=list(Modules),
        OutputHandlers=SimpleNamespace(states=make_handler("state"),
                                       reports=make_handler("report"),
                                       wellSols=make_handler("wellSols")))


packSimulationProblem = pack_simulation_problem

__all__ = ["packSimulationProblem", "pack_simulation_problem"]


if __name__ == "__main__":
    import tempfile

    def test_results_go_to_directory_and_name():
        with tempfile.TemporaryDirectory() as tmp:
            problem = pack_simulation_problem({"pressure": [1.0]}, SimpleNamespace(),
                                              None, "case", Directory=tmp, useHash=False)
            assert problem.Name == "SimpleNamespace"
            assert problem.Description == "SimpleNamespace"
            path = problem.OutputHandlers.states.get_data_path()
            assert path == Path(tmp)/"SimpleNamespace"
            assert problem.OutputHandlers.wellSols.data_prefix == "wellSols"
        print("ok test_results_go_to_directory_and_name")

    def test_a_hashed_name_changes_with_the_problem():
        with tempfile.TemporaryDirectory() as tmp:
            a = pack_simulation_problem({"pressure": [1.0]}, SimpleNamespace(), None,
                                        "case", Directory=tmp, useHash=True)
            b = pack_simulation_problem({"pressure": [2.0]}, SimpleNamespace(), None,
                                        "case", Directory=tmp, useHash=True)
            assert a.Name != b.Name and a.Name.startswith("SimpleNamespace_")
        print("ok test_a_hashed_name_changes_with_the_problem")

    def test_ministep_output_gets_the_sub_prefix():
        with tempfile.TemporaryDirectory() as tmp:
            problem = pack_simulation_problem(
                {}, SimpleNamespace(), None, "case", Directory=tmp, useHash=False,
                Name="named", ExtraArguments={"OutputMinisteps": True, "Verbose": True})
            assert problem.SimulatorSetup.OutputMinisteps
            assert problem.SimulatorSetup.ExtraArguments == {"Verbose": True}
            assert problem.OutputHandlers.states.data_prefix == "substate"
            assert problem.Description == "named_SimpleNamespace"
        print("ok test_ministep_output_gets_the_sub_prefix")

    test_results_go_to_directory_and_name()
    test_a_hashed_name_changes_with_the_problem()
    test_ministep_output_gets_the_sub_prefix()
    print("\nAll tests passed!")
