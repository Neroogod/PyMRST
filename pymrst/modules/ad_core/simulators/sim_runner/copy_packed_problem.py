"""MRST ``ad-core/simulators/sim_runner/copyPackedProblem.m``.

A copy of a packed problem under a new `BaseName` (and optionally `Name` and
`Description`), with fresh output handlers writing to `<Directory>/<Name>`.
`Directory` defaults to `<mrstOutputDirectory>/<newBaseName>`. A copy with the
same base name and name warns, and errors if it would write to the original's
output directory. As in MRST, the handlers use the prefixes `state`, `report`
and `wellSols` whether or not the problem outputs ministeps.

The copy is MATLAB's struct assignment: a new problem and a new
`OutputHandlers`, sharing the simulator setup with the original.

`copy_packed_problem_ad` (in `packed_collection.py`) is PyMRST's earlier
version, kept for its callers.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

from pymrst.modules.ad_core.utils.result_handler import ResultHandler
from pymrst.utils.settings_manager.mrst_settings import mrst_settings


def copy_packed_problem(problem: Any, newBaseName: str, *, Directory: str = "",
                        Name: str = "", Description: str = "") -> Any:
    """MRST `copyPackedProblem`."""
    if not Name:
        Name = problem.Name
    if not Description:
        Description = problem.Description
    if not Directory:
        Directory = str(Path(mrst_settings("get", "outputDirectory"))/newBaseName)
    if problem.BaseName == newBaseName and problem.Name == Name:
        warnings.warn("Original problem and copy have the same name.")
        if str(problem.OutputHandlers.states.get_data_path()) == str(Path(Directory)/Name):
            raise ValueError("Cannot create new problem with same output directory "
                             "as original problem.")
    problem_copy = copy.copy(problem)
    problem_copy.BaseName = newBaseName
    problem_copy.Name = Name
    problem_copy.Description = Description

    def make_handler(prefix: str) -> ResultHandler:
        return ResultHandler(data_prefix=prefix, write_to_disk=True, store_in_memory=False,
                             data_directory=Directory, data_folder=problem_copy.Name,
                             cleardir=False)

    problem_copy.OutputHandlers = copy.copy(problem.OutputHandlers)
    problem_copy.OutputHandlers.states = make_handler("state")
    problem_copy.OutputHandlers.reports = make_handler("report")
    problem_copy.OutputHandlers.wellSols = make_handler("wellSols")
    return problem_copy


copyPackedProblem = copy_packed_problem

__all__ = ["copyPackedProblem", "copy_packed_problem"]


if __name__ == "__main__":
    import tempfile

    from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
        pack_simulation_problem,
    )

    def _packed(root: str):
        return pack_simulation_problem({"pressure": [1.0]}, object(), object(), "base",
                                       Directory=root, Name="case", useHash=False,
                                       Description="original")

    def test_copy_gets_new_identity_and_handlers():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _packed(tmp)
            problem.OutputHandlers.states[1] = {"time": 1.0}
            twin = copy_packed_problem(problem, "other", Directory=tmp)
            assert (twin.BaseName, twin.Name, twin.Description) == ("other", "case", "original")
            assert twin.OutputHandlers.states.get_data_path() == Path(tmp)/"case"
            assert twin.OutputHandlers.reports.data_prefix == "report"
            assert problem.BaseName == "base"
            assert problem.OutputHandlers.states is not twin.OutputHandlers.states
            assert twin.SimulatorSetup is problem.SimulatorSetup
        print("ok test_copy_gets_new_identity_and_handlers")

    def test_directory_defaults_to_output_directory_and_base_name():
        with tempfile.TemporaryDirectory() as tmp:
            twin = copy_packed_problem(_packed(tmp), "elsewhere", Name="n2",
                                       Description="d2")
            expected = Path(mrst_settings("get", "outputDirectory"))/"elsewhere"/"n2"
            assert twin.OutputHandlers.wellSols.get_data_path() == expected
            assert twin.Description == "d2"
            expected.rmdir()
            expected.parent.rmdir()
        print("ok test_directory_defaults_to_output_directory_and_base_name")

    def test_same_name_and_directory_is_an_error():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _packed(tmp)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    copy_packed_problem(problem, "base", Directory=tmp)
                except ValueError:
                    pass
                else:
                    raise AssertionError("expected a ValueError")
            assert "same name" in str(caught[0].message)
        print("ok test_same_name_and_directory_is_an_error")

    test_copy_gets_new_identity_and_handlers()
    test_directory_defaults_to_output_directory_and_base_name()
    test_same_name_and_directory_is_an_error()
    print("\nAll tests passed!")
