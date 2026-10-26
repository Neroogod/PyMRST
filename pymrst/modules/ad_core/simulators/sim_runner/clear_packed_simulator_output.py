"""MRST ``ad-core/simulators/sim_runner/clearPackedSimulatorOutput.m``.

Delete the stored states, reports and well solutions of packed problems: all
of them, or report steps `start..min(stop, number of states)` (1-based and
inclusive, as in MRST). A problem with nothing stored is skipped. With
`Prompt` (the default) it asks at the terminal first, and an empty answer
takes `Default`. MRST prints its progress when prompting or under
`mrstVerbose`; PyMRST has no global verbosity, so progress is printed when
prompting, which is what MRST does with `mrstVerbose` off (its default).

`clear_packed_simulator_output_ad` (in `packed_output.py`) is PyMRST's earlier,
zero-based version without the prompt, kept for its callers.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def clear_packed_simulator_output(problems: Any | Sequence[Any], *, Prompt: bool = True,
                                  Default: str = "n", start: int = 1,
                                  stop: float = math.inf) -> None:
    """MRST `clearPackedSimulatorOutput`."""
    if not isinstance(problems, (list, tuple)):
        problems = [problems]
    for problem in problems:
        s = problem.OutputHandlers.states
        r = problem.OutputHandlers.reports
        w = problem.OutputHandlers.wellSols
        ns, nr, nw = s.numel_data(), r.numel_data(), w.numel_data()
        if ns == 0 and nr == 0 and nw == 0:
            continue
        steps = np.arange(start, math.floor(min(stop, ns)) + 1)
        if Prompt:
            answer = input(f"Do you want to delete {steps.size} states and reports \n"
                           f"for {problem.BaseName} [{problem.Name}]? y/n [{Default}]: ")
            if answer.lower() != "y" and not (Default.lower() == "y" and not answer):
                print("Ok, will not remove files.")
                continue
        do_print = Prompt
        if do_print:
            print("Removing files...", end="")
        if start == 1 and steps.size == ns:
            s.reset_data()
            r.reset_data()
            w.reset_data()
        else:
            s.reset_data(steps)
            r.reset_data(steps[steps <= nr])
            w.reset_data(steps[steps <= nw])
        if do_print:
            print(" Files removed.")


clearPackedSimulatorOutput = clear_packed_simulator_output

__all__ = ["clearPackedSimulatorOutput", "clear_packed_simulator_output"]


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace
    from unittest import mock

    from pymrst.modules.ad_core.utils.result_handler import ResultHandler

    def _problem(root: str, nsteps: int = 4, nreports: int = 4) -> SimpleNamespace:
        def make(prefix: str) -> ResultHandler:
            return ResultHandler(data_prefix=prefix, write_to_disk=True,
                                 store_in_memory=False, data_directory=root,
                                 data_folder="case")
        handlers = SimpleNamespace(states=make("state"), reports=make("report"),
                                   wellSols=make("wellSols"))
        for i in range(1, nsteps + 1):
            handlers.states[i] = {"time": float(i)}
            handlers.wellSols[i] = [i]
        for i in range(1, nreports + 1):
            handlers.reports[i] = {"Iterations": i}
        return SimpleNamespace(BaseName="base", Name="case", OutputHandlers=handlers)

    def test_clears_everything_without_prompt():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _problem(tmp)
            clear_packed_simulator_output(problem, Prompt=False)
            handlers = vars(problem.OutputHandlers).values()
            assert all(h.numel_data() == 0 for h in handlers)
        print("ok test_clears_everything_without_prompt")

    def test_clears_a_range_and_respects_shorter_handlers():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _problem(tmp, nsteps=4, nreports=2)
            clear_packed_simulator_output([problem], Prompt=False, start=2)
            assert problem.OutputHandlers.states.get_valid_ids() == [1]
            assert problem.OutputHandlers.reports.get_valid_ids() == [1]
            assert problem.OutputHandlers.wellSols.get_valid_ids() == [1]
        print("ok test_clears_a_range_and_respects_shorter_handlers")

    def test_stop_is_capped_at_the_number_of_states():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _problem(tmp)
            clear_packed_simulator_output(problem, Prompt=False, start=2, stop=3)
            assert problem.OutputHandlers.states.get_valid_ids() == [1, 4]
        print("ok test_stop_is_capped_at_the_number_of_states")

    def test_prompt_no_keeps_and_empty_answer_takes_default():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _problem(tmp)
            with mock.patch("builtins.input", return_value=""):
                clear_packed_simulator_output(problem)
            assert problem.OutputHandlers.states.numel_data() == 4
            with mock.patch("builtins.input", return_value=""):
                clear_packed_simulator_output(problem, Default="y")
            assert problem.OutputHandlers.states.numel_data() == 0
        print("ok test_prompt_no_keeps_and_empty_answer_takes_default")

    def test_empty_problem_is_not_prompted_for():
        with tempfile.TemporaryDirectory() as tmp:
            problem = _problem(tmp, nsteps=0, nreports=0)
            with mock.patch("builtins.input", side_effect=AssertionError):
                clear_packed_simulator_output(problem)
        print("ok test_empty_problem_is_not_prompted_for")

    test_clears_everything_without_prompt()
    test_clears_a_range_and_respects_shorter_handlers()
    test_stop_is_capped_at_the_number_of_states()
    test_prompt_no_keeps_and_empty_answer_takes_default()
    test_empty_problem_is_not_prompted_for()
    print("\nAll tests passed!")
