"""MRST ``ad-core/simulators/sim_runner/simulatePackedProblemStandalone.m``.

Simulate the packed problem that `simulatePackedProblemBackground` stored in
`pth`: load `problem.pkl` (MRST's `problem.mat`), limit the computational
threads if `maxNumCompThreads` was finite, and run `simulatePackedProblem`
with a lock file `lock.mrst` in place, deleting it afterwards. As in MRST, an
error in the simulation is swallowed (it is in the log and the stored output
stops short), and the "already running" check looks for a *directory* named
`lock.mrst` while the lock written is a file, so it never fires.

MRST first resets its module list to the stored `modlist`; PyMRST has no
module system, so the list is only read. The thread limit uses `threadpoolctl`,
which is needed only when a finite limit is given.

`simulate_packed_problem_standalone_ad` (in `background.py`) is PyMRST's
earlier version for the legacy models, kept for its callers.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)


def simulate_packed_problem_standalone(pth: str | Path) -> None:
    """MRST `simulatePackedProblemStandalone`."""
    with (Path(pth)/"problem.pkl").open("rb") as stream:
        tmp = pickle.load(stream)
    nthread = tmp["opt"]["maxNumCompThreads"]
    if math.isfinite(nthread):
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(nthread))
    lockpath = Path(pth)/"lock.mrst"
    if lockpath.is_dir():
        raise RuntimeError("This simulation is already running!")
    lockpath.open("w").close()
    try:
        simulate_packed_problem(tmp["problem"])
    except Exception:
        pass
    lockpath.unlink()


simulatePackedProblemStandalone = simulate_packed_problem_standalone

__all__ = ["simulatePackedProblemStandalone", "simulate_packed_problem_standalone"]


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace
    from unittest import mock

    MODULE = __name__

    def _store(root: Path, **opt) -> None:
        payload = {"problem": SimpleNamespace(Name="case"), "modlist": ["ad-core"],
                   "opt": {"maxNumCompThreads": math.inf, **opt}}
        with (root/"problem.pkl").open("wb") as stream:
            pickle.dump(payload, stream)

    def test_runs_under_a_lock_and_removes_it():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store(root)
            seen = []

            def fake(problem):
                seen.append((problem.Name, (root/"lock.mrst").is_file()))
            with mock.patch(f"{MODULE}.simulate_packed_problem", fake):
                simulate_packed_problem_standalone(root)
            assert seen == [("case", True)]
            assert not (root/"lock.mrst").exists()
        print("ok test_runs_under_a_lock_and_removes_it")

    def test_a_failed_simulation_is_swallowed_and_unlocked():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store(root)
            with mock.patch(f"{MODULE}.simulate_packed_problem",
                            side_effect=RuntimeError("diverged")):
                simulate_packed_problem_standalone(root)
            assert not (root/"lock.mrst").exists()
        print("ok test_a_failed_simulation_is_swallowed_and_unlocked")

    def test_a_lock_directory_means_already_running():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store(root)
            (root/"lock.mrst").mkdir()
            try:
                simulate_packed_problem_standalone(root)
            except RuntimeError as error:
                assert "already running" in str(error)
            else:
                raise AssertionError("expected a RuntimeError")
        print("ok test_a_lock_directory_means_already_running")

    test_runs_under_a_lock_and_removes_it()
    test_a_failed_simulation_is_swallowed_and_unlocked()
    test_a_lock_directory_means_already_running()
    print("\nAll tests passed!")
