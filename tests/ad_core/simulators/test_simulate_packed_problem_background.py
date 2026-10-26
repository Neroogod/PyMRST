"""MRST simulatePackedProblemBackground end to end: SPE1 simulated by a
separate interpreter, which runs simulatePackedProblemStandalone, matches the
same problem simulated in-process, and monitorBackgroundSimulations follows it
to completion."""

import numpy as np
import pytest

from pymrst.modules.ad_core.simulators.sim_runner import (
    copy_packed_problem,
    monitor_background_simulations,
    simulate_packed_problem,
    simulate_packed_problem_background,
)
from pymrst.params.gravity.gravity import isolated_gravity

pytest.importorskip("cloudpickle")


def test_background_run_matches_in_process_run(tmp_path, capsys):
    from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
        pack_time_integration_problems,
    )
    with isolated_gravity():
        fim = pack_time_integration_problems("spe1")[0][0]
        here = copy_packed_problem(fim, "here", Directory=str(tmp_path/"here"))
        there = copy_packed_problem(fim, "there", Directory=str(tmp_path/"there"))
        simulate_packed_problem(here)
        info = simulate_packed_problem_background(there, workdir=tmp_path/"work")
        monitor_background_simulations([there], pause=0.5, dynamicText=False)
    assert "Complete!" in capsys.readouterr().out
    assert not (info.path/"lock.mrst").exists()
    a, b = here.OutputHandlers, there.OutputHandlers
    assert b.states.numel_data() == a.states.numel_data() == 120
    for i in range(1, 121):
        np.testing.assert_array_equal(b.states[i].pressure, a.states[i].pressure)
        np.testing.assert_array_equal(b.states[i].s, a.states[i].s)
    assert ([r["Iterations"] for r in b.reports.get_data()]
            == [r["Iterations"] for r in a.reports.get_data()])
