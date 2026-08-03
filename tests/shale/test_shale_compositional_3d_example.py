import numpy as np

from pymrst.modules.shale import (
    run_compositional_3d_case,
    setup_compositional_3d_case,
)


def test_compositional_3d_setup_has_all_three_fractures_and_connection_types():
    case = setup_compositional_3d_case()
    assert case.grid.cartDims == (5, 3, 2)
    assert len(case.fracture_planes) == 3
    assert case.global_grid.n_fracture > 0
    assert any(kind.startswith("fracmat") for kind in case.global_grid.nnc.type)
    assert "fracfrac" in case.global_grid.nnc.type
    assert [well.name for well in case.wells] == ["Injector", "Producer"]
    assert case.wells[0].type == "rate"
    assert case.wells[1].type == "bhp"
    assert case.mixture.names == ("CarbonDioxide", "n-Decane")


def test_compositional_3d_reduced_case_runs_a_complete_report_step():
    case = run_compositional_3d_case(nstep=1, max_cuts=10)
    assert len(case.states) == 1
    assert case.result.iterations[0] > 0
    state = case.result.final_state
    assert np.all(np.isfinite(state.pressure))
    assert np.all(np.isfinite(state.z))
    assert state.qWs[0] > 0
    assert state.qOs[1] < 0
    assert state.bhp[1] == case.wells[1].val
