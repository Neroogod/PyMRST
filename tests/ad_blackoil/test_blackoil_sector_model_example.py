import numpy as np

from pymrst.modules.ad_blackoil.examples.blackoil_sector_model_example import (
    build_blackoil_sector_model_example,
    run_blackoil_sector_model_example,
)
from pymrst.utils.units import BARSA, DAY


def _quick_case():
    return dict(
        dims=(4, 4, 1),
        physical_dims=(80.0, 80.0, 8.0),
        total_time=2.0 * DAY,
        max_timestep=DAY,
        rampup_steps=1,
    )


def test_blackoilSectorModelExample_builds_mrst_forces():
    case = build_blackoil_sector_model_example(**_quick_case())
    assert case.open_model.G.cells.num == 16
    assert case.producer_cell == 5
    assert case.open_model.facility.wells[0].type == "orat"
    assert case.open_model.facility.wells[0].lims == {"bhp": 100.0 * BARSA}
    assert case.open_model.bc.face.size == 16
    assert np.array_equal(
        case.open_model.bc.dissolution,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 200.0, 1.0]],
    )
    assert case.closed_model.bc is None


def test_blackoilSectorModelExample_runs_open_and_closed_cases():
    result = run_blackoil_sector_model_example(**_quick_case())
    assert len(result.open_boundary.states) == len(result.closed_boundary.states) == 3
    p_open = result.open_boundary.history["pressure"][-1].mean()
    p_closed = result.closed_boundary.history["pressure"][-1].mean()
    assert p_open > p_closed
    assert p_open < 300.0 * BARSA
    assert set(result.gas_at_producer) == {"open", "closed"}
