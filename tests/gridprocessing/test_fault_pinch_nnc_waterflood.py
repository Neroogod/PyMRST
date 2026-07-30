import numpy as np

from pymrst.gridprocessing.examples.fault_pinch_nnc_waterflood import (
    build_fault_pinch_nnc_waterflood,
    run_fault_pinch_nnc_waterflood,
)


def test_combined_gridprocessing_pipeline_builds_all_features():
    deck, grid, faults, model = build_fault_pinch_nnc_waterflood()
    grid.check()
    assert grid.cells.num == 58
    assert len(faults) == 2
    assert [fault.name for fault in faults] == ["F_X", "F_Y"]
    assert grid.nnc["cells"].shape == (3, 2)
    assert model.ops.N.shape[0] == np.count_nonzero(
        np.all(grid.faces.neighbors >= 0, axis=1)
    ) + 3
    assert np.count_nonzero(np.asarray(deck["ACTNUM"]) == 0) == 2


def test_combined_gridprocessing_pipeline_simulates_schedule():
    result = run_fault_pinch_nnc_waterflood()
    assert result.pressure.shape == (6, 58)
    assert result.saturation.shape == (6, 58)
    assert np.all(np.isfinite(result.pressure))
    assert np.all((result.saturation >= 0) & (result.saturation <= 1))
    assert result.saturation[-1].max() > result.saturation[0].max()
