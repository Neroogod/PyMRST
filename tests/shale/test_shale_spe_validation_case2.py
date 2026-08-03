import numpy as np
import pytest

from pymrst.modules.shale.spe201243pa.fig_15_17_18_validation_case_2 import (
    run_fig_15_17_18_validation_case_2,
    setup_fig_15_17_18_validation_case_2,
)
from pymrst.modules.shale.spe201243pa.fig_16_validation_case_2_structured import (
    setup_fig_16_validation_case_2_structured,
)
from pymrst.utils.units import DARCY, NANO


@pytest.mark.parametrize(
    "setup,structured,nmatrix",
    [(setup_fig_15_17_18_validation_case_2, False, 6405),
     (setup_fig_16_validation_case_2_structured, True, 6615)],
)
def test_case2_builds_exact_barrier_fracture_well_and_method_matrix(
    setup, structured, nmatrix,
):
    case = setup()
    assert case.structured is structured
    assert case.matrix_grid.cells.num == nmatrix
    assert case.global_grid.n_fracture == 105
    assert case.well_connections.fracture_index.tolist() == [1]
    assert case.pedfm.model.ops.N.shape[0] > case.edfm.model.ops.N.shape[0]
    assert case.explicit.grid.cells.num == 6405
    assert case.explicit_well_cells.tolist() == [3202]
    np.testing.assert_allclose(
        [plane.permeability for plane in case.fracture_planes],
        [NANO*DARCY, 50*DARCY, NANO*DARCY],
    )


def test_case2_live_three_method_compositional_acceptance():
    case = run_fig_15_17_18_validation_case_2(timesteps=[1e-7], max_cuts=2)
    for variant in (case.pedfm, case.edfm, case.explicit):
        assert variant.result.reports[0]["Converged"]
        assert len(variant.states) == 1
        assert np.isfinite(variant.states[0].pressure).all()
        assert variant.states[0].qGs[0] < 0
