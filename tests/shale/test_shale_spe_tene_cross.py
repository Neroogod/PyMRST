import numpy as np
import pytest

from pymrst.modules.shale.spe201243pa.fig_10_to_13_tene_high_kf import (
    run_fig_10_to_13_tene_high_kf,
)
from pymrst.modules.shale.spe201243pa.fig_10_to_13_tene_low_kf import (
    run_fig_10_to_13_tene_low_kf,
    setup_fig_10_to_13_tene_low_kf,
)
from pymrst.utils.units import DARCY


def test_tene_cross_builds_three_distinct_reference_discretizations():
    case = setup_fig_10_to_13_tene_low_kf(reduced_explicit=True)
    assert case.matrix_grid.cells.num == 27*27*3
    assert case.global_grid.n_fracture == 90
    assert case.pedfm.model.operators.N.shape[0] > case.edfm.model.operators.N.shape[0]
    assert case.explicit.grid.cells.num == 45*45*2
    assert case.explicit_fracture_cells.size == 98
    assert np.all(case.pedfm.model.pv_ref > 0)


@pytest.mark.parametrize(
    "runner,kf", [(run_fig_10_to_13_tene_low_kf, 1e-8),
                   (run_fig_10_to_13_tene_high_kf, 1e3)],
)
def test_tene_cross_live_acceptance_advances_all_three_methods(runner, kf):
    case = runner(reduced_explicit=True, timesteps=[1e-6])
    assert case.fracture_permeability == pytest.approx(kf*DARCY)
    for variant in (case.pedfm, case.edfm, case.explicit):
        assert variant.reports[0].converged
        assert len(variant.states) == 1
        assert np.isfinite(variant.states[0].pressure).all()
