import numpy as np

from pymrst.modules.shale.spe201243pa.fig_21_22_23_bakken_edfm import (
    setup_fig_21_22_23_bakken_edfm,
)
from pymrst.modules.shale.spe201243pa.fig_21_22_23_bakken_pedfm import (
    setup_fig_21_22_23_bakken_pedfm,
)


def test_bakken_full_input_is_seeded_reproducible_and_reference_sized():
    first = setup_fig_21_22_23_bakken_edfm(seed=12345)
    second = setup_fig_21_22_23_bakken_edfm(seed=12345)
    assert first.matrix_grid.cells.num == 4004
    assert len(first.hydraulic_fractures) == 24
    assert len(first.natural_fractures) == 1000
    assert first.source == "native-seeded-adfne"
    np.testing.assert_allclose(
        first.natural_fractures[0].points, second.natural_fractures[0].points,
    )
    assert {plane.metadata["family"] for plane in first.natural_fractures} == {0, 1}
    assert np.all(first.natural_fractures[0].points > 0)
    assert np.all(first.natural_fractures[0].points < [1200, 300, 90])


def test_bakken_edfm_and_pedfm_build_distinct_live_operator_stacks():
    edfm = setup_fig_21_22_23_bakken_edfm(
        natural_fracture_limit=0, build_edfm=True,
    )
    pedfm = setup_fig_21_22_23_bakken_pedfm(
        natural_fracture_limit=0, build_edfm=True,
    )
    assert edfm.global_grid.n_fracture == pedfm.global_grid.n_fracture == 672
    assert edfm.well_connections.fracture_cells.size == 24
    assert pedfm.operators.N.shape[0] > edfm.operators.N.shape[0]
    assert np.all(edfm.operators.pv > 0) and np.all(pedfm.operators.pv > 0)
    state, converged, _ = edfm.model.step(edfm.initial_state, 1e-10)
    assert converged
    assert state.qOs[0] < 0
    assert np.isfinite(state.pressure).all()
