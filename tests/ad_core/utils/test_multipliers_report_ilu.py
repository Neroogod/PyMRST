from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    Control,
    Schedule,
    convert_report_to_schedule,
    get_multipliers,
    reorder_for_ilu,
)


def test_get_multipliers_current_previous_and_defaults():
    fluid = SimpleNamespace(
        pvMultR=lambda p: 1.0 + 0.01 * np.asarray(p),
        tranMultR=lambda p: 2.0 + 0.02 * np.asarray(p),
        transMult=lambda p: 3.0 + 0.03 * np.asarray(p),
    )
    values = get_multipliers(fluid, np.array([10, 20]), np.array([5, 15]))
    assert np.allclose(values[0], [1.1, 1.2])
    assert np.allclose(values[1], [3.3, 3.6])
    assert np.allclose(values[2], [2.2, 2.4])
    assert np.allclose(values[3], [1.05, 1.15])
    assert get_multipliers({}, [1], [0]) == (1.0,) * 6


def test_convert_report_to_schedule_keeps_only_converged_steps():
    schedule = Schedule(
        step_val=[10, 20],
        step_control=[0, 1],
        controls=[Control(W=["A"]), Control(W=["B"])],
    )
    report = SimpleNamespace(ControlstepReports=[
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=False, Timestep=6.0),
            SimpleNamespace(Converged=True, Timestep=4.0),
            SimpleNamespace(Converged=True, Timestep=6.0),
        ]),
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=True, Timestep=20.0),
        ]),
    ])
    converted, timesteps, counts = convert_report_to_schedule(report, schedule)
    assert np.array_equal(timesteps, [4.0, 6.0, 20.0])
    assert np.array_equal(converted.step_control, [0, 0, 1])
    assert np.array_equal(counts, [2, 1])


def test_reorder_for_ilu_dense_and_sparse_with_cell_offset():
    matrix = np.array([[4.0, 0, 0], [0, 0, 1], [0, 1, 2]])
    rhs = np.array([10.0, 20.0, 30.0])
    reordered, reordered_rhs = reorder_for_ilu(matrix, rhs, 1)
    assert np.array_equal(reordered, matrix[[0, 2, 1], :])
    assert np.array_equal(reordered_rhs, [10.0, 30.0, 20.0])
    sparse, _ = reorder_for_ilu(sp.csr_matrix(matrix), rhs, 1)
    assert sp.issparse(sparse)
    assert np.array_equal(sparse.toarray(), reordered)
