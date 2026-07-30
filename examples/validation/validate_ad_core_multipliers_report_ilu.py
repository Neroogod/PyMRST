"""Validate ``getMultipliers``, ``convertReportToSchedule``, and
``reorderForILU`` against MRST.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    Control,
    Schedule,
    convert_report_to_schedule,
    get_multipliers,
    reorder_for_ilu,
)


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_multipliers_report_ilu.mat",
        squeeze_me=True,
    )
    fluid = SimpleNamespace(
        pvMultR=lambda p: 1.0 + 0.01 * np.asarray(p),
        tranMultR=lambda p: 2.0 + 0.02 * np.asarray(p),
        transMult=lambda p: 3.0 + 0.03 * np.asarray(p),
    )
    multipliers = get_multipliers(
        fluid, np.array([10.0, 20.0]), np.array([5.0, 15.0])
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
    schedule = Schedule(
        step_val=[10.0, 20.0],
        step_control=[0, 1],
        controls=[Control(W=["A"]), Control(W=["B"])],
    )
    converted, timesteps, counts = convert_report_to_schedule(report, schedule)
    matrix = np.array([[4.0, 0, 0], [0, 0, 1], [0, 1, 2]])
    rhs = np.array([10.0, 20.0, 30.0])
    dense_a, dense_b = reorder_for_ilu(matrix, rhs, 1)
    sparse_a, sparse_b = reorder_for_ilu(sp.csr_matrix(matrix), rhs, 1)
    labels = ["pv", "trans", "mob", "pv0", "trans0", "mob0"]
    checks = {
        **{
            label: (actual, ref[label])
            for label, actual in zip(labels, multipliers)
        },
        "converted values": (converted.step_val, ref["converted_values"]),
        "converted controls": (
            converted.step_control + 1,
            ref["converted_controls"],
        ),
        "timesteps": (timesteps, ref["timesteps"]),
        "number of steps": (counts, ref["numsteps"]),
        "dense matrix": (dense_a, ref["A_reordered"]),
        "dense rhs": (dense_b, ref["b_reordered"]),
        "sparse matrix": (sparse_a.toarray(), ref["A_sparse"]),
        "sparse rhs": (sparse_b, ref["b_sparse"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
