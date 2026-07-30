"""Validate ``getPerforationToWellMapping``, ``convertIncompWellSols`` and ``getReportMinisteps``."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.utils import (
    convert_incomp_well_sols,
    get_perforation_to_well_mapping,
    get_report_ministeps,
)
from pymrst.params.wells_and_bc import Well


def _well(name, cells, compi):
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells, type="rate", val=1.0, r=np.full(cells.size, 0.1),
        dir="z", WI=np.ones(cells.size), dZ=np.zeros(cells.size), name=name,
        compi=np.asarray(compi, float), refDepth=0.0, sign=1,
    )


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_well_report_mapping.mat",
        squeeze_me=True,
    )
    wells = [_well("A", [0, 1], [1, 0]), _well("B", [2], [0.25, 0.75])]
    mapping, incidence = get_perforation_to_well_mapping(wells)
    state = SimpleNamespace(wellSol=[
        SimpleNamespace(flux=np.array([-2.0, 3.0]), pressure=100.0),
        SimpleNamespace(flux=np.array([-4.0]), pressure=90.0),
    ])

    class Fluid:
        properties = lambda self, state: np.array([1.0, 2.0])
        saturation = lambda self, state: np.zeros((3, 2))
        relperm = lambda self, saturation, state: np.array([
            [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]
        ])

    rows = convert_incomp_well_sols(wells, [state], Fluid())[0]
    report = SimpleNamespace(ControlstepReports=[
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=False, Timestep=4.0),
            SimpleNamespace(Converged=True, Timestep=2.0),
        ]),
        SimpleNamespace(StepReports=[SimpleNamespace(Converged=True, Timestep=3.0)]),
    ])
    checks = {
        "perforation map": (mapping + 1, ref["perf2well"]),
        "incidence": (incidence.toarray(), ref["Rw_full"]),
        "qWs": ([row.qWs for row in rows], ref["qWs"]),
        "qOs": ([row.qOs for row in rows], ref["qOs"]),
        "bhp": ([row.bhp for row in rows], ref["bhp"]),
        "ministeps": (get_report_ministeps(report), ref["timesteps"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
