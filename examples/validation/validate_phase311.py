"""Phase 311: MRST FacilityModel setWellSolStatistics parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, cells, *, status: bool = True, sign: int = 1) -> Well:
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells,
        type="rate",
        val=1.0,
        r=np.full(cells.size, 0.1),
        dir="z",
        WI=np.ones(cells.size),
        dZ=np.zeros(cells.size),
        name=name,
        compi=np.array([1.0, 0.0, 0.0]),
        refDepth=0.0,
        sign=sign,
        status=status,
    )


def validate_set_well_sol_statistics() -> int:
    facility = FacilityModel([
        _well("I1", [0, 1], status=True, sign=1),
        _well("P1", [2], status=False, sign=-1),
        _well("P2", [3, 4], status=True, sign=-1),
    ])
    well_sol = [
        SimpleNamespace(status=True, qWs=10.0, qOs=20.0, qGs=30.0),
        SimpleNamespace(status=True, qWs=99.0, qOs=99.0, qGs=99.0),
        SimpleNamespace(status=True, qWs=-3.0, qOs=-7.0, qGs=-11.0),
    ]
    sources = {
        "phaseVolume": [
            np.array([1.0, 2.0, 3.0, 4.0]),   # W rows for active perforations [I1, I1, P2, P2]
            np.array([5.0, 6.0, 7.0, 8.0]),   # O
            np.array([9.0, 10.0, 11.0, 12.0]), # G
        ]
    }
    out = facility.setWellSolStatistics(well_sol, sources)

    bad = [SimpleNamespace(status=True, qWs=1.0, qOs=1.0, qGs=1.0)]
    bad_facility = FacilityModel([_well("I1", [0, 1])])
    bad_out = bad_facility.setWellSolStatistics(
        bad,
        {"phaseVolume": [np.array([1.0]), np.array([2.0]), np.array([3.0])]},
    )

    failures = 0
    failures += _check("active wells receive reservoir phase-rate sums",
                       np.isclose(out[0].qWr, 3.0)
                       and np.isclose(out[0].qOr, 11.0)
                       and np.isclose(out[0].qGr, 19.0)
                       and np.isclose(out[2].qWr, 7.0)
                       and np.isclose(out[2].qOr, 15.0)
                       and np.isclose(out[2].qGr, 23.0))
    failures += _check("inactive wells are not decorated",
                       not hasattr(out[1], "qTr")
                       and not hasattr(out[1], "flux"))
    failures += _check("surface and reservoir total rates match MRST fields",
                       np.isclose(out[0].qTs, 60.0)
                       and np.isclose(out[0].qTr, 33.0)
                       and np.isclose(out[2].qTs, -21.0)
                       and np.isclose(out[2].qTr, 45.0))
    failures += _check("phase cuts and GOR are computed from surface rates",
                       np.isclose(out[0].wcut, 10.0 / 30.0)
                       and np.isclose(out[0].gcut, 30.0 / 60.0)
                       and np.isclose(out[0].ocut, 20.0 / 60.0)
                       and np.isclose(out[0].gor, 30.0 / 20.0))
    failures += _check("per-well flux matrices keep active perforation rows and WOG columns",
                       np.allclose(out[0].flux, [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]])
                       and np.allclose(out[2].flux, [[3.0, 7.0, 11.0], [4.0, 8.0, 12.0]]))
    failures += _check("mismatched source rows return without adding statistics",
                       bad_out is bad
                       and not hasattr(bad[0], "qTr")
                       and not hasattr(bad[0], "flux"))
    return failures


def main() -> int:
    print("\nPhase 311: MRST FacilityModel setWellSolStatistics parity")
    failures = validate_set_well_sol_statistics()
    if failures:
        print(f"\nPhase 311 failed ({failures} checks).")
        return 1
    print("\nFacilityModel wellSol statistics are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
