"""Phase 319: FacilityModel getWellContributions gateway parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402
from pymrst.utils.adi import value  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class TinyReservoirModel:
    n_phases = 2
    n_components = 2
    gravity_magnitude = 0.0

    def get_props(self, state, *names):
        out = []
        for name in names:
            if name == "Mobility":
                out.append([np.ones(2), np.ones(2)])
            elif name == "ShrinkageFactors":
                out.append([np.ones(2), np.ones(2)])
            elif name == "Density":
                out.append([np.full(2, 1000.0), np.full(2, 800.0)])
            else:
                raise KeyError(name)
        return out[0] if len(out) == 1 else out

    def getPhaseNames(self):
        return ["W", "O"]


def _well() -> Well:
    return Well(
        cells=np.array([0, 1], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1, 0.1]),
        dir="z",
        WI=np.array([1.0, 2.0]),
        dZ=np.array([0.0, 0.0]),
        name="P1",
        compi=np.array([0.5, 0.5]),
        refDepth=0.0,
        sign=-1,
        status=True,
    )


def validate_get_well_contributions() -> int:
    model = TinyReservoirModel()
    facility = FacilityModel([_well()])
    facility.setReservoirModel(model)
    facility.addedEquationNames = ["extraWellEq"]
    facility.addedEquationTypes = ["constraint"]

    well_sol = [{"status": True, "qWs": 0.25, "qOs": 0.75, "bhp": 80.0}]
    wellvars, _, wellmap = facility.getAllPrimaryVariables(well_sol)
    pressure = np.array([100.0, 95.0])
    state = {"pressure": pressure, "wellSol": well_sol}

    sources, well_system, updated = facility.getWellContributions(
        well_sol,
        well_sol,
        wellvars,
        wellmap,
        pressure,
        comp=[np.ones(2), np.zeros(2)],
        state=state,
    )

    failures = 0
    failures += _check("getWellContributions returns the input wellSol object",
                       updated is well_sol)
    failures += _check("sources carry MRST-style phase source fields",
                       len(sources.phaseMass) == 2
                       and len(sources.phaseVolume) == 2
                       and sources.sourceCells.tolist() == [0, 1])
    failures += _check("phase source rows follow perforation count",
                       all(np.asarray(x).shape == (2,) for x in sources.phaseVolume)
                       and all(np.asarray(x).shape == (2,) for x in sources.phaseMass))
    failures += _check("component source rows follow component count",
                       len(sources.components) == 2
                       and all(np.asarray(x).shape == (2,) for x in sources.components))
    failures += _check("wellSystem carries base plus advertised extra equation metadata",
                       well_system.names == ["wellW", "wellO", "extraWellEq"]
                       and well_system.types == ["perf", "perf", "constraint"]
                       and len(well_system.wellEquations) == 3)
    failures += _check("control equation is assembled as a finite residual",
                       np.all(np.isfinite(np.asarray(value(well_system.controlEquation), dtype=float))))
    failures += _check("phase-mass sources use reservoir density scaling",
                       np.allclose(sources.phaseMass[0], sources.phaseVolume[0] * 1000.0)
                       and np.allclose(sources.phaseMass[1], sources.phaseVolume[1] * 800.0))
    return failures


def main() -> int:
    print("\nPhase 319: FacilityModel getWellContributions gateway parity")
    failures = validate_get_well_contributions()
    if failures:
        print(f"\nPhase 319 failed ({failures} checks).")
        return 1
    print("\ngetWellContributions gateway is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
