"""Phase 321: getWellContributions delegates to MRST-style well-model hooks."""

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


class HookReservoirModel:
    n_phases = 2
    n_components = 2
    extraWellSolOutput = False

    def getPhaseNames(self):
        return ["W", "O"]

    def getActivePhases(self):
        return np.array([True, True])

    def getNumberOfComponents(self):
        return 2

    def getComponentNames(self):
        return ["water", "oil"]


def _well(name: str, cell: int) -> Well:
    return Well(
        cells=np.array([cell], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.array([0.5, 0.5]),
        refDepth=0.0,
        sign=1,
        status=True,
    )


class HookWellModel:
    def __init__(self, well: Well, scale: float):
        self.W = well
        self.scale = float(scale)
        self.calls: list[str] = []

    def getExtraPrimaryVariableNames(self, reservoir_model):
        return [], []

    def getExtraEquationNames(self, reservoir_model):
        return ["segmentBalance"], ["seg"]

    def getWellEquationNames(self, reservoir_model):
        return ["wellW", "wellO"], ["perf", "perf"]

    def updateConnectionPressureDrop(self, ws0, ws, reservoir_model, qw, bh, packed, dt, iteration):
        self.calls.append("cdp")
        ws = dict(ws)
        ws["cdp_updated"] = True
        return ws

    def updateLimits(self, ws0, ws, reservoir_model, qw, bh, packed, dt, iteration):
        self.calls.append("limits")
        return qw, bh, ws, True

    def computeWellEquations(self, ws0, ws, reservoir_model, qw, bh, packed, dt, iteration):
        self.calls.append("well_eq")
        base = [
            np.array([10.0 * self.scale]),
            np.array([20.0 * self.scale]),
        ]
        ctrl = np.array([30.0 * self.scale])
        extra = [np.array([40.0 * self.scale])]
        mass = [
            np.array([100.0 * self.scale]),
            np.array([200.0 * self.scale]),
        ]
        vol = [
            np.array([1.0 * self.scale]),
            np.array([2.0 * self.scale]),
        ]
        ws = dict(ws)
        ws["well_eq_updated"] = True
        return base, ctrl, extra, ["segmentBalance"], mass, vol, ws

    def computeComponentContributions(self, ws0, ws, reservoir_model, qw, bh, packed, mass, vol, dt, iteration):
        self.calls.append("component")
        comp = [
            np.array([1000.0 * self.scale]),
            np.array([2000.0 * self.scale]),
        ]
        return [], comp, [], ws


def validate_hook_delegation() -> int:
    wells = [_well("I1", 0), _well("I2", 0)]
    wellmodels = [HookWellModel(wells[0], 1.0), HookWellModel(wells[1], 2.0)]
    facility = FacilityModel(wells)
    facility.setReservoirModel(HookReservoirModel())
    facility.setupWells(wells, wellmodels=wellmodels)
    well_sol = [
        {"status": True, "qWs": 1.0, "qOs": 2.0, "bhp": 100.0},
        {"status": True, "qWs": 3.0, "qOs": 4.0, "bhp": 110.0},
    ]
    wellvars, _, wellmap = facility.getAllPrimaryVariables(well_sol)
    sources, well_system, updated = facility.getWellContributions(
        well_sol,
        well_sol,
        wellvars,
        wellmap,
        p=np.array([100.0]),
        mob=[np.ones(1), np.ones(1)],
        rho=[np.ones(1), np.ones(1)],
        comp=[np.ones(1), np.ones(1)],
        dt=10.0,
        iteration=2,
    )

    failures = 0
    failures += _check("setupWells harvested hook-model extra equations",
                       facility.addedEquationNames == ["segmentBalance"]
                       and facility.addedEquationTypes == ["seg"])
    failures += _check("all well-model hook methods were called",
                       all(model.calls == ["cdp", "limits", "well_eq", "component"] for model in wellmodels))
    failures += _check("base well equations are combined by phase",
                       np.allclose(well_system.wellEquations[0], [10.0, 20.0])
                       and np.allclose(well_system.wellEquations[1], [20.0, 40.0]))
    failures += _check("extra equations are mapped by canonical equation name",
                       np.allclose(well_system.wellEquations[2], [40.0, 80.0])
                       and well_system.names == ["wellW", "wellO", "segmentBalance"]
                       and well_system.types == ["perf", "perf", "seg"])
    failures += _check("control equations are vertically assembled",
                       np.allclose(np.asarray(value(well_system.controlEquation), dtype=float), [30.0, 60.0]))
    failures += _check("repeated perforation cells are collapsed for phase source terms",
                       sources.sourceCells.tolist() == [0]
                       and np.allclose(sources.phaseMass[0], [300.0])
                       and np.allclose(sources.phaseVolume[1], [6.0]))
    failures += _check("component source terms are combined by component",
                       np.allclose(sources.components[0], [1000.0, 2000.0])
                       and np.allclose(sources.components[1], [2000.0, 4000.0]))
    failures += _check("updated wellSol entries come back from well-model hooks",
                       all(item["cdp_updated"] and item["well_eq_updated"] for item in updated))
    return failures


def main() -> int:
    print("\nPhase 321: getWellContributions delegates to MRST-style well-model hooks")
    failures = validate_hook_delegation()
    if failures:
        print(f"\nPhase 321 failed ({failures} checks).")
        return 1
    print("\ngetWellContributions well-model hook delegation is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
