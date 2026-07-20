"""Phase 336: MRST SimpleWell helper-method parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import SimpleWell  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well() -> Well:
    return Well(
        cells=np.array([0, 1, 2]),
        type="rate",
        val=3.0,
        r=np.array([0.1, 0.1, 0.1]),
        dir="z",
        WI=np.array([1.0, 2.0, 3.0]),
        dZ=np.array([0.0, 1.0, 3.0]),
        name="W",
        compi=np.array([0.2, 0.3, 0.5]),
        refDepth=0.0,
        sign=1,
        status=True,
    )


def validate_mapping_and_limits() -> int:
    simple = SimpleWell(_well())
    lims = simple.setMissingLimits({"bhp": 10.0}, ("bhp", "rate", "vrat"), np.inf)

    failures = 0
    failures += _check("getVariableField maps MRST well variables",
                       simple.getVariableField("bhp") == ("bhp", 0)
                       and simple.getVariableField("qWs") == ("qWs", 0)
                       and simple.getVariableField("well_massfractions") == ("massfractions", slice(None)))
    failures += _check("setMissingLimits preserves existing and fills absent controls",
                       lims["bhp"] == 10.0 and np.isinf(lims["rate"]) and np.isinf(lims["vrat"]))
    failures += _check("wb2in builds MRST default wellbore incidence",
                       np.allclose(simple.wb2in(simple.W), [[1.0, -1.0, 0.0],
                                                            [0.0, 1.0, -1.0],
                                                            [0.0, 0.0, 1.0]]))
    return failures


def validate_update_helpers() -> int:
    simple = SimpleWell(_well(), dpMaxAbs=2.0)
    wsol = SimpleNamespace(bhp=100.0, massfractions=np.array([0.2, 0.3, 0.5]), qWs=1.0, qOs=2.0, qGs=3.0)
    updated = simple.updateWellSol(wsol, ["bhp", "well_massfractions"], [np.array([10.0]), np.array([0.1, -0.5, 0.2])])
    after, shut = simple.updateWellSolAfterStep(None, SimpleNamespace(type="rate", status=True, qWs=-1.0, qOs=0.0, qGs=0.0),
                                                SimpleNamespace(type="bhp"))

    failures = 0
    failures += _check("updateWellSol applies pressure cap and normalizes mass fractions",
                       np.isclose(updated.bhp, 102.0)
                       and np.isclose(np.sum(updated.massfractions), 1.0)
                       and np.all(updated.massfractions >= 0.0))
    failures += _check("getTotalRate sums MRST qXs fields",
                       np.allclose(simple.getTotalRate(updated), [6.0]))
    failures += _check("updateWellSolAfterStep shuts down forbidden sign changes",
                       shut is True and after.status is False and after.switched is True)
    return failures


def validate_pressure_drop_state() -> int:
    simple = SimpleWell(_well())
    model = SimpleNamespace(gravity_magnitude=9.80665)
    wsol = SimpleNamespace(pressure=np.zeros(3))
    out = simple.updateConnectionPressureDropState(model, wsol, np.array([1000.0, 900.0, 800.0]))
    expected = np.cumsum(9.80665 * np.array([1000.0, 900.0, 800.0]) * np.array([0.0, 1.0, 2.0]))
    failures = 0
    failures += _check("updateConnectionPressureDropState computes hydrostatic cdp fallback",
                       np.allclose(out.cdp, expected))
    failures += _check("compVolRat computes independent-phase volume ratio",
                       np.allclose(simple.compVolRat([[0.25, 0.75]], None, [[2.0, 3.0]], None), [0.375]))
    return failures


def main() -> int:
    print("\nPhase 336: MRST SimpleWell helper-method parity")
    failures = 0
    failures += validate_mapping_and_limits()
    failures += validate_update_helpers()
    failures += validate_pressure_drop_state()
    if failures:
        print(f"\nPhase 336 failed ({failures} checks).")
        return 1
    print("\nSimpleWell helper-method parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
