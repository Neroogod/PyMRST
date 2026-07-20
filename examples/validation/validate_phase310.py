"""Phase 310: MRST FacilityModel valid driving-force keys."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_valid_driving_forces() -> int:
    facility = FacilityModel([])
    forces = facility.get_valid_driving_forces()
    alias = facility.getValidDrivingForces()
    failures = 0
    failures += _check("facility exposes MRST W/bc/src driving-force keys",
                       set(forces) == {"W", "bc", "src"})
    failures += _check("driving-force values are empty placeholders",
                       all(value is None for value in forces.values()))
    failures += _check("camelCase getValidDrivingForces alias matches",
                       alias == forces)
    return failures


def main() -> int:
    print("\nPhase 310: MRST FacilityModel valid driving-force keys")
    failures = validate_valid_driving_forces()
    if failures:
        print(f"\nPhase 310 failed ({failures} checks).")
        return 1
    print("\nFacilityModel driving-force keys are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
