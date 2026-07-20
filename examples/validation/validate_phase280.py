"""Phase 280: public MRST-style facility-group inspection surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.model import FacilityGroupControl, OverallCompositionModel  # noqa: E402


class DummyWell:
    def __init__(self, name: str, sign: int = -1, status: bool = True):
        self.name = name
        self.sign = sign
        self.status = status


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _model_with_groups():
    model = OverallCompositionModel.__new__(OverallCompositionModel)
    model.wells = [DummyWell("P1"), DummyWell("P2"), DummyWell("P3")]
    west = FacilityGroupControl("WEST", wells=["P1", "P2"], lims={"orat": 4.0e-5}, allocation="equal")
    east = FacilityGroupControl("EAST", wells=["P3"], lims={"orat": 2.0e-5})
    field = FacilityGroupControl(
        "FIELD",
        wells=["group:WEST", "group:EAST"],
        lims={"orat": 5.0e-5},
        allocation="fraction",
        fractions=[0.7, 0.3],
        extra={"source": "phase280"},
    )
    model.group_controls = [west, east, field]
    return model, west, east, field


def main() -> int:
    print("\nPhase 280: public MRST-style facility-group inspection surface")
    failures = 0
    model, west, east, field = _model_with_groups()

    west.groupName = "WEST"
    west.members = ["P1", "P2"]
    west.limits = {"orat": 4.0e-5, "grat": 1.0e-5}
    west.isActive = True
    west_dict = west.toDict()
    failures += _check(
        "FacilityGroupControl exposes MRST-style name/member/limit aliases",
        west.groupName == "WEST"
        and list(west.members) == ["P1", "P2"]
        and np.isclose(west.limits["grat"], 1.0e-5)
        and west.isActive is True
        and west_dict["groupName"] == "WEST"
        and west_dict["limits"]["orat"] == west.lims["orat"],
    )

    failures += _check(
        "resolveFacilityGroup flattens nested group trees in leaf order",
        model.resolveFacilityGroup("FIELD") == [0, 1, 2]
        and model.facilityGroupLeafWellNames(field) == ["P1", "P2", "P3"],
    )

    summary = {row["name"]: row for row in model.facilityGroupSummary()}
    failures += _check(
        "facilityGroupSummary includes leaf wells, indices, allocation and extras",
        summary["FIELD"]["leaf_wells"] == ["P1", "P2", "P3"]
        and summary["FIELD"]["leaf_indices"] == [0, 1, 2]
        and summary["FIELD"]["allocation"] == "fraction"
        and summary["FIELD"]["fractions"] == [0.7, 0.3]
        and summary["FIELD"]["extra"] == {"source": "phase280"},
    )

    try:
        model.resolveFacilityGroup("MISSING")
        missing_error = False
    except ValueError as err:
        missing_error = "unknown facility group" in str(err)
    failures += _check("unknown group names raise a clear ValueError", missing_error)

    if failures:
        print(f"\nPhase 280 failed ({failures} checks).")
        return 1
    print("\nFacility group trees now have a public MRST-style inspection API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
