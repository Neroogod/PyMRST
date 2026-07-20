"""Phase 162: GECON/GRUPNET duplicate replacement parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import (  # noqa: E402
    facility_group_controls_from_deck_control,
    facility_group_controls_from_deck_schedule,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_gecon_same_group_replacement() -> int:
    groups = facility_group_controls_from_deck_control({
        "GECON": [
            ["FIELD", 1.0, 0.0, 0.0, 0.0, 0.0, "NONE", "NO", 0],
            ["FIELD", 2.0, 0.0, 0.0, 3.0, 0.0, "CON", "YES", 4],
        ]
    })
    group = groups[0]
    meta = group.extra["GECON"]

    failures = 0
    failures += _check("duplicate GECON keeps last metadata row",
                       np.isclose(meta["min_oil_rate"], 2.0)
                       and np.isclose(meta["max_gas_oil_ratio"], 3.0)
                       and meta["workover"] == "CON"
                       and meta["end_run"] is True)
    failures += _check("duplicate GECON keeps last raw row",
                       meta["raw"][1] == 2.0 and meta["raw"][8] == 4)
    return failures


def validate_grupnet_same_group_replacement() -> int:
    groups = facility_group_controls_from_deck_control({
        "GRUPNET": [
            ["FIELD", 1.0, 0, 0, "NO", "NO", "NONE"],
            ["FIELD", 2.0, 3, 4, "YES", "YES", "NODE1"],
        ]
    })
    group = groups[0]
    meta = group.extra["GRUPNET"]

    failures = 0
    failures += _check("duplicate GRUPNET keeps last metadata row",
                       np.isclose(meta["guide_rate"], 2.0)
                       and np.isclose(meta["min_vfp_table"], 3.0)
                       and np.isclose(meta["max_vfp_table"], 4.0)
                       and meta["use_network"] is True
                       and meta["terminal"] == "NODE1")
    failures += _check("duplicate GRUPNET keeps last raw row",
                       meta["raw"] == ["FIELD", 2.0, 3, 4, "YES", "YES", "NODE1"])
    return failures


def validate_distinct_groups_survive() -> int:
    groups = facility_group_controls_from_deck_control({
        "GECON": [
            ["FIELD", 1.0],
            ["SATELLITE", 2.0],
        ],
        "GRUPNET": [
            ["FIELD", 3.0],
            ["SATELLITE", 4.0],
        ],
    })
    by_name = {g.name: g for g in groups}

    failures = 0
    failures += _check("distinct GECON/GRUPNET groups survive",
                       set(by_name) == {"FIELD", "SATELLITE"})
    failures += _check("distinct groups keep their own replacement targets",
                       by_name["FIELD"].extra["GECON"]["raw"][1] == 1.0
                       and by_name["SATELLITE"].extra["GECON"]["raw"][1] == 2.0
                       and by_name["FIELD"].extra["GRUPNET"]["raw"][1] == 3.0
                       and by_name["SATELLITE"].extra["GRUPNET"]["raw"][1] == 4.0)
    return failures


def validate_schedule_replacement() -> int:
    converted = facility_group_controls_from_deck_schedule({
        "control": [
            {
                "GECON": [["FIELD", 1.0], ["FIELD", 2.0]],
                "GRUPNET": [["FIELD", 3.0], ["FIELD", 4.0]],
            }
        ]
    })
    group = converted[0][0]

    failures = 0
    failures += _check("schedule conversion applies GECON/GRUPNET replacement",
                       group.extra["GECON"]["raw"][1] == 2.0
                       and group.extra["GRUPNET"]["raw"][1] == 4.0)
    return failures


def main() -> int:
    print("\nPhase 162: GECON/GRUPNET duplicate replacement parity")
    failures = validate_gecon_same_group_replacement()
    failures += validate_grupnet_same_group_replacement()
    failures += validate_distinct_groups_survive()
    failures += validate_schedule_replacement()
    if failures:
        print(f"\nPhase 162 failed ({failures} checks).")
        return 1
    print("\nGECON and GRUPNET imports now mirror MRST same-group replacement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
