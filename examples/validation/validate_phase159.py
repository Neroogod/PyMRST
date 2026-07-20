"""Phase 159: GCONINJE duplicate replacement parity."""

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


def validate_same_group_phase_replacement() -> int:
    groups = facility_group_controls_from_deck_control(
        {
            "GCONINJE": [
                ["GINJ", "GAS", "RATE", 1.0e-4, np.inf, np.inf, np.inf, "YES", 1.0],
                ["GINJ", "GAS", "RATE", 2.0e-4, np.inf, np.inf, np.inf, "NO", 2.0],
            ],
        }
    )
    group = groups[0]
    rows = group.extra.get("GCONINJE", [])

    failures = 0
    failures += _check("duplicate GCONINJE keeps one metadata row",
                       len(rows) == 1)
    failures += _check("duplicate GCONINJE keeps the last native target",
                       np.isclose(group.lims["grat"], 2.0e-4)
                       and group.status is False)
    failures += _check("duplicate GCONINJE keeps the last metadata row",
                       np.isclose(rows[0]["surface_rate"], 2.0e-4)
                       and np.isclose(rows[0]["guide_rate"], 2.0))
    return failures


def validate_different_phase_rows_survive() -> int:
    groups = facility_group_controls_from_deck_control(
        {
            "GCONINJE": [
                ["GINJ", "WATER", "RATE", 1.0e-4, np.inf, np.inf, np.inf, "YES", 1.0],
                ["GINJ", "GAS", "RATE", 2.0e-4, np.inf, np.inf, np.inf, "YES", 2.0],
            ],
        }
    )
    group = groups[0]
    phases = [row["phase"] for row in group.extra.get("GCONINJE", [])]

    failures = 0
    failures += _check("different-phase GCONINJE rows are preserved",
                       phases == ["WATER", "GAS"])
    failures += _check("different-phase GCONINJE rows keep distinct limits",
                       np.isclose(group.lims["wrat"], 1.0e-4)
                       and np.isclose(group.lims["grat"], 2.0e-4))
    return failures


def validate_schedule_replacement() -> int:
    schedule = {
        "control": [
            {
                "GCONINJE": [
                    ["GINJ", "GAS", "RATE", 1.0, np.inf, np.inf, np.inf, "YES", 1.0],
                    ["GINJ", "GAS", "RATE", 3.0, np.inf, np.inf, np.inf, "YES", 3.0],
                ]
            }
        ]
    }
    converted = facility_group_controls_from_deck_schedule(schedule)
    row = converted[0][0].extra["GCONINJE"][0]

    failures = 0
    failures += _check("schedule conversion applies GCONINJE replacement",
                       len(converted[0][0].extra["GCONINJE"]) == 1
                       and row["surface_rate"] == 3.0)
    return failures


def main() -> int:
    print("\nPhase 159: GCONINJE duplicate replacement parity")
    failures = validate_same_group_phase_replacement()
    failures += validate_different_phase_rows_survive()
    failures += validate_schedule_replacement()
    if failures:
        print(f"\nPhase 159 failed ({failures} checks).")
        return 1
    print("\nGCONINJE imports now mirror MRST same-group/same-phase replacement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
