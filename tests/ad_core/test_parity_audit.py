import csv
from pathlib import Path
import subprocess
import sys


def test_deprecated_classic_well_stack_is_not_reported_as_missing():
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "scripts/audit_ad_core_parity.py"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    with (repo / "docs/ad_core/parity.csv").open() as stream:
        rows = {row["symbol"]: row for row in csv.DictReader(stream)}
    # Genuinely deprecated: their only callers are each other and WellModel.
    names = {
        "Well" + "Model",
        "updateConnection" + "DP",
    }
    for name in names:
        assert rows[name]["port_status"] == "excluded-deprecated"
        assert "FacilityModel/SimpleWell" in rows[name]["port_evidence"]

    # These two were in that group and should not have been. Both are called
    # by `UniformFacilityModel`, which is *not* deprecated, so the classic
    # stack is reachable from a live class and porting that model required
    # them. Verified against MRST 2024a: each has exactly two callers,
    # WellModel.m (deprecated) and UniformFacilityModel.m (not).
    #
    # This corrects an earlier audit of my own, which confirmed the
    # deprecation for `updateConnectionDP` and generalised it to the whole
    # group without checking each member. The generalisation was wrong twice.
    for name in ("computeWell" + "ContributionsNew",
                 "setupWell" + "ControlEquations"):
        assert rows[name]["port_status"] == "ported", (
            f"{name} is reachable from UniformFacilityModel and is ported")
        assert rows[name]["python_counterpart"], (
            f"{name} is marked ported with no counterpart file")
