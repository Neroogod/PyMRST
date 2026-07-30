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
    names = {
        "Well" + "Model",
        "computeWell" + "ContributionsNew",
        "setupWell" + "ControlEquations",
        "updateConnection" + "DP",
    }
    for name in names:
        assert rows[name]["port_status"] == "excluded-deprecated"
        assert "FacilityModel/SimpleWell" in rows[name]["port_evidence"]
