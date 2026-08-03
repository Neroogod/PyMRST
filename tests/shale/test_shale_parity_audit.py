import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_shale_audit_is_reproducible_and_has_complete_layout():
    subprocess.run(
        [sys.executable, "scripts/audit_shale_parity.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    with (ROOT / "docs/shale/parity.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    owners = [row for row in rows if row["kind"] == "owner"]
    assets = [row for row in rows if row["kind"] == "asset"]
    assert len(owners) == 48
    assert len(assets) == 6
    assert all(row["layout_status"] == "present" for row in rows)
    assert all(row["sha256_match"] == "true" for row in assets)
