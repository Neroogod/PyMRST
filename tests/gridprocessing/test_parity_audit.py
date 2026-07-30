"""Behavior tests for the gridprocessing parity scanner."""

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_gridprocessing_parity.py"
SPEC = importlib.util.spec_from_file_location("grid_parity_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_snake_case_handles_gridprocessing_names():
    assert AUDIT.snake_case("processGRDECL") == "process_grdecl"
    assert AUDIT.snake_case("get2DGridFromFaces") == "get2_d_grid_from_faces"
    assert AUDIT.snake_case("checkAndRepairZCORN") == "check_and_repair_zcorn"


def test_reviewed_consolidated_counterparts_exist():
    for counterparts, _reason in AUDIT.CONSOLIDATED.values():
        for path in counterparts.split("; "):
            assert (ROOT / path).is_file(), path


def test_reviewed_exclusions_have_reasons():
    for status, reason in AUDIT.EXCLUSIONS.values():
        assert status.startswith("excluded-")
        assert reason
