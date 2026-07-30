import csv
from pathlib import Path

from scripts.audit_ad_blackoil_parity import FIELDS, generate_rows


REPO = Path(__file__).resolve().parents[2]
MRST = REPO.parent / "mrst-2024a-Geomech" / "modules" / "ad-blackoil"
PYTHON = REPO / "pymrst" / "modules" / "ad_blackoil"


def test_generated_ledger_matches_checked_in_csv():
    generated = generate_rows(REPO, MRST, PYTHON)
    with (REPO / "docs/ad_blackoil/parity.csv").open(newline="", encoding="utf-8") as stream:
        checked_in = list(csv.DictReader(stream))
    assert checked_in == generated


def test_ledger_accounts_for_every_matlab_file_once():
    rows = generate_rows(REPO, MRST, PYTHON)
    matlab = sorted(str(path.relative_to(MRST)) for path in MRST.rglob("*.m"))
    assert sorted(row["mrst_path"] for row in rows) == matlab
    assert len({row["mrst_path"] for row in rows}) == len(rows)
    assert tuple(rows[0]) == FIELDS


def test_missing_utilities_and_guarded_examples_remain_honestly_pending():
    rows = {row["symbol"]: row for row in generate_rows(REPO, MRST, PYTHON)}
    assert rows["eggExample"]["port_status"] == "pending"
    assert rows["blackoilTimeIntegrationExample"]["port_status"] == "partial"
    assert rows["blackoilSectorModelExample"]["port_status"] == "ported"


def test_new_shrinkage_helpers_are_direct_ports():
    rows = {row["symbol"]: row for row in generate_rows(REPO, MRST, PYTHON)}
    assert rows["getbO_BO"]["port_status"] == "ported"
    assert rows["getbG_BO"]["port_status"] == "ported"
    assert rows["computeInitAquifer"]["port_status"] == "ported"
    assert rows["getFluxAndPropsOil_BO"]["port_status"] == "ported"
    assert rows["updateStateBlackOilGeneric"]["port_status"] == "ported"
    assert rows["equationsWaterThermal"]["port_status"] == "ported"
    assert rows["getPolymerShearMultiplier"]["port_status"] == "ported"
    assert rows["getWellPolymer"]["port_status"] == "ported"
