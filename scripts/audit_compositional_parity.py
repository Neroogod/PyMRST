#!/usr/bin/env python3
"""Generate the reviewed MRST compositional parity inventory.

The audit deliberately separates structural ownership from behavioral status.
A snake-case Python file is an owner, not proof of complete MATLAB behavior.
Use ``--require-complete`` only for the final acceptance gate.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MRST = ROOT.parent / "mrst-2024a-Geomech" / "modules" / "compositional"
DEFAULT_PYTHON = ROOT / "pymrst" / "modules" / "compositional"
DEFAULT_CSV = ROOT / "docs" / "compositional" / "parity.csv"

PARTIAL = {}

EXTERNAL_DATA = {}

EVIDENCE_BY_AREA = {
    "examples": "examples/validation/validate_phase115.py; examples/validation/validate_phase118.py",
    "fluids": "examples/validation/validate_phase110.py; examples/validation/validate_phase114.py",
    "models/eos": "examples/validation/validate_phase13.py",
    "models/natvars": "examples/validation/validate_phase80.py; examples/validation/validate_phase81.py; examples/validation/validate_phase83.py",
    "models/overall": "examples/validation/validate_phase53.py; examples/validation/validate_phase82.py",
    "models": "examples/validation/validate_phase53.py; examples/validation/validate_phase80.py",
    "properties": "examples/validation/validate_phase13.py; examples/validation/validate_phase50.py",
    "utils": "examples/validation/validate_phase13.py; examples/validation/validate_phase59.py",
}

OWNER_OVERRIDES = {
    "examples/compositionalExample3DSixComponents.m": (
        "examples/compositional_example_3d_six_components.py"
    ),
    "fluids/blackoil/convertRSRVtoKValue.m": (
        "fluids/blackoil/convert_rsrv_to_k_value.py"
    ),
}

EVIDENCE_OVERRIDES = {
    "examples/compositionalValidationSimple.m": (
        "tests/compositional/examples/test_simplecomp_reference.py"
    ),
    "fluids/TableCompositionalMixture.m": "examples/validation/validate_phase268.py",
    "utils/coolPropFluidsStructs.m": "examples/validation/validate_phase268.py",
    "utils/initDeckEOSModel.m": "examples/validation/validate_phase269.py",
    "utils/getPartialVolumes.m": (
        "examples/validation/validate_phase59.py; "
        "examples/validation/validate_phase275.py; "
        "examples/validation/validate_phase284.py"
    ),
    "models/natvars/sequential/PressureReducedLinearSystem.m": (
        "examples/validation/validate_phase59.py; "
        "examples/validation/validate_phase299.py"
    ),
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def expected_owner(relative: Path, python_root: Path) -> Path:
    override = OWNER_OVERRIDES.get(relative.as_posix())
    if override is not None:
        return python_root / override
    directories = ["book_ii" if part == "book-ii" else part for part in relative.parts[:-1]]
    return python_root.joinpath(*directories, snake_case(relative.stem) + ".py")


def evidence_for(relative: Path) -> str:
    override = EVIDENCE_OVERRIDES.get(relative.as_posix())
    if override is not None:
        return override
    parts = relative.parts
    candidates = []
    if len(parts) >= 2:
        candidates.append("/".join(parts[:2]))
    candidates.append(parts[0])
    for key in candidates:
        evidence = EVIDENCE_BY_AREA.get(key)
        if evidence:
            return evidence
    return ""


def audit(mrst_root: Path, python_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for matlab in sorted(mrst_root.rglob("*.m")):
        relative = matlab.relative_to(mrst_root)
        rel = relative.as_posix()
        if matlab.name == "Contents.m":
            owner = python_root.joinpath(
                *("book_ii" if p == "book-ii" else p for p in relative.parts[:-1]),
                "contents.py",
            )
            # Root examples has a catalog. Other directories follow the same rule.
            status = "catalog" if owner.is_file() else "missing"
            note = "MATLAB package catalog; Python catalog owner."
            validation = "not-applicable"
        else:
            owner = expected_owner(relative, python_root)
            if not owner.is_file():
                status, note = "missing", "No matching Python owner."
            elif rel in PARTIAL:
                status, note = "partial", PARTIAL[rel]
            elif rel in EXTERNAL_DATA:
                status, note = "external-data", EXTERNAL_DATA[rel]
            else:
                status = "broad"
                note = "Named owner exists; option-level completion remains under review."
                if rel == "models/eos/EquationOfStateModel.m":
                    note += " Zudkevitch-Joffe refusal matches MRST 2024a."
                elif rel == "utils/initDeckEOSModel.m":
                    note += " Region-one fallback matches and is validated against MRST 2024a."
                elif rel == "fluids/TableCompositionalMixture.m":
                    note += " The complete stored MRST CoolProp table is present and validated."
                elif rel == "models/natvars/sequential/PressureReducedLinearSystem.m":
                    note += " Weighted assembly and natural-variable reconstruction are implemented."
            validation = evidence_for(relative)
        rows.append(
            {
                "mrst_path": rel,
                "kind": "catalog" if matlab.name == "Contents.m" else "executable",
                "status": status,
                "python_owner": owner.relative_to(ROOT).as_posix(),
                "validation_evidence": validation,
                "notes": note,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrst-root", type=Path, default=DEFAULT_MRST)
    parser.add_argument("--python-root", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    mrst_root = args.mrst_root.resolve()
    python_root = args.python_root.resolve()
    csv_path = args.csv.resolve()
    if not mrst_root.is_dir():
        parser.error(f"MRST compositional tree not found: {mrst_root}")
    if not python_root.is_dir():
        parser.error(f"PyMRST compositional tree not found: {python_root}")

    rows = audit(mrst_root, python_root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["status"] for row in rows)
    executable = [row for row in rows if row["kind"] == "executable"]
    print(f"MRST compositional files: {len(rows)}")
    print(f"Executable owners present: {sum(row['status'] != 'missing' for row in executable)}/{len(executable)}")
    print(f"Catalog owners present: {counts['catalog']}/{sum(row['kind'] == 'catalog' for row in rows)}")
    print(
        f"Behavioral status: broad={counts['broad']}, partial={counts['partial']}, "
        f"external-data={counts['external-data']}, complete={counts['complete']}"
    )
    print(f"Ledger: {csv_path.relative_to(ROOT)}")

    if counts["missing"]:
        return 1
    if args.require_complete and any(
        row["status"] not in {"complete", "external-data"} for row in executable
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
