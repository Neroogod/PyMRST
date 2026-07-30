#!/usr/bin/env python3
"""Generate the MRST ``gridprocessing`` to PyMRST parity ledger.

Port status and validation status are intentionally independent. A Python
counterpart is not considered validated merely because its filename exists.
Ambiguous numerical entries remain ``pending-review`` until implemented or
given an explicit reviewed classification.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


FIELDS = (
    "mrst_path",
    "symbol",
    "subsystem",
    "port_status",
    "python_counterpart",
    "port_evidence",
    "validation_status",
    "validation_files",
    "notes",
)

# Reviewed mappings for helpers intentionally consolidated into a Python
# consumer rather than duplicated as one module per MATLAB private function.
CONSOLIDATED = {
    "buildCornerPtNodes": (
        "pymrst/gridprocessing/process_grdecl.py",
        "corner-node construction is part of process_grdecl",
    ),
    "extractSubgrid": (
        "pymrst/gridprocessing/grid_structure.py",
        "Grid subsetting is implemented by extract_subgrid",
    ),
    "get2DGridFromFaces": (
        "pymrst/gridprocessing/get_2d_grid_from_faces.py",
        "Python filename uses an unambiguous numeric token",
    ),
    "glue2DGrid": (
        "pymrst/gridprocessing/glue_2d_grid.py",
        "Python filename uses an unambiguous numeric token",
    ),
    "grdeclXYZ": (
        "pymrst/gridprocessing/refine_grdecl.py",
        "GRDECL coordinate unpacking is shared with refine_grdecl",
    ),
    "copyFaces": (
        "pymrst/gridprocessing/make_internal_boundary.py",
        "private face-copy helper is internal to boundary splitting",
    ),
    "computeCpGeometry": (
        "pymrst/modules/ad_core/utils/compute_cp_geometry.py",
        "shared corner-point geometry utility lives with its AD consumer",
    ),
    "extractGridInfoFromPolygons": (
        "pymrst/gridprocessing/compute_grid_slice_polygons.py;"
        " pymrst/gridprocessing/slice_grid.py",
        "polygon extraction is split between slice geometry and topology",
    ),
    "insertInPackedData": (
        "pymrst/gridprocessing/private/packed_data.py",
        "shared packed-data insertion helper",
    ),
    "removeFromPackedData": (
        "pymrst/gridprocessing/private/packed_data.py",
        "shared packed-data removal helper",
    ),
    "sortEdges": (
        "pymrst/gridprocessing/pebi.py",
        "edge/cell-face ordering is internal to the PEBI construction",
    ),
}

EXCLUSIONS = {
    "computeGeometryOrig": (
        "excluded-deprecated",
        "MRST deprecated geometry implementation superseded by computeGeometry",
    ),
    "extended_grid_structure": (
        "excluded-documentation",
        "MATLAB documentation function describing optional edge-augmented grids",
    ),
    "removeIntGrid": (
        "excluded-matlab-compatibility",
        "legacy MATLAB int32-to-double grid storage conversion",
    ),
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _source_index(paths: list[Path]) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in paths
    }


def _token(symbol: str) -> re.Pattern:
    alternatives = "|".join(
        re.escape(value) for value in sorted({symbol, snake_case(symbol)})
    )
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])",
        re.I,
    )


def _relative_list(paths: list[Path], root: Path) -> str:
    return "; ".join(
        str(path.relative_to(root)) for path in sorted(set(paths))
    )


def classify(
    matlab_file: Path,
    matlab_root: Path,
    repo: Path,
    python_sources: dict[Path, str],
) -> tuple[str, str, str]:
    rel = matlab_file.relative_to(matlab_root)
    symbol = matlab_file.stem

    if symbol == "Contents":
        return "excluded-catalog", "", "MATLAB package catalog"
    if rel.parts[0] == "testgrids":
        stem = snake_case(symbol)
        exact = [path for path in python_sources if path.stem == stem]
        if exact:
            return (
                "ported",
                _relative_list(exact, repo),
                "ported deterministic test-grid fixture",
            )
        return (
            "test-fixture",
            "",
            "MATLAB test-grid fixture not yet ported",
        )
    if symbol in EXCLUSIONS:
        status, reason = EXCLUSIONS[symbol]
        counterpart = ""
        if symbol == "computeGeometryOrig":
            counterpart = (
                "pymrst/gridprocessing/deprecated/compute_geometry_orig.py"
            )
            if not (repo / counterpart).is_file():
                raise FileNotFoundError(
                    f"deprecated compatibility counterpart missing: {counterpart}"
                )
        return status, counterpart, reason
    if symbol in CONSOLIDATED:
        counterparts, reason = CONSOLIDATED[symbol]
        for value in counterparts.split("; "):
            if not (repo / value).is_file():
                raise FileNotFoundError(f"reviewed counterpart is missing: {value}")
        return "ported-consolidated", counterparts, reason

    stem = snake_case(symbol)
    exact = [path for path in python_sources if path.stem == stem]
    if exact:
        return "ported", _relative_list(exact, repo), "filename"

    token = _token(symbol)
    mentions = [
        path for path, text in python_sources.items() if token.search(text)
    ]
    if mentions:
        return (
            "ported-consolidated",
            _relative_list(mentions, repo),
            "explicit symbol reference",
        )
    return "pending-review", "", ""


def generate_rows(
    repo: Path, matlab_root: Path, python_root: Path
) -> list[dict[str, str]]:
    python_files = sorted(python_root.rglob("*.py"))
    python_sources = _source_index(python_files)

    validation_files = sorted((repo / "tests").rglob("*.py"))
    validation_files += sorted((repo / "examples" / "validation").rglob("*.py"))
    # Ledger integrity tests mention classifications and symbols but do not
    # constitute numerical evidence.
    validation_files = [
        path for path in validation_files if "parity" not in path.stem
    ]
    validation_sources = _source_index(validation_files)

    rows = []
    for matlab_file in sorted(matlab_root.rglob("*.m")):
        rel = matlab_file.relative_to(matlab_root)
        symbol = matlab_file.stem
        status, counterparts, evidence = classify(
            matlab_file, matlab_root, repo, python_sources
        )
        token = _token(symbol)
        validations = [
            path for path, text in validation_sources.items()
            if token.search(text)
        ]
        rows.append(
            {
                "mrst_path": str(rel),
                "symbol": symbol,
                "subsystem": rel.parts[0] if len(rel.parts) > 1 else "root",
                "port_status": status,
                "python_counterpart": counterparts,
                "port_evidence": evidence,
                "validation_status": (
                    "validated-reference" if validations else "unvalidated"
                ),
                "validation_files": _relative_list(validations, repo),
                "notes": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    counts = Counter(row["port_status"] for row in rows)
    validation = Counter(row["validation_status"] for row in rows)
    pending = [
        row for row in rows if row["port_status"] == "pending-review"
    ]
    path.write_text(
        "# Gridprocessing parity ledger\n\n"
        "Generated by `scripts/audit_gridprocessing_parity.py` from the local "
        "MRST 2024a reference tree. The CSV ledger is authoritative; port and "
        "validation status are tracked independently.\n\n"
        "## Current scan\n\n"
        f"- MRST MATLAB files inventoried: **{len(rows)}**\n"
        f"- Direct filename counterparts: **{counts['ported']}**\n"
        f"- Reviewed consolidated counterparts: "
        f"**{counts['ported-consolidated']}**\n"
        f"- Pending implementation/review: **{counts['pending-review']}**\n"
        f"- Test-grid fixtures: **{counts['test-fixture']}**\n"
        f"- Catalogs excluded: **{counts['excluded-catalog']}**\n"
        f"- Documentation excluded: **{counts['excluded-documentation']}**\n"
        f"- Deprecated implementations excluded: "
        f"**{counts['excluded-deprecated']}**\n"
        f"- MATLAB compatibility helpers excluded: "
        f"**{counts['excluded-matlab-compatibility']}**\n"
        f"- Entries named by tests/reference validators: "
        f"**{validation['validated-reference']}**\n\n"
        "## Pending entries\n\n"
        + (
            "| MRST file | Symbol |\n|---|---|\n"
            + "".join(
                f"| `{row['mrst_path']}` | `{row['symbol']}` |\n"
                for row in pending
            )
            if pending else "None.\n"
        )
        + "\n## Status policy\n\n"
        "- `ported`: a snake-case Python filename counterpart exists.\n"
        "- `ported-consolidated`: a reviewed Python module implements the "
        "MATLAB function as part of a shared or public consumer.\n"
        "- `pending-review`: no defensible counterpart or exclusion exists.\n"
        "- `validated-reference`: a non-ledger test or validation script names "
        "the MATLAB symbol or Python spelling; validation depth remains subject "
        "to review.\n"
        "- Exclusion and fixture statuses require an explicit reviewed reason "
        "in the CSV.\n\n"
        "Regenerate with:\n\n"
        "```bash\npython3 scripts/audit_gridprocessing_parity.py\n```\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrst-root",
        type=Path,
        default=Path("../mrst-2024a-Geomech/gridprocessing"),
    )
    parser.add_argument(
        "--python-root",
        type=Path,
        default=Path("pymrst/gridprocessing"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/gridprocessing/parity.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("GRIDPROCESSING_PARITY.md"),
    )
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    matlab_root = args.mrst_root.resolve()
    python_root = args.python_root.resolve()
    if not matlab_root.is_dir():
        parser.error(f"MRST gridprocessing tree not found: {matlab_root}")
    if not python_root.is_dir():
        parser.error(f"PyMRST gridprocessing tree not found: {python_root}")

    rows = generate_rows(repo, matlab_root, python_root)
    csv_path = (repo / args.csv).resolve()
    markdown_path = (repo / args.markdown).resolve()
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    print(f"Wrote {csv_path.relative_to(repo)} ({len(rows)} rows)")
    print(f"Wrote {markdown_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
