#!/usr/bin/env python3
"""Generate the MRST ``ad-core`` to PyMRST parity ledger.

The scanner is deliberately conservative.  It only calls an item ``ported``
when it finds either a filename match or an explicit reference to the MATLAB
symbol in a Python source file.  Validation is tracked independently by looking
for the symbol in tests and validation scripts.  Ambiguous entries remain
``pending-review`` instead of being guessed complete.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def source_index(paths: list[Path]) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in paths
    }


def find_counterparts(
    matlab_file: Path,
    matlab_root: Path,
    python_root: Path,
    python_sources: dict[Path, str],
) -> tuple[str, list[Path], str]:
    rel = matlab_file.relative_to(matlab_root)
    symbol = matlab_file.stem
    stem = snake_case(symbol)

    if symbol.lower() == "contents":
        return "excluded-catalog", [], "MATLAB package catalog"
    if "mex" in {part.lower() for part in rel.parts[:-1]}:
        return (
            "excluded-platform",
            [],
            "MATLAB MEX build/runtime implementation; Python fallback required",
        )
    if symbol == "getSparseBlocksFixedWidth":
        return (
            "excluded-platform",
            [],
            "private converter for MRST FixedWidthJacobian/MEX storage",
        )
    if symbol == "getDeckOPMData":
        return (
            "excluded-external",
            [],
            "optional opm-data/opm-tests repository and restart-output reader",
        )
    if symbol in {
        "AGMGSolverAD",
        "AMGCLSolverAD",
        "AMGCLSolverBlockAD",
        "AMGCL_CPRSolverAD",
        "AMGCL_CPRSolverBlockAD",
    } and rel.parts[0] == "solvers":
        return (
            "excluded-external-solver",
            [],
            "wrapper for optional external AGMG/AMGCL binary backend",
        )
    if symbol in {
        "WellModel",
        "computeWellContributionsNew",
        "setupWellControlEquations",
        "updateConnectionDP",
    } and rel.parts[:2] == ("models", "wells"):
        return (
            "excluded-deprecated",
            [],
            "deprecated classic well stack; superseded by FacilityModel/SimpleWell",
        )
    if symbol in {"plotRegionContacts", "terniaryWellPlot",
                  "plotPackedProblem"}:
        return (
            "excluded-visualization",
            [],
            "MATLAB graphics-only helper; no simulation or numerical output",
        )
    if rel.parts[0] == "plotting":
        return (
            "excluded-visualization",
            [],
            "MATLAB figure/UI helper; no simulation or numerical output",
        )

    exact = [
        path for path in python_sources
        if path.stem == stem
    ]
    if exact:
        return "ported", exact, "filename"

    token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", re.I)
    mentions = [path for path, text in python_sources.items() if token.search(text)]
    if mentions:
        return "ported-consolidated", mentions, "symbol-reference"

    return "pending-review", [], ""


def validation_matches(
    symbol: str,
    validation_sources: dict[Path, str],
) -> list[Path]:
    token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", re.I)
    return [path for path, text in validation_sources.items() if token.search(text)]


def relative_list(paths: list[Path], root: Path) -> str:
    return "; ".join(str(path.relative_to(root)) for path in sorted(set(paths)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrst-root",
        type=Path,
        default=Path("../mrst-2024a-Geomech/modules/ad-core"),
    )
    parser.add_argument(
        "--python-root",
        type=Path,
        default=Path("pymrst/modules/ad_core"),
    )
    parser.add_argument("--csv", type=Path, default=Path("docs/ad_core/parity.csv"))
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("AD_CORE_PARITY.md"),
    )
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    mrst_root = args.mrst_root.resolve()
    python_root = args.python_root.resolve()
    if not mrst_root.is_dir():
        parser.error(f"MRST ad-core tree not found: {mrst_root}")
    if not python_root.is_dir():
        parser.error(f"PyMRST ad_core tree not found: {python_root}")

    python_files = sorted(python_root.rglob("*.py"))
    python_sources = source_index(python_files)
    validation_files = sorted((repo / "tests").rglob("*.py"))
    validation_files += sorted((repo / "examples" / "validation").rglob("*.py"))
    validation_sources = source_index(validation_files)

    rows = []
    for matlab_file in sorted(mrst_root.rglob("*.m")):
        status, counterparts, evidence = find_counterparts(
            matlab_file, mrst_root, python_root, python_sources
        )
        validations = validation_matches(matlab_file.stem, validation_sources)
        validation = "validated-reference" if validations else "unvalidated"
        rows.append(
            {
                "mrst_path": str(matlab_file.relative_to(mrst_root)),
                "symbol": matlab_file.stem,
                "subsystem": matlab_file.relative_to(mrst_root).parts[0],
                "port_status": status,
                "python_counterpart": relative_list(counterparts, repo),
                "port_evidence": evidence,
                "validation_status": validation,
                "validation_files": relative_list(validations, repo),
                "notes": "",
            }
        )

    csv_path = (repo / args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["port_status"] for row in rows)
    validated = Counter(row["validation_status"] for row in rows)
    subsystems = {}
    for row in rows:
        sub = subsystems.setdefault(row["subsystem"], Counter())
        sub[row["port_status"]] += 1
        sub["total"] += 1
        if row["validation_status"] == "validated-reference":
            sub["validated"] += 1

    md_path = (repo / args.markdown).resolve()
    md_path.write_text(
        "# AD-core parity ledger\n\n"
        "Generated by `scripts/audit_ad_core_parity.py` from the local MRST "
        "2024a reference tree. The CSV is the authoritative file-level ledger. "
        "Scanner results are conservative candidates and must be reviewed before "
        "an item is declared complete.\n\n"
        "## Current scan\n\n"
        f"- MRST files inventoried: **{len(rows)}**\n"
        f"- Direct filename counterparts: **{counts['ported']}**\n"
        f"- Consolidated/symbol-reference counterparts: "
        f"**{counts['ported-consolidated']}**\n"
        f"- Excluded MATLAB catalogs: **{counts['excluded-catalog']}**\n"
        f"- Excluded MATLAB MEX implementation files: "
        f"**{counts['excluded-platform']}**\n"
        f"- Excluded optional external-data integrations: "
        f"**{counts['excluded-external']}**\n"
        f"- Excluded deprecated legacy implementations: "
        f"**{counts['excluded-deprecated']}**\n"
        f"- Excluded optional external solver wrappers: "
        f"**{counts['excluded-external-solver']}**\n"
        f"- Excluded MATLAB-only visualization helpers: "
        f"**{counts['excluded-visualization']}**\n"
        f"- Pending manual review or implementation: "
        f"**{counts['pending-review']}**\n"
        f"- Files mentioned by reference validators: "
        f"**{validated['validated-reference']}**\n\n"
        "## Subsystems\n\n"
        "| Subsystem | MRST files | Candidate ported | Pending review | "
        "Reference-validated |\n"
        "|---|---:|---:|---:|---:|\n"
        + "".join(
            f"| `{name}` | {data['total']} | "
            f"{data['ported'] + data['ported-consolidated']} | "
            f"{data['pending-review']} | {data['validated']} |\n"
            for name, data in sorted(subsystems.items())
        )
        + "\n## Status policy\n\n"
        "- `ported`: a snake-case filename counterpart exists.\n"
        "- `ported-consolidated`: the MATLAB symbol is explicitly referenced "
        "inside a consolidated Python module.\n"
        "- `validated-reference`: a test or validation script explicitly names "
        "the MATLAB symbol. This does not yet grade validation depth.\n"
        "- `excluded-catalog`: MATLAB `Contents.m` catalog only.\n"
        "- `excluded-platform`: MATLAB MEX build/runtime source; its numerical "
        "fallback must still be ported and validated.\n"
        "- `excluded-external`: optional external repository/data integration, "
        "outside the self-contained numerical core.\n"
        "- `excluded-deprecated`: MRST explicitly deprecated the implementation; "
        "its active replacement behavior is provided by the facility layer.\n"
        "- `excluded-external-solver`: wrapper requiring an optional external "
        "binary solver; PyMRST's native CPR/direct solvers remain available.\n"
        "- `excluded-visualization`: MATLAB graphics-only helper.\n"
        "- `pending-review`: no defensible automatic mapping; requires manual "
        "classification or implementation.\n\n"
        "Regenerate with:\n\n"
        "```bash\npython3 scripts/audit_ad_core_parity.py\n```\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path.relative_to(repo)} ({len(rows)} rows)")
    print(f"Wrote {md_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
