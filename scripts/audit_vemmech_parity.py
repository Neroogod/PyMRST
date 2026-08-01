#!/usr/bin/env python3
"""Generate the MRST ``vemmech`` to PyMRST parity ledger.

Port status and validation status are intentionally independent. A Python
counterpart is not considered validated merely because its filename exists.
Ambiguous numerical entries remain ``pending-review`` until implemented or
given an explicit reviewed classification.

Pending entries carry the phase of ``VEMMECH_implementation.md`` that retires
them, so the ledger and the plan cannot drift apart.
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
# consumer rather than duplicated as one module per MATLAB function.
CONSOLIDATED = {
    "lincompTPFA": (
        "pymrst/modules/vemmech/models/linear_compressibility_model.py",
        "lincomp_tpfa is the solver behind LinearCompressibilityModel",
    ),
    "ssparsemul": (
        "pymrst/modules/vemmech/ad_version/__init__.py",
        "sparse contraction is native to SparseMultiArray.__mul__",
    ),
    "tbinaryop": (
        "pymrst/modules/vemmech/ad_version/__init__.py",
        "elementwise tensor ops are SparseMultiArray operators",
    ),
    "tcontract": (
        "pymrst/modules/vemmech/ad_version/__init__.py",
        "index contraction is SparseMultiArray.__mul__",
    ),
}

EXCLUSIONS = {
    "getjac": (
        "excluded-mex",
        "MATLAB MEX accessor for ADI Jacobian internals; no Python analogue",
    ),
    "getval": (
        "excluded-mex",
        "MATLAB MEX accessor for ADI values; no Python analogue",
    ),
    "makeadi": (
        "excluded-mex",
        "MATLAB MEX ADI constructor; PyMRST builds AD objects directly",
    ),
}

# Which phase of VEMMECH_implementation.md retires each outstanding entry.
PENDING_PHASE = {
    "squareTest": "phase-2 fixture",
    "squareLayersTest": "phase-2 fixture",
    "squareGrid": "phase-2 fixture",
    "makeCompactionTest": "phase-2 fixture",
    "complex3DGrid": "phase-2 fixture",
    "linearDisplacement": "phase-2 fixture",
    "exploreSquareGrid": "phase-2 fixture",
    "ENu2LMu_3D": "phase-2 fixture",
    "addFluidContribMechVEM": "phase-3 poroelastic boundary coupling",
    "example_2D": "phase-4 example",
    "example_2D_layers": "phase-4 example",
    "example_2D_complex_grid": "phase-4 example",
    "compactionTest2D": "phase-4 example",
    "compactionTest3D": "phase-4 example",
    "example_poroelasticity": "phase-4 example",
    "simple_matching_example": "phase-4 example",
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _normalized(name: str) -> str:
    """Collapse a name to letters and digits for spelling-insensitive matching.

    MATLAB embeds dimension tokens (``plotFaces2D``, ``VEM3D_div``) that no
    single snake-case rule spells the way a human would. Comparing without
    separators keeps those honest filename ports out of the consolidated
    bucket, where they would misrepresent the layout.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


_DOCSTRING = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_COMMENT = re.compile(r"#[^\n]*")


def _code_only(text: str) -> str:
    """Drop docstrings and comments so prose cannot pass as an implementation.

    A module that merely *mentions* `squareTest` while explaining what it is
    for must not be reported as implementing it; without this the consolidated
    fallback promotes documentation into parity.
    """
    return _COMMENT.sub("", _DOCSTRING.sub("", text))


def _source_index(paths: list[Path]) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in paths
    }


def _token(symbol: str, *extra: str) -> re.Pattern:
    """Match `symbol` under any of its spellings, including its real ports.

    Passing the Python counterpart stems matters: `plotFaces2D` is spelled
    `plot_faces_2d`, which no snake-case rule derives, so a validator using it
    would otherwise be missed and the entry wrongly reported as unvalidated.
    """
    names = {symbol, snake_case(symbol)} | {value for value in extra if value}
    alternatives = "|".join(re.escape(value) for value in sorted(names))
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
    symbol = matlab_file.stem

    if symbol == "Contents":
        return "excluded-catalog", "", "MATLAB package catalog"
    if symbol in EXCLUSIONS:
        status, reason = EXCLUSIONS[symbol]
        return status, "", reason
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

    target = _normalized(symbol)
    near = [
        path for path in python_sources if _normalized(path.stem) == target
    ]
    if near:
        return (
            "ported",
            _relative_list(near, repo),
            "filename, dimension-token spelling",
        )

    token = _token(symbol)
    mentions = [
        path for path, text in python_sources.items()
        if token.search(_code_only(text))
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
        stems = [Path(value).stem for value in counterparts.split("; ") if value]
        token = _token(symbol, *stems)
        validations = [
            path for path, text in validation_sources.items()
            if token.search(text)
        ]
        note = PENDING_PHASE.get(symbol, "") if status == "pending-review" else ""
        rows.append(
            {
                "mrst_path": str(rel),
                "symbol": symbol,
                "subsystem": "/".join(rel.parts[:-1]) or "root",
                "port_status": status,
                "python_counterpart": counterparts,
                "port_evidence": evidence,
                "validation_status": (
                    "validated-reference" if validations else "unvalidated"
                ),
                "validation_files": _relative_list(validations, repo),
                "notes": note,
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
    pending = [row for row in rows if row["port_status"] == "pending-review"]
    unvalidated = [
        row for row in rows
        if row["port_status"] in {"ported", "ported-consolidated"}
        and row["validation_status"] != "validated-reference"
    ]
    phases = Counter(row["notes"] for row in pending)

    lines = [
        "# vemmech parity ledger",
        "",
        "Generated by `scripts/audit_vemmech_parity.py` from the local MRST "
        "2024a reference tree. The CSV ledger is authoritative; port and "
        "validation status are tracked independently. The plan that retires "
        "the pending entries is `VEMMECH_implementation.md`.",
        "",
        "## Current scan",
        "",
        f"- MRST MATLAB files inventoried: **{len(rows)}**",
        f"- Direct filename counterparts: **{counts['ported']}**",
        f"- Reviewed consolidated counterparts: "
        f"**{counts['ported-consolidated']}**",
        f"- Pending implementation/review: **{counts['pending-review']}**",
        f"- Catalogs excluded: **{counts['excluded-catalog']}**",
        f"- MATLAB MEX helpers excluded: **{counts['excluded-mex']}**",
        f"- Entries named by tests/reference validators: "
        f"**{validation['validated-reference']}**",
        f"- Ported entries still unvalidated: **{len(unvalidated)}**",
        "",
        "## Outstanding work by plan phase",
        "",
    ]
    if pending:
        lines += ["| Phase | Entries |", "|---|---:|"]
        for phase, count in sorted(phases.items()):
            lines.append(f"| {phase or 'unassigned'} | {count} |")
    else:
        lines.append("None.")
    lines += ["", "## Pending entries", ""]
    if pending:
        lines += ["| MRST file | Symbol | Phase |", "|---|---|---|"]
        lines += [
            f"| `{row['mrst_path']}` | `{row['symbol']}` | "
            f"{row['notes'] or 'unassigned'} |"
            for row in pending
        ]
    else:
        lines.append("None.")
    lines += ["", "## Ported but unvalidated", ""]
    if unvalidated:
        lines += ["| MRST file | Symbol |", "|---|---|"]
        lines += [
            f"| `{row['mrst_path']}` | `{row['symbol']}` |"
            for row in unvalidated
        ]
    else:
        lines.append("None.")
    lines += [
        "",
        "## Status policy",
        "",
        "- `ported`: a Python filename counterpart exists. MATLAB dimension "
        "tokens (`plotFaces2D`, `VEM3D_div`) are matched ignoring separators, "
        "so an honest filename port is not misreported as consolidated.",
        "- `ported-consolidated`: a reviewed Python module implements the "
        "MATLAB function as part of a shared or public consumer.",
        "- `excluded-mex`: a MATLAB MEX helper with no Python analogue, "
        "because PyMRST's AD objects are built directly.",
        "- `pending-review`: no defensible counterpart or exclusion exists; "
        "the `notes` column names the plan phase that retires it.",
        "- `validated-reference`: a non-ledger test or validation script names "
        "the MATLAB symbol or Python spelling; validation depth remains "
        "subject to review.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python3 scripts/audit_vemmech_parity.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrst-root",
        type=Path,
        default=Path("../mrst-2024a-Geomech/modules/vemmech"),
    )
    parser.add_argument(
        "--python-root",
        type=Path,
        default=Path("pymrst/modules/vemmech"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/vemmech/parity.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("VEMMECH_PARITY.md"),
    )
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    matlab_root = args.mrst_root.resolve()
    python_root = args.python_root.resolve()
    if not matlab_root.is_dir():
        parser.error(f"MRST vemmech tree not found: {matlab_root}")
    if not python_root.is_dir():
        parser.error(f"PyMRST vemmech tree not found: {python_root}")

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
