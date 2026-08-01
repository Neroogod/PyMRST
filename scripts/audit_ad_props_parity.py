#!/usr/bin/env python3
"""Generate the MRST ``ad-props`` to PyMRST parity ledger.

Port status and validation status are tracked independently, and a filename
match is not by itself evidence of a port. This module in particular carries
several MRST-named files that only re-export an implementation living
elsewhere, so those are reported as ``ported-shim``: a filename-only scanner
would score them as complete and overstate the module considerably.

Pending entries carry the phase of ``ad-prop_implementation.md`` that retires
them, so the ledger and the plan cannot drift apart.
"""

from __future__ import annotations

import argparse
import ast
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

# Reviewed mappings for helpers intentionally folded into a Python consumer
# rather than given a module of their own.
CONSOLIDATED: dict[str, tuple[str, str]] = {}

EXCLUSIONS: dict[str, tuple[str, str]] = {}

# Which phase of ad-prop_implementation.md retires each outstanding entry.
_PHASE3_UTILS = (
    "interpPVT", "interpRegular2DTable", "interp2DTable", "handleRegions",
    "getRegMap", "interp2DPVT", "interpReg", "interpRegPVT",
    "preprocessTablePVT", "extendTab", "dinterpReg",
)
_PHASE5_PVT = (
    "assignPVTO", "assignPVTG", "assignPVTW", "assignPVDO", "assignPVDG",
    "assignPVDS", "assignPVCDO", "assignRSCONSTT", "assignDENSITY",
    "assignSDENSITY", "assignGRAVITY", "assignROCK", "assignROCKTAB",
)
_PHASE5_SATFUNC = (
    "assignSWOF", "assignSGOF", "assignSLGOF", "assignSWFN", "assignSGFN",
    "assignSGWFN", "assignSOF2", "assignSOF3", "assignSSFN", "assignMSFN",
    "assignRelPerm", "assignRelPermScal", "assignEHYSTR",
)
_PHASE6_EOR = (
    "assignPLYADS", "assignPLYMAX", "assignPLYROCK", "assignPLYSHEAR",
    "assignPLYSHLOG", "assignPLYVISC", "assignPLMIXPAR", "assignTLMIXPAR",
    "assignSHRATE", "assignSURFADS", "assignSURFCAPD", "assignSURFROCK",
    "assignSURFST", "assignSURFVISC", "assignMISC", "assignPMISC",
    "assignSPECHEAT", "assignSPECROCK", "assignWATERVIST", "assignOILVIST",
    "assignVISCREF",
)
_PHASE7_DISPATCH = (
    "initDeckADIFluid", "initSimpleADIFluid", "imposeRelpermScaling",
)

PENDING_PHASE = {
    **{name: "phase-3 interpolation and region core" for name in _PHASE3_UTILS},
    **{name: "phase-5 core PVT" for name in _PHASE5_PVT},
    **{name: "phase-5 saturation functions" for name in _PHASE5_SATFUNC},
    **{name: "phase-6 EOR and thermal" for name in _PHASE6_EOR},
    **{name: "phase-7 dispatch and scaling" for name in _PHASE7_DISPATCH},
    "coreyPhaseRelpermAD": "phase-2 layout refactor",
    "tableByPressureLinearAD": "phase-2 layout refactor",
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _normalized(name: str) -> str:
    """Collapse to letters and digits, for spelling-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_DOCSTRING = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_COMMENT = re.compile(r"#[^\n]*")


def _code_only(text: str) -> str:
    """Drop docstrings and comments so prose cannot pass as an implementation.

    A module that merely *mentions* `assignPVTO` while explaining what it is
    for must not be reported as implementing it.
    """
    return _COMMENT.sub("", _DOCSTRING.sub("", text))


def is_reexport_shim(path: Path, package: str) -> tuple[bool, str]:
    """Is this module only forwarding to an implementation elsewhere?

    True when the module defines nothing of its own, or when every function it
    defines is a one-line forward and it imports from a sibling module. Both
    shapes satisfy a filename match while adding no implementation, which is
    exactly what this ledger must not count as a port.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return False, ""

    imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        and package in node.module
    }
    if not imports:
        return False, ""

    definitions = [n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef))]
    target = "; ".join(sorted(imports))
    if not definitions:
        return True, f"re-exports {target}"

    def trivial(node) -> bool:
        body = [s for s in node.body if not (isinstance(s, ast.Expr)
                                             and isinstance(s.value, ast.Constant))]
        return len(body) <= 1
    if all(isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and trivial(d)
           for d in definitions):
        return True, f"forwards to {target}"
    return False, ""


def _source_index(paths: list[Path]) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in paths
    }


def _token(symbol: str, *extra: str) -> re.Pattern:
    """Match `symbol` under any of its spellings, including its real ports."""
    names = {symbol, snake_case(symbol)} | {value for value in extra if value}
    alternatives = "|".join(re.escape(value) for value in sorted(names))
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])",
        re.I,
    )


def _relative_list(paths: list[Path], root: Path) -> str:
    return "; ".join(str(path.relative_to(root)) for path in sorted(set(paths)))


def classify(
    matlab_file: Path,
    repo: Path,
    python_sources: dict[Path, str],
    package: str,
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

    target = _normalized(symbol)
    stem = snake_case(symbol)
    matches = [path for path in python_sources
               if path.stem == stem or _normalized(path.stem) == target]
    if matches:
        shims = [(path,) + is_reexport_shim(path, package) for path in matches]
        if all(flag for _, flag, _ in shims):
            reason = next(note for _, _, note in shims if note)
            return "ported-shim", _relative_list(matches, repo), reason
        real = [path for path, flag, _ in shims if not flag]
        return "ported", _relative_list(real, repo), "filename"

    token = _token(symbol)
    mentions = [path for path, text in python_sources.items()
                if token.search(_code_only(text))]
    if mentions:
        return ("ported-consolidated", _relative_list(mentions, repo),
                "explicit symbol reference")
    return "pending-review", "", ""


def generate_rows(repo: Path, matlab_root: Path,
                  python_root: Path) -> list[dict[str, str]]:
    python_files = sorted(python_root.rglob("*.py"))
    python_sources = _source_index(python_files)
    package = python_root.name

    validation_files = sorted((repo / "tests").rglob("*.py"))
    validation_files += sorted((repo / "examples" / "validation").rglob("*.py"))
    validation_files = [p for p in validation_files if "parity" not in p.stem]
    validation_sources = _source_index(validation_files)

    rows = []
    for matlab_file in sorted(matlab_root.rglob("*.m")):
        rel = matlab_file.relative_to(matlab_root)
        symbol = matlab_file.stem
        status, counterparts, evidence = classify(
            matlab_file, repo, python_sources, package
        )
        stems = [Path(v).stem for v in counterparts.split("; ") if v]
        token = _token(symbol, *stems)
        validations = [path for path, text in validation_sources.items()
                       if token.search(text)]
        note = (PENDING_PHASE.get(symbol, "")
                if status in {"pending-review", "ported-shim"} else "")
        rows.append({
            "mrst_path": str(rel),
            "symbol": symbol,
            "subsystem": "/".join(rel.parts[:-1]) or "root",
            "port_status": status,
            "python_counterpart": counterparts,
            "port_evidence": evidence,
            "validation_status": ("validated-reference" if validations
                                  else "unvalidated"),
            "validation_files": _relative_list(validations, repo),
            "notes": note,
        })
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
    outstanding = [row for row in rows
                   if row["port_status"] in {"pending-review", "ported-shim"}]
    shims = [row for row in rows if row["port_status"] == "ported-shim"]
    unvalidated = [row for row in rows
                   if row["port_status"] in {"ported", "ported-consolidated"}
                   and row["validation_status"] != "validated-reference"]
    phases = Counter(row["notes"] for row in outstanding)
    production = len(rows) - counts["excluded-catalog"]
    done = counts["ported"] + counts["ported-consolidated"]

    lines = [
        "# ad-props parity ledger",
        "",
        "Generated by `scripts/audit_ad_props_parity.py` from the local MRST "
        "2024a reference tree. The CSV ledger is authoritative; port and "
        "validation status are tracked independently. The plan that retires "
        "the outstanding entries is `ad-prop_implementation.md`.",
        "",
        "## Current scan",
        "",
        f"- MRST MATLAB files inventoried: **{len(rows)}**",
        f"- Production files (excluding catalogs): **{production}**",
        f"- Direct filename counterparts: **{counts['ported']}**",
        f"- Reviewed consolidated counterparts: "
        f"**{counts['ported-consolidated']}**",
        f"- **Re-export shims (counterpart in name only): "
        f"{counts['ported-shim']}**",
        f"- Pending implementation/review: **{counts['pending-review']}**",
        f"- Catalogs excluded: **{counts['excluded-catalog']}**",
        f"- Entries named by tests/reference validators: "
        f"**{validation['validated-reference']}**",
        f"- Ported entries still unvalidated: **{len(unvalidated)}**",
        "",
        f"**{done} of {production} production files are implemented.**",
        "",
        "## Outstanding work by plan phase",
        "",
    ]
    if outstanding:
        lines += ["| Phase | Entries |", "|---|---:|"]
        for phase, count in sorted(phases.items()):
            lines.append(f"| {phase or 'unassigned'} | {count} |")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Re-export shims",
        "",
        "These carry an MRST name but forward to an implementation living "
        "outside its MRST-shaped path. A filename-only scanner counts them as "
        "ported; this ledger does not. Phase 2 of the plan retires them.",
        "",
    ]
    if shims:
        lines += ["| MRST file | Python file | What it does |", "|---|---|---|"]
        lines += [f"| `{r['mrst_path']}` | `{r['python_counterpart']}` | "
                  f"{r['port_evidence']} |" for r in shims]
    else:
        lines.append("None.")

    lines += ["", "## Pending entries", ""]
    pending = [r for r in rows if r["port_status"] == "pending-review"]
    if pending:
        lines += ["| MRST file | Symbol | Phase |", "|---|---|---|"]
        lines += [f"| `{r['mrst_path']}` | `{r['symbol']}` | "
                  f"{r['notes'] or 'unassigned'} |" for r in pending]
    else:
        lines.append("None.")

    lines += ["", "## Ported but unvalidated", ""]
    if unvalidated:
        lines += ["| MRST file | Symbol |", "|---|---|"]
        lines += [f"| `{r['mrst_path']}` | `{r['symbol']}` |"
                  for r in unvalidated]
    else:
        lines.append("None.")

    lines += [
        "",
        "## Status policy",
        "",
        "- `ported`: a Python file at the MRST-shaped path implements the "
        "symbol. MATLAB dimension tokens are matched ignoring separators, so "
        "an honest filename port is not misreported.",
        "- `ported-shim`: a file with the right name that only re-exports or "
        "forwards. Detected by parsing the module: it defines nothing, or "
        "every function it defines is a one-line forward to a sibling import. "
        "**Not counted as implemented.**",
        "- `ported-consolidated`: a reviewed Python module implements the "
        "MATLAB function as part of a shared or public consumer. Matched "
        "against code only - docstrings and comments are stripped first, so "
        "prose describing a symbol cannot pass as an implementation.",
        "- `pending-review`: no counterpart or exclusion exists; the `notes` "
        "column names the plan phase that retires it.",
        "- `validated-reference`: a non-ledger test or validation script names "
        "the symbol; validation depth remains subject to review.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python3 scripts/audit_ad_props_parity.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrst-root", type=Path,
                        default=Path("../mrst-2024a-Geomech/modules/ad-props"))
    parser.add_argument("--python-root", type=Path,
                        default=Path("pymrst/modules/ad_props"))
    parser.add_argument("--csv", type=Path,
                        default=Path("docs/ad_props/parity.csv"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("AD_PROPS_PARITY.md"))
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    matlab_root = args.mrst_root.resolve()
    python_root = args.python_root.resolve()
    if not matlab_root.is_dir():
        parser.error(f"MRST ad-props tree not found: {matlab_root}")
    if not python_root.is_dir():
        parser.error(f"PyMRST ad_props tree not found: {python_root}")

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
