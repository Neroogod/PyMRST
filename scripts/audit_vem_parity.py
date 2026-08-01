#!/usr/bin/env python3
"""Generate the MRST ``vem`` to PyMRST parity ledger.

Port status and validation status are tracked independently: a Python file
existing is not evidence that its numbers are right.

This auditor adds a classification the other module auditors do not have -
**misplaced**. The `vem` port implements most of MRST's functions but keeps
several of them in files named after something else, so a filename scan
reports them missing while a symbol scan reports them present. Neither answer
is useful on its own. `misplaced` says what is actually true: the code exists,
the layout does not match, and moving it is a rename rather than a rewrite.

That distinction is the whole point of the ledger for this module, because it
separates the work that is genuinely outstanding from the work that is only
filed wrongly.
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

# Which phase of vem_implementation.md retires each outstanding entry.
PENDING_PHASE = {
    "incompVEM": "phase-2 solver at its own path",
    "computeVEMGeometry": "phase-1 layout",
    "polyDim": "phase-1 layout",
    "triangleQuadRule": "phase-1 layout",
    "tetrahedronQuadRule": "phase-1 layout",
    "polygonInt": "phase-1 layout",
    "polygonInt3D": "phase-1 layout",
    "polyhedronInt": "phase-1 layout",
    "addBCVEM": "phase-3 boundary conditions",
    "conserveFlux": "phase-4 flux postprocessing",
    "squeezeBlockDiag": "phase-1 layout",
    "producerInjectorGrid": "phase-5 examples",
    "vemExample1": "phase-5 examples",
    "vemExample2": "phase-5 examples",
}

# Python files with no MRST counterpart. Recorded so the ledger accounts for
# the whole tree, not only the parts MRST names.
EXTRA_NOTES = {
    "vem_poisson.py": "PyMRST-only Poisson demo; MRST solves through incompVEM",
}

_DOCSTRING = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_COMMENT = re.compile(r"#[^\n]*")


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _normalized(name: str) -> str:
    """Collapse to letters and digits, for spelling-insensitive matching.

    MATLAB embeds dimension and acronym tokens - ``polygonInt3D``,
    ``computeVEMGeometry`` - that no single snake-case rule spells the way a
    human would. Comparing without separators keeps an honest filename port
    out of the misplaced bucket, where it would overstate the work left.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _code_only(text: str) -> str:
    """Drop docstrings and comments so prose cannot pass as an implementation.

    A module that merely *mentions* `incompVEM` while explaining what it does
    must not be reported as implementing it.
    """
    return _COMMENT.sub("", _DOCSTRING.sub("", text))


_DEFINITION = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
                         re.M)


def _defines(text: str, symbol: str) -> bool:
    """Is ``symbol`` actually defined here, rather than referred to?

    A definition is `def name(` or `class name`. A call or an import is not:
    the file that *uses* `poly_dim` is not the file that provides it, and
    counting the former would put the counterpart in the wrong place.

    Names are compared with separators removed, because snake_case cannot
    reproduce every MATLAB spelling. `polygonInt3D` becomes `polygon_int3_d`
    under any positional rule, while the port is `polygon_int_3d`; comparing
    literally reports a function that exists as missing.
    """
    target = _normalized(symbol)
    return any(_normalized(name) == target
               for name in _DEFINITION.findall(_code_only(text)))


def _source_index(paths: list[Path]) -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8", errors="ignore") for p in paths}


def _token(symbol: str, *extra: str) -> re.Pattern:
    names = {symbol, snake_case(symbol)} | {v for v in extra if v}
    alternatives = "|".join(re.escape(v) for v in sorted(names))
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])", re.I)


def _relative(paths: list[Path], root: Path) -> str:
    return "; ".join(str(p.relative_to(root)) for p in sorted(set(paths)))


def classify(symbol: str, repo: Path,
             python_sources: dict[Path, str]) -> tuple[str, str, str]:
    if symbol == "Contents":
        return "excluded-catalog", "", "MATLAB package catalog"

    stem = snake_case(symbol)
    exact = [p for p in python_sources if p.stem == stem]
    if exact:
        return "ported", _relative(exact, repo), "filename"

    target = _normalized(symbol)
    near = [p for p in python_sources if _normalized(p.stem) == target]
    if near:
        return "ported", _relative(near, repo), "filename, token spelling"

    # Defined, but in a file named after something else.
    defined = [p for p, text in python_sources.items()
               if _defines(text, symbol)]
    if defined:
        return "misplaced", _relative(defined, repo), "symbol defined elsewhere"

    return "pending-review", "", ""


def generate_rows(repo: Path, matlab_root: Path,
                  python_root: Path) -> list[dict[str, str]]:
    python_sources = _source_index(sorted(python_root.rglob("*.py")))

    validation_files = sorted((repo / "tests").rglob("*.py"))
    validation_files += sorted((repo / "examples").rglob("*.py"))
    validation_files = [p for p in validation_files if "parity" not in p.stem]
    validation_sources = _source_index(validation_files)

    rows = []
    for matlab_file in sorted(matlab_root.rglob("*.m")):
        rel = matlab_file.relative_to(matlab_root)
        symbol = matlab_file.stem
        status, counterparts, evidence = classify(symbol, repo, python_sources)
        stems = [Path(v).stem for v in counterparts.split("; ") if v]
        token = _token(symbol, *stems)
        validations = [p for p, text in validation_sources.items()
                       if token.search(text)]
        # Code that does not exist cannot be validated. A validator may still
        # *name* an unported symbol - `validate_phase406` compares against an
        # MRST run that used `addBCVEM` - but that validates PyMRST's
        # substitute, not a port of the function, and recording it as
        # validated would hide the gap.
        if status == "pending-review":
            validations = []
        rows.append({
            "mrst_path": str(rel),
            "symbol": symbol,
            "subsystem": "/".join(rel.parts[:-1]) or "root",
            "port_status": status,
            "python_counterpart": counterparts,
            "port_evidence": evidence,
            "validation_status": ("validated-reference" if validations
                                  else "unvalidated"),
            "validation_files": _relative(validations, repo),
            "notes": PENDING_PHASE.get(symbol, "")
            if status in ("pending-review", "misplaced") else "",
        })
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], destination: Path,
                   python_root: Path, repo: Path) -> None:
    """Render the ledger to ``destination``.

    The output path is named `destination`, not `path`: an earlier version of
    this function used `path` and also looped `for path in ...` while building
    the tables, which rebound the name and wrote the ledger over a source file
    in `pymrst/modules/vem/`. Loop variables here must not be called `path`.
    """
    counts = Counter(row["port_status"] for row in rows)
    production = [r for r in rows if r["port_status"] != "excluded-catalog"]
    ported = counts["ported"]
    misplaced = counts["misplaced"]
    pending = counts["pending-review"]
    unvalidated = [r for r in production
                   if r["validation_status"] == "unvalidated"]

    lines = [
        "# vem parity ledger",
        "",
        "Generated by `scripts/audit_vem_parity.py` from the local MRST 2024a "
        "reference tree. The CSV ledger is authoritative; port and validation "
        "status are tracked independently. The plan that retires the "
        "outstanding entries is `vem_implementation.md`.",
        "",
        "## Current scan",
        "",
        f"- MRST MATLAB files inventoried: **{len(rows)}**",
        f"- Production files (excluding catalogs): **{len(production)}**",
        f"- At an MRST-shaped path: **{ported}**",
        f"- **Implemented but misplaced: {misplaced}**",
        f"- Not implemented: **{pending}**",
        f"- Catalogs excluded: **{counts['excluded-catalog']}**",
        f"- Production entries still unvalidated: **{len(unvalidated)}**",
        "",
        f"**{ported} of {len(production)} production files sit at an "
        f"MRST-shaped path.**",
        "",
        "`misplaced` means the code exists and is reachable, but under a "
        "filename MRST does not use. It is a rename, not a rewrite - which is "
        "why it is counted apart from work that has not been done at all.",
        "",
        "## Outstanding work by plan phase",
        "",
        "| Phase | Entries |",
        "|---|---:|",
    ]
    phases = Counter(r["notes"] for r in production if r["notes"])
    for phase, count in sorted(phases.items()):
        lines.append(f"| {phase} | {count} |")

    for title, status in (("Implemented but misplaced", "misplaced"),
                          ("Not implemented", "pending-review")):
        entries = [r for r in rows if r["port_status"] == status]
        lines += ["", f"## {title}", ""]
        if not entries:
            lines.append("None.")
            continue
        lines += ["| MRST file | Symbol | Currently at | Phase |",
                  "|---|---|---|---|"]
        for row in entries:
            where = row["python_counterpart"] or "-"
            lines.append(f"| `{row['mrst_path']}` | `{row['symbol']}` | "
                         f"`{where}` | {row['notes'] or '-'} |")

    lines += ["", "## Python files with no MRST counterpart", ""]
    named = [f for f in python_root.rglob("*.py") if f.name in EXTRA_NOTES]
    if named:
        lines += ["| File | Note |", "|---|---|"]
        for source in sorted(named):
            lines.append(f"| `{source.relative_to(repo)}` | "
                         f"{EXTRA_NOTES[source.name]} |")
    else:
        lines.append("None.")

    lines += ["", "## Full ledger", "",
              "| MRST file | Status | Python counterpart | Validation |",
              "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| `{row['mrst_path']}` | {row['port_status']} | "
            f"`{row['python_counterpart'] or '-'}` | "
            f"{row['validation_status']} |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--mrst", type=Path,
        default=Path("/Users/Shared/PyMRST_workspace"
                     "/mrst-2024a-Geomech/modules/vem"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    python_root = repo / "pymrst" / "modules" / "vem"
    if not args.mrst.is_dir():
        print(f"MRST reference tree not found: {args.mrst}")
        return 2

    rows = generate_rows(repo, args.mrst, python_root)
    write_csv(rows, repo / "docs" / "vem" / "parity.csv")
    write_markdown(rows, repo / "VEM_PARITY.md", python_root, repo)
    print(f"Wrote docs/vem/parity.csv ({len(rows)} rows)")
    print("Wrote VEM_PARITY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
