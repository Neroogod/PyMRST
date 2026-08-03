#!/usr/bin/env python3
"""Generate the MRST ``utils`` to PyMRST parity ledger.

MRST's `utils/` is 469 files, and most of them are not numerics. A ledger that
counted all of them would report a number that means nothing: PyMRST will never
port a vendored Boost Graph Library binding or MATLAB's module-path machinery,
and pretending otherwise buries the work that is real.

So exclusions here are **explicit and reasoned**, one entry per directory or
symbol, and they are part of the ledger's output rather than a filter applied
before it. A reader can disagree with any of them by name.

Three port states are distinguished, because for this module the difference
matters more than a single count:

- ``ported`` - a counterpart at the MRST-shaped path under ``pymrst/utils``;
- ``ported-elsewhere`` - a counterpart exists in PyMRST but at another path,
  which is either a deliberate deviation or a layout bug, and either way needs
  a decision rather than a checkmark;
- ``consolidated`` - the symbol is defined but no file bears its name.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

FIELDS = ("mrst_path", "symbol", "subsystem", "port_status",
          "python_counterpart", "port_evidence", "validation_status",
          "validation_files", "notes")

# Whole directories with no Python counterpart, and why not.
EXCLUDED_DIRS = {
    "3rdparty": "vendored MATLAB BGL (Boost Graph Library) bindings; "
                "Python uses scipy.sparse.csgraph and networkx",
    "octave_only": "Octave compatibility shims for MATLAB builtins",
    "mex": "MEX compilation helpers; PyMRST has no compiled extensions",
    "paper_manager": "bibliography of MRST publications",
}

# Individual root files that are MATLAB-environment plumbing, not numerics.
EXCLUDED_SYMBOLS = {
    # Its own comment: "This function exists only as an Octave compatibility
    # shim.  MATLAB's REGEXP is already sufficiently capable." Python's bare
    # str.split() splits on runs of whitespace and drops empties, which is
    # exactly what the shim provides, so there is nothing to port.
    "splitString": "Octave shim for whitespace splitting; str.split() is this",
    # readVector dispatches on the platform: readVectorOld under Octave, the
    # TEXTSCAN implementation under MATLAB. PyMRST targets neither - readVector
    # itself is consolidated into deckformat's reader - so both halves of that
    # dispatch are accounted for there rather than ported separately.
    "readVectorOld": "Octave branch of readVector; PyMRST does not target Octave",
    "readVector_textscan": "MATLAB branch of readVector, which is itself "
                           "consolidated into deckformat's array reader",
    # Its own Contents.m calls it "simplified version of"; the current reader is
    # modules/deckformat/deckinput/readSCHEDULE.m, and nothing but that catalog
    # entry references it. PyMRST reads schedules in read_eclipse_deck.py.
    "readSCHEDULE_orig": "superseded by readSCHEDULE; referenced only from "
                         "Contents.m",
    "ROOTDIR": "MATLAB install path", "buildmex": "MEX compilation",
    "mrstModule": "MATLAB path management",
    "mrstPath": "MATLAB path management",
    "mrstExtraDirs": "MATLAB path management",
    "mrstExamples": "MATLAB example discovery",
    "mrstStartupMessage": "MATLAB console banner",
    "mrstPlatform": "MATLAB platform query",
    "mrstIsLiveEditorDir": "MATLAB editor query",
    "mrstDebug": "MATLAB debug flag", "mrstVerbose": "MATLAB verbosity flag",
    "mrstConfigureMETISLocations": "external solver configuration",
    "mrstDefaultMexFlags": "MEX compilation",
    "mrstOutputDirectory": "MATLAB output path",
    "mrstDataDirectory": "MATLAB data path",
    "mrstWebSave": "MATLAB download wrapper",
    "githubDownload": "MATLAB download wrapper",
    "md5sum_fallback": "MATLAB checksum shim; Python has hashlib",
    "getCanonicalPath": "MATLAB path normalisation; Python has pathlib",
    "findFilesSubfolders": "MATLAB directory walk; Python has pathlib.rglob",
    "isBatchMode": "MATLAB session query",
    "moduleGUI": "MATLAB GUI", "ternaryAxis": "MATLAB plotting axis",
    "tetrahedralAxis": "MATLAB plotting axis",
    "merge_options": "MATLAB option-struct parsing; Python uses keyword "
                     "arguments, which is why no PyMRST function takes one",
    "merge_options_relaxed": "as merge_options",
    "mrstNargInCheck": "MATLAB argument-count check; Python raises TypeError",
    "msgid": "MATLAB warning identifier construction",
    "require": "MATLAB module dependency assertion",
    "dispif": "conditional printing; Python uses logging",
    "ticif": "conditional timing", "tocif": "conditional timing",
    "propertynames": "MATLAB class introspection",
    "evalFunWrapper": "MATLAB function-handle plumbing",
    "evalFunStandalone": "MATLAB function-handle plumbing",
    "applyFunction": "MATLAB cellfun wrapper; Python has comprehensions",
    "interpTableMEX": "MEX build of interpTable",
    "mrstExpandModuleAlias": "MATLAB module aliasing",
    "mrstExploreModules": "MATLAB GUI", "mrstSplitText": "MATLAB GUI text",
    "normaliseTextBox": "MATLAB GUI text",
    "removeCarriageReturn": "MATLAB GUI text",
}

# Which phase of utils_implementation.md retires each outstanding entry.
PHASE = {
    "units": "phase-3 units",
    "gridtools": "phase-4 grid tools",
    "inout": "phase-5 deck io",
    "tensorAssembly": "phase-6 tensor assembly",
    "equil": "phase-7 remainder",
    "explore": "phase-7 remainder",
    "ad_utils": "phase-7 remainder",
    "private": "phase-7 remainder",
    "(root)": "phase-2 core numerics",
}

# Name collisions between unrelated things. Checked before filename matching,
# because a filename scan cannot tell a unit of force from a nonlinear solver.
FALSE_POSITIVES = {
    "units/Newton.m": "pymrst/backends/newton.py is the Newton solver, not "
                      "the SI unit of force",
    "units/Pascal.m": "any pascal.py would be the unit, but check before "
                      "trusting a match",
}

# Counterparts placed away from MRST's layout on purpose.
REVIEWED_DEVIATIONS = {
    "boundaryFaces": "MRST files it under plotting/; placed in utils/gridtools "
                     "so importing it does not pull in the PyVista stack",
    "ADI": "one MATLAB classdef becomes a Python package: utils/adi/ holds the "
           "class, its operators and initVariablesADI",
    "initVariablesADI": "part of the utils/adi package, with the ADI class it "
                        "constructs",
    "value": "MRST keeps it at utils/ root; it strips AD derivatives and "
             "nothing else, so it belongs with the ADI class that defines them",
    "numelValue": "as value: AD-aware, so it lives beside the ADI class",
}

_DOCSTRING = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_COMMENT = re.compile(r"#[^\n]*")
_DEFINITION = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)", re.M)
# Module-level constants count as definitions. MRST's `units/` is one file per
# unit - `darcy.m`, `barsa.m` - and PyMRST states them as constants in a single
# module, so a def/class-only scan reports 48 units as missing when they are
# all present.
_CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=", re.M)


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def variants(name: str) -> set:
    """Every spelling a counterpart might reasonably use.

    MRST writes "to" as a digit - `faceFlux2cellFlux`, `transform3Dto2Dgrid` -
    and Python spells it out. Without this the three flux converters, which
    exist and are exercised, were reported missing.
    """
    base = normalized(name)
    out = {base}
    # a lone 2 between letters is "to"
    out.add(re.sub(r"(?<=[a-z])2(?=[a-z])", "to", base))
    out.add(re.sub(r"(?<=[a-z])to(?=[a-z])", "2", base))
    return out


def code_only(text: str) -> str:
    return _COMMENT.sub("", _DOCSTRING.sub("", text))


def classify(rel: Path, repo: Path, python_files, definitions):
    symbol = rel.stem
    top = rel.parts[0] if len(rel.parts) > 1 else "(root)"

    if top in EXCLUDED_DIRS:
        return "excluded-directory", "", EXCLUDED_DIRS[top]
    if symbol == "Contents":
        return "excluded-catalog", "", "MATLAB package catalog"
    if symbol in EXCLUDED_SYMBOLS:
        return "excluded-environment", "", EXCLUDED_SYMBOLS[symbol]

    key = normalized(symbol)
    if str(rel) in FALSE_POSITIVES:
        # Fall through to the symbol index only; the filename match is known
        # to be a different thing entirely.
        defined = definitions.get(key, [])
        if defined:
            return "consolidated", relative(defined, repo), "symbol defined"
        return "pending-review", "", ""
    matches = python_files.get(key, [])
    if matches:
        # Is it at the path MRST's layout implies, under pymrst/utils?
        # Directories follow the same naming rule as filenames: MRST's
        # `tensorAssembly/` becomes `tensor_assembly/`, just as
        # `faceFlux2cellFlux.m` becomes `face_flux_to_cell_flux.py`. Every
        # other utils directory is already lowercase, so this is a no-op for
        # them.
        expected = Path("pymrst") / "utils" / Path(
            *(snake_case(part) for part in rel.parent.parts))
        at_path = [p for p in matches if p.parent == expected]
        if at_path:
            return "ported", relative(at_path, repo), "filename and location"
        under_utils = [p for p in matches
                       if p.parts[:2] == ("pymrst", "utils")]
        if under_utils:
            note = REVIEWED_DEVIATIONS.get(symbol, "")
            where = expected.as_posix().rstrip("/.")
            return ("ported" if note else "ported-elsewhere",
                    relative(under_utils, repo),
                    note or f"under pymrst/utils, but MRST's layout puts it "
                            f"at {where}")
        return ("ported-elsewhere", relative(matches, repo),
                "counterpart outside pymrst/utils")

    defined = definitions.get(key, [])
    if defined:
        return "consolidated", relative(defined, repo), "symbol defined"
    return "pending-review", "", ""


def relative(paths, repo: Path) -> str:
    return "; ".join(sorted(str(p) for p in set(paths)))


def build_index(repo: Path):
    files, definitions = {}, {}
    for path in (repo / "pymrst").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo)
        if path.stem != "__init__":
            for key in variants(path.stem):
                files.setdefault(key, []).append(rel)
        body = code_only(path.read_text(errors="ignore"))
        for name in _DEFINITION.findall(body) + _CONSTANT.findall(body):
            for key in variants(name):
                definitions.setdefault(key, []).append(rel)
    return files, definitions


def generate_rows(repo: Path, matlab_root: Path):
    files, definitions = build_index(repo)

    validation = []
    for root in ("tests", "examples"):
        validation += [p for p in (repo / root).rglob("*.py")
                       if "parity" not in p.stem]
    # The MATLAB-reference validators live in scripts/ and are the strongest
    # evidence there is, so they count too. Restricted to validate_*.py: the
    # auditors themselves name every symbol they classify and would otherwise
    # mark the whole tree validated.
    validation += sorted((repo / "scripts").glob("validate_*.py"))
    sources = {p: p.read_text(errors="ignore") for p in validation}

    rows = []
    for matlab_file in sorted(matlab_root.rglob("*.m")):
        rel = matlab_file.relative_to(matlab_root)
        symbol = matlab_file.stem
        status, counterpart, evidence = classify(rel, repo, files, definitions)

        names = {symbol, snake_case(symbol)}
        names |= {Path(v).stem for v in counterpart.split("; ") if v}
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])"
            % "|".join(re.escape(n) for n in sorted(names)), re.I)
        hits = ([p for p, text in sources.items() if pattern.search(text)]
                if status.startswith(("ported", "consolidated")) else [])

        top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        rows.append({
            "mrst_path": str(rel),
            "symbol": symbol,
            "subsystem": top,
            "port_status": status,
            "python_counterpart": counterpart,
            "port_evidence": evidence,
            "validation_status": ("validated-reference" if hits
                                  else "unvalidated"),
            "validation_files": relative(
                [p.relative_to(repo) for p in hits], repo),
            "notes": (PHASE.get(top, "") if status in
                      ("pending-review", "ported-elsewhere", "consolidated")
                      else ""),
        })
    return rows


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, destination: Path) -> None:
    counts = Counter(r["port_status"] for r in rows)
    excluded = (counts["excluded-directory"] + counts["excluded-catalog"]
                + counts["excluded-environment"])
    scope = [r for r in rows if not r["port_status"].startswith("excluded")]
    ported = counts["ported"]

    lines = [
        "# utils parity ledger",
        "",
        "Generated by `scripts/audit_utils_parity.py` from the local MRST "
        "2024a reference tree. The CSV ledger is authoritative. The plan that "
        "retires the outstanding entries is `utils_implementation.md`.",
        "",
        "## Scope",
        "",
        f"MRST's `utils/` holds **{len(rows)}** MATLAB files. Most are not "
        "numerics, and counting them would make the ledger meaningless, so "
        f"**{excluded}** are excluded by an explicit, reasoned rule and "
        f"**{len(scope)}** are in scope.",
        "",
        "| Exclusion | Files | Reason |",
        "|---|---:|---|",
    ]
    for name, reason in sorted(EXCLUDED_DIRS.items()):
        n = sum(1 for r in rows if r["subsystem"] == name)
        lines.append(f"| `{name}/` | {n} | {reason} |")
    lines.append(f"| MATLAB environment | {counts['excluded-environment']} | "
                 f"path, module, option and console plumbing with no Python "
                 f"analogue |")
    lines.append(f"| `Contents.m` | {counts['excluded-catalog']} | "
                 f"MATLAB package catalogs |")

    lines += [
        "",
        "## Current scan",
        "",
        f"- In scope: **{len(scope)}**",
        f"- At an MRST-shaped path under `pymrst/utils`: **{ported}**",
        f"- **Ported but elsewhere: {counts['ported-elsewhere']}**",
        f"- Consolidated into another module: **{counts['consolidated']}**",
        f"- Not implemented: **{counts['pending-review']}**",
        "",
        f"**{ported} of {len(scope)} in-scope files sit at an MRST-shaped "
        f"path.**",
        "",
        "`ported-elsewhere` is counted apart from both success and absence: "
        "the code exists and works, but a reader following MRST's layout will "
        "not find it. Each one is either a deliberate deviation with a stated "
        "reason or a layout bug, and the plan decides which.",
        "",
        "## In-scope work by subsystem",
        "",
        "| Subsystem | In scope | Ported | Elsewhere | Missing | Phase |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for subsystem in sorted({r["subsystem"] for r in scope}):
        group = [r for r in scope if r["subsystem"] == subsystem]
        lines.append(
            f"| `{subsystem}` | {len(group)} "
            f"| {sum(1 for r in group if r['port_status'] == 'ported')} "
            f"| {sum(1 for r in group if r['port_status'] == 'ported-elsewhere')} "
            f"| {sum(1 for r in group if r['port_status'] == 'pending-review')} "
            f"| {PHASE.get(subsystem, '-')} |")

    for title, status in (("Ported, but not where MRST puts it",
                           "ported-elsewhere"),
                          ("Consolidated into another module", "consolidated")):
        entries = [r for r in rows if r["port_status"] == status]
        lines += ["", f"## {title}", ""]
        if not entries:
            lines.append("None.")
            continue
        lines += ["| MRST file | Currently at | Note |", "|---|---|---|"]
        for row in entries:
            lines.append(f"| `{row['mrst_path']}` | "
                         f"`{row['python_counterpart']}` | "
                         f"{row['port_evidence']} |")

    lines += ["", "## In-scope entries", "",
              "| MRST file | Status | Python counterpart | Phase |",
              "|---|---|---|---|"]
    for row in scope:
        lines.append(f"| `{row['mrst_path']}` | {row['port_status']} | "
                     f"`{row['python_counterpart'] or '-'}` | "
                     f"{row['notes'] or '-'} |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mrst", type=Path,
                        default=Path("/Users/Shared/"
                                     "PyMRST_workspace/"
                                     "mrst-2024a-Geomech/utils"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not args.mrst.is_dir():
        print(f"MRST reference tree not found: {args.mrst}")
        return 2

    rows = generate_rows(repo, args.mrst)
    write_csv(rows, repo / "docs" / "utils" / "parity.csv")
    write_markdown(rows, repo / "UTILS_PARITY.md")
    print(f"Wrote docs/utils/parity.csv ({len(rows)} rows)")
    print("Wrote UTILS_PARITY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
