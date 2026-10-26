#!/usr/bin/env python3
"""Generate the MRST ``ad-blackoil`` to PyMRST parity ledger."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


FIELDS = (
    "mrst_path", "symbol", "subsystem", "port_status",
    "python_counterpart", "port_evidence", "mrst_methods",
    "methods_absent", "depth", "validation_status",
    "validation_files", "notes",
)

# A matching filename is insufficient for these examples: the current Python
# module explicitly refuses the workflow or implements only a reduced subset.
PLACEHOLDERS = {
}

PARTIAL = {
}

# Validated ports whose reference comparison is not exact throughout, and why.
NOTES = {
    "blackoilTutorialMexAcceleration": (
        "20x20x10: the run with a direct solver identical in iterations, states to 3e-13; "
        "AMGCL_CPRSolverAD's prepared system (matrix, rhs, DRS weights) MRST's to 1e-14; the "
        "AMGCL solve is MRST's own C++ (amgcl_matlab.cpp) compiled for Python, as MRST's MEX "
        "cannot be built here; no OpenMP, so the 4-thread run is serial"
    ),
    "blackoilTimeIntegrationExample": (
        "name = 'spe1' (the script's default): fully implicit, explicit and AIM over 120 steps "
        "with identical iterations and ministeps, AIM's implicit fractions identical, states to "
        "1e-7. 'spe9': fully implicit and AIM over 35 steps likewise, states to 7e-7; explicit "
        "identical through step 7, then parted by GMRES noise on singular systems - MRST parts "
        "from itself the same way on a one-ulp change - and both stop on the substep limit"
    ),
    "tenthCSP_Model_I": (
        "setup to 1e-14, steps 1-2 to 1e-13, step 3 from MRST's step-2 state to 3e-10; the full "
        "131 steps with identical ministeps and controls, iterations 603 vs 609, states to 6e-5 - "
        "the direct solvers' round-off sign in sG at the SGOF knot so = 1 parts the runs in step 3"
    ),
    "setupAquifertest": (
        "MRST 2024a's function cannot finish (AquiferBlackOilModel no longer exists); the "
        "port stops with the same error, and its steps before it match MRST exactly"
    ),
    "fieldModelNorneExample": (
        "initial state to 1e-12, first system and first Newton update to round-off; "
        "direct solver: step 1 MRST's iterations, pressure 1e-6; step 2 8 vs 9 iterations, "
        "pressure 2e-5 - MATLAB's complex rv (RvMax's power of a negative factor) in up to six "
        "cells, of which the port keeps the real part; CPR/GMRES well curves within 1e-4"
    ),
    "simulatorWorkflowExample": (
        "MRST's makeModel3 draws passed in (MATLAB's randn cannot be reproduced); fine, simply "
        "upscaled and flow-upscaled runs with identical iterations, states to 1e-9; upscaleTrans "
        "to 4e-11 (fine pressure-solve round-off); diagnostics to 4e-12; the closing "
        "interactiveDiagnostics viewer is not ported, its computations (well pairs, F-Phi, Lorenz, "
        "sweep, well TOFs, first arrival) are, validated in validate_diagnostics_utils.py"
    ),
    "blackoilTutorialSPE9": (
        "steps 1-17 identical iterations, states to 1e-5; step 18 converges or cuts on a "
        "1-ulp change in MRST itself, so it is checked on MRST's ministeps (to 1e-9) and "
        "steps 18-35 are reported"
    ),
}

REVIEWED_NAMES = {
    "getbG_BO": "pymrst/modules/ad_blackoil/utils/get_b_g_bo.py",
    "getbO_BO": "pymrst/modules/ad_blackoil/utils/get_b_o_bo.py",
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def token(symbol: str) -> re.Pattern:
    """Match a symbol or its snake_case form, case-insensitively.

    Known false positive, left in place because changing it silently rewrites
    every module's ledger: generic MATLAB names collide with ordinary code. The
    nine `Contents.m` catalog entries are all marked validated-reference on the
    strength of `from pymrst.modules.shale.examples.contents import ...`, which
    is a different module's contents and validates nothing. Those rows carry
    port_status `excluded-catalog`, so a validation status on them is
    meaningless either way; treat validation_files on a catalog row as noise.
    """
    variants = sorted({symbol, snake_case(symbol)})
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{'|'.join(map(re.escape, variants))})(?![A-Za-z0-9_])",
        re.I,
    )


def source_index(paths):
    return {p: p.read_text(encoding="utf-8", errors="ignore") for p in paths}


def relative_list(paths, root):
    return "; ".join(str(p.relative_to(root)) for p in sorted(set(paths)))


def classify(matlab_file, matlab_root, repo, python_sources):
    rel = matlab_file.relative_to(matlab_root)
    symbol = matlab_file.stem
    if symbol == "Contents":
        return "excluded-catalog", "", "MATLAB package catalog"

    if symbol in REVIEWED_NAMES:
        counterpart = REVIEWED_NAMES[symbol]
        if not (repo / counterpart).is_file():
            raise FileNotFoundError(f"reviewed counterpart is missing: {counterpart}")
        return "ported", counterpart, "reviewed Python naming"

    stem = snake_case(symbol)
    exact = [path for path in python_sources if path.stem == stem]
    counterpart = relative_list(exact, repo)
    if symbol in PLACEHOLDERS:
        return "pending", counterpart, PLACEHOLDERS[symbol]
    if symbol in PARTIAL:
        return "partial", counterpart, PARTIAL[symbol]
    if exact:
        return "ported", counterpart, "filename"
    return "pending", "", "no reviewed Python owner"


def matlab_methods(path: Path) -> list:
    """Method names of a MATLAB `classdef`, or `[]` for a plain function file.

    A `.m` file that is not a class has no protocol to be missing, so it is
    excluded rather than counted as depth 1.00 - which would let plain
    functions inflate the average and hide a shallow class.
    """
    text = path.read_text(errors="ignore")
    if not re.search(r"^\s*classdef\b", text, re.M):
        return []
    return sorted(set(re.findall(
        r"^\s*function\s+(?:\[[^\]]*\]\s*=\s*|\w[\w.]*\s*=\s*)?(\w+)\s*\(",
        text, re.M)))


def python_names(python_root: Path) -> set:
    """Every name defined or aliased anywhere in the *whole* Python package.

    Not just `ad_blackoil`. A method inherited from `ad_core` - and after the
    `ThreePhaseBlackOilModel` migration most of that protocol is inherited -
    is genuinely available on the class, so scanning only this module reported
    five methods absent that `hasattr` finds. The scan follows what a caller
    can actually call.
    """
    names = set()
    root = python_root
    while root.name != "pymrst" and root.parent != root:
        root = root.parent
    for f in root.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        text = f.read_text(errors="ignore")
        names |= set(re.findall(r"^\s*(?:def|class)\s+(\w+)", text, re.M))
        names |= set(re.findall(r"^\s*(\w+)\s*=\s*[A-Za-z_][\w.]*\s*$", text, re.M))
    return names


def _defined(method: str, defined: set) -> bool:
    """Whether a MATLAB method name is reachable under any Python spelling.

    The `get_`-dropped form matters: this port names accessors after the thing
    they return, so `getComponentDensity` is `component_density`. Without it
    the ledger reports a method absent that is defined two lines away.
    """
    snake = snake_case(method)
    candidates = {method, snake}
    for prefix in ("get_", "set_"):
        if snake.startswith(prefix):
            candidates.add(snake[len(prefix):])
    return bool(candidates & defined)


def measure_depth(matlab_file: Path, defined: set) -> tuple:
    """`(n_methods, absent, depth)` for one MATLAB class.

    This is the check the ledger did not have, and its absence is why 20
    absent methods sat unreported behind rows marked `ported`: a file-level
    scan asks whether a counterpart *exists*, which a class with one method
    out of fifteen satisfies as readily as a complete one.

    A constructor is skipped - MATLAB names it after the class, Python calls
    it `__init__`, and counting it would report every class as one method
    short forever.
    """
    methods = [m for m in matlab_methods(matlab_file)
               if m != matlab_file.stem]
    if not methods:
        return "", "", ""
    absent = [m for m in methods if not _defined(m, defined)]
    depth = (len(methods) - len(absent))/len(methods)
    return str(len(methods)), " ".join(absent), f"{depth:.2f}"


def generate_rows(repo: Path, matlab_root: Path, python_root: Path):
    python_sources = source_index(sorted(python_root.rglob("*.py")))
    validation_paths = sorted((repo / "tests").rglob("*.py"))
    validation_paths += sorted((repo / "examples" / "validation").rglob("*.py"))
    validation_paths = [p for p in validation_paths if "parity" not in p.stem]
    validations = source_index(validation_paths)
    defined = python_names(python_root)
    rows = []
    for matlab_file in sorted(matlab_root.rglob("*.m")):
        rel = matlab_file.relative_to(matlab_root)
        symbol = matlab_file.stem
        status, counterpart, evidence = classify(
            matlab_file, matlab_root, repo, python_sources
        )
        n_methods, absent, depth = measure_depth(matlab_file, defined)
        pattern = token(symbol)
        validation_files = [p for p, text in validations.items() if pattern.search(text)]
        rows.append({
            "mrst_path": str(rel),
            "symbol": symbol,
            "subsystem": rel.parts[0] if len(rel.parts) > 1 else "root",
            "port_status": status,
            "python_counterpart": counterpart,
            "port_evidence": evidence,
            "mrst_methods": n_methods,
            "methods_absent": absent,
            "depth": depth,
            "validation_status": "validated-reference" if validation_files else "unvalidated",
            "validation_files": relative_list(validation_files, repo),
            "notes": NOTES.get(symbol, ""),
        })
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    counts = Counter(row["port_status"] for row in rows)
    validation = Counter(row["validation_status"] for row in rows)
    incomplete = [r for r in rows if r["port_status"] in {"pending", "partial"}]
    lines = [
        "# AD-blackoil parity ledger", "",
        "Generated by `scripts/audit_ad_blackoil_parity.py` from the local MRST 2024a tree.",
        "The CSV ledger is authoritative; filename presence does not make a guarded example complete.",
        "", "## Current scan", "",
        f"- MRST MATLAB files inventoried: **{len(rows)}**",
        f"- Ported: **{counts['ported']}**",
        f"- Partial: **{counts['partial']}**",
        f"- Pending: **{counts['pending']}**",
        f"- Catalogs excluded: **{counts['excluded-catalog']}**",
        f"- Reference-validated entries: **{validation['validated-reference']}**",
        f"- Unvalidated entries: **{validation['unvalidated']}**",
        "", "## Incomplete executable entries", "",
    ]
    if incomplete:
        lines += [
            f"- `{row['mrst_path']}` — **{row['port_status']}**: {row['port_evidence']}"
            for row in incomplete
        ]
    else:
        lines.append("None.")
    lines += ["", "See `docs/ad_blackoil/parity.csv` for per-file evidence.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_mrst_root(repo):
    return repo.parent / "mrst-2024a-Geomech" / "modules" / "ad-blackoil"


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrst-root", type=Path, default=default_mrst_root(repo))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    python_root = repo / "pymrst" / "modules" / "ad_blackoil"
    rows = generate_rows(repo, args.mrst_root, python_root)
    csv_path = repo / "docs" / "ad_blackoil" / "parity.csv"
    md_path = repo / "AD_BLACKOIL_PARITY.md"
    if args.check:
        return int(any(r["port_status"] in {"pending", "partial"} for r in rows))
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path.relative_to(repo)} ({len(rows)} rows)")
    print(f"Wrote {md_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
